#!/usr/bin/env -S uv run --env-file .env
# /// script
# requires-python = ">=3.11"
# dependencies = ["PyGithub", "tqdm", "python-dotenv", "requests"]
# ///
"""Extract Java before/after pairs from RefactoringMiner benchmark data.
uv run scripts/extract_rminer_data.py --data /path/to/data.json --max-commits 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import requests
from dotenv import load_dotenv
from github import Auth, Github
from github.GithubException import (
    GithubException,
    RateLimitExceededException,
    UnknownObjectException,
)
from github.Repository import Repository
from tqdm import tqdm

# GitHub helpers -------------------------------------------------------------

gh_client: Github | None = None


@lru_cache(maxsize=256)
def get_repo(repo_name: str) -> Optional[Repository]:
    try:
        return gh_client.get_repo(repo_name) if gh_client else None
    except UnknownObjectException:
        return None


def repo_from_url(url: str) -> str:
    return url.replace("https://github.com/", "").replace(".git", "").strip("/")


def ensure_rate_limit(flag: list[bool]):
    if flag[0]:
        time.sleep(60)
        flag[0] = False


def fetch_commit(repo: Repository, sha: str, rate_flag: list[bool]):
    try:
        return repo.get_commit(sha)
    except RateLimitExceededException:
        rate_flag[0] = True
        return None
    except (UnknownObjectException, GithubException):
        return None


def iter_java_files(commit) -> Iterable:
    for file in commit.files:
        if file.filename.endswith(".java") and file.status != "added":
            yield file


def fetch_raw(headers: dict[str, str], repo: str, sha: str, path: str) -> Optional[str]:
    url = f"https://raw.githubusercontent.com/{repo}/{sha}/{path}"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.content.decode("utf-8", errors="replace")
    except requests.RequestException:
        return None


# Manifest helpers -----------------------------------------------------------


def load_manifest(path: Path) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    try:
        data = json.load(path.open())
    except json.JSONDecodeError:
        return [], set()
    pairs = data.get("pairs", [])
    seen = {item.get("commit_sha") for item in pairs if item.get("commit_sha")}
    return pairs, {sha for sha in seen if sha}


def save_manifest(path: Path, pairs: list[dict], config: dict, failures: list[dict]):
    payload = {
        "total_pairs": len(pairs),
        "failed_commits": len(failures),
        "config": config,
        "pairs": pairs,
        "failures": failures,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def reset_output_files(output_dir: Path) -> tuple[Path, Path]:
    manifest_path = output_dir / "manifest.json"
    failures_path = output_dir / "failures.json"
    manifest_path.write_text(
        json.dumps(
            {
                "total_pairs": 0,
                "failed_commits": 0,
                "config": {},
                "pairs": [],
                "failures": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    failures_path.write_text("[]\n", encoding="utf-8")
    return manifest_path, failures_path


def save_failures(path: Path, failures: list[dict]):
    path.write_text(json.dumps(failures, indent=2), encoding="utf-8")


# Core extraction ------------------------------------------------------------


def _fetch_pair(
    headers: dict[str, str],
    repo_name: str,
    parent_sha: str,
    sha: str,
    prev_name: str,
    current_name: str,
    status: str,
):
    before = fetch_raw(headers, repo_name, parent_sha, prev_name)
    after = fetch_raw(headers, repo_name, sha, current_name)
    if not before or not after or before == after:
        return None

    pid = hashlib.sha256(f"{sha}:{current_name}".encode()).hexdigest()[:12]
    return pid, prev_name, current_name, status, before, after


def process_commit(
    repo_name: str,
    commit_meta: dict,
    output_dir: Path,
    only_tp: bool,
    headers: dict[str, str],
    rate_flag: list[bool],
    failures: list[dict],
    pairs: list[dict],
    executor: ThreadPoolExecutor,
):
    repo = get_repo(repo_name)
    sha = commit_meta.get("sha1")
    if not repo or not sha:
        failures.append({"sha": sha or "unknown", "error": "repo_not_found"})
        return

    commit = fetch_commit(repo, sha, rate_flag)
    if not commit or not commit.parents:
        failures.append({"sha": sha, "error": "commit_not_found"})
        return

    parent_sha = commit.parents[0].sha
    refactorings = commit_meta.get("refactorings", [])
    if only_tp:
        refactorings = [r for r in refactorings if r.get("validation") == "TP"]
    if not refactorings:
        return

    ref_types = sorted({r.get("type", "") for r in refactorings})
    ref_descr = "\n".join(r.get("description", "") for r in refactorings)

    commit_dir = output_dir / "pairs" / sha
    commit_dir.mkdir(parents=True, exist_ok=True)

    futures = []
    for file in iter_java_files(commit):
        prev_name = file.previous_filename or file.filename
        futures.append(
            executor.submit(
                _fetch_pair,
                headers,
                repo_name,
                parent_sha,
                sha,
                prev_name,
                file.filename,
                file.status,
            )
        )

    for future in as_completed(futures):
        result = future.result()
        if not result:
            continue
        pid, prev_name, current_name, status, before_content, after_content = result
        (commit_dir / f"{pid}_before.java").write_text(before_content, encoding="utf-8")
        (commit_dir / f"{pid}_after.java").write_text(after_content, encoding="utf-8")

        pairs.append(
            {
                "id": pid,
                "commit_sha": sha,
                "parent_sha": parent_sha,
                "repository": commit_meta.get("repository"),
                "file_path": prev_name,
                "file_path_after": current_name,
                "refactoring_type": "|".join(t for t in ref_types if t),
                "refactoring_description": ref_descr,
                "status": status,
                "before_file": f"pairs/{sha}/{pid}_before.java",
                "after_file": f"pairs/{sha}/{pid}_after.java",
            }
        )


# Main entry point -----------------------------------------------------------


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    parser = argparse.ArgumentParser(description="Extract refactoring file pairs")
    parser.add_argument("--data", required=True, help="Path to data.json")
    parser.add_argument("--output", default="./rminer_data", help="Output directory")
    parser.add_argument("--max-commits", type=int, help="Limit commits processed")
    parser.add_argument(
        "--include-fp", action="store_true", help="Include false positives"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(4, (os.cpu_count() or 1) * 2),
        help="Number of threads used to download file contents",
    )
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 1

    output_dir = Path(args.output)
    (output_dir / "pairs").mkdir(parents=True, exist_ok=True)

    manifest_path, failures_path = reset_output_files(output_dir)
    existing_pairs: list[dict] = []
    seen_commits: set[str] = set()

    with open(args.data, encoding="utf-8") as fh:
        raw = json.load(fh)
    commits = list(raw.values()) if isinstance(raw, dict) else list(raw)
    commits = [c for c in commits if c.get("sha1") not in seen_commits]
    if args.max_commits:
        commits = commits[: args.max_commits]

    config = {
        "only_tp": not args.include_fp,
        "source": args.data,
        "max_commits": args.max_commits,
        "workers": args.workers,
    }

    if not commits:
        save_manifest(
            manifest_path,
            existing_pairs,
            config,
            [],
        )
        save_failures(failures_path, [])
        print("Nothing to do.")
        return 0

    global gh_client
    gh_client = Github(auth=Auth.Token(token), per_page=100)
    rate = gh_client.get_rate_limit()
    print(f"GitHub API: {rate}")

    headers = {
        "User-Agent": "smellai-rminer-extractor",
        "Accept": "application/vnd.github.v3.raw",
        "Authorization": f"Bearer {token}",
    }

    new_pairs: list[dict] = []
    failures: list[dict] = []
    rate_flag = [False]

    def flush_progress():
        save_manifest(manifest_path, existing_pairs + new_pairs, config, failures)
        save_failures(failures_path, failures)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for commit_meta in tqdm(commits, desc="Extracting"):
                repo_name = repo_from_url(commit_meta.get("repository", ""))
                if not repo_name:
                    failures.append(
                        {
                            "sha": commit_meta.get("sha1", "unknown"),
                            "error": "invalid_repo",
                        }
                    )
                    flush_progress()
                    continue
                process_commit(
                    repo_name,
                    commit_meta,
                    output_dir,
                    only_tp=not args.include_fp,
                    headers=headers,
                    rate_flag=rate_flag,
                    failures=failures,
                    pairs=new_pairs,
                    executor=executor,
                )
                flush_progress()
    except KeyboardInterrupt:
        print("Interrupted by user. Progress saved to manifest.", file=sys.stderr)
        return 130
    finally:
        if gh_client:
            gh_client.close()
        flush_progress()

    print(f"Pairs added: {len(new_pairs)}")
    print(f"Failures: {len(failures)}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
