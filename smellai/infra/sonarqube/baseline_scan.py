import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Consensus rule mapping (conceptual smell name)
RULE_NAME_MAP = {
    "java:S1541": "Complex Method",
    "java:S138": "Long Method",
    "java:S107": "Long Parameter List",
    "java:S1067": "Conditional Complexity",
    "java:S1200": "God Class",  # heuristic approximation
    "java:S110": "Large Class",
    "java:S1871": "Duplicated Conditions",
    "java:S106": "Print Statements",
}

SEVERITY_MAP = {
    "BLOCKER": "HIGH",
    "CRITICAL": "HIGH",
    "MAJOR": "MEDIUM",
    "MINOR": "LOW",
    "INFO": "LOW",
}

SONAR_URL = os.environ.get("SONAR_URL", "http://localhost:9000")
SONAR_TOKEN = os.environ.get("SONAR_TOKEN")

CUTOFF_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def run(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> str:
    """Run a shell command and return stdout."""
    proc = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if check and proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return proc.stdout.strip()


def derive_project_key(repo_url: str) -> str:
    # github.com/<org>/<repo>.git? -> org_repo
    m = re.search(
        r"github.com[:/](?P<org>[\w.-]+)/(?P<repo>[\w.-]+)(?:\.git)?", repo_url
    )
    if not m:
        raise ValueError(f"Cannot parse repo URL: {repo_url}")
    return f"{m.group('org')}_{m.group('repo')}".lower()


def get_last_commit_before(repo_url: str, before: datetime) -> str:
    """Clone shallow, fetch history, and find latest commit before date (UTC)."""
    tmp = Path(tempfile.mkdtemp(prefix="repo_"))
    try:
        run(["git", "clone", "--no-tags", "--depth", "2000", repo_url, str(tmp)])
        # Ensure we have enough depth; adapt if needed by checking oldest commit date.
        log = run(["git", "log", "--pretty=%H %cI"], cwd=tmp)
        chosen: Optional[str] = None
        for line in log.splitlines():
            sha, commit_date = line.split(" ", 1)
            dt = datetime.fromisoformat(commit_date.replace("Z", "+00:00"))
            if dt < before:
                chosen = sha
                break  # log is reverse chronological
        if not chosen:
            raise RuntimeError("No commit found before cutoff date")
        return chosen
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def write_sonar_properties(path: Path, project_key: str) -> None:
    content = (
        f"sonar.projectKey={project_key}\n"
        f"sonar.projectName={project_key}\n"
        "sonar.sources=.\n"
        f"sonar.host.url={SONAR_URL}\n"
        f"sonar.login={SONAR_TOKEN}\n"
        "sonar.java.binaries=.\n"  # Allow analysis without compiled classes
    )
    (path / "sonar-project.properties").write_text(content)


def poll_analysis_completion(project_key: str, timeout_sec: int = 600) -> None:
    """Poll SonarQube CE until the current analysis succeeds or timeout."""
    if not SONAR_TOKEN:
        raise RuntimeError("SONAR_TOKEN env not set")
    import time

    session = requests.Session()
    session.auth = (SONAR_TOKEN, "")

    # Try to get analysis ID from component endpoint
    try:
        resp = session.get(
            f"{SONAR_URL}/api/ce/component",
            params={"component": project_key},
            timeout=30,
        )
        resp.raise_for_status()
        analysis_id = resp.json().get("current", {}).get("analysisId")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            print(
                "[WARN] Insufficient permissions to poll analysis status. Waiting 30 seconds for server processing..."
            )
            time.sleep(30)
            return
        raise

    if not analysis_id:
        print(
            "[WARN] No current analysisId found. Analysis may have already completed."
        )
        return

    start = time.time()
    while True:
        task_resp = session.get(
            f"{SONAR_URL}/api/ce/task", params={"id": analysis_id}, timeout=30
        )
        task_resp.raise_for_status()
        status = task_resp.json().get("task", {}).get("status")
        if status == "SUCCESS":
            return
        if status in {"FAILED", "CANCELED"}:
            raise RuntimeError(f"Analysis ended with status {status}")
        if time.time() - start > timeout_sec:
            raise TimeoutError("Timeout waiting for analysis completion")
        time.sleep(5)


def fetch_issues(project_key: str) -> List[Dict[str, Any]]:
    if not SONAR_TOKEN:
        raise RuntimeError("SONAR_TOKEN env not set")
    session = requests.Session()
    session.auth = (SONAR_TOKEN, "")
    all_issues: List[Dict[str, Any]] = []
    page = 1
    rule_list = ",".join(RULE_NAME_MAP.keys())
    while True:
        resp = session.get(
            f"{SONAR_URL}/api/issues/search",
            params={
                "componentKeys": project_key,
                "types": "CODE_SMELL",
                "rules": rule_list,
                "p": page,
                "ps": 500,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("issues", [])
        all_issues.extend(batch)
        total = data.get("total", 0)
        if page * 500 >= total:
            break
        page += 1
    return all_issues


def normalize_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
    rule = issue.get("rule")
    smell_type = RULE_NAME_MAP.get(rule, rule)
    sev = SEVERITY_MAP.get(issue.get("severity"), "LOW")
    location = (
        f"{issue.get('component')}:{issue.get('line')}"
        if issue.get("line")
        else issue.get("component")
    )
    description = issue.get("message")
    return {
        "smell_type": smell_type,
        "location": location,
        "severity": sev,
        "description": description,
        "refactoring_suggestion": f"Refer to Sonar rule {rule} guidance.",
        "confidence": 1.0,
        "rule": rule,
        "raw_severity": issue.get("severity"),
        "raw": issue,
    }


def run_sonar_scanner_docker(clone_dir: Path, project_key: str) -> None:
    """Run sonar-scanner via Docker container."""
    scanner_opts = (
        f"-Dsonar.projectKey={project_key} "
        f"-Dsonar.java.binaries=. "
        f"-Dsonar.language=java "
        f"-Dsonar.verbose=true"
    )
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{clone_dir.absolute()}:/usr/src",
        "--network=host",
        "-e",
        f"SONAR_HOST_URL={SONAR_URL}",
        "-e",
        f"SONAR_SCANNER_OPTS={scanner_opts}",
        "-e",
        f"SONAR_TOKEN={SONAR_TOKEN}",
        "sonarsource/sonar-scanner-cli",
    ]
    run(cmd)


def scan_repository(
    repo_url: str,
    cutoff: datetime,
    work_dir: Path,
    dry_run: bool = False,
    use_docker: bool = True,
) -> Dict[str, Any]:
    project_key = derive_project_key(repo_url)
    print(f"[INFO] Project key: {project_key}")
    commit_sha = get_last_commit_before(repo_url, cutoff)
    print(f"[INFO] Selected commit before {cutoff.date()}: {commit_sha}")

    clone_dir = work_dir / f"scan_{project_key}"
    run(["git", "clone", repo_url, str(clone_dir)])
    run(["git", "checkout", commit_sha], cwd=clone_dir)

    if dry_run:
        print("[INFO] Dry run: skipping sonar-scanner execution.")
        return {"project_key": project_key, "commit_sha": commit_sha, "issues": []}

    print(
        f"[INFO] Running sonar-scanner {'via Docker' if use_docker else 'locally'}..."
    )
    try:
        if use_docker:
            run_sonar_scanner_docker(clone_dir, project_key)
        else:
            write_sonar_properties(clone_dir, project_key)
            run(["sonar-scanner"], cwd=clone_dir)
    except Exception as e:
        raise RuntimeError(f"sonar-scanner failed: {e}")

    print("[INFO] Polling analysis completion...")
    poll_analysis_completion(project_key)

    print("[INFO] Fetching issues...")
    issues = fetch_issues(project_key)
    normalized = [normalize_issue(i) for i in issues]
    return {"project_key": project_key, "commit_sha": commit_sha, "issues": normalized}


def write_output(result: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{result['project_key']}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Wrote {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="SonarQube baseline scanning script")
    parser.add_argument(
        "repo", help="GitHub repository URL (https://github.com/org/repo)"
    )
    parser.add_argument(
        "--output",
        default="eval_results/sonarqube_baseline",
        help="Output directory for JSON",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not run sonar-scanner"
    )
    parser.add_argument(
        "--local-scanner",
        action="store_true",
        help="Use local sonar-scanner instead of Docker",
    )
    args = parser.parse_args()

    if not SONAR_TOKEN and not args.dry_run:
        parser.error("SONAR_TOKEN environment variable required unless --dry-run")

    work_dir = Path(tempfile.mkdtemp(prefix="baseline_work_"))
    try:
        result = scan_repository(
            args.repo,
            CUTOFF_DATE,
            work_dir,
            dry_run=args.dry_run,
            use_docker=not args.local_scanner,
        )
        write_output(result, Path(args.output))
        if args.dry_run:
            print("[INFO] Dry run completed. No issues collected.")
        else:
            print(f"[INFO] Collected {len(result['issues'])} issues.")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
