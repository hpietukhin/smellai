"""SonarQube scanning for specific git commits.

This module provides functionality to:
1. Checkout a specific commit in a temporary directory
2. Run SonarQube analysis on that commit
3. Fetch and normalize issues for specific files
4. Cache results to avoid redundant scans
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Reuse rule mapping from baseline_scan
RULE_NAME_MAP = {
    "java:S1541": "Complex Method",
    "java:S138": "Long Method",
    "java:S107": "Long Parameter List",
    "java:S1067": "Conditional Complexity",
    "java:S1200": "God Class",
    "java:S110": "Large Class",
    "java:S1871": "Duplicated Conditions",
    "java:S106": "Print Statements",
}

# TODO SPEC-012: Verify this severity mapping table exists in codebase and document exact location.
# Mapping shown in specification but location needs verification.
# LOW priority - informational verification task.
# (See TECHNICAL_SPECIFICATION.md §5.1)
SEVERITY_MAP = {
    "BLOCKER": "HIGH",
    "CRITICAL": "HIGH",
    "MAJOR": "MEDIUM",
    "MINOR": "LOW",
    "INFO": "LOW",
}


def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    check: bool = True,
    verbose: bool = False,
) -> str:
    """Run a shell command and return stdout."""
    logging.info(f"Running command: {' '.join(cmd)}")
    if cwd:
        logging.info(f"Working directory: {cwd}")

    if verbose:
        # Run with output streaming to console
        proc = subprocess.run(cmd, cwd=cwd, text=True)
        if check and proc.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
        return ""
    else:
        proc = subprocess.run(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if proc.stdout:
            logging.info(f"Command stdout:\n{proc.stdout}")
        if proc.stderr:
            logging.warning(f"Command stderr:\n{proc.stderr}")

        if check and proc.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")
        return proc.stdout.strip()


def derive_project_key(repo_url: str, commit_sha: str) -> str:
    """Generate unique project key for repo+commit."""
    m = re.search(r"github.com[:/](?P<org>[\w.-]+)/(?P<repo>[\w.-]+)", repo_url)
    if not m:
        raise ValueError(f"Cannot parse repo URL: {repo_url}")
    repo = m.group("repo").removesuffix(".git")
    short_sha = commit_sha[:8]
    return f"{m.group('org')}_{repo}_{short_sha}".lower()


def _compile_java_project(project_path: Path) -> Path | None:
    """Attempt to compile the Java project. Returns the classes directory or None."""
    if (project_path / "pom.xml").exists():
        run_command(
            ["mvn", "compile", "-q", "-B", "--fail-at-end"],
            cwd=project_path,
            check=False,
        )
        classes_dir = project_path / "target" / "classes"
    elif (project_path / "build.gradle").exists() or (
        project_path / "build.gradle.kts"
    ).exists():
        gradle = "./gradlew" if (project_path / "gradlew").exists() else "gradle"
        run_command(
            [gradle, "compileJava", "-x", "test", "-q"],
            cwd=project_path,
            check=False,
        )
        classes_dir = project_path / "build" / "classes" / "java" / "main"
    else:
        return None
    return classes_dir if classes_dir.exists() else None


def run_sonar_scanner_local(
    clone_dir: Path, project_key: str, sonar_url: str, sonar_token: str
) -> str:
    """Run sonar-scanner locally via CLI. Returns the CE task ID."""
    classes_dir = _compile_java_project(clone_dir)
    if classes_dir:
        binaries = str(classes_dir)
    else:
        logging.warning(
            "Could not find compiled classes for %s; Java bytecode analysis will be degraded",
            project_key,
        )
        binaries = "."

    props_content = (
        f"sonar.projectKey={project_key}\n"
        f"sonar.projectName={project_key}\n"
        "sonar.sources=.\n"
        f"sonar.host.url={sonar_url}\n"
        f"sonar.token={sonar_token}\n"
        f"sonar.java.binaries={binaries}\n"
    )
    (clone_dir / "sonar-project.properties").write_text(props_content)
    logging.info("Starting sonar-scanner (this may take several minutes)...")
    run_command(["sonar-scanner"], cwd=clone_dir, verbose=True)

    report_file = clone_dir / ".scannerwork" / "report-task.txt"
    if not report_file.exists():
        raise RuntimeError("sonar-scanner did not produce .scannerwork/report-task.txt")
    for line in report_file.read_text().splitlines():
        if line.startswith("ceTaskId="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("ceTaskId not found in .scannerwork/report-task.txt")


POLL_INTERVAL_SEC = 5


def poll_analysis_completion(
    task_id: str, sonar_url: str, sonar_token: str, timeout_sec: int = 600
) -> None:
    """Poll SonarQube CE task by ID until it succeeds or times out."""
    logging.info("Polling CE task %s for completion", task_id)
    session = requests.Session()
    session.auth = (sonar_token, "")

    start = time.monotonic()
    poll_count = 0
    while True:
        poll_count += 1
        resp = session.get(
            f"{sonar_url}/api/ce/task", params={"id": task_id}, timeout=30
        )
        resp.raise_for_status()
        status = resp.json()["task"]["status"]
        elapsed = int(time.monotonic() - start)
        logging.info("Poll #%d (%ds): status=%s", poll_count, elapsed, status)

        if status == "SUCCESS":
            logging.info("Analysis completed in %ds", elapsed)
            return
        if status in {"FAILED", "CANCELED"}:
            raise RuntimeError(f"Analysis ended with status {status}")
        if elapsed >= timeout_sec:
            raise TimeoutError("Timeout waiting for analysis completion")
        time.sleep(POLL_INTERVAL_SEC)


def fetch_issues_for_file(
    project_key: str, file_path: str, sonar_url: str, sonar_token: str
) -> List[Dict[str, Any]]:
    """Fetch SonarQube issues for a specific file in the project."""
    session = requests.Session()
    session.auth = (sonar_token, "")

    all_issues: List[Dict[str, Any]] = []
    page = 1
    rule_list = ",".join(RULE_NAME_MAP.keys())

    # Construct component key: projectKey:filepath
    component_key = f"{project_key}:{file_path}"

    while True:
        resp = session.get(
            f"{sonar_url}/api/issues/search",
            params={
                "components": component_key,
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


def fetch_all_project_issues(
    project_key: str, sonar_url: str, sonar_token: str
) -> List[Dict[str, Any]]:
    """Fetch all code smell issues for a project from SonarQube API (paginated).

    Args:
        project_key: SonarQube project key
        sonar_url: SonarQube server URL
        sonar_token: SonarQube authentication token

    Returns:
        List of raw issue dicts from the SonarQube API
    """
    session = requests.Session()
    session.auth = (sonar_token, "")

    all_issues: List[Dict[str, Any]] = []
    page = 1
    rule_list = ",".join(RULE_NAME_MAP.keys())

    while True:
        resp = session.get(
            f"{sonar_url}/api/issues/search",
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
    """Normalize SonarQube issue to simplified format."""
    rule = issue.get("rule")
    smell_type = RULE_NAME_MAP.get(rule, rule)
    sev = SEVERITY_MAP.get(issue.get("severity"), "LOW")

    return {
        "smell_type": smell_type,
        "line": issue.get("line"),
        "severity": sev,
        "message": issue.get("message"),
        "rule": rule,
        "raw_severity": issue.get("severity"),
    }


def scan_commit_file(
    repo_url: str,
    commit_sha: str,
    file_path: str,
    sonar_url: str,
    sonar_token: str,
    cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Scan a specific file at a specific commit with SonarQube.

    Args:
        repo_url: Repository URL
        commit_sha: Commit SHA to checkout
        file_path: Relative path to file to analyze
        sonar_url: SonarQube server URL
        sonar_token: SonarQube authentication token
        cache_dir: Optional cache directory for scan results

    Returns:
        List of normalized issues for the file
    """
    cache_file: Path | None = None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{commit_sha}_{file_path.replace('/', '_')}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

    project_key = derive_project_key(repo_url, commit_sha)

    from repo_utils import checkout_repo, clone_repository as clone_repo

    work_dir = Path(tempfile.mkdtemp(prefix="sonar_commit_scan_"))
    try:
        clone_dir = work_dir / "repo"
        repo_name = clone_repo(repo_url, clone_dir)
        repo_path = clone_dir / repo_name
        checkout_repo(repo_path, commit_sha)

        if not (repo_path / file_path).exists():
            return []

        task_id = run_sonar_scanner_local(repo_path, project_key, sonar_url, sonar_token)
        poll_analysis_completion(task_id, sonar_url, sonar_token)

        issues = fetch_issues_for_file(project_key, file_path, sonar_url, sonar_token)
        normalized = [normalize_issue(i) for i in issues]

        if cache_file is not None:
            cache_file.write_text(json.dumps(normalized, indent=2))

        return normalized

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def scan_commit(
    repo_url: str,
    commit_sha: str,
    sonar_url: str,
    sonar_token: str,
    cache_dir: Optional[Path] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Scan entire commit with SonarQube.

    Returns:
        Dictionary mapping file paths to lists of issues
    """
    # Check cache first
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{commit_sha}_full.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

    project_key = derive_project_key(repo_url, commit_sha)

    from repo_utils import checkout_repo, clone_repository as clone_repo

    # Clone and checkout
    work_dir = Path(tempfile.mkdtemp(prefix="sonar_commit_scan_"))
    try:
        clone_dir = work_dir / "repo"
        repo_name = clone_repo(repo_url, clone_dir)
        repo_path = clone_dir / repo_name
        checkout_repo(repo_path, commit_sha)

        task_id = run_sonar_scanner_local(
            repo_path, project_key, sonar_url, sonar_token
        )
        poll_analysis_completion(task_id, sonar_url, sonar_token)

        all_issues = fetch_all_project_issues(project_key, sonar_url, sonar_token)

        # Group by file
        issues_by_file: Dict[str, List[Dict[str, Any]]] = {}
        for issue in all_issues:
            # Extract file path from component key (format: projectKey:filepath)
            component = issue.get("component", "")
            if ":" in component:
                file_path = component.split(":", 1)[1]
            else:
                continue

            normalized = normalize_issue(issue)
            if file_path not in issues_by_file:
                issues_by_file[file_path] = []
            issues_by_file[file_path].append(normalized)

        # Cache results
        if cache_dir:
            cache_file.write_text(json.dumps(issues_by_file, indent=2))

        return issues_by_file

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SonarQube scans for a commit")
    parser.add_argument("--repo", required=True, help="Git repository URL")
    parser.add_argument("--commit", required=True, help="Commit SHA to analyze")
    parser.add_argument("--sonar-url", required=True, help="SonarQube server URL")
    parser.add_argument(
        "--sonar-token", required=True, help="SonarQube authentication token"
    )
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        help="Specific file path to scan (can be repeated)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory to store cached scan results",
    )
    return parser.parse_args()


def main() -> int:
    from logging_config import setup_logging

    setup_logging()

    args = parse_args()
    cache_dir = args.cache_dir

    if args.files:
        results: Dict[str, List[Dict[str, Any]]] = {}
        for rel_path in args.files:
            issues = scan_commit_file(
                repo_url=args.repo,
                commit_sha=args.commit,
                file_path=rel_path,
                sonar_url=args.sonar_url,
                sonar_token=args.sonar_token,
                cache_dir=cache_dir,
            )
            results[rel_path] = issues
        print(json.dumps(results, indent=2))
    else:
        issues = scan_commit(
            repo_url=args.repo,
            commit_sha=args.commit,
            sonar_url=args.sonar_url,
            sonar_token=args.sonar_token,
            cache_dir=cache_dir,
        )
        print(json.dumps(issues, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
