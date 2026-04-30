"""LangChain tool for SonarQube code smell detection.

Provides a tool interface to scan git commits for code smells using SonarQube.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from sonarqube.commit_scan import scan_commit, scan_commit_file

DEFAULT_SONAR_URL = "http://localhost:9000"


class SonarScanArgs(BaseModel):
    """Input schema for the `scan_commit_smells` tool."""

    repo_url: str = Field(
        ...,
        description="Git repository URL (e.g., https://github.com/org/repo)",
    )
    commit_sha: str = Field(
        ...,
        description="Commit SHA to analyze",
    )
    file_paths: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific file paths to scan. If None, scans entire commit.",
    )
    sonar_url: str = Field(
        default=DEFAULT_SONAR_URL,
        description="SonarQube server URL",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Optional directory path for caching scan results",
    )


def _ensure_sonarqube_running(sonar_url: str) -> bool:
    """Check if SonarQube is accessible."""
    import requests

    try:
        resp = requests.get(f"{sonar_url}/api/system/status", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _start_sonarqube() -> None:
    """Start SonarQube via docker compose."""
    project_root = Path(__file__).resolve().parents[1]
    compose_file = project_root / "sonarqube/docker-compose.yml"

    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "up", "-d"],
        check=True,
        capture_output=True,
    )

    import time

    for _ in range(60):
        if _ensure_sonarqube_running(DEFAULT_SONAR_URL):
            return
        time.sleep(2)

    raise RuntimeError("SonarQube failed to start within timeout")


@tool("scan_commit_smells", args_schema=SonarScanArgs)
def scan_commit_smells(
    repo_url: str,
    commit_sha: str,
    file_paths: Optional[List[str]] = None,
    sonar_url: str = DEFAULT_SONAR_URL,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Scan a git commit for code smells using SonarQube.

    This tool checks out a specific commit and runs SonarQube analysis to detect
    code smells. Returns a dictionary mapping file paths to lists of detected smells.

    Each smell includes:
        - smell_type: Human-readable smell name (e.g., "Long Method", "God Class")
        - line: Line number where smell was detected
        - severity: HIGH, MEDIUM, or LOW
        - message: Detailed description of the issue
        - rule: SonarQube rule identifier

    The tool automatically starts SonarQube if it's not running.
    """
    sonar_token = os.getenv("SONAR_TOKEN")
    if not sonar_token:
        return {
            "error": "SONAR_TOKEN environment variable not set. Please configure your SonarQube token."
        }

    # Ensure SonarQube is running
    if not _ensure_sonarqube_running(sonar_url):
        try:
            _start_sonarqube()
        except Exception as e:
            return {
                "error": f"Failed to start SonarQube: {e}. Start manually with: ./workflows/sonarqube_server.sh start"
            }

    cache_path = Path(cache_dir) if cache_dir else None

    try:
        if file_paths:
            results: Dict[str, List[Dict[str, Any]]] = {}
            for file_path in file_paths:
                issues = scan_commit_file(
                    repo_url=repo_url,
                    commit_sha=commit_sha,
                    file_path=file_path,
                    sonar_url=sonar_url,
                    sonar_token=sonar_token,
                    cache_dir=cache_path,
                )
                results[file_path] = issues

            total_smells = sum(len(v) for v in results.values())
            return {
                "commit_sha": commit_sha,
                "files_scanned": len(results),
                "total_smells": total_smells,
                "smells_by_file": results,
            }
        else:
            issues_by_file = scan_commit(
                repo_url=repo_url,
                commit_sha=commit_sha,
                sonar_url=sonar_url,
                sonar_token=sonar_token,
                cache_dir=cache_path,
            )

            total_smells = sum(len(v) for v in issues_by_file.values())
            return {
                "commit_sha": commit_sha,
                "files_scanned": len(issues_by_file),
                "total_smells": total_smells,
                "smells_by_file": issues_by_file,
            }

    except Exception as e:
        return {
            "error": f"Scan failed: {str(e)}",
            "commit_sha": commit_sha,
        }
