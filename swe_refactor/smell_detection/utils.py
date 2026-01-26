"""Utilities for smell detection using SonarQube.

Provides functions to scan local projects and compare smell sets between iterations.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List

import requests

from sonarqube.commit_scan import (
    RULE_NAME_MAP,
    SEVERITY_MAP,
    poll_analysis_completion,
    run_sonar_scanner_local,
)
from swe_refactor.persistence.models import SmellAction, SmellEvent

LOGGER = logging.getLogger(__name__)


def scan_local_project(
    project_path: Path,
    project_key: str,
    sonar_url: str,
    session_id: str,
    iteration: int,
) -> List[SmellEvent]:
    """Scan a local project (already cloned) with SonarQube.

    This function:
    1. Runs sonar-scanner locally on the project
    2. Polls until analysis completes
    3. Fetches all issues from SonarQube API
    4. Converts to SmellEvent objects

    Args:
        project_path: Path to cloned project directory
        project_key: Unique SonarQube project key
        sonar_url: SonarQube server URL
        session_id: Session identifier for logging
        iteration: Current refactoring iteration

    Returns:
        List of SmellEvent objects detected

    Raises:
        ValueError: If SONAR_TOKEN not set
        RuntimeError: If scanner fails or analysis times out
    """
    sonar_token = os.getenv("SONAR_TOKEN")
    if not sonar_token:
        raise ValueError("SONAR_TOKEN environment variable not set")

    # Run scanner locally
    LOGGER.info("Running SonarQube scanner for %s", project_key)
    run_sonar_scanner_local(
        clone_dir=project_path,
        project_key=project_key,
        sonar_url=sonar_url,
        sonar_token=sonar_token,
    )

    # Poll until complete
    LOGGER.info("Waiting for analysis to complete...")
    poll_analysis_completion(project_key, sonar_url, sonar_token)

    # Fetch all issues for project
    LOGGER.info("Fetching issues from SonarQube API...")
    session = requests.Session()
    session.auth = (sonar_token, "")

    all_issues = []
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

        LOGGER.info("Fetched %d/%d issues (page %d)", len(all_issues), total, page)

        if page * 500 >= total:
            break
        page += 1

    # Convert to SmellEvent
    events = []
    for issue in all_issues:
        # Extract file path from component (format: projectKey:filepath)
        component = issue.get("component", "")
        if ":" in component:
            file_path = component.split(":", 1)[1]
        else:
            continue

        rule = issue.get("rule")
        smell_type = RULE_NAME_MAP.get(rule, rule)
        severity = SEVERITY_MAP.get(issue.get("severity"), "LOW")
        line = issue.get("line", 0)

        smell_id = f"{smell_type}:{file_path}:{line}"

        events.append(
            SmellEvent(
                session_id=session_id,
                iteration=iteration,
                smell_id=smell_id,
                smell_type=smell_type,
                severity=severity,
                file_path=file_path,
                line_number=line,
                action=SmellAction.DETECTED,
            )
        )

    LOGGER.info("Detected %d smells in iteration %d", len(events), iteration)
    return events


def compare_smell_sets(
    before: List[SmellEvent], after: List[SmellEvent]
) -> Dict[str, List[str]]:
    """Compare two sets of smells and identify changes.

    Args:
        before: Smells detected before refactoring
        after: Smells detected after refactoring

    Returns:
        Dictionary with keys:
        - "resolved": List of smell_ids completely removed
        - "created": List of smell_ids newly introduced
        - "persisted": List of smell_ids still present
    """
    before_ids = {s.smell_id for s in before}
    after_ids = {s.smell_id for s in after}

    return {
        "resolved": list(before_ids - after_ids),
        "created": list(after_ids - before_ids),
        "persisted": list(before_ids & after_ids),
    }


def calculate_smell_diff_summary(diff: Dict[str, List[str]]) -> str:
    """Generate human-readable summary of smell changes.

    Args:
        diff: Output from compare_smell_sets()

    Returns:
        Formatted string summary
    """
    resolved_count = len(diff["resolved"])
    created_count = len(diff["created"])
    persisted_count = len(diff["persisted"])

    summary = f"Smell changes: {resolved_count} resolved, {created_count} created, {persisted_count} persisted"

    if resolved_count > 0:
        summary += f"\n  Resolved: {', '.join(diff['resolved'][:3])}"
        if resolved_count > 3:
            summary += f" (and {resolved_count - 3} more)"

    if created_count > 0:
        summary += f"\n  Created: {', '.join(diff['created'][:3])}"
        if created_count > 3:
            summary += f" (and {created_count - 3} more)"

    return summary
