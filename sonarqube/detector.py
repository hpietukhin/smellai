"""SonarQube-backed implementation of the domain `SmellDetector` interface.

Lives in `sonarqube/` (infrastructure), not `domain/`, so domain code never
depends on SonarQube specifics.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

from domain.detector import (
    DetectorConfigError,
    DetectorExecutionError,
    DetectorUnavailableError,
    SmellDetectionError,
    SmellDetector,
)
from domain.models import SmellEvent

LOGGER = logging.getLogger(__name__)


class SonarQubeDetector(SmellDetector):
    """Detect smells via a running SonarQube instance."""

    def __init__(
        self,
        sonar_url: str = "http://localhost:9000",
        sonar_token: str | None = None,
    ) -> None:
        self.sonar_url = sonar_url
        self.sonar_token = os.getenv("SONAR_TOKEN") if sonar_token is None else sonar_token

    def detect(self, project_path: Path) -> list[SmellEvent]:
        if not self.sonar_token:
            msg = "SONAR_TOKEN is not set for SonarQubeDetector"
            raise DetectorConfigError(msg)

        try:
            from sonarqube.commit_scan import (
                fetch_all_project_issues,
                poll_analysis_completion,
                run_sonar_scanner_local,
            )
        except Exception as exc:  # pragma: no cover - defensive import guard
            raise DetectorUnavailableError(
                "Failed to import SonarQube scanning dependencies"
            ) from exc

        project_key = f"smellai_{project_path.name}"

        try:
            LOGGER.info("Running SonarQube scanner for %s", project_key)
            task_id = run_sonar_scanner_local(
                clone_dir=project_path,
                project_key=project_key,
                sonar_url=self.sonar_url,
                sonar_token=self.sonar_token,
            )
            LOGGER.info("Waiting for analysis to complete...")
            poll_analysis_completion(task_id, self.sonar_url, self.sonar_token)
            LOGGER.info("Fetching issues from SonarQube API...")
            raw_issues = fetch_all_project_issues(
                project_key, self.sonar_url, self.sonar_token,
            )
        except SmellDetectionError:
            raise
        except FileNotFoundError as exc:
            raise DetectorUnavailableError(
                "SonarQube scanner executable was not found"
            ) from exc
        except Exception as exc:
            raise DetectorExecutionError(
                f"SonarQube scan failed for {project_key}"
            ) from exc

        events = _normalize_issues(raw_issues)
        LOGGER.info("Detected %d smells", len(events))
        return events


def _normalize_issues(raw_issues: Iterable[dict]) -> list[SmellEvent]:
    """Normalize raw SonarQube issue dicts into domain SmellEvents."""
    from sonarqube.commit_scan import normalize_issue

    events: list[SmellEvent] = []
    for issue in raw_issues:
        component = issue.get("component", "")
        if ":" not in component:
            continue

        file_path = component.split(":", 1)[1]
        n = normalize_issue(issue)
        line = n.get("line") or 0

        events.append(
            SmellEvent(
                smell_id=f"{n['smell_type']}:{file_path}:{line}",
                smell_type=n["smell_type"],
                severity=n["severity"],
                file_path=file_path,
                line_number=line,
            )
        )
    return events
