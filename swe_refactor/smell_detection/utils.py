"""Backend-agnostic helpers for smell detection workflows.

This module intentionally keeps orchestration logic thin. The canonical smell
backend abstraction lives in `store.detector`.
"""

from __future__ import annotations

from pathlib import Path

from domain.detector import SmellDetector, SonarQubeDetector
from domain.models import SmellEvent


def scan_local_project(
    project_path: Path | str,
    project_key: str,
    sonar_url: str,
    session_id: str,  # noqa: ARG001 — kept for call-site compatibility
    iteration: int,   # noqa: ARG001 — kept for call-site compatibility
    cache_dir: str | None = None,
    detector: SmellDetector | None = None,
) -> list[SmellEvent]:
    """Compatibility wrapper around the detector abstraction."""
    del project_key, cache_dir, session_id, iteration
    smell_detector = detector or SonarQubeDetector(sonar_url=sonar_url)
    return smell_detector.detect(Path(project_path))


def compare_smell_sets(
    before: list[SmellEvent], after: list[SmellEvent],
) -> dict[str, list[str]]:
    """Compare two smell sets — resolved / created / persisted."""
    return SmellDetector.compare(before, after)


def _format_smell_list(label: str, smell_ids: list[str]) -> str | None:
    if not smell_ids:
        return None

    suffix = f" (and {len(smell_ids) - 3} more)" if len(smell_ids) > 3 else ""
    return f"\n  {label}: {', '.join(smell_ids[:3])}{suffix}"


def calculate_smell_diff_summary(diff: dict[str, list[str]]) -> str:
    """Generate human-readable summary of smell changes."""
    resolved = diff["resolved"]
    created = diff["created"]
    persisted = diff["persisted"]

    summary = (
        f"Smell changes: {len(resolved)} resolved, "
        f"{len(created)} created, {len(persisted)} persisted"
    )
    for section in (
        _format_smell_list("Resolved", resolved),
        _format_smell_list("Created", created),
    ):
        if section:
            summary += section

    return summary
