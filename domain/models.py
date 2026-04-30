"""Core smell domain types.

These are pure data structures with no dependency on SQLModel, LangGraph,
or any external infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SmellAction(str, Enum):
    """Lifecycle action performed on a smell during the refactoring workflow."""

    DETECTED = "detected"
    RESOLVED = "resolved"
    CREATED = "created"
    PERSISTED = "persisted"


@dataclass
class SmellEvent:
    """A single code smell instance detected in a project.

    Intentionally free of ORM fields (id, session_id, iteration, timestamp).
    Those belong in ``SmellEventRecord`` (swe_refactor.persistence.models).
    """

    smell_id: str                        # Composite key: {type}:{file}:{line}
    smell_type: str                      # "Long Method", "God Class", etc.
    severity: str                        # "HIGH", "MEDIUM", "LOW"
    file_path: str
    line_number: int = 0
    action: SmellAction = field(default=SmellAction.DETECTED)

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.line_number}"

    @property
    def severity_score(self) -> int:
        """Numeric severity for PZ formula (1–3). Spec §4.4 Eq. 2."""
        s = self.severity.upper()
        if s in ("BLOCKER", "CRITICAL", "HIGH"):
            return 3
        if s in ("MAJOR", "MEDIUM"):
            return 2
        return 1
