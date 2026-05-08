"""Core smell domain types.

These are pure data structures with no dependency on SQLModel, LangGraph,
or any external infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SmellEvent:
    """A single code smell instance detected in a project.

    Intentionally free of ORM fields (id, session_id, iteration, timestamp).
    Those belong in ``SmellEventRecord`` (swe_refactor.persistence.models).

    Optional fields (class_name, method_signature, project, commit_hash,
    end_line) carry dataset-level provenance when populated from external
    sources such as the Composite Refactorings 2020 Neo4j graph.
    """

    smell_id: str                        # Composite key: {type}:{file}:{line}
    smell_type: str                      # "Long Method", "God Class", etc.
    severity: str                        # "HIGH", "MEDIUM", "LOW"
    file_path: str
    line_number: int = 0
    action: str = "detected"

    # --- Optional provenance / dataset fields ---
    class_name: str | None = None        # FQN, e.g. "org.apache.Foo"
    method_signature: str | None = None  # e.g. "action(ActionCode,Object)"
    project: str | None = None           # Project name from dataset
    commit_hash: str | None = None       # Commit hash where smell observed
    end_line: int | None = None          # End line of smelly region
    detection_reason: str | None = None  # Metric threshold rule, e.g. "MLOC > 6.87"

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.line_number}"

    @property
    def class_context(self) -> str:
        """Best available class-level identifier for locality grouping.

        Prefers explicit class_name; falls back to file_path.
        Used by SmellGraph edge construction for same-class matching.
        """
        return self.class_name or self.file_path

    @property
    def severity_score(self) -> int:
        """Numeric severity for PZ formula (1–3). Spec §4.4 Eq. 2."""
        s = self.severity.upper()
        if s in ("BLOCKER", "CRITICAL", "HIGH"):
            return 3
        if s in ("MAJOR", "MEDIUM"):
            return 2
        return 1
