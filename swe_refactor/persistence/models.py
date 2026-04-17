"""SQLModel data models for analytics persistence.

This module defines the database schema using SQLModel (Pydantic + SQLAlchemy ORM).
Tables store structured events for MLFlow export, visualization, and analysis.
"""
# pylint: disable=duplicate-code  # _TestCountsBase mirrors TestCounts (agents/tools/java_test_tools.py).
# They serve different framework requirements (SQLModel vs dataclass) and cannot share a base.

from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class SmellAction(str, Enum):
    """Action performed on a smell during refactoring workflow."""

    DETECTED = "detected"  # Smell found in current iteration
    RESOLVED = "resolved"  # Smell completely removed by refactoring
    CREATED = "created"  # Smell introduced by refactoring
    PERSISTED = "persisted"  # Smell still exists (may have changed severity)


class ToolCall(SQLModel, table=True):
    """Records tool invocations during agent execution.

    Used for debugging, performance analysis, and replay visualization.
    """

    __tablename__ = "tool_calls"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    iteration: int = Field(index=True)
    node_name: str  # e.g., "A1_detect", "A5_generate"
    tool_name: str  # e.g., "sonar.scan", "llm.invoke"
    arguments: str  # JSON-serialized dict
    result: Optional[str] = None  # JSON-serialized result
    duration_ms: float
    timestamp: datetime = Field(default_factory=_utc_now)


class SmellEvent(SQLModel, table=True):
    """Records smell detection/resolution events during workflow.

    Each event represents a state change for a specific smell instance.
    session_id and iteration default to "" / 0 so the model can be used
    as a plain in-memory value without a database context (e.g. in SmellPrioritizer).
    """

    __tablename__ = "smell_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(default="", index=True)
    iteration: int = Field(default=0, index=True)
    smell_id: str  # Composite key: {type}:{file}:{line}
    smell_type: str  # "Long Method", "God Class", etc.
    severity: str  # "HIGH", "MEDIUM", "LOW" (normalized from SonarQube)
    file_path: str
    line_number: int = Field(default=0)
    action: SmellAction = Field(default=SmellAction.DETECTED)
    timestamp: datetime = Field(default_factory=_utc_now)

    @property
    def location(self) -> str:
        """Composite location string used by SmellPrioritizer."""
        return f"{self.file_path}:{self.line_number}"

    @property
    def severity_score(self) -> int:
        """Numeric severity for PZ prioritization formula (1–3)."""
        s = self.severity.upper()
        if s in ("BLOCKER", "CRITICAL", "HIGH"):
            return 3
        if s in ("MAJOR", "MEDIUM"):
            return 2
        return 1


class SmellDependency(SQLModel, table=True):
    """Records dependency relationships between smells.

    Tracks positive dependencies (solving A removes B) and negative dependencies
    (solving A creates B) for prioritization analysis.
    """

    __tablename__ = "smell_dependencies"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    iteration: int
    source_smell: str  # smell_id of the source smell
    target_smell: str  # smell_id of the dependent smell
    relationship: str  # "positive" | "negative"
    timestamp: datetime = Field(default_factory=_utc_now)


class RefactoringAttempt(SQLModel, table=True):
    """Records refactoring attempts and their outcomes.

    Each row represents one complete refactoring cycle:
    select smell → generate code → verify → measure impact.
    """

    __tablename__ = "refactoring_attempts"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    iteration: int = Field(index=True)
    smell_id: str  # Target smell being refactored
    refactoring_type: str  # "Extract Method", "Rename Variable", etc.
    outcome: str  # "success" | "test_failed" | "build_failed" | "max_retries"
    retries: int = 0  # Number of retry attempts
    smells_resolved: int = 0  # Count of smells removed
    smells_created: int = 0  # Count of new smells introduced
    code_diff: Optional[str] = None  # Git diff showing code changes
    timestamp: datetime = Field(default_factory=_utc_now)


class TokenUsage(SQLModel, table=True):
    """Records LLM token usage per node invocation.

    Aggregated per session for cost tracking and analysis.
    """

    __tablename__ = "token_usage"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    iteration: int = Field(index=True)
    node_name: str  # e.g., "A4_map", "A5_generate"
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str  # e.g., "gpt-4o-mini"
    timestamp: datetime = Field(default_factory=_utc_now)


# These fields mirror TestCounts in agents/tools/java_test_tools.py.
# Defined separately here because SQLModel (Pydantic) and dataclass
# cannot share a base without cross-layer coupling.
class _TestCountsBase(SQLModel):  # pylint: disable=duplicate-code
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration: float = 0.0


class TestRun(_TestCountsBase, table=True):
    """Records test execution results per refactoring iteration.

    Stores test run summary and individual test results for visualization
    and behavior verification tracking.
    """

    __tablename__ = "test_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    iteration: int = Field(index=True)
    success: bool = True
    failed_tests: Optional[str] = None  # JSON list of failed test info
    test_names: Optional[str] = None  # JSON list of all test names
    timestamp: datetime = Field(default_factory=_utc_now)
