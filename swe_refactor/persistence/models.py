"""SQLModel data models for analytics persistence."""
# pylint: disable=duplicate-code  # _TestCountsBase mirrors TestCounts (agents/tools/java_test_tools.py).
# They serve different framework requirements (SQLModel vs dataclass) and cannot share a base.

from __future__ import annotations

from datetime import UTC, datetime
from sqlmodel import Field, SQLModel

from domain.models import SmellAction, SmellEvent  # re-export SmellAction; SmellEvent used in from_domain


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class ToolCall(SQLModel, table=True):

    __tablename__ = "tool_calls"

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    iteration: int = Field(index=True)
    node_name: str  # e.g., "A1_detect", "A5_generate"
    tool_name: str  # e.g., "sonar.scan", "llm.invoke"
    arguments: str  # JSON-serialized dict
    result: str | None = None  # JSON-serialized result
    duration_ms: float
    timestamp: datetime = Field(default_factory=_utc_now)


class SmellEventRecord(SQLModel, table=True):
    """ORM record for persisting SmellEvent analytics to SQLite.

    Use ``SmellEvent`` (domain.models) for in-memory smell logic.
    Use this class only when reading/writing the analytics DB.
    """

    __tablename__ = "smell_events"

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(default="", index=True)
    iteration: int = Field(default=0, index=True)
    smell_id: str
    smell_type: str
    severity: str
    file_path: str
    line_number: int = Field(default=0)
    action: SmellAction = Field(default=SmellAction.DETECTED)
    timestamp: datetime = Field(default_factory=_utc_now)

    @classmethod
    def from_domain(
        cls,
        event: SmellEvent,
        *,
        session_id: str,
        iteration: int,
    ) -> SmellEventRecord:
        """Wrap a domain SmellEvent in an ORM record ready for DB insertion."""
        return cls(
            session_id=session_id,
            iteration=iteration,
            smell_id=event.smell_id,
            smell_type=event.smell_type,
            severity=event.severity,
            file_path=event.file_path,
            line_number=event.line_number,
            action=event.action,
        )


class SmellDependency(SQLModel, table=True):

    __tablename__ = "smell_dependencies"

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    iteration: int
    source_smell: str  # smell_id of the source smell
    target_smell: str  # smell_id of the dependent smell
    relationship: str  # "positive" | "negative"
    timestamp: datetime = Field(default_factory=_utc_now)


class RefactoringAttempt(SQLModel, table=True):

    __tablename__ = "refactoring_attempts"

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    iteration: int = Field(index=True)
    smell_id: str  # Target smell being refactored
    refactoring_type: str  # "Extract Method", "Rename Variable", etc.
    outcome: str  # "success" | "test_failed" | "build_failed" | "max_retries"
    retries: int = 0  # Number of retry attempts
    smells_resolved: int = 0  # Count of smells removed
    smells_created: int = 0  # Count of new smells introduced
    code_diff: str | None = None  # Git diff showing code changes
    timestamp: datetime = Field(default_factory=_utc_now)


class TokenUsage(SQLModel, table=True):

    __tablename__ = "token_usage"

    id: int | None = Field(default=None, primary_key=True)
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

    __tablename__ = "test_runs"

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    iteration: int = Field(index=True)
    success: bool = True
    failed_tests: str | None = None  # JSON list of failed test info
    test_names: str | None = None  # JSON list of all test names
    timestamp: datetime = Field(default_factory=_utc_now)
