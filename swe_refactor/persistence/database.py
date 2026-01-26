"""Database manager for analytics persistence using SQLModel ORM."""

from pathlib import Path
from typing import Optional

from sqlmodel import Session, SQLModel, create_engine, select, func

from swe_refactor.persistence.models import (
    RefactoringAttempt,
    SmellDependency,
    SmellEvent,
    ToolCall,
    TokenUsage,
)


class AnalyticsDB:
    """Analytics database manager using SQLModel ORM.

    Manages SQLite database for storing:
    - Tool call logs
    - Smell detection events
    - Smell dependencies
    - Refactoring attempts
    - Token usage metrics

    Separate from LangGraph checkpoints (SqliteSaver) for clean separation:
    - Checkpoints: workflow state for replay
    - Analytics: structured queries for MLFlow, visualization, reports
    """

    def __init__(self, db_path: str = "analytics.db"):
        """Initialize database connection and create tables.

        Args:
            db_path: Path to SQLite database file
        """
        self.engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(self.engine)

    def log_tool_call(self, tool_call: ToolCall) -> None:
        """Log a tool invocation with timing."""
        with Session(self.engine) as session:
            session.add(tool_call)
            session.commit()

    def log_smell_event(self, event: SmellEvent) -> None:
        """Log a smell detection/resolution event."""
        with Session(self.engine) as session:
            session.add(event)
            session.commit()

    def log_smell_dependency(self, dep: SmellDependency) -> None:
        """Log a dependency relationship between smells."""
        with Session(self.engine) as session:
            session.add(dep)
            session.commit()

    def log_refactoring_attempt(self, attempt: RefactoringAttempt) -> None:
        """Log a complete refactoring attempt with outcome."""
        with Session(self.engine) as session:
            session.add(attempt)
            session.commit()

    def log_token_usage(self, usage: TokenUsage) -> None:
        """Log LLM token usage for a node."""
        with Session(self.engine) as session:
            session.add(usage)
            session.commit()

    def get_session_summary(self, session_id: str) -> dict:
        """Get aggregated summary for a session.

        Returns:
            Dictionary with:
            - total_tokens: Total tokens used
            - total_iterations: Number of refactoring iterations
            - successful_refactorings: Count of successful attempts
            - smells_resolved: Total smells resolved
            - smells_created: Total smells created
        """
        with Session(self.engine) as session:
            total_tokens = session.exec(
                select(func.sum(TokenUsage.total_tokens)).where(
                    TokenUsage.session_id == session_id
                )
            ).one()

            attempts = session.exec(
                select(RefactoringAttempt).where(
                    RefactoringAttempt.session_id == session_id
                )
            ).all()

            return {
                "total_tokens": total_tokens or 0,
                "total_iterations": len(attempts),
                "successful_refactorings": sum(
                    1 for a in attempts if a.outcome == "success"
                ),
                "smells_resolved": sum(a.smells_resolved for a in attempts),
                "smells_created": sum(a.smells_created for a in attempts),
            }

    def get_tokens_by_node(self, session_id: str) -> dict[str, int]:
        """Get token usage breakdown by node.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary mapping node_name to total_tokens
        """
        with Session(self.engine) as session:
            results = session.exec(
                select(TokenUsage.node_name, func.sum(TokenUsage.total_tokens))
                .where(TokenUsage.session_id == session_id)
                .group_by(TokenUsage.node_name)
            ).all()

            return {node_name: int(total) for node_name, total in results}

    def get_smell_events(
        self, session_id: str, iteration: Optional[int] = None
    ) -> list[SmellEvent]:
        """Get smell events for a session, optionally filtered by iteration.

        Args:
            session_id: Session identifier
            iteration: Specific iteration (None = all iterations)

        Returns:
            List of SmellEvent objects
        """
        with Session(self.engine) as session:
            query = select(SmellEvent).where(SmellEvent.session_id == session_id)
            if iteration is not None:
                query = query.where(SmellEvent.iteration == iteration)

            return session.exec(query).all()
