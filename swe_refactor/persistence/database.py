"""Database manager for analytics persistence using SQLModel ORM."""

from __future__ import annotations

from sqlmodel import Session, SQLModel, create_engine, func, select

from swe_refactor.persistence.models import (
    RefactoringAttempt,
    SmellDependency,
    SmellEventRecord,
    ToolCall,
    TokenUsage,
)


class AnalyticsDB:
    """Analytics database manager.

    Separate from LangGraph checkpoints: checkpoints for replay,
    analytics for MLflow, visualization, and reports.
    """

    def __init__(self, db_path: str = "analytics.db"):
        self.engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(self.engine)

    def _log(self, record: SQLModel) -> None:
        with Session(self.engine) as session:
            session.add(record)
            session.commit()

    def log_tool_call(self, tool_call: ToolCall) -> None:
        self._log(tool_call)

    def log_smell_event(self, event: SmellEventRecord) -> None:
        self._log(event)

    def log_smell_dependency(self, dep: SmellDependency) -> None:
        self._log(dep)

    def log_refactoring_attempt(self, attempt: RefactoringAttempt) -> None:
        self._log(attempt)

    def log_token_usage(self, usage: TokenUsage) -> None:
        self._log(usage)

    def get_session_summary(self, session_id: str) -> dict:
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

            successful_attempts = sum(1 for a in attempts if a.outcome == "success")
            return {
                "total_tokens": total_tokens or 0,
                "total_iterations": len(attempts),
                "successful_refactorings": successful_attempts,
                "smells_resolved": sum(a.smells_resolved for a in attempts),
                "smells_created": sum(a.smells_created for a in attempts),
            }

    def get_tokens_by_node(self, session_id: str) -> dict[str, int]:
        with Session(self.engine) as session:
            results = session.exec(
                select(TokenUsage.node_name, func.sum(TokenUsage.total_tokens))
                .where(TokenUsage.session_id == session_id)
                .group_by(TokenUsage.node_name)
            ).all()

            return {node_name: int(total) for node_name, total in results}

    def get_smell_events(
        self, session_id: str, iteration: int | None = None,
    ) -> list[SmellEventRecord]:
        with Session(self.engine) as session:
            query = select(SmellEventRecord).where(SmellEventRecord.session_id == session_id)
            if iteration is not None:
                query = query.where(SmellEventRecord.iteration == iteration)

            return session.exec(query).all()
