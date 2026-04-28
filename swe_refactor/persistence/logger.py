"""Context manager for automatic tool call logging with timing."""

import json
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from swe_refactor.persistence.database import AnalyticsDB
from swe_refactor.persistence.models import ToolCall


@contextmanager
def log_tool_call(
    db: AnalyticsDB,
    session_id: str,
    iteration: int,
    node_name: str,
    tool_name: str,
    arguments: dict,
):
    """Context manager that times a tool call and logs it on exit."""
    start = perf_counter()
    result_holder: dict[str, Any] = {}

    def log_result(result: Any) -> None:
        """Store result for logging when context exits."""
        result_holder["value"] = result

    try:
        yield log_result
    finally:
        duration_ms = (perf_counter() - start) * 1000

        result_json: str | None = None
        if "value" in result_holder:
            try:
                result_json = json.dumps(result_holder["value"], default=str)
            except (TypeError, ValueError):
                result_json = str(result_holder["value"])

        tool_call = ToolCall(
            session_id=session_id,
            iteration=iteration,
            node_name=node_name,
            tool_name=tool_name,
            arguments=json.dumps(arguments, default=str),
            result=result_json,
            duration_ms=duration_ms,
        )
        db.log_tool_call(tool_call)
