"""Shared observability helpers for MLflow-backed agent/workflow actions."""

from __future__ import annotations

import logging
from contextlib import contextmanager, suppress
from typing import Iterator

import mlflow

LOGGER = logging.getLogger(__name__)


class MLflowAction:
    """Small adapter around an MLflow span used by existing workflow code."""

    def __init__(self, span: object | None = None) -> None:
        self._span = span

    def addSuccessFields(self, **fields: object) -> None:
        if self._span is None:
            return
        with suppress(AttributeError, RuntimeError, TypeError):
            self._span.set_attributes(fields)  # type: ignore[attr-defined]


@contextmanager
def start_action(action_type: str, span_type: str = "UNKNOWN", **attributes: object) -> Iterator[MLflowAction]:
    """Create an MLflow span when a run is active; otherwise yield a no-op action."""
    if mlflow.active_run() is None:
        yield MLflowAction()
        return

    span_attributes = {"action_type": action_type, **attributes}
    try:
        span_cm = mlflow.start_span(action_type, span_type=span_type, attributes=span_attributes)
    except (AttributeError, RuntimeError, TypeError):
        LOGGER.debug("MLflow span creation skipped", exc_info=True)
        yield MLflowAction()
        return

    with span_cm as span:
        yield MLflowAction(span)
