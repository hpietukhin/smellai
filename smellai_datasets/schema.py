"""Unified evaluation-sample schema shared across all dataset sources.

Design note
-----------
The three research datasets (RMiner, SWE-Refactor, TDD) have different raw
schemas. A single rigid row model would either drop information or collapse into
dict[str, Optional[Any]]. Instead we use a two-layer approach:

1. Per-source raw pandas DataFrames (produced by source-specific loaders).
2. One rigid ``EvalSample`` pydantic model — the generalised form. Its shape
   matches the MLflow GenAI evaluate contract: every sample carries ``inputs``,
   ``expectations``, and ``tags`` dicts plus a ``source`` discriminator and a
   stable ``sample_id``. Source-specific keys live *inside* those dicts.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DatasetSource = Literal["rminer", "swe", "tdd"]


class EvalSample(BaseModel):
    """One evaluation sample in the generalised, source-agnostic form."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: DatasetSource
    sample_id: str = Field(..., min_length=1)
    inputs: dict[str, Any] = Field(default_factory=dict)
    expectations: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, Any] = Field(default_factory=dict)


def rminer_sample(
    pair_id: str,
    before_code: str,
    file_path: str,
    refactoring_types: list,
    refactoring_descriptions: list,
    diff_hunks: list,
    sonar_issues: list | None = None,
    expectations: "dict[str, Any] | None" = None,
    tags: "dict[str, Any] | None" = None,
) -> EvalSample:
    """Build a minimal EvalSample for an RMiner record."""
    return EvalSample(
        source="rminer",
        sample_id=f"rminer:{pair_id}",
        inputs={
            "pair_id": pair_id,
            "before_code": before_code,
            "file_path": file_path,
            "refactoring_types": refactoring_types,
            "refactoring_descriptions": refactoring_descriptions,
            "diff_hunks": diff_hunks,
            "sonar_issues": sonar_issues or [],
        },
        expectations=expectations or {},
        tags=tags or {},
    )


__all__ = ["DatasetSource", "EvalSample", "rminer_sample"]
