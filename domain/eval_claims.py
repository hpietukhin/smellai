"""Claim-safety helpers for evaluation wording and mode constraints."""

from __future__ import annotations

from domain.experiment_axes import ComparisonMode, ExperimentSpec


def validate_comparison_mode(mode: str) -> ComparisonMode:
    allowed = {"planner_vs_planner", "planner_vs_dev_reference"}
    assert mode in allowed, f"unsupported comparison_mode={mode!r}"
    return mode  # type: ignore[return-value]


def describe_developer_reference(spec: ExperimentSpec) -> str:
    validate_comparison_mode(spec.comparison_mode)
    if spec.comparison_mode == "planner_vs_dev_reference":
        return "Comparison uses observed developer trajectory (reference) as an empirical anchor only."
    return "Comparison is planner-vs-planner under identical setup."
