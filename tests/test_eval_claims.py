import pytest

from domain.experiment_axes import ExperimentSpec
from domain.eval_claims import (
    describe_developer_reference,
    validate_comparison_mode,
)


def test_validate_comparison_mode_allows_supported_values():
    validate_comparison_mode("planner_vs_planner")
    validate_comparison_mode("planner_vs_dev_reference")


def test_validate_comparison_mode_rejects_unknown():
    with pytest.raises(AssertionError):
        validate_comparison_mode("planner_vs_dev_optimal")


def test_developer_wording_is_reference_not_optimal():
    spec = ExperimentSpec(
        heuristic="range-based",
        planner="befs",
        comparison_mode="planner_vs_dev_reference",
    )
    text = describe_developer_reference(spec)
    assert "observed developer trajectory (reference)" in text
    assert "optimal" not in text.lower()
