from __future__ import annotations

from hypothesis import given, strategies as st

from dataset.neo4j_graph import CompositeStep, RefactoringRecord
from dataset.planner_eval import PlannerEvalResult, evaluate_composite
from domain.models import SmellEvent


def _smell(smell_id: str, class_name: str, smell_type: str = "Long Method") -> SmellEvent:
    return SmellEvent(
        smell_id=smell_id,
        smell_type=smell_type,
        severity="HIGH",
        file_path=f"src/{class_name.split('.')[-1]}.java",
        line_number=10,
        class_name=class_name,
    )


@given(
    dev_steps=st.integers(min_value=0, max_value=500),
    dev_matched_steps=st.integers(min_value=0, max_value=500),
)
def test_dev_match_rate_metric_is_bounded_and_consistent(dev_steps: int, dev_matched_steps: int) -> None:
    dev_matched_steps = min(dev_matched_steps, dev_steps)
    rate = dev_matched_steps / max(dev_steps, 1)

    result = PlannerEvalResult(
        project="p",
        initial_commit="c",
        initial_commit_order=1,
        smells_initial=1,
        smells_after_empirical=1,
        befs_reduces=False,
        greedy_reduces=False,
        dev_reduces=False,
        befs_eta=0.0,
        greedy_eta=0.0,
        befs_rho=0.0,
        greedy_rho=0.0,
        h_initial=1.0,
        h_befs_final=1.0,
        h_greedy_final=1.0,
        h_dev_final=1.0,
        befs_h_trace=(1.0,),
        greedy_h_trace=(1.0,),
        dev_h_trace=(1.0,),
        dev_steps=dev_steps,
        dev_matched_steps=dev_matched_steps,
        dev_match_rate=rate,
    )

    metrics = result.to_mlflow_metrics()
    assert 0.0 <= metrics["dev_match_rate"] <= 1.0
    assert metrics["dev_matched_steps"] <= metrics["dev_steps"]
    assert metrics["dev_match_rate"] == metrics["dev_matched_steps"] / max(metrics["dev_steps"], 1.0)


@given(n_unrelated_refs=st.integers(min_value=1, max_value=40))
def test_evaluate_composite_never_matches_outside_initial_smells(n_unrelated_refs: int) -> None:
    initial_smells = [
        _smell("s1", "org.example.InitialA"),
        _smell("s2", "org.example.InitialB"),
    ]

    unrelated_refs = [
        RefactoringRecord(
            ref_type="Extract Method",
            hash_id=f"r{i}",
            commit_hash="c1",
            commit_order=1,
            changed_elements=["org.example.UnrelatedClass.someMethod()"],
            produced_elements=[],
        )
        for i in range(n_unrelated_refs)
    ]

    steps = [
        CompositeStep(
            commit_hash="c0",
            commit_order=0,
            smells=initial_smells,
            refactorings=[],
        ),
        CompositeStep(
            commit_hash="c1",
            commit_order=1,
            smells=[*initial_smells, _smell("s3", "org.example.UnrelatedClass")],
            refactorings=unrelated_refs,
        ),
    ]

    result = evaluate_composite(steps, project="synthetic")
    assert result is not None

    assert result.dev_steps == n_unrelated_refs
    assert result.dev_matched_steps == 0
    assert result.dev_match_rate == 0.0
    assert 0.0 <= result.dev_match_rate <= 1.0
    assert result.dev_matched_steps <= result.dev_steps
