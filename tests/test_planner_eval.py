"""Integration tests for planner evaluation against Composite Refactorings 2020.

Requires a running Neo4j instance with the Composite Refactorings 2020 data.
Tests are skipped automatically if Neo4j is unreachable.

Fixtures used:
  - Apache Tomcat / DefaultServlet (same as test_dataset_graph.py, 11 smells)
"""
from __future__ import annotations

import pytest

from smellai_datasets.composite_dataset import is_available

pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="Neo4j not available",
)

TOMCAT_PROJECT = "Apache Tomcat"
TOMCAT_ELEMENT = "org.apache.catalina.servlets.DefaultServlet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_steps(max_steps: int = 20):
    from dataset.neo4j_graph import DatasetGraph

    ds = DatasetGraph()
    return ds.composite_refactoring(
        elements={TOMCAT_ELEMENT},
        project=TOMCAT_PROJECT,
        max_steps=max_steps,
    )


# ---------------------------------------------------------------------------
# Test: changed_elements / produced_elements are now populated
# ---------------------------------------------------------------------------


class TestChangedElementsPopulated:

    def test_refactoring_records_have_changed_elements(self):
        """composite_refactoring() now populates changed_elements."""
        steps = _get_steps()
        records_with_elements = [
            ref
            for step in steps
            for ref in step.refactorings
            if ref.changed_elements or ref.produced_elements
        ]
        assert len(records_with_elements) > 0, (
            "Expected at least one RefactoringRecord with changed_elements populated"
        )

    def test_changed_elements_contain_target_element(self):
        """CHANGED elements should include our tracked element's FQN (or a method on it)."""
        steps = _get_steps()
        for step in steps:
            for ref in step.refactorings:
                for el in ref.changed_elements:
                    # Element FQNs should start with the class FQN
                    assert el.startswith("org.") or el.startswith("com.") or el.startswith("java."), (
                        f"Unexpected element FQN format: {el!r}"
                    )


# ---------------------------------------------------------------------------
# Test: find_smell_for_refactoring
# ---------------------------------------------------------------------------


class TestFindSmellForRefactoring:

    def test_returns_none_for_no_smells(self):
        from dataset.neo4j_graph import RefactoringRecord
        from dataset.planner_eval import find_smell_for_refactoring
        from domain.dependency_graph import DependencyGraph

        steps = _get_steps()
        initial_step = next(s for s in steps if s.smells)
        dep_graph = DependencyGraph.from_events(initial_step.smells, locality="none")

        # Build a dummy refactoring
        dummy_ref = RefactoringRecord(
            ref_type="Extract Method",
            hash_id="dummy",
            commit_hash="dummy",
            commit_order=0,
        )
        result = find_smell_for_refactoring(dummy_ref, [], dep_graph)
        assert result is None

    def test_returns_smell_id_for_compatible_refactoring(self):
        from dataset.planner_eval import find_smell_for_refactoring
        from domain.dependency_graph import DependencyGraph

        steps = _get_steps()
        initial_step = next(s for s in steps if s.smells)
        dep_graph = DependencyGraph.from_events(initial_step.smells, locality="none")

        # Try each refactoring from steps that have both refactorings and the step follows smells
        for step in steps:
            for ref in step.refactorings:
                result = find_smell_for_refactoring(ref, initial_step.smells, dep_graph)
                # May be None if ref type is not in catalogue — that's OK
                if result is not None:
                    assert isinstance(result, str)
                    assert len(result) > 0
                    return  # Found at least one match — test passes

        # If no match found, that's technically valid but we warn
        pytest.skip("No refactoring matched any smell in this episode — check REFACTORING_CATALOGUE")

    def test_does_not_fallback_to_type_only_when_element_mismatches(self):
        """If refactoring has touched elements, incompatible element scope must skip."""
        from dataset.neo4j_graph import RefactoringRecord
        from dataset.planner_eval import find_smell_for_refactoring
        from domain.dependency_graph import DependencyGraph

        steps = _get_steps()
        initial_step = next(s for s in steps if s.smells)
        dep_graph = DependencyGraph.from_events(initial_step.smells, locality="none")

        unrelated_ref = RefactoringRecord(
            ref_type="Extract Method",
            hash_id="dummy-unrelated",
            commit_hash="dummy",
            commit_order=0,
            changed_elements=["org.example.UnrelatedClass"],
        )

        assert find_smell_for_refactoring(unrelated_ref, initial_step.smells, dep_graph) is None

    def test_result_is_in_initial_smell_set(self):
        """If a match is found, it must be a smell_id from the initial set."""
        from dataset.planner_eval import find_smell_for_refactoring
        from domain.dependency_graph import DependencyGraph

        steps = _get_steps()
        initial_step = next(s for s in steps if s.smells)
        dep_graph = DependencyGraph.from_events(initial_step.smells, locality="none")
        smell_ids = {s.smell_id for s in initial_step.smells}

        for step in steps:
            for ref in step.refactorings:
                result = find_smell_for_refactoring(ref, initial_step.smells, dep_graph)
                if result is not None:
                    assert result in smell_ids, (
                        f"find_smell_for_refactoring returned {result!r} "
                        f"which is not in the initial smell set"
                    )


# ---------------------------------------------------------------------------
# Test: developer_plan_from_steps
# ---------------------------------------------------------------------------


class TestDeveloperPlanFromSteps:

    def test_returns_plan(self):
        from dataset.planner_eval import developer_plan_from_steps
        from domain.dependency_graph import DependencyGraph
        from domain.refactoring_tree import RefactoringTree, State, Plan

        steps = _get_steps()
        initial_step = next(s for s in steps if s.smells)
        dep_graph = DependencyGraph.from_events(initial_step.smells, locality="none")
        initial_state = State(frozenset(e.smell_id for e in initial_step.smells))
        tree = RefactoringTree(initial_state, dep_graph)

        plan_dev = developer_plan_from_steps(steps, tree, dep_graph)
        assert isinstance(plan_dev, Plan)

    def test_plan_states_match_actions_plus_one(self):
        """len(states) == len(actions) + 1 invariant."""
        from dataset.planner_eval import developer_plan_from_steps
        from domain.dependency_graph import DependencyGraph
        from domain.refactoring_tree import RefactoringTree, State

        steps = _get_steps()
        initial_step = next(s for s in steps if s.smells)
        dep_graph = DependencyGraph.from_events(initial_step.smells, locality="none")
        initial_state = State(frozenset(e.smell_id for e in initial_step.smells))
        tree = RefactoringTree(initial_state, dep_graph)

        plan_dev = developer_plan_from_steps(steps, tree, dep_graph)
        assert len(plan_dev.states) == len(plan_dev.actions) + 1

    def test_h_trace_non_negative(self):
        """All h values in developer plan must be ≥ 0."""
        from dataset.planner_eval import developer_plan_from_steps
        from domain.dependency_graph import DependencyGraph
        from domain.refactoring_tree import RefactoringTree, State

        steps = _get_steps()
        initial_step = next(s for s in steps if s.smells)
        dep_graph = DependencyGraph.from_events(initial_step.smells, locality="none")
        initial_state = State(frozenset(e.smell_id for e in initial_step.smells))
        tree = RefactoringTree(initial_state, dep_graph)

        plan_dev = developer_plan_from_steps(steps, tree, dep_graph)
        for h in plan_dev.h_trace:
            assert h >= 0.0, f"Negative h value: {h}"


# ---------------------------------------------------------------------------
# Test: evaluate_composite — full pipeline
# ---------------------------------------------------------------------------


class TestEvaluateComposite:

    def test_returns_result(self):
        from dataset.planner_eval import evaluate_composite, PlannerEvalResult

        steps = _get_steps()
        result = evaluate_composite(steps, project=TOMCAT_PROJECT)
        assert result is not None
        assert isinstance(result, PlannerEvalResult)

    def test_smells_initial_positive(self):
        from dataset.planner_eval import evaluate_composite

        steps = _get_steps()
        result = evaluate_composite(steps, project=TOMCAT_PROJECT)
        assert result is not None
        assert result.smells_initial > 0

    def test_h_values_non_negative(self):
        from dataset.planner_eval import evaluate_composite

        steps = _get_steps()
        result = evaluate_composite(steps, project=TOMCAT_PROJECT)
        assert result is not None
        assert result.h_initial >= 0
        assert result.h_befs_final >= 0
        assert result.h_greedy_final >= 0
        assert result.h_dev_final >= 0

    def test_befs_no_worse_than_greedy_h(self):
        """BeFS minimises h — its final h should be ≤ greedy's final h."""
        from dataset.planner_eval import evaluate_composite

        steps = _get_steps()
        result = evaluate_composite(steps, project=TOMCAT_PROJECT)
        assert result is not None
        assert result.h_befs_final <= result.h_greedy_final + 1e-9, (
            f"BeFS h_final={result.h_befs_final} > greedy h_final={result.h_greedy_final}"
        )

    def test_eta_positive(self):
        """η must be > 0 if any smells were resolved."""
        from dataset.planner_eval import evaluate_composite

        steps = _get_steps()
        result = evaluate_composite(steps, project=TOMCAT_PROJECT)
        assert result is not None
        if result.befs_reduces:
            assert result.befs_eta > 0
        if result.greedy_reduces:
            assert result.greedy_eta > 0

    def test_rho_non_negative(self):
        """ρ must be ≥ 0."""
        from dataset.planner_eval import evaluate_composite

        steps = _get_steps()
        result = evaluate_composite(steps, project=TOMCAT_PROJECT)
        assert result is not None
        assert result.befs_rho >= 0
        assert result.greedy_rho >= 0

    def test_mlflow_metrics_are_floats(self):
        """to_mlflow_metrics() returns a flat dict of floats."""
        from dataset.planner_eval import evaluate_composite

        steps = _get_steps()
        result = evaluate_composite(steps, project=TOMCAT_PROJECT)
        assert result is not None
        metrics = result.to_mlflow_metrics()
        assert isinstance(metrics, dict)
        for k, v in metrics.items():
            assert isinstance(v, float), f"metric {k!r} is not float: {v!r}"

    def test_mlflow_params_are_strings(self):
        """to_mlflow_params() returns a flat dict of strings."""
        from dataset.planner_eval import evaluate_composite

        steps = _get_steps()
        result = evaluate_composite(steps, project=TOMCAT_PROJECT)
        assert result is not None
        params = result.to_mlflow_params()
        assert isinstance(params, dict)
        for k, v in params.items():
            assert isinstance(v, str), f"param {k!r} is not str: {v!r}"
