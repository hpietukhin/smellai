"""TDD tests for DatasetGraph — read-only Neo4j accessor.

Requires running Neo4j with Composite Refactorings 2020 data.
Tests are skipped if Neo4j is unreachable.

Vertical slices:
1. Connection + smell_state at a commit
2. Refactorings at a commit
3. Range-based composite extraction with intermediate smell states
"""
from __future__ import annotations

import pytest

from smellai_datasets.composite_dataset import is_available

pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="Neo4j not available",
)


# Known good data from probe: Apache Tomcat, commit order 16119
# Episode: Apache Tomcat:24c8d8c635a694d9e8832c8de4fef508c49c6987:16119
# 11 smells before, 2 Extract Method refactorings
TOMCAT_PROJECT = "Apache Tomcat"
TOMCAT_COMMIT = "24c8d8c635a694d9e8832c8de4fef508c49c6987"
TOMCAT_ELEMENT = "org.apache.catalina.servlets.DefaultServlet"


class TestConnection:

    def test_can_connect(self):
        from dataset.neo4j_graph import DatasetGraph
        dg = DatasetGraph()
        assert dg.is_available()


class TestSmellState:

    def test_smell_state_returns_smell_events(self):
        from dataset.neo4j_graph import DatasetGraph
        dg = DatasetGraph()
        smells = dg.smell_state({TOMCAT_ELEMENT}, TOMCAT_COMMIT)
        assert len(smells) > 0
        # All results should be SmellEvents
        from domain.models import SmellEvent
        assert all(isinstance(s, SmellEvent) for s in smells)

    def test_smell_state_for_known_episode_has_expected_types(self):
        from dataset.neo4j_graph import DatasetGraph
        dg = DatasetGraph()
        smells = dg.smell_state({TOMCAT_ELEMENT}, TOMCAT_COMMIT)
        smell_types = {s.smell_type for s in smells}
        # Should contain at least GodClass and LongMethod (known from probe)
        assert "God Class" in smell_types or "GodClass" in smell_types

    def test_smell_state_empty_for_nonexistent_commit(self):
        from dataset.neo4j_graph import DatasetGraph
        dg = DatasetGraph()
        smells = dg.smell_state({TOMCAT_ELEMENT}, "nonexistent_hash_000")
        assert smells == []


class TestRefactorings:

    def test_refactorings_at_commit(self):
        """The known Tomcat commit has 2 Extract Method refactorings."""
        from dataset.neo4j_graph import DatasetGraph
        dg = DatasetGraph()
        refs = dg.refactorings_at_start(TOMCAT_COMMIT)
        assert len(refs) >= 2
        assert all(r.ref_type for r in refs)


class TestCompositeRefactoring:
    """Range-based composite refactoring with intermediate smell states."""

    def test_composite_has_multiple_steps(self):
        from dataset.neo4j_graph import DatasetGraph, CompositeStep
        dg = DatasetGraph()
        steps = dg.composite_refactoring(
            elements={TOMCAT_ELEMENT},
            project=TOMCAT_PROJECT,
            max_steps=10,
        )
        assert len(steps) >= 2
        for step in steps:
            assert isinstance(step, CompositeStep)
            assert step.commit_hash
            assert step.commit_order > 0

    def test_steps_ordered_by_commit(self):
        from dataset.neo4j_graph import DatasetGraph
        dg = DatasetGraph()
        steps = dg.composite_refactoring(
            elements={TOMCAT_ELEMENT},
            project=TOMCAT_PROJECT,
            max_steps=5,
        )
        orders = [s.commit_order for s in steps]
        assert orders == sorted(orders)

    def test_steps_have_smell_state(self):
        from dataset.neo4j_graph import DatasetGraph
        dg = DatasetGraph()
        steps = dg.composite_refactoring(
            elements={TOMCAT_ELEMENT},
            project=TOMCAT_PROJECT,
            max_steps=10,
        )
        assert any(len(s.smells) > 0 for s in steps)

    def test_steps_have_refactorings(self):
        from dataset.neo4j_graph import DatasetGraph
        dg = DatasetGraph()
        steps = dg.composite_refactoring(
            elements={TOMCAT_ELEMENT},
            project=TOMCAT_PROJECT,
            max_steps=20,
        )
        assert any(len(s.refactorings) > 0 for s in steps)


class TestFullPipeline:
    """End-to-end: DatasetGraph → DependencyGraph → RefactoringTree → Plan."""

    def test_composite_to_plan(self):
        from dataset.neo4j_graph import DatasetGraph
        from domain.dependency_graph import DependencyGraph
        from domain.refactoring_tree import RefactoringTree, State

        ds = DatasetGraph()
        steps = ds.composite_refactoring(
            elements={TOMCAT_ELEMENT},
            project=TOMCAT_PROJECT,
            max_steps=10,
        )

        initial_step = next((s for s in steps if s.smells), None)
        assert initial_step is not None

        dep_graph = DependencyGraph.from_events(initial_step.smells, locality="none")
        assert len(dep_graph) > 0

        initial = State(frozenset(e.smell_id for e in initial_step.smells))
        tree = RefactoringTree(initial, dep_graph)

        plan_greedy = tree.greedy()
        plan_befs = tree.befs()

        assert plan_greedy.h_trace[-1] == 0
        assert plan_befs.h_trace[-1] == 0
        assert len(plan_befs.actions) <= len(plan_greedy.actions)
