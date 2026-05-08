"""Tests for store package: DependencyGraph, SmellStore, SmellDetector."""

from typing import TypedDict

import pytest
from langgraph.graph import StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from domain.detector import (
    DetectorConfigError,
    SmellDetector,
    StaticDetector,
)
from sonarqube.detector import SonarQubeDetector
from domain.dependency_graph import DependencyGraph
from domain.refactoring_tree import RefactoringTree, State
from store.smell_store import SmellStore
from domain.models import SmellEvent


def _smell(
    smell_id: str,
    smell_type: str = "Long Method",
    file_path: str = "src/A.java",
    line: int = 10,
    severity: str = "HIGH",
) -> SmellEvent:
    return SmellEvent(
        smell_id=smell_id,
        smell_type=smell_type,
        file_path=file_path,
        line_number=line,
        severity=severity,
    )


SAMPLES = [
    _smell("LM:A:10", "Long Method", "src/A.java", 10, "HIGH"),
    _smell("GC:A:1", "God Class", "src/A.java", 1, "HIGH"),
    _smell("LPL:A:20", "Long Parameter List", "src/A.java", 20, "LOW"),
    _smell("LM:B:5", "Long Method", "src/B.java", 5, "MEDIUM"),
]


# === DependencyGraph ======================================================


class TestDependencyGraphStore:
    def test_from_events_contains_all(self):
        dg = DependencyGraph.from_events(SAMPLES)
        assert len(dg) == len(SAMPLES)
        assert all(s.smell_id in dg for s in SAMPLES)

    def test_positive_edges_same_file(self):
        """LM:A:10 should have positive edge to LPL:A:20 (same file, default locality=none)."""
        dg = DependencyGraph.from_events(SAMPLES)
        pos = dg.positive_neighbors("LM:A:10")
        assert "LPL:A:20" in pos

    def test_cross_file_no_edges_with_class_locality(self):
        dg = DependencyGraph.from_events(SAMPLES, locality="class")
        assert dg.positive_neighbors("LM:B:5") == []

    def test_serialization_roundtrip(self):
        dg = DependencyGraph.from_events(SAMPLES)
        dg2 = DependencyGraph.from_dict(dg.to_dict())
        assert len(dg2) == len(dg)
        assert set(dg2.all_smell_ids()) == set(dg.all_smell_ids())
        assert dg2.positive_neighbors("LM:A:10") == dg.positive_neighbors("LM:A:10")

    def test_greedy_plan_covers_all_smells(self):
        dg = DependencyGraph.from_events(SAMPLES)
        initial = State(frozenset(s.smell_id for s in SAMPLES))
        tree = RefactoringTree(initial, dg)
        plan = tree.greedy()
        assert plan.h_trace[-1] == 0

    def test_node_data(self):
        dg = DependencyGraph.from_events([_smell("s1", severity="MEDIUM")])
        d = dg.node_data("s1")
        assert d["severity"] == "MEDIUM"
        assert d["smell_type"] == "Long Method"


# === SmellStore ===========================================================


class TestSmellStore:
    def test_save_and_load_graph(self):
        ss = SmellStore(InMemoryStore())
        dg = DependencyGraph.from_events(SAMPLES)
        ss.save_graph("sess1", dg, iteration=3)

        loaded = ss.load_graph("sess1")
        assert loaded is not None
        assert len(loaded) == len(dg)
        assert set(loaded.all_smell_ids()) == set(dg.all_smell_ids())

    def test_load_nonexistent_returns_none(self):
        assert SmellStore(InMemoryStore()).load_graph("no_such") is None

    def test_meta(self):
        ss = SmellStore(InMemoryStore())
        dg = DependencyGraph.from_events(SAMPLES)
        ss.save_graph("sess1", dg, iteration=5)

        meta = ss.get_meta("sess1")
        assert meta is not None
        assert meta["iteration"] == 5  # noqa: PLR2004
        assert meta["node_count"] == len(SAMPLES)

    def test_priorities(self):
        ss = SmellStore(InMemoryStore())
        pq = [{"smell_id": "s1", "pz_score": 10}]
        ss.save_priorities("sess1", pq)
        assert ss.load_priorities("sess1") == pq

    def test_overwrite_graph(self):
        ss = SmellStore(InMemoryStore())
        dg1 = DependencyGraph.from_events([_smell("s1")])
        ss.save_graph("sess1", dg1)

        dg2 = DependencyGraph.from_events([_smell("s2"), _smell("s3", smell_type="God Class")])
        ss.save_graph("sess1", dg2, iteration=1)

        loaded = ss.load_graph("sess1")
        assert len(loaded) == 2  # noqa: PLR2004
        assert "s1" not in loaded


# === SmellDetector ========================================================


class TestSmellDetector:
    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            SmellDetector()  # type: ignore[abstract]

    def test_sonarqube_detector_is_subclass(self):
        assert issubclass(SonarQubeDetector, SmellDetector)

    def test_static_detector_is_subclass(self):
        assert issubclass(StaticDetector, SmellDetector)

    def test_compare(self):
        before = [_smell("s1"), _smell("s2"), _smell("s3")]
        after = [_smell("s2"), _smell("s4")]
        diff = SmellDetector.compare(before, after)
        assert set(diff["resolved"]) == {"s1", "s3"}
        assert diff["created"] == ["s4"]
        assert diff["persisted"] == ["s2"]

    def test_static_detector_returns_normalized_events(self, tmp_path):
        detector = StaticDetector([_smell("s1", file_path="A.java", line=7)])

        events = detector.detect(tmp_path)

        assert len(events) == 1
        assert events[0].smell_id == "s1"
        assert events[0].file_path == "A.java"
        assert events[0].line_number == 7

    def test_sonarqube_detector_requires_token(self, tmp_path):
        detector = SonarQubeDetector(sonar_url="http://localhost:9000", sonar_token="")

        with pytest.raises(DetectorConfigError, match="SONAR_TOKEN"):
            detector.detect(tmp_path)


# === LangGraph integration ================================================


class TestLangGraphIntegration:
    def test_node_with_store_injection(self):
        class S(TypedDict):
            priority_queue: list

        def prioritize_node(_state: S, *, store: BaseStore) -> dict:
            dg = SmellStore(store).load_graph("int_test")
            if dg is None:
                return {"priority_queue": []}
            initial = State(frozenset(dg.all_smell_ids()))
            tree = RefactoringTree(initial, dg)
            plan = tree.greedy()
            return {"priority_queue": [a.smell_id for a in plan.actions]}

        raw_store = InMemoryStore()
        SmellStore(raw_store).save_graph(
            "int_test", DependencyGraph.from_events(SAMPLES),
        )

        builder = StateGraph(S)
        builder.add_node("prioritize", prioritize_node)
        builder.set_entry_point("prioritize")
        builder.set_finish_point("prioritize")
        app = builder.compile(store=raw_store)

        result = app.invoke({"priority_queue": []})
        assert len(result["priority_queue"]) > 0
