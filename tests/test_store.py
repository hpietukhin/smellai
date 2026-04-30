"""Tests for store package: SmellGraph, SmellStore, SmellDetector."""

from typing import TypedDict

import pytest
from langgraph.graph import StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from domain.detector import (
    DetectorConfigError,
    SmellDetector,
    SonarQubeDetector,
    StaticDetector,
)
from domain.graph import SmellGraph
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


# === SmellGraph ===========================================================


class TestSmellGraph:
    def test_add_and_contains(self):
        g = SmellGraph()
        g.add_smell(_smell("s1"))
        assert "s1" in g
        assert len(g) == 1

    def test_remove_smell(self):
        g = SmellGraph()
        g.add_smell(_smell("s1"))
        g.remove_smell("s1")
        assert "s1" not in g
        assert len(g) == 0

    def test_remove_nonexistent_is_noop(self):
        SmellGraph().remove_smell("nope")

    def test_file_index(self):
        g = SmellGraph()
        g.add_smell(_smell("s1", file_path="A.java"))
        g.add_smell(_smell("s2", file_path="A.java"))
        g.add_smell(_smell("s3", file_path="B.java"))
        assert set(g.smells_for_file("A.java")) == {"s1", "s2"}
        assert g.smells_for_file("B.java") == ["s3"]
        assert g.smells_for_file("C.java") == []

    def test_file_index_after_remove(self):
        g = SmellGraph()
        g.add_smell(_smell("s1", file_path="A.java"))
        g.add_smell(_smell("s2", file_path="A.java"))
        g.remove_smell("s1")
        assert g.smells_for_file("A.java") == ["s2"]

    def test_dependencies(self):
        g = SmellGraph()
        g.add_smell(_smell("s1"))
        g.add_smell(_smell("s2"))
        g.add_smell(_smell("s3"))
        g.add_dependency("s1", "s2", "positive")
        g.add_dependency("s1", "s3", "negative")
        assert g.positive_neighbors("s1") == ["s2"]
        assert g.negative_neighbors("s1") == ["s3"]
        assert g.successors("s1") == ["s2", "s3"]
        assert g.predecessors("s2") == ["s1"]

    def test_from_smells_wires_edges(self):
        g = SmellGraph.from_smells(SAMPLES)
        assert len(g) == len(SAMPLES)
        pos = g.positive_neighbors("LM:A:10")
        assert "LPL:A:20" in pos
        assert g.positive_neighbors("LM:B:5") == []

    def test_serialization_roundtrip(self):
        g = SmellGraph.from_smells(SAMPLES)
        g2 = SmellGraph.from_dict(g.to_dict())
        assert len(g2) == len(g)
        assert set(g2.all_smell_ids()) == set(g.all_smell_ids())
        assert g2.positive_neighbors("LM:A:10") == g.positive_neighbors("LM:A:10")

    def test_calculate_priorities(self):
        g = SmellGraph.from_smells(SAMPLES)
        seq = g.calculate_priorities()
        assert len(seq) == len(SAMPLES)
        assert seq[0]["order"] == 1
        ids = {item["smell_id"] for item in seq}
        assert ids == {s.smell_id for s in SAMPLES}

    def test_node_data(self):
        g = SmellGraph()
        g.add_smell(_smell("s1", severity="MEDIUM"))
        d = g.node_data("s1")
        assert d["severity"] == "MEDIUM"
        assert d["smell_type"] == "Long Method"


# === SmellStore ===========================================================


class TestSmellStore:
    def test_save_and_load_graph(self):
        ss = SmellStore(InMemoryStore())
        g = SmellGraph.from_smells(SAMPLES)
        ss.save_graph("sess1", g, iteration=3)

        loaded = ss.load_graph("sess1")
        assert loaded is not None
        assert len(loaded) == len(g)
        assert set(loaded.all_smell_ids()) == set(g.all_smell_ids())

    def test_load_nonexistent_returns_none(self):
        assert SmellStore(InMemoryStore()).load_graph("no_such") is None

    def test_meta(self):
        ss = SmellStore(InMemoryStore())
        g = SmellGraph.from_smells(SAMPLES)
        ss.save_graph("sess1", g, iteration=5)

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
        g1 = SmellGraph()
        g1.add_smell(_smell("s1"))
        ss.save_graph("sess1", g1)

        g2 = SmellGraph()
        g2.add_smell(_smell("s2"))
        g2.add_smell(_smell("s3"))
        ss.save_graph("sess1", g2, iteration=1)

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
            graph = SmellStore(store).load_graph("int_test")
            if graph is None:
                return {"priority_queue": []}
            seq = graph.calculate_priorities()
            return {"priority_queue": [p["smell_id"] for p in seq]}

        raw_store = InMemoryStore()
        SmellStore(raw_store).save_graph(
            "int_test", SmellGraph.from_smells(SAMPLES),
        )

        builder = StateGraph(S)
        builder.add_node("prioritize", prioritize_node)
        builder.set_entry_point("prioritize")
        builder.set_finish_point("prioritize")
        app = builder.compile(store=raw_store)

        result = app.invoke({"priority_queue": []})
        assert len(result["priority_queue"]) == len(SAMPLES)
