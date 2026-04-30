"""Tests for agents.dependency_analysis.agent."""

from langgraph.store.memory import InMemoryStore

from agents.dependency_analysis.agent import (
    analyze_dependencies,
    build_smell_graph,
    issue_to_smell_event,
    issues_to_smell_events,
    prioritize_smells,
)
from store.smell_store import SmellStore


SONAR_ISSUES = [
    {
        "rule": "java:S138",
        "severity": "CRITICAL",
        "line": 10,
        "component": "proj:src/A.java",
        "message": "Method has too many lines",
    },
    {
        "rule": "java:S107",
        "severity": "MAJOR",
        "line": 20,
        "component": "proj:src/A.java",
        "message": "Too many parameters",
    },
    {
        "rule": "java:S1200",
        "severity": "BLOCKER",
        "line": 1,
        "component": "proj:src/B.java",
        "message": "God class",
    },
]


def test_issue_to_smell_event_converts_issue():
    event = issue_to_smell_event(SONAR_ISSUES[0])

    assert event is not None
    assert event.smell_type == "Long Method"
    assert event.file_path == "src/A.java"
    assert event.severity == "HIGH"
    assert event.smell_id == "Long Method:src/A.java:10"



def test_issues_to_smell_events_skips_invalid_items():
    events = issues_to_smell_events(
        SONAR_ISSUES + [{"rule": "java:S138", "component": "missingpath"}],
    )

    assert len(events) == 3
    assert {event.smell_type for event in events} == {
        "Long Method",
        "Long Parameter List",
        "God Class",
    }



def test_build_smell_graph_and_persist_to_store():
    store = InMemoryStore()

    graph = build_smell_graph(
        SONAR_ISSUES,
        store=store,
        session_id="sess1",
        iteration=4,
    )

    assert len(graph) == 3
    assert "Long Parameter List:src/A.java:20" in graph.positive_neighbors(
        "Long Method:src/A.java:10"
    )

    loaded = SmellStore(store).load_graph("sess1")
    assert loaded is not None
    assert set(loaded.all_smell_ids()) == set(graph.all_smell_ids())
    assert SmellStore(store).get_meta("sess1") == {
        "iteration": 4,
        "node_count": 3,
        "edge_count": graph.graph.number_of_edges(),
    }



def test_analyze_dependencies_uses_graph_and_keeps_shape():
    results = analyze_dependencies(SONAR_ISSUES)

    by_smell = {item.smell_type: item for item in results}
    assert set(by_smell) == {
        "Long Method",
        "Long Parameter List",
        "God Class",
    }
    assert by_smell["Long Method"].rule_id == "java:S138"
    assert "Long Parameter List" in by_smell["Long Method"].positive_dependencies
    assert by_smell["Long Parameter List"].negative_dependencies == ["Data Class"]



def test_prioritize_smells_persists_queue_when_store_provided():
    store = InMemoryStore()

    priorities = prioritize_smells(
        SONAR_ISSUES,
        store=store,
        session_id="sess2",
        iteration=1,
    )

    assert len(priorities) == 3
    saved = SmellStore(store).load_priorities("sess2")
    assert saved == priorities
    assert priorities[0]["smell_type"] == "Long Method"
