"""LangGraph Store adapter for persisting SmellGraph between agent steps.

Usage in a LangGraph node::

    def a2_prioritize(state, *, store: BaseStore):
        ss = SmellStore(store)
        graph = ss.load_graph(session_id)
        priorities = graph.calculate_priorities()
        return {"priority_queue": [p["smell_id"] for p in priorities]}
"""

from __future__ import annotations

from typing import Any

from langgraph.store.base import BaseStore

from store.graph import SmellGraph

_NS_PREFIX = "smell_graph"


def _namespace(session_id: str, *parts: str) -> tuple[str, ...]:
    return (_NS_PREFIX, session_id, *parts)


class SmellStore:
    """Domain adapter over LangGraph ``BaseStore``."""

    def __init__(self, store: BaseStore) -> None:
        self._store = store

    def save_graph(
        self,
        session_id: str,
        graph: SmellGraph,
        *,
        iteration: int = 0,
    ) -> None:
        self._store.put(_namespace(session_id, "snapshot"), "latest", graph.to_dict())
        self._store.put(_namespace(session_id, "meta"), "info", {
            "iteration": iteration,
            "node_count": len(graph),
            "edge_count": graph.graph.number_of_edges(),
        })

    def load_graph(self, session_id: str) -> SmellGraph | None:
        item = self._store.get(_namespace(session_id, "snapshot"), "latest")
        if item is None:
            return None
        return SmellGraph.from_dict(item.value)

    def get_meta(self, session_id: str) -> dict[str, Any] | None:
        item = self._store.get(_namespace(session_id, "meta"), "info")
        return item.value if item else None

    def save_priorities(
        self,
        session_id: str,
        priorities: list[dict[str, Any]],
    ) -> None:
        self._store.put(_namespace(session_id, "priorities"), "latest", {"queue": priorities})

    def load_priorities(
        self, session_id: str,
    ) -> list[dict[str, Any]] | None:
        item = self._store.get(_namespace(session_id, "priorities"), "latest")
        if item is None:
            return None
        return item.value.get("queue", [])
