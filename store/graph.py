"""In-memory smell dependency graph built on networkx.

Spec reference: §4.4, Eq. 1-2, Algorithms 1-2.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import networkx as nx

from agents.dependency_analysis.scorer import STANDARD_SCORE, ScoringContext
from store.rules import DEPENDENCY_RULES
from swe_refactor.persistence.models import SmellEvent


class _SmellProxy(NamedTuple):
    severity_score: int


def _count_by_relation(g: nx.MultiDiGraph, node: str, relation: str) -> int:
    return sum(
        1 for _, _, d in g.out_edges(node, data=True)
        if d.get("relation") == relation
    )


class SmellGraph:
    """Thin wrapper around ``nx.MultiDiGraph`` specialised for smell instances."""

    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()
        self._file_index: dict[str, set[str]] = defaultdict(set)

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self._g

    def __len__(self) -> int:
        return self._g.number_of_nodes()

    def __contains__(self, smell_id: str) -> bool:
        return smell_id in self._g

    # -- Mutation ----------------------------------------------------------

    def add_smell(self, smell: SmellEvent) -> None:
        self._g.add_node(
            smell.smell_id,
            smell_type=smell.smell_type,
            file_path=smell.file_path,
            line_number=smell.line_number,
            severity=smell.severity,
            severity_score=smell.severity_score,
        )
        self._file_index[smell.file_path].add(smell.smell_id)

    def remove_smell(self, smell_id: str) -> None:
        if smell_id not in self._g:
            return
        file_path = self._g.nodes[smell_id].get("file_path", "")
        self._g.remove_node(smell_id)
        if file_path in self._file_index:
            self._file_index[file_path].discard(smell_id)
            if not self._file_index[file_path]:
                del self._file_index[file_path]

    def add_dependency(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0,
    ) -> None:
        self._g.add_edge(source, target, relation=relation, weight=weight)

    # -- Queries -----------------------------------------------------------

    def successors(self, smell_id: str) -> list[str]:
        return list(self._g.successors(smell_id))

    def predecessors(self, smell_id: str) -> list[str]:
        return list(self._g.predecessors(smell_id))

    def positive_neighbors(self, smell_id: str) -> list[str]:
        return [
            dst
            for _, dst, d in self._g.out_edges(smell_id, data=True)
            if d.get("relation") == "positive"
        ]

    def negative_neighbors(self, smell_id: str) -> list[str]:
        return [
            dst
            for _, dst, d in self._g.out_edges(smell_id, data=True)
            if d.get("relation") == "negative"
        ]

    def smells_for_file(self, file_path: str) -> list[str]:
        return list(self._file_index.get(file_path, set()))

    def node_data(self, smell_id: str) -> dict[str, Any]:
        return dict(self._g.nodes[smell_id])

    def all_smell_ids(self) -> list[str]:
        return list(self._g.nodes)

    # -- Construction ------------------------------------------------------

    @classmethod
    def from_smells(cls, smells: Sequence[SmellEvent]) -> SmellGraph:
        """Build graph with edges derived from ``DEPENDENCY_RULES``.

        Two smells in the same file are connected when the source type
        lists the target type as a positive or negative dependency.
        """
        g = cls()
        for smell in smells:
            g.add_smell(smell)

        for src in smells:
            rules = DEPENDENCY_RULES.get(src.smell_type, {})
            positive = rules.get("positive", [])
            negative = rules.get("negative", [])
            for tgt in smells:
                if src.smell_id == tgt.smell_id or src.file_path != tgt.file_path:
                    continue
                if tgt.smell_type in positive:
                    g.add_dependency(src.smell_id, tgt.smell_id, "positive")
                if tgt.smell_type in negative:
                    g.add_dependency(src.smell_id, tgt.smell_id, "negative")
        return g

    # -- Prioritization (greedy, Algorithm 1) ------------------------------

    def calculate_priorities(
        self, score_fn: Callable = STANDARD_SCORE,
    ) -> list[dict[str, Any]]:
        """Greedy planner: pick highest-scoring smell, remove, repeat."""
        freq_map = Counter(
            self._g.nodes[n].get("smell_type") for n in self._g.nodes
        )
        working = self._g.copy()
        sequence: list[dict[str, Any]] = []

        while working.number_of_nodes() > 0:
            scores: dict[str, float] = {}
            for node in working.nodes():
                attrs = working.nodes[node]
                context = ScoringContext(
                    freq=freq_map.get(attrs.get("smell_type", ""), 1),
                    pos_out=_count_by_relation(working, node, "positive"),
                    neg_out=_count_by_relation(working, node, "negative"),
                )
                scores[node] = score_fn(
                    _SmellProxy(attrs.get("severity_score", 1)),
                    context,
                )

            best = max(scores, key=scores.get)  # type: ignore[arg-type]
            best_attrs = working.nodes[best]
            sequence.append({
                "order": len(sequence) + 1,
                "smell_id": best,
                "smell_type": best_attrs.get("smell_type", ""),
                "file_path": best_attrs.get("file_path", ""),
                "pz_score": scores[best],
                "positive_impacts": _count_by_relation(
                    working, best, "positive",
                ),
                "negative_impacts": _count_by_relation(
                    working, best, "negative",
                ),
            })
            working.remove_node(best)

        return sequence

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        nodes = [
            {"smell_id": nid, **attrs}
            for nid, attrs in self._g.nodes(data=True)
        ]
        edges = [
            {"source": src, "target": dst, **attrs}
            for src, dst, attrs in self._g.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SmellGraph:
        g = cls()
        for node in data.get("nodes", []):
            smell_id = node["smell_id"]
            attrs = {k: v for k, v in node.items() if k != "smell_id"}
            g._g.add_node(smell_id, **attrs)
            fp = attrs.get("file_path", "")
            if fp:
                g._file_index[fp].add(smell_id)

        for edge in data.get("edges", []):
            src, tgt = edge["source"], edge["target"]
            attrs = {
                k: v for k, v in edge.items()
                if k not in ("source", "target")
            }
            g._g.add_edge(src, tgt, **attrs)
        return g
