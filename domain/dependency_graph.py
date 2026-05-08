"""Inter-smell dependency graph built from SmellEvents + Markovič rules.

Immutable after construction. No planning logic — that belongs in
RefactoringTree. Uses networkx MultiDiGraph internally.

Spec reference: §III-B, Eq. 2 (conf_Pietukhin_10_3_rev2-2.pdf).
"""
from __future__ import annotations

from typing import Any, Sequence

import networkx as nx

from collections import Counter

from domain.models import SmellEvent
from domain.rules import DEPENDENCY_RULES


class DependencyGraph:
    """Inter-smell dependency graph. Immutable after from_events()."""

    __slots__ = ("_g", "_freq")

    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()
        self._freq: dict[str, int] = {}

    def __len__(self) -> int:
        return self._g.number_of_nodes()

    def __contains__(self, smell_id: str) -> bool:
        return smell_id in self._g

    # --- Queries ---

    def smell_type_of(self, smell_id: str) -> str:
        return self._g.nodes[smell_id]["smell_type"]

    def severity_of(self, smell_id: str) -> int:
        return self._g.nodes[smell_id]["severity_score"]

    def positive_neighbors(self, smell_id: str) -> list[str]:
        return [
            dst for _, dst, d in self._g.out_edges(smell_id, data=True)
            if d.get("relation") == "positive"
        ]

    @property
    def graph(self) -> nx.MultiDiGraph:
        """Raw networkx graph for visualization/introspection."""
        return self._g

    def all_smell_ids(self) -> list[str]:
        return list(self._g.nodes)

    def node_data(self, smell_id: str) -> dict[str, Any]:
        return dict(self._g.nodes[smell_id])

    def negative_neighbors(self, smell_id: str) -> list[str]:
        return [
            dst for _, dst, d in self._g.out_edges(smell_id, data=True)
            if d.get("relation") == "negative"
        ]

    def resolved_by(self, smell_id: str, active: frozenset[str]) -> frozenset[str]:
        """Smells resolved if we refactor smell_id: itself + positive neighbors in active."""
        result = {smell_id}
        for neighbor in self.positive_neighbors(smell_id):
            if neighbor in active:
                result.add(neighbor)
        return frozenset(result)

    def created_by(self, smell_id: str, active: frozenset[str]) -> frozenset[str]:
        """Smells that might be introduced: negative neighbors in active set."""
        return frozenset(
            n for n in self.negative_neighbors(smell_id) if n in active
        )

    def score(self, smell_id: str, *, w_sev: float = 0.33, w_neg: float = 0.5) -> float:
        """P_i^conc = f_i * w_sev * sev(s_i) + Σ pos_out^conc - w_neg * Σ neg_out^abs.

        Eq. 2 from paper. freq is counted over all nodes in the graph.
        """
        node = self._g.nodes[smell_id]
        sev = node["severity_score"]
        smell_type = node["smell_type"]
        freq = self._freq[smell_type]
        pos_out = len(self.positive_neighbors(smell_id))
        neg_out = len(self.negative_neighbors(smell_id))
        return freq * w_sev * sev + pos_out - w_neg * neg_out

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        nodes = [
            {"smell_id": nid, **attrs}
            for nid, attrs in self._g.nodes(data=True)
        ]
        edges = [
            {"source": src, "target": dst, **attrs}
            for src, dst, attrs in self._g.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges, "freq": self._freq}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DependencyGraph:
        dg = cls()
        dg._freq = dict(data.get("freq", {}))
        for node in data.get("nodes", []):
            smell_id = node["smell_id"]
            attrs = {k: v for k, v in node.items() if k != "smell_id"}
            dg._g.add_node(smell_id, **attrs)
        for edge in data.get("edges", []):
            src, tgt = edge["source"], edge["target"]
            attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
            dg._g.add_edge(src, tgt, **attrs)
        return dg

    # --- Construction ---

    @classmethod
    def from_events(
        cls,
        events: Sequence[SmellEvent],
        *,
        locality: str = "none",
    ) -> DependencyGraph:
        dg = cls()
        dg._freq = dict(Counter(s.smell_type for s in events))
        for smell in events:
            dg._g.add_node(
                smell.smell_id,
                smell_type=smell.smell_type,
                severity=smell.severity,
                severity_score=smell.severity_score,
                file_path=smell.file_path,
                line_number=smell.line_number,
                class_name=smell.class_name,
            )

        # Add edges from Markovič dependency rules
        for src in events:
            rules = DEPENDENCY_RULES.get(src.smell_type, {})
            pos_types = rules.get("positive", [])
            neg_types = rules.get("negative", [])
            for tgt in events:
                if src.smell_id == tgt.smell_id:
                    continue
                if not _same_locality(src, tgt, locality):
                    continue
                if tgt.smell_type in pos_types:
                    dg._g.add_edge(src.smell_id, tgt.smell_id, relation="positive")
                if tgt.smell_type in neg_types:
                    dg._g.add_edge(src.smell_id, tgt.smell_id, relation="negative")

        return dg


def _same_locality(src: SmellEvent, tgt: SmellEvent, locality: str) -> bool:
    if locality == "none":
        return True
    if locality == "file":
        return src.file_path == tgt.file_path
    if locality == "class":
        return src.class_context == tgt.class_context
    raise ValueError(f"Unsupported locality={locality!r}")
