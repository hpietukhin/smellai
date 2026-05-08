"""SmellDetector backed by Neo4j dataset snapshots.

Returns the exact smell state recorded in the Composite Refactorings 2020
dataset at a given commit for a set of elements. No live analysis — this
provides ground truth for planner evaluation.

Usage:
    detector = DatasetDetector(
        elements={"org.apache.Foo"},
        commit_hash="abc123",
    )
    smells = detector.detect(Path("/unused"))  # path ignored
"""
from __future__ import annotations

from pathlib import Path

from domain.detector import SmellDetector
from domain.models import SmellEvent


class DatasetDetector(SmellDetector):
    """Ground truth smell detector using Neo4j dataset snapshots.

    The ``project_path`` argument to ``detect()`` is ignored — smells come
    from the dataset graph, not from live code analysis.
    """

    def __init__(
        self,
        elements: set[str],
        commit_hash: str,
        *,
        neo4j_uri: str = "http://localhost:7474",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "boil2.eat",
    ) -> None:
        self._elements = elements
        self._commit_hash = commit_hash
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password

    def detect(self, project_path: Path) -> list[SmellEvent]:
        """Return dataset smell state at the configured commit.

        ``project_path`` is ignored — smells come from Neo4j.
        """
        from dataset.neo4j_graph import DatasetGraph

        ds = DatasetGraph(
            uri=self._neo4j_uri,
            user=self._neo4j_user,
            password=self._neo4j_password,
        )
        return ds.smell_state(self._elements, self._commit_hash)
