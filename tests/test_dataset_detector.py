"""TDD tests for DatasetDetector — SmellDetector backed by Neo4j snapshots.

Provides ground truth smell detection for planner evaluation:
instead of running SonarQube/Organic on live code, returns the exact
smell state that the dataset recorded at a given commit.

Requires running Neo4j.
"""
from __future__ import annotations

import pytest

from smellai_datasets.composite_dataset import is_available

pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="Neo4j not available",
)

TOMCAT_ELEMENT = "org.apache.catalina.servlets.DefaultServlet"
TOMCAT_COMMIT = "24c8d8c635a694d9e8832c8de4fef508c49c6987"


class TestDatasetDetector:

    def test_implements_smell_detector_protocol(self):
        from dataset.dataset_detector import DatasetDetector
        from domain.detector import SmellDetector
        assert issubclass(DatasetDetector, SmellDetector)

    def test_detect_returns_smell_events(self):
        from dataset.dataset_detector import DatasetDetector
        from domain.models import SmellEvent
        from pathlib import Path

        detector = DatasetDetector(
            elements={TOMCAT_ELEMENT},
            commit_hash=TOMCAT_COMMIT,
        )
        smells = detector.detect(Path("/unused"))
        assert len(smells) > 0
        assert all(isinstance(s, SmellEvent) for s in smells)

    def test_detect_returns_normalized_smell_types(self):
        """Smell types should be canonical (e.g. 'God Class' not 'GodClass')."""
        from dataset.dataset_detector import DatasetDetector
        from pathlib import Path

        detector = DatasetDetector(
            elements={TOMCAT_ELEMENT},
            commit_hash=TOMCAT_COMMIT,
        )
        smells = detector.detect(Path("/unused"))
        smell_types = {s.smell_type for s in smells}
        # Should use canonical names from rules.py, not raw Neo4j types
        assert "God Class" in smell_types or "Long Method" in smell_types
        assert not any(t[0].islower() for t in smell_types)  # no camelCase

    def test_detect_with_unknown_commit_returns_empty(self):
        from dataset.dataset_detector import DatasetDetector
        from pathlib import Path

        detector = DatasetDetector(
            elements={TOMCAT_ELEMENT},
            commit_hash="nonexistent_hash_0000",
        )
        smells = detector.detect(Path("/unused"))
        assert smells == []

    def test_plugs_into_dependency_graph(self):
        """Full pipeline: DatasetDetector → DependencyGraph → score."""
        from dataset.dataset_detector import DatasetDetector
        from domain.dependency_graph import DependencyGraph
        from pathlib import Path

        detector = DatasetDetector(
            elements={TOMCAT_ELEMENT},
            commit_hash=TOMCAT_COMMIT,
        )
        smells = detector.detect(Path("/unused"))
        dg = DependencyGraph.from_events(smells, locality="none")
        assert len(dg) == len(smells)
        # All smells should have valid scores
        for sid in dg.all_smell_ids():
            assert dg.score(sid) is not None
