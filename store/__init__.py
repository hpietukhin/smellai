"""Shared smell graph, LangGraph Store adapter, and detector interface.

Usage in LangGraph nodes::

    def my_node(state, *, store: BaseStore):
        smell_store = SmellStore(store)
        graph = smell_store.load_graph(session_id)
        ...
        smell_store.save_graph(session_id, graph)
"""

from store.detector import (
    DetectorConfigError,
    DetectorExecutionError,
    DetectorUnavailableError,
    SmellDetectionError,
    SmellDetector,
    SonarQubeDetector,
    StaticDetector,
)
from store.graph import SmellGraph
from store.rules import DEPENDENCY_RULES
from store.smell_store import SmellStore

__all__ = [
    "DEPENDENCY_RULES",
    "DetectorConfigError",
    "DetectorExecutionError",
    "DetectorUnavailableError",
    "SmellDetectionError",
    "SmellDetector",
    "SmellGraph",
    "SmellStore",
    "SonarQubeDetector",
    "StaticDetector",
]
