"""LangGraph Store adapter for smell graphs.

Usage in LangGraph nodes::

    def my_node(state, *, store: BaseStore):
        smell_store = SmellStore(store)
        graph = smell_store.load_graph(session_id)
        ...
        smell_store.save_graph(session_id, graph)
"""

from store.smell_store import SmellStore

__all__ = ["SmellStore"]
