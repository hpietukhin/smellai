"""Dependency analysis package exports.

Kept lazy to avoid import cycles with ``store.graph`` during module loading.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DEPENDENCY_RULES",
    "DependencyAnalysis",
    "analyze_dependencies",
    "build_smell_graph",
    "issue_to_smell_event",
    "issues_to_smell_events",
    "prioritize_smells",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(".agent", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
