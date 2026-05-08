"""Smell domain — core models, dependency graph, refactoring tree, rules, detector, and scoring.

This package is the domain layer: it has no dependency on LangGraph, MLflow,
or any external infrastructure. Infrastructure adapters (e.g. SmellStore) live
in ``store/`` and depend on this package, not the other way around.
"""

from domain.models import SmellEvent
from domain.detector import (
    DetectorConfigError,
    DetectorExecutionError,
    DetectorUnavailableError,
    SmellDetectionError,
    SmellDetector,
    StaticDetector,
)
from domain.dependency_graph import DependencyGraph
from domain.refactoring_tree import (
    RefactoringTree,
    State,
    RefactoringAction,
    Plan,
)
from domain.rules import (
    DEPENDENCY_RULES,
    REFACTORING_CATALOGUE,
    SMELL_GROUPS,
    SMELL_GROUP_MEMBERSHIP,
    get_default_severity,
    get_refactoring_types,
    get_smell_groups,
    normalize_dataset_smell_type,
    smells_in_group,
)
from domain.scorer import STANDARD_SCORE, STANDARD_H, ScoringContext

__all__ = [
    # rules
    "DEPENDENCY_RULES",
    "REFACTORING_CATALOGUE",
    "SMELL_GROUPS",
    "SMELL_GROUP_MEMBERSHIP",
    "get_default_severity",
    "get_refactoring_types",
    "get_smell_groups",
    "normalize_dataset_smell_type",
    "smells_in_group",
    # detector
    "DetectorConfigError",
    "DetectorExecutionError",
    "DetectorUnavailableError",
    "SmellDetectionError",
    "SmellDetector",
    "StaticDetector",
    # models
    "SmellEvent",
    # dependency graph
    "DependencyGraph",
    # refactoring tree
    "RefactoringTree",
    "State",
    "RefactoringAction",
    "Plan",
    # scorer
    "STANDARD_H",
    "STANDARD_SCORE",
    "ScoringContext",
]
