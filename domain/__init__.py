"""Smell domain — core models, graph, rules, detector, and scoring.

This package is the domain layer: it has no dependency on LangGraph, MLflow,
or any external infrastructure. Infrastructure adapters (e.g. SmellStore) live
in ``store/`` and depend on this package, not the other way around.
"""

from domain.models import SmellAction, SmellEvent
from domain.detector import (
    DetectorConfigError,
    DetectorExecutionError,
    DetectorUnavailableError,
    SmellDetectionError,
    SmellDetector,
    StaticDetector,
)
from domain.graph import SmellGraph
from domain.rules import DEPENDENCY_RULES
from domain.scorer import STANDARD_SCORE, STANDARD_H, ScoringContext

__all__ = [
    "DEPENDENCY_RULES",
    "DetectorConfigError",
    "DetectorExecutionError",
    "DetectorUnavailableError",
    "SmellAction",
    "SmellDetectionError",
    "SmellDetector",
    "SmellEvent",
    "SmellGraph",
    "STANDARD_H",
    "STANDARD_SCORE",
    "ScoringContext",
    "StaticDetector",
]
