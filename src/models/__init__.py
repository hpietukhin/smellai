"""Data models for code smell detection and evaluation."""

from .entities import (
    DACOSSample,
    EvaluationResult,
    EvaluationScore,
    SmellAnnotation,
    SmellDetection,
    SmellEvaluation,
)

__all__ = [
    "SmellAnnotation",
    "SmellDetection",
    "SmellEvaluation",
    "EvaluationResult",
    "DACOSSample",
    "EvaluationScore",
]
