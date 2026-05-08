"""Smell detection interface and pure-domain detector implementations.

`SmellDetector` is the architectural boundary for smell detection. High-level
workflow code should depend on this module, not on infrastructure backends.
Concrete backends live alongside their infrastructure (e.g. `sonarqube.detector`).

Contract for `SmellDetector.detect(...)`:
- accepts a checked-out local project path;
- returns a normalized `list[SmellEvent]` (domain type, no ORM fields);
- may raise `SmellDetectionError` subclasses for configuration, availability,
  or backend execution failures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from domain.models import SmellEvent


class SmellDetectionError(RuntimeError):
    """Base class for detector failures."""


class DetectorConfigError(SmellDetectionError):
    """Detector is misconfigured (missing token, invalid settings, etc.)."""


class DetectorUnavailableError(SmellDetectionError):
    """Detector backend is unavailable (server down, CLI missing, etc.)."""


class DetectorExecutionError(SmellDetectionError):
    """Detector backend failed while scanning the project."""


class SmellDetector(ABC):
    """Strategy interface for smell detection backends."""

    @abstractmethod
    def detect(self, project_path: Path) -> list[SmellEvent]:
        """Scan ``project_path`` and return detected domain SmellEvents."""
        ...

    @staticmethod
    def compare(
        before: list[SmellEvent],
        after: list[SmellEvent],
    ) -> dict[str, list[str]]:
        """Compare two smell sets — resolved / created / persisted."""
        before_ids = {s.smell_id for s in before}
        after_ids = {s.smell_id for s in after}
        return {
            "resolved": list(before_ids - after_ids),
            "created": list(after_ids - before_ids),
            "persisted": list(before_ids & after_ids),
        }


class StaticDetector(SmellDetector):
    """Simple test/dry-run detector that returns a predefined smell set."""

    def __init__(self, smells: Sequence[SmellEvent] | None = None) -> None:
        self._smells = list(smells or [])

    def detect(self, project_path: Path) -> list[SmellEvent]:
        del project_path
        return [replace(smell) for smell in self._smells]
