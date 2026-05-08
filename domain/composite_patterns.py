"""Composite-smell pattern extraction primitives.

Paper alignment:
- RQ3: identify removal and creational patterns.
- Findings 6/7: effect inference depends on relation-aware grouping and must keep
  negative patterns visible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal


PatternKind = Literal["removal", "creational"]


@dataclass(frozen=True)
class CompositePatternEvent:
    refactoring_types: tuple[str, ...]
    smell_type: str
    kind: PatternKind


def build_pattern_event(
    refactoring_types: list[str],
    smell_type: str,
    before_count: int,
    after_count: int,
) -> CompositePatternEvent | None:
    if after_count == before_count:
        return None
    kind: PatternKind = "removal" if after_count < before_count else "creational"
    return CompositePatternEvent(tuple(refactoring_types), smell_type, kind)


def mine_pattern_frequencies(events: list[CompositePatternEvent]) -> dict[CompositePatternEvent, int]:
    return dict(Counter(events))
