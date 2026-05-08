"""Composite outcome classification for smell-incidence experiments.

Paper alignment:
- RQ2 classification: positive / neutral / negative by smell-count delta.
- Finding 7: negative composites are non-negligible, so keep this as first-class
  metric rather than secondary logging detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Outcome = Literal["positive", "neutral", "negative"]


@dataclass(frozen=True)
class OutcomeDelta:
    before_count: int
    after_count: int

    @property
    def outcome(self) -> Outcome:
        if self.after_count < self.before_count:
            return "positive"
        if self.after_count > self.before_count:
            return "negative"
        return "neutral"


def classify_smell_incidence(before_count: int, after_count: int) -> Outcome:
    return OutcomeDelta(before_count=before_count, after_count=after_count).outcome
