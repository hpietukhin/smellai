"""Typed experiment specification for modular methodology runs.

Encodes the main axes used in the paper-style evaluation design:
- synthesis heuristic (element/commit/range)
- planner algorithm
- locality mode
- optional completion-risk tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Heuristic = Literal["element-based", "commit-based", "range-based"]
Planner = Literal["greedy", "befs", "developer"]
Locality = Literal["none", "class", "file"]
ComparisonMode = Literal["planner_vs_planner", "planner_vs_dev_reference"]


@dataclass(frozen=True)
class ExperimentSpec:
    heuristic: Heuristic
    planner: Planner
    locality: Locality = "none"
    track_completion_risk: bool = True
    comparison_mode: ComparisonMode = "planner_vs_planner"

    def __post_init__(self) -> None:
        assert self.comparison_mode in {"planner_vs_planner", "planner_vs_dev_reference"}
