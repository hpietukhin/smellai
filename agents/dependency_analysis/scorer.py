"""Functional scoring primitives for smell prioritization.

Layer 1 — scoring: smell × ScoringContext → float

Composers build callable scorers from independent feature functions:
  scorer(*terms)          → (smell, ctx) → float   (used by Greedy)
  heuristic(feature, agg) → state → float           (used by BeFS)

Spec reference: Eq. 1–2, Algorithms 1–2 (conf_Pietukhin_10_3_rev2-2.pdf).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from operator import attrgetter
from typing import Callable


@dataclass(frozen=True)
class ScoringContext:
    """Graph-derived features for one smell instance."""

    freq: int       # occurrence count of this smell_type in the codebase
    pos_out: int    # outgoing positive edges (concrete, co-located instances)
    neg_out: int    # outgoing negative edges (abstract catalogue rules)
    files_affected: int = 1


# ---------------------------------------------------------------------------
# Feature functions  (smell, ScoringContext) → float
# ---------------------------------------------------------------------------

def freq_severity(smell, ctx: ScoringContext, *, w_sev: float) -> float:
    """f_i · w_sev · sev(s_i)  — intrinsic value term."""
    return ctx.freq * w_sev * smell.severity_score


def pos_out_fn(smell, ctx: ScoringContext) -> float:
    """Σ pos_out^conc  — positive dependency bonus."""
    return float(ctx.pos_out)


def neg_out_fn(smell, ctx: ScoringContext, *, w_neg: float) -> float:
    """−w_neg · Σ neg_out^abs  — negative dependency penalty."""
    return -w_neg * ctx.neg_out


# ---------------------------------------------------------------------------
# Composers
# ---------------------------------------------------------------------------

def scorer(*terms: Callable) -> Callable:
    """Returns (smell, ctx) → float as sum of independent terms."""
    return lambda smell, ctx: sum(f(smell, ctx) for f in terms)


def heuristic(
    feature_fn: Callable = attrgetter("severity_score"),
    agg: Callable = sum,
) -> Callable:
    """Returns state → float by aggregating feature_fn over all smells in state."""
    return lambda state: agg(feature_fn(s) for s in state)


# ---------------------------------------------------------------------------
# Spec defaults — Eq. 2: P_i^conc = f_i·w_sev·sev + Σpos_out^conc − w_neg·Σneg_out^abs
# ---------------------------------------------------------------------------

STANDARD_SCORE = scorer(
    partial(freq_severity, w_sev=0.33),
    pos_out_fn,
    partial(neg_out_fn, w_neg=0.5),
)

#: h(S) = Σ sev(s)  — BeFS state heuristic (Algorithm 2)
STANDARD_H = heuristic(attrgetter("severity_score"))
