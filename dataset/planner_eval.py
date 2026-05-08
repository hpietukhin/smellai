"""Planner evaluation utilities against the Composite Refactorings 2020 dataset.

Bridges the DatasetGraph (empirical) and RefactoringTree (theoretical) layers
to compare planner plans against developer-committed refactoring sequences.

Three h-traces are compared for each composite episode:
  - plan_befs    : BeFS planner (Algorithm 2)
  - plan_greedy  : Greedy planner (Algorithm 1)
  - plan_dev     : Developer's actual sequence replayed via simulate()

Usage::

    from dataset.neo4j_graph import DatasetGraph
    from dataset.planner_eval import evaluate_composite

    ds = DatasetGraph()
    steps = ds.composite_refactoring(
        elements={"org.junit.runners.BlockJUnit4ClassRunner", "org.junit.runners.ParentRunner"},
        project="JUnit4",
        max_steps=20,
    )
    result = evaluate_composite(steps)
    # result is a dict of metrics ready for MLflow logging
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain.composite_completion import detect_missing_follow_ups
from domain.composite_evaluation import classify_smell_incidence
from domain.composite_synthesis import (
    CommitBasedSynthesizer,
    ElementBasedSynthesizer,
    RangeBasedSynthesizer,
    RefactoringOccurrence,
)
from domain.dependency_graph import DependencyGraph
from domain.experiment_axes import ExperimentSpec
from domain.refactoring_tree import RefactoringAction, RefactoringTree, State, Plan
from domain.rules import REFACTORING_CATALOGUE

if TYPE_CHECKING:
    from dataset.neo4j_graph import CompositeStep, RefactoringRecord
    from domain.models import SmellEvent


# ---------------------------------------------------------------------------
# find_smell_for_refactoring
# ---------------------------------------------------------------------------


def find_smell_for_refactoring(
    ref: "RefactoringRecord",
    active_smells: list["SmellEvent"],
    dep_graph: DependencyGraph,
) -> str | None:
    """Map a developer's RefactoringRecord to a smell_id in the active set.

    Matching strategy (in order):
    1. Element overlap  — smell.class_name overlaps ref.changed_elements ∪ produced_elements
    2. Type compatibility — ref.ref_type in REFACTORING_CATALOGUE[smell.smell_type]
    3. Tie-break        — highest dep_graph.score()

    Returns ``None`` if no smell matches (the refactoring targets something
    outside S₀, e.g. a pure architectural operation).  The caller should
    treat ``None`` as a no-op in simulate().
    """
    if not active_smells:
        return None

    touched_elements: set[str] = set(ref.changed_elements) | set(ref.produced_elements)

    def _type_compatible(smell: "SmellEvent") -> bool:
        ops = [op for op, _ in REFACTORING_CATALOGUE.get(smell.smell_type, [])]
        return ref.ref_type in ops

    def _element_matches(smell: "SmellEvent") -> bool:
        if not touched_elements:
            # No element info available — skip element filter
            return True
        smell_class = smell.class_name or ""
        # Match on element FQN: ref element starts with smell class FQN
        # (handles both class-level and method-level element names in dataset)
        for el in touched_elements:
            if el and smell_class and (
                el == smell_class
                or el.startswith(smell_class + ".")
                or el.startswith(smell_class + "(")
                or smell_class == el.split("(")[0].rsplit(".", 1)[0]
            ):
                return True
        return False

    candidates = [
        s for s in active_smells
        if _element_matches(s) and _type_compatible(s)
    ]

    if not candidates:
        return None

    # Tiebreak: highest priority score, then smell_id for determinism
    best = max(candidates, key=lambda s: (dep_graph.score(s.smell_id), s.smell_id))
    return best.smell_id


# ---------------------------------------------------------------------------
# developer_plan_from_steps
# ---------------------------------------------------------------------------


def developer_plan_from_steps(
    steps: list["CompositeStep"],
    tree: RefactoringTree,
    dep_graph: DependencyGraph,
) -> Plan:
    """Replay the developer's refactoring sequence through the RefactoringTree.

    Maps each RefactoringRecord in ``steps`` to a RefactoringAction using
    ``find_smell_for_refactoring``.  Records that don't match any active smell
    are silently skipped (no-op).

    Returns a ``Plan`` produced by ``tree.simulate(dev_actions)``.
    """
    initial_step = next((s for s in steps if s.smells), None)
    if initial_step is None:
        return tree.simulate([])

    # Smell lookup by ID for the initial set
    smell_by_id: dict[str, "SmellEvent"] = {s.smell_id: s for s in initial_step.smells}

    dev_actions: list[RefactoringAction] = []
    current_state = State(frozenset(smell_by_id))

    for step in steps:
        for ref in step.refactorings:
            # Only consider smells currently active
            active_smells = [
                smell_by_id[sid]
                for sid in current_state.active
                if sid in smell_by_id
            ]
            smell_id = find_smell_for_refactoring(ref, active_smells, dep_graph)
            if smell_id is not None:
                action = RefactoringAction(smell_id=smell_id, ref_type=ref.ref_type)
                dev_actions.append(action)
                # Advance state so subsequent refactorings see updated active set
                if smell_id in current_state.active:
                    current_state = tree.transition(current_state, action)

    return tree.simulate(dev_actions)


# ---------------------------------------------------------------------------
# evaluate_composite
# ---------------------------------------------------------------------------


@dataclass
class PlannerEvalResult:
    """Metrics for one composite episode."""

    # Episode identity
    project: str
    initial_commit: str
    initial_commit_order: int

    # Smell counts
    smells_initial: int
    smells_after_empirical: int

    # Binary outcome: did each approach reduce smell count?
    befs_reduces: bool
    greedy_reduces: bool
    dev_reduces: bool          # empirical: last dataset step has fewer smells than first

    # Plan efficiency: steps / smells_resolved (lower = more efficient)
    befs_eta: float
    greedy_eta: float

    # Negative dependency rate: new smells introduced / steps (lower = better)
    befs_rho: float
    greedy_rho: float

    # h values (sum of severities)
    h_initial: float
    h_befs_final: float
    h_greedy_final: float
    h_dev_final: float         # h after simulate(dev_actions)

    # Full h-traces for downstream comparison / artifact logging
    befs_h_trace: tuple[float, ...]
    greedy_h_trace: tuple[float, ...]
    dev_h_trace: tuple[float, ...]

    # Developer sequence length / mapping coverage
    dev_steps: int
    dev_matched_steps: int
    dev_match_rate: float

    # Methodology-aligned metadata
    heuristic: str = "range-based"
    locality: str = "none"
    outcome_class: str = "neutral"
    synthesized_composites: int = 0
    completion_risk_count: int = 0

    def to_mlflow_metrics(self) -> dict[str, float]:
        """Flat float dict suitable for ``mlflow.log_metrics()``."""
        return {
            "smells_initial": float(self.smells_initial),
            "smells_after_empirical": float(self.smells_after_empirical),
            "befs_reduces": float(self.befs_reduces),
            "greedy_reduces": float(self.greedy_reduces),
            "dev_reduces": float(self.dev_reduces),
            "befs_eta": float(self.befs_eta),
            "greedy_eta": float(self.greedy_eta),
            "befs_rho": float(self.befs_rho),
            "greedy_rho": float(self.greedy_rho),
            "h_initial": float(self.h_initial),
            "h_befs_final": float(self.h_befs_final),
            "h_greedy_final": float(self.h_greedy_final),
            "h_dev_final": float(self.h_dev_final),
            "dev_steps": float(self.dev_steps),
            "dev_matched_steps": float(self.dev_matched_steps),
            "dev_match_rate": float(self.dev_match_rate),
            "synthesized_composites": float(self.synthesized_composites),
            "completion_risk_count": float(self.completion_risk_count),
        }

    def to_mlflow_params(self) -> dict[str, str]:
        """Identifiers suitable for ``mlflow.log_params()``."""
        return {
            "project": self.project,
            "initial_commit": self.initial_commit,
            "initial_commit_order": str(self.initial_commit_order),
            "heuristic": self.heuristic,
            "locality": self.locality,
            "outcome_class": self.outcome_class,
        }


def _count_created(plan: Plan) -> int:
    """Count total smells introduced by negative deps across all plan steps."""
    created = 0
    for i, state_after in enumerate(plan.states[1:], start=1):
        state_before = plan.states[i - 1]
        new_ids = state_after.active - state_before.active
        created += len(new_ids)
    return created


def _synthesize_count(steps: list["CompositeStep"], heuristic: str) -> int:
    occs: list[RefactoringOccurrence] = []
    for step in steps:
        for ref in step.refactorings:
            scope = frozenset((ref.changed_elements or []) + (ref.produced_elements or []))
            occs.append(
                RefactoringOccurrence(
                    ref_id=ref.hash_id or f"{step.commit_hash}:{ref.ref_type}",
                    ref_type=ref.ref_type,
                    commit_hash=step.commit_hash,
                    commit_order=step.commit_order,
                    scope=scope,
                )
            )
    if heuristic == "element-based":
        return len(ElementBasedSynthesizer().synthesize(occs))
    if heuristic == "commit-based":
        return len(CommitBasedSynthesizer().synthesize(occs))
    return len(RangeBasedSynthesizer().synthesize(occs))


def evaluate_composite(
    steps: list["CompositeStep"],
    *,
    project: str = "",
    spec: ExperimentSpec | None = None,
) -> PlannerEvalResult | None:
    """Evaluate planner plans against a composite refactoring episode.

    Returns ``None`` if the episode has no initial smells (nothing to plan).

    Args:
        steps:   Ordered list of CompositeSteps from DatasetGraph.composite_refactoring().
        project: Project name for logging (used in PlannerEvalResult).
    """
    initial_step = next((s for s in steps if s.smells), None)
    if initial_step is None:
        return None

    S0_events = initial_step.smells
    smells_initial = len(S0_events)

    spec = spec or ExperimentSpec(heuristic="range-based", planner="befs", locality="none")

    dep_graph = DependencyGraph.from_events(S0_events, locality=spec.locality)
    initial_state = State(frozenset(e.smell_id for e in S0_events))
    tree = RefactoringTree(initial_state, dep_graph)

    # --- Run planners ---
    plan_befs = tree.befs()
    plan_greedy = tree.greedy()

    # --- Replay developer sequence ---
    plan_dev = developer_plan_from_steps(steps, tree, dep_graph)

    # --- Binary outcome ---
    befs_final = len(plan_befs.states[-1].active)
    greedy_final = len(plan_greedy.states[-1].active)
    befs_reduces = befs_final < smells_initial
    greedy_reduces = greedy_final < smells_initial

    # Empirical before/after: compare the planner input state with the last
    # observed dataset step.  Do not filter out empty smell states: a final
    # state with zero smells is exactly the strongest improvement signal.
    final_step = steps[-1]
    smells_after_empirical = len(final_step.smells)
    dev_reduces = smells_after_empirical < smells_initial
    outcome_class = classify_smell_incidence(smells_initial, smells_after_empirical)

    # --- Efficiency η = steps / max(smells_resolved, 1) ---
    befs_resolved = max(smells_initial - befs_final, 0)
    greedy_resolved = max(smells_initial - greedy_final, 0)
    befs_eta = len(plan_befs.actions) / max(befs_resolved, 1)
    greedy_eta = len(plan_greedy.actions) / max(greedy_resolved, 1)

    # --- Negative dependency rate ρ = created / steps ---
    befs_created = _count_created(plan_befs)
    greedy_created = _count_created(plan_greedy)
    befs_rho = befs_created / max(len(plan_befs.actions), 1)
    greedy_rho = greedy_created / max(len(plan_greedy.actions), 1)

    # --- h values / traces ---
    h_initial = plan_befs.h_trace[0]  # same for all plans
    h_befs_final = plan_befs.h_trace[-1]
    h_greedy_final = plan_greedy.h_trace[-1]
    h_dev_final = plan_dev.h_trace[-1]
    befs_h_trace = plan_befs.h_trace
    greedy_h_trace = plan_greedy.h_trace
    dev_h_trace = plan_dev.h_trace

    # --- Developer steps / mapping coverage ---
    dev_steps = sum(len(s.refactorings) for s in steps)
    dev_matched_steps = len(plan_dev.actions)
    dev_match_rate = dev_matched_steps / max(dev_steps, 1)

    synthesized_count = _synthesize_count(steps, spec.heuristic)
    completion_risk_count = 0
    if spec.track_completion_risk:
        all_ref_types = [r.ref_type for s in steps for r in s.refactorings]
        completion_risk_count = len(detect_missing_follow_ups(all_ref_types))

    return PlannerEvalResult(
        project=project,
        initial_commit=initial_step.commit_hash,
        initial_commit_order=initial_step.commit_order,
        smells_initial=smells_initial,
        smells_after_empirical=smells_after_empirical,
        befs_reduces=befs_reduces,
        greedy_reduces=greedy_reduces,
        dev_reduces=dev_reduces,
        befs_eta=befs_eta,
        greedy_eta=greedy_eta,
        befs_rho=befs_rho,
        greedy_rho=greedy_rho,
        h_initial=h_initial,
        h_befs_final=h_befs_final,
        h_greedy_final=h_greedy_final,
        h_dev_final=h_dev_final,
        befs_h_trace=befs_h_trace,
        greedy_h_trace=greedy_h_trace,
        dev_h_trace=dev_h_trace,
        dev_steps=dev_steps,
        dev_matched_steps=dev_matched_steps,
        dev_match_rate=dev_match_rate,
        heuristic=spec.heuristic,
        locality=spec.locality,
        outcome_class=outcome_class,
        synthesized_composites=synthesized_count,
        completion_risk_count=completion_risk_count,
    )
