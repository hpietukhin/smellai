"""State-space search over smell configurations.

Implements Algorithms 1 (Greedy) and 2 (BeFS) from the paper.
Each node is a State (frozenset of active smell_ids).
Each edge is a RefactoringAction.

Spec reference: Algorithms 1-2 (conf_Pietukhin_10_3_rev2-2.pdf).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain.rules import REFACTORING_CATALOGUE

if TYPE_CHECKING:
    from domain.dependency_graph import DependencyGraph


@dataclass(frozen=True)
class State:
    """Immutable set of active smells. Hashable for use in CLOSED sets."""
    active: frozenset[str]

    def h(self, dg: DependencyGraph) -> float:
        """h(S) = Σ sev(s) for s in active. BeFS heuristic."""
        return sum(dg.severity_of(s) for s in self.active)


@dataclass(frozen=True)
class RefactoringAction:
    """Atomic refactoring step: refactor smell_id using ref_type."""
    smell_id: str
    ref_type: str


@dataclass
class Plan:
    """Ordered refactoring plan with full state trace."""
    actions: tuple[RefactoringAction, ...]
    states: tuple[State, ...]       # S₀, S₁, ..., Sₙ (len = actions + 1)
    h_trace: tuple[float, ...]      # h(S₀), h(S₁), ..., h(Sₙ)


class RefactoringTree:
    """Search tree over smell states. Uses DependencyGraph for transition rules."""

    def __init__(self, initial: State, dep_graph: DependencyGraph) -> None:
        self._initial = initial
        self._dg = dep_graph

    def transition(self, state: State, action: RefactoringAction) -> State:
        """S' = (S \\ resolved(action.smell)) ∪ created(action.smell).
        
        Core of Algorithms 1-2, line 9/13.
        """
        resolved = self._dg.resolved_by(action.smell_id, state.active)
        created = self._dg.created_by(action.smell_id, state.active)
        new_active = (state.active - resolved) | created
        return State(frozenset(new_active))

    def _pick_ref_type(self, smell_id: str) -> str:
        """Pick the top-ranked refactoring type for a smell from the catalogue."""
        smell_type = self._dg.smell_type_of(smell_id)
        ops = REFACTORING_CATALOGUE.get(smell_type, [])
        return ops[0][0] if ops else "Refactor"

    def greedy(self) -> Plan:
        """Algorithm 1: at each step pick argmax P(s_i), apply transition."""
        state = self._initial
        actions: list[RefactoringAction] = []
        states: list[State] = [state]
        h_trace: list[float] = [state.h(self._dg)]

        while state.active:
            # Pick smell with highest score (tiebreak by smell_id for determinism)
            best_id = max(state.active, key=lambda s: (self._dg.score(s), s))
            action = RefactoringAction(
                smell_id=best_id,
                ref_type=self._pick_ref_type(best_id),
            )
            state = self.transition(state, action)
            actions.append(action)
            states.append(state)
            h_trace.append(state.h(self._dg))

        return Plan(
            actions=tuple(actions),
            states=tuple(states),
            h_trace=tuple(h_trace),
        )

    def befs(self, *, max_expansions: int = 10_000) -> Plan:
        """Algorithm 2: Best-First Search minimizing h(S).

        OPEN = priority queue ordered by h(S).
        CLOSED = set of visited states (frozenset).
        Returns the plan that reaches the lowest h.
        """
        if not self._initial.active:
            return Plan(actions=(), states=(self._initial,), h_trace=(0.0,))

        counter = 0  # tiebreaker for heapq
        # (h, counter, state, actions_tuple)
        initial_h = self._initial.h(self._dg)
        open_set: list[tuple[float, int, State, tuple[RefactoringAction, ...]]] = [
            (initial_h, counter, self._initial, ())
        ]
        closed: set[frozenset[str]] = set()
        best_h = initial_h
        best_actions: tuple[RefactoringAction, ...] = ()
        best_state = self._initial
        expansions = 0

        while open_set and expansions < max_expansions:
            h, _, state, actions = heapq.heappop(open_set)

            if state.active in closed:
                continue
            closed.add(state.active)
            expansions += 1

            if h < best_h:
                best_h = h
                best_actions = actions
                best_state = state

            if not state.active:
                break  # goal: h = 0

            for smell_id in state.active:
                action = RefactoringAction(
                    smell_id=smell_id,
                    ref_type=self._pick_ref_type(smell_id),
                )
                next_state = self.transition(state, action)
                if next_state.active not in closed:
                    counter += 1
                    next_h = next_state.h(self._dg)
                    heapq.heappush(open_set, (
                        next_h, counter, next_state, actions + (action,),
                    ))

        # Reconstruct full state trace from best_actions
        return self._build_plan(best_actions)

    def simulate(self, actions: list[RefactoringAction]) -> Plan:
        """Replay a developer's actual sequence through transition function.

        Actions targeting smells no longer active are treated as no-ops
        (the state doesn't change but the action is still recorded).
        """
        state = self._initial
        states: list[State] = [state]
        h_trace: list[float] = [state.h(self._dg)]

        for action in actions:
            if action.smell_id in state.active:
                state = self.transition(state, action)
            # else: no-op, smell already resolved
            states.append(state)
            h_trace.append(state.h(self._dg))

        return Plan(
            actions=tuple(actions),
            states=tuple(states),
            h_trace=tuple(h_trace),
        )

    def _build_plan(self, actions: tuple[RefactoringAction, ...]) -> Plan:
        """Replay actions from initial state to build full state/h trace."""
        state = self._initial
        states: list[State] = [state]
        h_trace: list[float] = [state.h(self._dg)]

        for action in actions:
            state = self.transition(state, action)
            states.append(state)
            h_trace.append(state.h(self._dg))

        return Plan(
            actions=actions,
            states=tuple(states),
            h_trace=tuple(h_trace),
        )
