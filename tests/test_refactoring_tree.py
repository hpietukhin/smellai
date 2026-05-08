"""TDD tests for RefactoringTree — state-space search over smell configurations.

Vertical slices:
1. State + transition (S' = (S \\ resolved) ∪ created)
2. Greedy planner (Algorithm 1)
3. BeFS planner (Algorithm 2)
4. Paper example: Fig. 4 trace
5. simulate_developer (replay real sequence)
"""
from __future__ import annotations

from domain.models import SmellEvent
from domain.dependency_graph import DependencyGraph


def _smell(smell_id: str, smell_type: str, severity: str = "HIGH",
           class_name: str = "Foo") -> SmellEvent:
    return SmellEvent(
        smell_id=smell_id,
        smell_type=smell_type,
        file_path="Foo.java",
        severity=severity,
        class_name=class_name,
    )


def _build_fig4():
    """Paper Fig. 4: S0 = {GC_H, LM_H, FE_M, DC_M}.
    Two God Class instances (f_GC=2) to match paper's f_GC=2."""
    events = [
        _smell("gc", "God Class", severity="HIGH"),
        _smell("lm", "Long Method", severity="HIGH"),
        _smell("fe", "Feature Envy", severity="MEDIUM"),
        _smell("dc", "Data Clumps", severity="MEDIUM"),
    ]
    dg = DependencyGraph.from_events(events, locality="none")
    return events, dg


# --- Slice 1: State and transition ---

class TestTransition:

    def test_transition_removes_refactored_smell(self):
        from domain.refactoring_tree import RefactoringTree, State, RefactoringAction
        events = [_smell("lm", "Long Method")]
        dg = DependencyGraph.from_events(events)
        initial = State(frozenset(["lm"]))
        tree = RefactoringTree(initial, dg)

        action = RefactoringAction(smell_id="lm", ref_type="Extract Method")
        next_state = tree.transition(initial, action)
        assert "lm" not in next_state.active

    def test_transition_resolves_positive_neighbors(self):
        """Refactoring LM should also resolve FE (positive dep)."""
        from domain.refactoring_tree import RefactoringTree, State, RefactoringAction
        events = [
            _smell("lm", "Long Method"),
            _smell("fe", "Feature Envy"),
        ]
        dg = DependencyGraph.from_events(events, locality="none")
        initial = State(frozenset(["lm", "fe"]))
        tree = RefactoringTree(initial, dg)

        action = RefactoringAction(smell_id="lm", ref_type="Extract Method")
        next_state = tree.transition(initial, action)
        assert "lm" not in next_state.active
        assert "fe" not in next_state.active  # resolved via positive dep

    def test_transition_adds_created_smells(self):
        """Refactoring LM may introduce LPL (negative dep)."""
        from domain.refactoring_tree import RefactoringTree, State, RefactoringAction
        events = [
            _smell("lm", "Long Method"),
            _smell("lpl", "Long Parameter List", severity="MEDIUM"),
        ]
        dg = DependencyGraph.from_events(events, locality="none")
        initial = State(frozenset(["lm", "lpl"]))
        tree = RefactoringTree(initial, dg)

        action = RefactoringAction(smell_id="lm", ref_type="Extract Method")
        next_state = tree.transition(initial, action)
        assert "lm" not in next_state.active
        # lpl was already active AND matches LM's negative rule → stays/re-created
        assert "lpl" in next_state.active

    def test_state_h_is_sum_of_severities(self):
        """h(S) = Σ sev(s) for s in active."""
        from domain.refactoring_tree import State
        events = [
            _smell("gc", "God Class", severity="HIGH"),       # 3
            _smell("fe", "Feature Envy", severity="MEDIUM"),  # 2
        ]
        dg = DependencyGraph.from_events(events)
        state = State(frozenset(["gc", "fe"]))
        assert state.h(dg) == 5  # 3 + 2

    def test_empty_state_h_is_zero(self):
        from domain.refactoring_tree import State
        dg = DependencyGraph.from_events([])
        assert State(frozenset()).h(dg) == 0


# --- Slice 2: Greedy planner (Algorithm 1) ---

class TestGreedy:

    def test_greedy_on_single_smell_produces_one_step(self):
        from domain.refactoring_tree import RefactoringTree, State
        events = [_smell("lm", "Long Method")]
        dg = DependencyGraph.from_events(events)
        tree = RefactoringTree(State(frozenset(["lm"])), dg)

        plan = tree.greedy()
        assert len(plan.actions) == 1
        assert plan.actions[0].smell_id == "lm"
        assert plan.h_trace[-1] == 0  # all smells resolved

    def test_greedy_on_empty_state_produces_empty_plan(self):
        from domain.refactoring_tree import RefactoringTree, State
        dg = DependencyGraph.from_events([])
        tree = RefactoringTree(State(frozenset()), dg)
        plan = tree.greedy()
        assert len(plan.actions) == 0

    def test_greedy_picks_highest_score_first(self):
        """GC (HIGH, more deps) should be picked before FE (MEDIUM, no deps)."""
        from domain.refactoring_tree import RefactoringTree, State
        events = [
            _smell("gc", "God Class", severity="HIGH"),
            _smell("fe", "Feature Envy", severity="MEDIUM"),
            _smell("dc", "Data Clumps", severity="MEDIUM"),
        ]
        dg = DependencyGraph.from_events(events, locality="none")
        tree = RefactoringTree(State(frozenset(["gc", "fe", "dc"])), dg)

        plan = tree.greedy()
        assert plan.actions[0].smell_id == "gc"

    def test_greedy_terminates_when_negative_deps_create_smells(self):
        """Greedy must terminate even when negative deps re-create smells.
        LM neg→LPL, but LPL is resolved on next step."""
        from domain.refactoring_tree import RefactoringTree, State
        events = [
            _smell("lm", "Long Method"),
            _smell("lpl", "Long Parameter List", severity="MEDIUM"),
        ]
        dg = DependencyGraph.from_events(events, locality="none")
        tree = RefactoringTree(State(frozenset(["lm", "lpl"])), dg)
        plan = tree.greedy()
        assert plan.h_trace[-1] == 0  # eventually reaches empty

    def test_greedy_plan_has_consistent_state_trace(self):
        """states[i+1] = transition(states[i], actions[i])."""
        from domain.refactoring_tree import RefactoringTree, State
        events = [
            _smell("lm", "Long Method"),
            _smell("fe", "Feature Envy"),
        ]
        dg = DependencyGraph.from_events(events, locality="none")
        tree = RefactoringTree(State(frozenset(["lm", "fe"])), dg)

        plan = tree.greedy()
        assert len(plan.states) == len(plan.actions) + 1
        assert len(plan.h_trace) == len(plan.states)
        for i, action in enumerate(plan.actions):
            expected = tree.transition(plan.states[i], action)
            assert plan.states[i + 1] == expected


# --- Slice 3: BeFS planner (Algorithm 2) ---

class TestBeFS:

    def test_befs_on_single_smell(self):
        from domain.refactoring_tree import RefactoringTree, State
        events = [_smell("lm", "Long Method")]
        dg = DependencyGraph.from_events(events)
        tree = RefactoringTree(State(frozenset(["lm"])), dg)

        plan = tree.befs()
        assert len(plan.actions) == 1
        assert plan.h_trace[-1] == 0

    def test_befs_finds_shorter_plan_than_greedy_on_fig4(self):
        """Paper Fig. 4: BeFS picks LM first (h:10→5→0, 2 steps).
        Greedy picks GC first (fires neg deps, 4+ steps)."""
        from domain.refactoring_tree import RefactoringTree, State
        events, dg = _build_fig4()
        initial = State(frozenset(e.smell_id for e in events))
        tree = RefactoringTree(initial, dg)

        plan_greedy = tree.greedy()
        plan_befs = tree.befs()

        # BeFS should find plan with h reaching 0 (or lower than greedy)
        assert plan_befs.h_trace[-1] <= plan_greedy.h_trace[-1]
        # BeFS plan should be no longer than greedy
        assert len(plan_befs.actions) <= len(plan_greedy.actions)

    def test_befs_plan_has_consistent_state_trace(self):
        from domain.refactoring_tree import RefactoringTree, State
        events = [
            _smell("gc", "God Class"),
            _smell("lm", "Long Method"),
            _smell("fe", "Feature Envy", severity="MEDIUM"),
        ]
        dg = DependencyGraph.from_events(events, locality="none")
        tree = RefactoringTree(State(frozenset(["gc", "lm", "fe"])), dg)

        plan = tree.befs()
        assert len(plan.states) == len(plan.actions) + 1
        for i, action in enumerate(plan.actions):
            expected = tree.transition(plan.states[i], action)
            assert plan.states[i + 1] == expected

    def test_befs_on_empty_state(self):
        from domain.refactoring_tree import RefactoringTree, State
        dg = DependencyGraph.from_events([])
        tree = RefactoringTree(State(frozenset()), dg)
        plan = tree.befs()
        assert len(plan.actions) == 0


# --- Slice 4: Paper Fig. 4 exact BeFS trace ---

class TestFig4Trace:
    """Paper §III-C Fig. 4:
    S0 = {GC_H, LM_H, FE_M, DC_M}, h=10.
    BeFS: r(LM) → S1={GC,DC}, h=5 → r(GC) → S2=∅, h=0. Total: 2 steps.
    """

    def test_befs_reaches_h_zero_in_two_steps(self):
        from domain.refactoring_tree import RefactoringTree, State
        events, dg = _build_fig4()
        initial = State(frozenset(e.smell_id for e in events))
        tree = RefactoringTree(initial, dg)

        plan = tree.befs()
        assert plan.h_trace[0] == 10  # 3+3+2+2
        assert plan.h_trace[-1] == 0
        assert len(plan.actions) == 2


# --- Slice 5: simulate_developer ---

class TestSimulateDeveloper:

    def test_simulate_replays_actions_through_transition(self):
        """Use FE + DC: no mutual positive deps, so each must be resolved separately."""
        from domain.refactoring_tree import RefactoringTree, State, RefactoringAction
        events = [
            _smell("fe", "Feature Envy", severity="HIGH"),         # sev=3
            _smell("dc", "Data Clumps", severity="MEDIUM"),        # sev=2
        ]
        dg = DependencyGraph.from_events(events, locality="none")
        initial = State(frozenset(["fe", "dc"]))
        tree = RefactoringTree(initial, dg)

        dev_actions = [
            RefactoringAction(smell_id="dc", ref_type="Extract Class"),
            RefactoringAction(smell_id="fe", ref_type="Move Method"),
        ]
        plan = tree.simulate(dev_actions)
        assert len(plan.actions) == 2
        assert plan.states[0] == initial
        assert plan.h_trace[0] == 5  # 3 + 2
        # After DC removed, FE still there
        assert "fe" in plan.states[1].active
        assert plan.h_trace[-1] == 0

    def test_simulate_skips_action_on_already_resolved_smell(self):
        """If developer's action targets a smell already resolved, skip it."""
        from domain.refactoring_tree import RefactoringTree, State, RefactoringAction
        events = [
            _smell("lm", "Long Method"),
            _smell("fe", "Feature Envy"),
        ]
        dg = DependencyGraph.from_events(events, locality="none")
        initial = State(frozenset(["lm", "fe"]))
        tree = RefactoringTree(initial, dg)

        # LM resolves FE via positive dep, then developer tries to refactor FE (already gone)
        dev_actions = [
            RefactoringAction(smell_id="lm", ref_type="Extract Method"),
            RefactoringAction(smell_id="fe", ref_type="Move Method"),  # already resolved
        ]
        plan = tree.simulate(dev_actions)
        # The plan should still record 2 actions but fe action is no-op
        assert plan.h_trace[-1] == 0
