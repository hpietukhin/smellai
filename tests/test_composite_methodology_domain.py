from domain.composite_completion import detect_missing_follow_ups
from domain.composite_evaluation import classify_smell_incidence
from domain.composite_patterns import build_pattern_event, mine_pattern_frequencies
from domain.composite_synthesis import (
    CommitBasedSynthesizer,
    ElementBasedSynthesizer,
    RangeBasedSynthesizer,
    RefactoringOccurrence,
)


def _occ(ref_id: str, commit: str, order: int, scope: set[str], ref_type: str = "Extract Method"):
    return RefactoringOccurrence(
        ref_id=ref_id,
        ref_type=ref_type,
        commit_hash=commit,
        commit_order=order,
        scope=frozenset(scope),
    )


def test_synthesis_strategies_differ_as_expected():
    occurrences = [
        _occ("r1", "c1", 1, {"A"}),
        _occ("r2", "c1", 1, {"B"}),
        _occ("r3", "c2", 2, {"B", "C"}),
    ]

    element = ElementBasedSynthesizer().synthesize(occurrences)
    commit = CommitBasedSynthesizer().synthesize(occurrences)
    range_based = RangeBasedSynthesizer().synthesize(occurrences)

    assert any(c.ref_ids == ("r2", "r3") for c in element)
    assert any(c.ref_ids == ("r1", "r2") for c in commit)
    assert any(c.ref_ids == ("r2", "r3") for c in range_based)


def test_outcome_classification():
    assert classify_smell_incidence(10, 8) == "positive"
    assert classify_smell_incidence(10, 10) == "neutral"
    assert classify_smell_incidence(10, 12) == "negative"


def test_pattern_mining_counts_only_non_neutral_events():
    e1 = build_pattern_event(["Extract Method", "Move Method"], "Feature Envy", 3, 2)
    e2 = build_pattern_event(["Extract Method", "Move Method"], "Feature Envy", 4, 3)
    e3 = build_pattern_event(["Extract Method"], "Feature Envy", 2, 2)

    assert e3 is None
    freqs = mine_pattern_frequencies([e1, e2])
    assert list(freqs.values()) == [2]


def test_incomplete_composite_detection_flags_missing_follow_up():
    missing = detect_missing_follow_ups(["Extract Method", "Rename Method"])
    assert any(r.required_follow_up == "Move Method" for r in missing)
