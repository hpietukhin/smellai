#!/usr/bin/env python3
"""
Test for smell prioritization logic.

Tests that the SmellPrioritizer correctly:
1. Creates SmellEvent objects usable in-memory (no DB session needed)
2. Calculates PZ scores (severity + positive impact * 2)
3. Returns smells ordered by highest PZ score first
"""

import pytest
from swe_refactor.persistence.models import SmellEvent
from scripts.prioritize_smells import SmellPrioritizer


def _smell(smell_id: str, smell_type: str, file_path: str, severity: str) -> SmellEvent:
    """Helper: build an in-memory SmellEvent (no DB context needed)."""
    return SmellEvent(smell_id=smell_id, smell_type=smell_type, file_path=file_path, severity=severity)


def test_smell_event_in_memory():
    """Test that SmellEvent can be used in-memory with defaults."""
    smell = SmellEvent(
        smell_id="Long Method:OrderProcessor.java:23",
        smell_type="Long Method",
        file_path="OrderProcessor.java",
        line_number=23,
        severity="HIGH",
    )

    assert smell.smell_id == "Long Method:OrderProcessor.java:23"
    assert smell.smell_type == "Long Method"
    assert smell.location == "OrderProcessor.java:23"
    assert smell.severity == "HIGH"
    assert smell.severity_score == 3


def test_severity_score_mapping():
    """Test severity string to numeric score mapping."""
    assert _smell("1", "Test", "loc", "HIGH").severity_score == 3
    assert _smell("2", "Test", "loc", "CRITICAL").severity_score == 3
    assert _smell("3", "Test", "loc", "MEDIUM").severity_score == 2
    assert _smell("4", "Test", "loc", "major").severity_score == 2
    assert _smell("5", "Test", "loc", "LOW").severity_score == 1


def test_prioritization_calculates_pz_correctly():
    """Test priority score follows spec formula: P = freq * w_sev * sev + pos_out - w_neg * neg_out."""
    smells = [
        _smell("1", "Long Method",     "OrderProcessor.java", "HIGH"),
        _smell("2", "Duplicated Code", "OrderProcessor.java", "MEDIUM"),
        _smell("3", "Complex Method",  "OrderProcessor.java", "MEDIUM"),
    ]

    prioritizer = SmellPrioritizer(smells)
    sequence = prioritizer.calculate_priorities()

    # Long Method: freq=1, w_sev=0.33, sev=3, pos_out=1 (Dup.Code), neg_out=0
    # P = 1 * 0.33 * 3 + 1 - 0.5 * 0 = 1.99  (highest of the three)
    assert len(sequence) == 3
    assert sequence[0]["smell_type"] == "Long Method"
    assert sequence[0]["pz_score"] > sequence[1]["pz_score"]


def test_prioritization_returns_highest_pz_first():
    """Test that smells are ordered by PZ score descending."""
    smells = [
        _smell("1", "God Class",       "ReportGenerator.java", "CRITICAL"),
        _smell("2", "Long Method",     "ReportGenerator.java", "HIGH"),
        _smell("3", "Data Clumps",     "ReportGenerator.java", "MEDIUM"),
        _smell("4", "Feature Envy",    "ReportGenerator.java", "MEDIUM"),
        _smell("5", "Print Statements","ReportGenerator.java", "LOW"),
    ]

    prioritizer = SmellPrioritizer(smells)
    sequence = prioritizer.calculate_priorities()

    pz_scores = [item["pz_score"] for item in sequence]
    assert pz_scores == sorted(pz_scores, reverse=True), (
        f"Expected descending PZ scores, got: {pz_scores}"
    )
    assert sequence[0]["smell_type"] in ["God Class", "Large Class"], (
        f"Expected high-priority smell first, got: {sequence[0]['smell_type']}"
    )


def test_prioritization_considers_dependencies():
    """Test that positive dependencies increase PZ score."""
    smells = [
        _smell("1", "Long Method",    "Test.java", "MEDIUM"),
        _smell("2", "Duplicated Code","Test.java", "MEDIUM"),
    ]

    prioritizer = SmellPrioritizer(smells)
    sequence = prioritizer.calculate_priorities()

    long_method_item = next(s for s in sequence if s["smell_type"] == "Long Method")
    duplicated_item  = next(s for s in sequence if s["smell_type"] == "Duplicated Code")

    assert long_method_item["pz_score"] > duplicated_item["pz_score"], (
        f"Long Method PZ ({long_method_item['pz_score']}) should be > Duplicated Code PZ ({duplicated_item['pz_score']})"
    )


def test_prioritization_ignores_different_files():
    """Test that dependencies only apply within same file/class context."""
    smells = [
        _smell("1", "Long Method",    "FileA.java", "HIGH"),
        _smell("2", "Duplicated Code","FileB.java", "HIGH"),
    ]

    prioritizer = SmellPrioritizer(smells)
    sequence = prioritizer.calculate_priorities()

    assert sequence[0]["pz_score"] == sequence[1]["pz_score"], (
        "Different files should not have dependency relationships"
    )


def test_prioritization_sequence_format():
    """Test that calculate_priorities() returns correct format."""
    smell = SmellEvent(
        smell_id="1",
        smell_type="Long Method",
        file_path="Test.java",
        line_number=10,
        severity="HIGH",
    )

    prioritizer = SmellPrioritizer([smell])
    sequence = prioritizer.calculate_priorities()

    assert len(sequence) == 1
    item = sequence[0]

    assert "order" in item
    assert "smell_id" in item
    assert "smell_type" in item
    assert "location" in item
    assert "pz_score" in item
    assert "positive_impacts" in item
    assert "negative_impacts" in item

    assert item["order"] == 1
    assert item["smell_id"] == "1"
    assert item["smell_type"] == "Long Method"
    assert item["location"] == "Test.java:10"
    assert isinstance(item["pz_score"], (int, float))


def test_agent_integration_format():
    """Test that SmellEvent objects are passed directly to SmellPrioritizer (no conversion needed)."""
    detected_smells = [
        SmellEvent(
            smell_id="Long Method:OrderProcessor.java:23",
            smell_type="Long Method",
            file_path="OrderProcessor.java",
            line_number=23,
            severity="MAJOR",
        ),
        SmellEvent(
            smell_id="Complex Method:OrderProcessor.java:23",
            smell_type="Complex Method",
            file_path="OrderProcessor.java",
            line_number=23,
            severity="MAJOR",
        ),
    ]

    prioritizer = SmellPrioritizer(detected_smells)
    priority_sequence = prioritizer.calculate_priorities()
    priority_ids = [item["smell_id"] for item in priority_sequence]

    assert len(priority_ids) == 2
    assert priority_ids[0] == "Long Method:OrderProcessor.java:23"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
