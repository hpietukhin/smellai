#!/usr/bin/env python3
"""
Test for smell prioritization logic.

Tests that the SmellPrioritizer correctly:
1. Creates SmellEvent objects usable in-memory (no DB session needed)
2. Calculates PZ scores via Eq. 2
3. Returns smells ordered by highest PZ score first
4. Greedy planner resolves positive dependencies (fewer steps)
"""

import pytest
from domain.models import SmellEvent
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


def test_prioritization_picks_long_method_first():
    """LM has highest score (positive deps to Dup.Code + Complex Method)."""
    smells = [
        _smell("1", "Long Method",     "OrderProcessor.java", "HIGH"),
        _smell("2", "Duplicated Code", "OrderProcessor.java", "MEDIUM"),
        _smell("3", "Complex Method",  "OrderProcessor.java", "MEDIUM"),
    ]

    prioritizer = SmellPrioritizer(smells)
    sequence = prioritizer.calculate_priorities()

    # LM resolves Dup.Code and Complex Method via positive deps → 1 step
    assert sequence[0]["smell_type"] == "Long Method"
    assert sequence[0]["pz_score"] > 0


def test_prioritization_resolves_positive_deps_in_fewer_steps():
    """Greedy with transition: resolving LM also resolves its positive neighbors."""
    smells = [
        _smell("1", "Long Method",     "OrderProcessor.java", "HIGH"),
        _smell("2", "Duplicated Code", "OrderProcessor.java", "MEDIUM"),
        _smell("3", "Complex Method",  "OrderProcessor.java", "MEDIUM"),
    ]

    prioritizer = SmellPrioritizer(smells)
    sequence = prioritizer.calculate_priorities()

    # LM positive deps include Dup.Code and Complex Method (same file, locality=none)
    # So greedy resolves all 3 in 1 step
    assert len(sequence) < 3


def test_prioritization_returns_highest_pz_first():
    """Test that first step has the highest PZ score."""
    smells = [
        _smell("1", "God Class",       "ReportGenerator.java", "CRITICAL"),
        _smell("2", "Long Method",     "ReportGenerator.java", "HIGH"),
        _smell("3", "Data Clumps",     "ReportGenerator.java", "MEDIUM"),
        _smell("4", "Feature Envy",    "ReportGenerator.java", "MEDIUM"),
        _smell("5", "Print Statements","ReportGenerator.java", "LOW"),
    ]

    prioritizer = SmellPrioritizer(smells)
    sequence = prioritizer.calculate_priorities()

    # First action should be highest-scoring smell
    assert len(sequence) > 0
    assert sequence[0]["smell_type"] in ["God Class", "Long Method"]


def test_prioritization_considers_dependencies():
    """LM with higher severity scores more than DC with lower severity."""
    smells = [
        _smell("1", "Long Method",    "Test.java", "HIGH"),     # sev=3
        _smell("2", "Duplicated Code","Test.java", "MEDIUM"),   # sev=2
    ]

    prioritizer = SmellPrioritizer(smells)
    sequence = prioritizer.calculate_priorities()

    # LM has higher severity → picked first; resolves DC via positive dep → 1 step
    assert sequence[0]["smell_type"] == "Long Method"
    assert len(sequence) == 1


def test_prioritization_mutual_positive_deps_resolve_in_one_step():
    """When LM and DC have mutual positive deps, one step resolves both."""
    smells = [
        _smell("1", "Long Method",    "FileA.java", "HIGH"),
        _smell("2", "Duplicated Code","FileB.java", "HIGH"),
    ]

    prioritizer = SmellPrioritizer(smells)
    sequence = prioritizer.calculate_priorities()

    # With locality=none, LM↔DC have mutual positive deps → 1 step
    assert len(sequence) == 1


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
    """SmellEvent objects are passed directly to SmellPrioritizer."""
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

    # LM and CM have mutual positive deps → one step resolves both
    assert len(priority_sequence) == 1
    # Either can be picked first (same severity, mutual deps)
    assert priority_sequence[0]["smell_id"] in (
        "Long Method:OrderProcessor.java:23",
        "Complex Method:OrderProcessor.java:23",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
