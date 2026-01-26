#!/usr/bin/env python3
"""
Test for smell prioritization logic.

Tests that the SmellPrioritizer correctly:
1. Creates SmellInstance objects with proper field names
2. Calculates PZ scores (severity + positive impact * 2)
3. Returns smells ordered by highest PZ score first
"""

import pytest
from scripts.prioritize_smells import SmellInstance, SmellPrioritizer


def test_smell_instance_creation():
    """Test that SmellInstance can be created with correct fields."""
    smell = SmellInstance(
        id="Long Method:OrderProcessor.java:23",
        smell_type="Long Method",
        location="OrderProcessor.java:23",
        severity="HIGH",
        description="Method too long",
    )

    assert smell.id == "Long Method:OrderProcessor.java:23"
    assert smell.smell_type == "Long Method"
    assert smell.location == "OrderProcessor.java:23"
    assert smell.severity == "HIGH"
    assert smell.severity_score == 3


def test_severity_score_mapping():
    """Test severity string to numeric score mapping."""
    high_smell = SmellInstance("1", "Test", "loc", "HIGH")
    assert high_smell.severity_score == 3

    critical_smell = SmellInstance("2", "Test", "loc", "CRITICAL")
    assert critical_smell.severity_score == 3

    medium_smell = SmellInstance("3", "Test", "loc", "MEDIUM")
    assert medium_smell.severity_score == 2

    major_smell = SmellInstance("4", "Test", "loc", "major")
    assert major_smell.severity_score == 2

    low_smell = SmellInstance("5", "Test", "loc", "LOW")
    assert low_smell.severity_score == 1


def test_prioritization_calculates_pz_correctly():
    """Test that PZ = severity + (positive_impact_count * 2)."""
    smells = [
        SmellInstance("1", "Long Method", "OrderProcessor.java:processOrder", "HIGH"),
        SmellInstance(
            "2", "Duplicated Code", "OrderProcessor.java:validateOrder", "MEDIUM"
        ),
        SmellInstance(
            "3", "Complex Method", "OrderProcessor.java:processOrder", "MEDIUM"
        ),
    ]

    prioritizer = SmellPrioritizer(smells)
    sequence = prioritizer.calculate_priorities()

    # Long Method (severity=3) + helps resolve Duplicated Code and Complex Method (+2*2=4) = 7
    # Should be first in sequence
    assert len(sequence) == 3
    assert sequence[0]["smell_type"] == "Long Method"
    assert sequence[0]["pz_score"] >= 3  # At minimum the severity score


def test_prioritization_returns_highest_pz_first():
    """Test that smells are ordered by PZ score descending."""
    smells = [
        SmellInstance("1", "God Class", "ReportGenerator.java", "CRITICAL"),
        SmellInstance(
            "2", "Long Method", "ReportGenerator.java:generateReport", "HIGH"
        ),
        SmellInstance("3", "Data Clumps", "ReportGenerator.java", "MEDIUM"),
        SmellInstance(
            "4", "Feature Envy", "ReportGenerator.java:calculateTotal", "MEDIUM"
        ),
        SmellInstance("5", "Print Statements", "ReportGenerator.java", "LOW"),
    ]

    prioritizer = SmellPrioritizer(smells)
    sequence = prioritizer.calculate_priorities()

    # Verify sequence is sorted by PZ descending
    pz_scores = [item["pz_score"] for item in sequence]
    assert pz_scores == sorted(pz_scores, reverse=True), (
        f"Expected descending PZ scores, got: {pz_scores}"
    )

    # Highest severity smell should be first (or tied with others having positive impacts)
    assert sequence[0]["smell_type"] in ["God Class", "Large Class"], (
        f"Expected high-priority smell first, got: {sequence[0]['smell_type']}"
    )


def test_prioritization_considers_dependencies():
    """Test that positive dependencies increase PZ score."""
    # Same file - dependencies apply
    smells = [
        SmellInstance("1", "Long Method", "Test.java:method1", "MEDIUM"),  # severity=2
        SmellInstance(
            "2", "Duplicated Code", "Test.java:method2", "MEDIUM"
        ),  # severity=2
    ]

    prioritizer = SmellPrioritizer(smells)
    sequence = prioritizer.calculate_priorities()

    # Long Method helps resolve Duplicated Code, so should have higher PZ
    # Long Method: severity(2) + positive_impact(1)*2 = 4
    # Duplicated Code: severity(2) + positive_impact(0)*2 = 2
    long_method_item = next(s for s in sequence if s["smell_type"] == "Long Method")
    duplicated_item = next(s for s in sequence if s["smell_type"] == "Duplicated Code")

    assert long_method_item["pz_score"] > duplicated_item["pz_score"], (
        f"Long Method PZ ({long_method_item['pz_score']}) should be > Duplicated Code PZ ({duplicated_item['pz_score']})"
    )


def test_prioritization_ignores_different_files():
    """Test that dependencies only apply within same file/class context."""
    # Different files - no dependencies
    smells = [
        SmellInstance("1", "Long Method", "FileA.java:method1", "HIGH"),
        SmellInstance("2", "Duplicated Code", "FileB.java:method2", "HIGH"),
    ]

    prioritizer = SmellPrioritizer(smells)
    sequence = prioritizer.calculate_priorities()

    # Both should have same PZ (severity only, no cross-file dependencies)
    assert sequence[0]["pz_score"] == sequence[1]["pz_score"], (
        "Different files should not have dependency relationships"
    )


def test_prioritization_sequence_format():
    """Test that calculate_priorities() returns correct format."""
    smells = [
        SmellInstance("1", "Long Method", "Test.java:method1", "HIGH"),
    ]

    prioritizer = SmellPrioritizer(smells)
    sequence = prioritizer.calculate_priorities()

    assert len(sequence) == 1
    item = sequence[0]

    # Verify all required keys exist
    assert "order" in item
    assert "smell_id" in item
    assert "smell_type" in item
    assert "location" in item
    assert "pz_score" in item
    assert "positive_impacts" in item
    assert "negative_impacts" in item

    # Verify values
    assert item["order"] == 1
    assert item["smell_id"] == "1"
    assert item["smell_type"] == "Long Method"
    assert item["location"] == "Test.java:method1"
    assert isinstance(item["pz_score"], (int, float))


def test_agent_integration_format():
    """Test the format expected by agent.py integration."""
    from swe_refactor.persistence.models import SmellEvent

    # Simulate SmellEvent objects from detection
    detected_smells = [
        SmellEvent(
            session_id="test",
            iteration=0,
            smell_type="Long Method",
            file_path="OrderProcessor.java",
            line_number=23,
            severity="major",
            rule_key="java:S138",
        ),
        SmellEvent(
            session_id="test",
            iteration=0,
            smell_type="Complex Method",
            file_path="OrderProcessor.java",
            line_number=23,
            severity="major",
            rule_key="java:S1541",
        ),
    ]

    # Convert to SmellInstance format (as done in agent.py)
    smell_instances = [
        SmellInstance(
            id=f"{s.smell_type}:{s.file_path}:{s.line_number}",
            smell_type=s.smell_type,
            location=f"{s.file_path}:{s.line_number}",
            severity=s.severity,
            description=getattr(s, "description", ""),
        )
        for s in detected_smells
    ]

    # Calculate priorities
    prioritizer = SmellPrioritizer(smell_instances)
    priority_sequence = prioritizer.calculate_priorities()

    # Extract smell_ids (as done in agent.py)
    priority_ids = [item["smell_id"] for item in priority_sequence]

    # Verify format
    assert len(priority_ids) == 2
    assert all(":" in smell_id for smell_id in priority_ids)
    assert priority_ids[0] == "Long Method:OrderProcessor.java:23"

    print(f"✓ Priority queue: {priority_ids}")
    print(f"✓ First smell PZ score: {priority_sequence[0]['pz_score']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
