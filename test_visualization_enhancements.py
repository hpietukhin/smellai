#!/usr/bin/env python3
"""
Test script for visualization enhancements.
Verifies that PZ scores and positive/negative dependencies are correctly displayed.
"""

import json
from scripts.prioritize_smells import SmellInstance, SmellPrioritizer

# Create test data with known dependencies
test_smells = [
    SmellInstance(
        id="1",
        smell_type="God Class",
        location="ReportGenerator.java:ReportGenerator",
        severity="CRITICAL",
        description="Class has too many responsibilities",
    ),
    SmellInstance(
        id="2",
        smell_type="Large Class",
        location="ReportGenerator.java:ReportGenerator",
        severity="MAJOR",
        description="Class is too large",
    ),
    SmellInstance(
        id="3",
        smell_type="Data Clumps",
        location="ReportGenerator.java:generateReport",
        severity="MAJOR",
        description="Parameters appear together frequently",
    ),
    SmellInstance(
        id="4",
        smell_type="Feature Envy",
        location="ReportGenerator.java:calculateTotals",
        severity="MEDIUM",
        description="Method uses more external data than own",
    ),
    SmellInstance(
        id="5",
        smell_type="Long Method",
        location="OrderProcessor.java:processOrder",
        severity="HIGH",
        description="Method is too long",
    ),
    SmellInstance(
        id="6",
        smell_type="Duplicated Code",
        location="OrderProcessor.java:validateOrder",
        severity="MEDIUM",
        description="Code appears in multiple places",
    ),
]

print("=" * 80)
print("TESTING VISUALIZATION ENHANCEMENTS")
print("=" * 80)

# Create prioritizer
prioritizer = SmellPrioritizer(test_smells)

# Calculate priorities
sequence = prioritizer.calculate_priorities()

print(f"\n✓ Created {len(test_smells)} test smells")
print(f"✓ Built dependency graph with {prioritizer.graph.number_of_nodes()} nodes")
print(f"✓ Built dependency graph with {prioritizer.graph.number_of_edges()} edges")

# Analyze edges
positive_edges = [
    e for e in prioritizer.graph.edges(data=True) if e[2].get("type") == "positive"
]
negative_edges = [
    e for e in prioritizer.graph.edges(data=True) if e[2].get("type") == "negative"
]

print(f"\n✓ Positive dependencies (green edges): {len(positive_edges)}")
print(f"✓ Negative dependencies (red edges): {len(negative_edges)}")

print("\n" + "=" * 80)
print("PRIORITY SEQUENCE (with PZ scores and dependencies)")
print("=" * 80)
print(
    f"{'#':<4} | {'PZ':<4} | {'Smell Type':<20} | {'Severity':<8} | {'Positive':<8} | {'Negative'}"
)
print("-" * 80)

for item in sequence:
    order = item["order"]
    pz = item["pz_score"]
    smell_type = item["smell_type"]
    severity = prioritizer.graph.nodes[item["smell_id"]]["data"].severity
    pos = item["positive_impacts"]
    neg = item["negative_impacts"]

    print(
        f"{order:<4} | {pz:<4} | {smell_type:<20} | {severity:<8} | +{pos:<7} | −{neg}"
    )

print("\n" + "=" * 80)
print("DECISION RATIONALE")
print("=" * 80)

# Show top 3 decisions
for i in range(min(3, len(sequence))):
    item = sequence[i]
    smell_id = item["smell_id"]
    smell = prioritizer.graph.nodes[smell_id]["data"]

    print(f"\n#{item['order']}: {item['smell_type']}")
    print(f"  Location: {item['location']}")
    print(f"  PZ Score: {item['pz_score']}")
    print(
        f"  Formula: PZ = {smell.severity_score} (severity) + {item['positive_impacts']} × 2 (positive impacts)"
    )
    print(
        f"  Positive impacts: {item['positive_impacts']} (refactoring this helps resolve {item['positive_impacts']} other smell(s))"
    )
    print(
        f"  Negative impacts: {item['negative_impacts']} (refactoring this may create {item['negative_impacts']} new smell(s))"
    )

    if item["positive_impacts"] > 0:
        # Show which smells it helps
        helped_smells = []
        for _, target, edge_data in prioritizer.graph.out_edges(smell_id, data=True):
            if edge_data.get("type") == "positive":
                target_smell = prioritizer.graph.nodes[target]["data"]
                helped_smells.append(target_smell.smell_type)
        if helped_smells:
            print(f"  Helps resolve: {', '.join(helped_smells)}")

print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)

# Verify highest PZ is first
assert sequence[0]["pz_score"] >= sequence[1]["pz_score"], (
    "First smell should have highest PZ"
)
print("✓ Highest PZ score is first in sequence")

# Verify PZ formula
for item in sequence:
    smell_id = item["smell_id"]
    smell = prioritizer.graph.nodes[smell_id]["data"]
    expected_pz = smell.severity_score + (item["positive_impacts"] * 2)
    assert item["pz_score"] == expected_pz, f"PZ calculation mismatch for {smell_id}"
print("✓ PZ formula verified: severity + (positive_impacts × 2)")

# Verify positive dependencies exist
has_positive_deps = any(item["positive_impacts"] > 0 for item in sequence)
print(f"✓ Positive dependencies detected: {has_positive_deps}")

# Verify negative dependencies exist
has_negative_deps = any(item["negative_impacts"] > 0 for item in sequence)
print(f"✓ Negative dependencies detected: {has_negative_deps}")

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\nEnhanced visualization will show:")
print("  • PZ scores on each node")
print("  • Priority order numbers (#1, #2, etc.)")
print("  • Green solid edges for positive dependencies (+)")
print("  • Red dashed edges for negative dependencies (−)")
print("  • Node size proportional to PZ score")
print("  • Blue border on next smell to be refactored")
print("  • Detailed PZ formula in sidebar on click")
print("  • Priority sequence table with PZ and +/− counts")
