"""
Test suite for validating code smell co-occurrence detection.

This test suite validates that the smell detection system correctly identifies
co-occurring smells in the test Java files and recognizes positive/negative
dependencies between smells.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "test_data" / "smell_cooccurrence"


@pytest.fixture
def smells_manifest(test_data_dir):
    """Load the smells manifest JSON."""
    manifest_path = test_data_dir / "smells_manifest.json"
    return json.loads(manifest_path.read_text())


class TestSmellCooccurrenceFiles:
    """Test that all smell co-occurrence files exist and are valid."""

    def test_all_files_exist(self, test_data_dir, smells_manifest):
        """Verify all documented Java files exist."""
        for file_info in smells_manifest["files"]:
            filename = file_info["filename"]
            java_file = test_data_dir / filename
            assert java_file.exists(), f"Missing file: {filename}"
            assert java_file.suffix == ".java", f"Not a Java file: {filename}"

    def test_files_not_empty(self, test_data_dir, smells_manifest):
        """Verify all Java files have content."""
        for file_info in smells_manifest["files"]:
            java_file = test_data_dir / file_info["filename"]
            content = java_file.read_text()
            assert len(content) > 100, f"File too small: {file_info['filename']}"
            assert "package com.example.smells;" in content

    def test_manifest_structure(self, smells_manifest):
        """Verify manifest has correct structure."""
        assert "files" in smells_manifest
        assert "smell_dependencies" in smells_manifest
        assert "positive" in smells_manifest["smell_dependencies"]
        assert "negative" in smells_manifest["smell_dependencies"]
        assert "sonarqube_rules" in smells_manifest

    def test_all_files_have_smells(self, smells_manifest):
        """Verify each file documents at least one smell."""
        for file_info in smells_manifest["files"]:
            assert len(file_info["smells"]) > 0, \
                f"No smells documented for {file_info['filename']}"


class TestPositiveDependencies:
    """Test detection of positive smell dependencies (solving one solves others)."""

    def test_long_method_dependencies(self, smells_manifest):
        """Test that Long Method has documented positive dependencies."""
        deps = smells_manifest["smell_dependencies"]["positive"]["Long Method"]
        expected = {"Duplicated Code", "Switch Statement", "Print Statements",
                   "Conditional Complexity", "Feature Envy"}
        assert set(deps) == expected

    def test_large_class_dependencies(self, smells_manifest):
        """Test that Large Class has documented positive dependencies."""
        deps = smells_manifest["smell_dependencies"]["positive"]["Large Class"]
        expected = {"Data Clumps", "Feature Envy", "Long Method"}
        assert set(deps) == expected

    def test_duplicated_conditions_dependencies(self, smells_manifest):
        """Test that Duplicated Conditions has documented positive dependencies."""
        deps = smells_manifest["smell_dependencies"]["positive"]["Duplicated Conditions"]
        assert "Duplicated Code" in deps
        assert "Complex Method" in deps


class TestNegativeDependencies:
    """Test detection of negative smell dependencies (fixing creates new smells)."""

    def test_long_parameter_list_creates_data_class(self, smells_manifest):
        """Test that fixing Long Parameter List can create Data Class."""
        negative_deps = smells_manifest["smell_dependencies"]["negative"]["Long Parameter List"]
        assert len(negative_deps) > 0

        dep = negative_deps[0]
        assert dep["refactoring"] == "Introduce Parameter Object"
        assert dep["creates"] == "Data Class"

    def test_customer_data_service_demonstrates_negative_dep(self, smells_manifest):
        """Test CustomerDataService.java demonstrates the negative dependency."""
        customer_service = next(
            f for f in smells_manifest["files"]
            if f["filename"] == "CustomerDataService.java"
        )

        smell_types = [s["type"] for s in customer_service["smells"]]
        assert "Long Parameter List" in smell_types
        assert "Data Class" in smell_types

        negative_deps = customer_service.get("negative_dependencies", [])
        assert len(negative_deps) > 0
        assert "Data Class" in negative_deps[0]


class TestOrderProcessor:
    """Test OrderProcessor.java smell detection."""

    def test_has_long_method(self, smells_manifest):
        """Test OrderProcessor has Long Method smell."""
        order_processor = next(
            f for f in smells_manifest["files"]
            if f["filename"] == "OrderProcessor.java"
        )

        smell_types = [s["type"] for s in order_processor["smells"]]
        assert "Long Method" in smell_types

    def test_has_multiple_cooccurring_smells(self, smells_manifest):
        """Test OrderProcessor has multiple co-occurring smells."""
        order_processor = next(
            f for f in smells_manifest["files"]
            if f["filename"] == "OrderProcessor.java"
        )

        smell_types = {s["type"] for s in order_processor["smells"]}
        expected_smells = {
            "Long Method",
            "Complex Method",
            "Conditional Complexity",
            "Duplicated Code",
            "Switch Statement",
            "Print Statements"
        }
        assert smell_types == expected_smells

    def test_has_positive_dependencies(self, smells_manifest):
        """Test OrderProcessor documents positive dependencies."""
        order_processor = next(
            f for f in smells_manifest["files"]
            if f["filename"] == "OrderProcessor.java"
        )

        assert "positive_dependencies" in order_processor
        assert len(order_processor["positive_dependencies"]) > 0


class TestReportGenerator:
    """Test ReportGenerator.java smell detection."""

    def test_is_god_class(self, smells_manifest):
        """Test ReportGenerator is detected as God Class."""
        report_gen = next(
            f for f in smells_manifest["files"]
            if f["filename"] == "ReportGenerator.java"
        )

        smell_types = [s["type"] for s in report_gen["smells"]]
        assert "God Class" in smell_types or "Large Class" in smell_types

    def test_has_data_clumps(self, smells_manifest):
        """Test ReportGenerator has Data Clumps smell."""
        report_gen = next(
            f for f in smells_manifest["files"]
            if f["filename"] == "ReportGenerator.java"
        )

        data_clump_smell = next(
            s for s in report_gen["smells"]
            if s["type"] == "Data Clumps"
        )

        assert "parameters" in data_clump_smell
        params = data_clump_smell["parameters"]
        assert "startDate" in params
        assert "endDate" in params
        assert "region" in params
        assert "category" in params


class TestPaymentValidator:
    """Test PaymentValidator.java smell detection."""

    def test_has_duplicated_conditions(self, smells_manifest):
        """Test PaymentValidator has Duplicated Conditions smell."""
        payment_validator = next(
            f for f in smells_manifest["files"]
            if f["filename"] == "PaymentValidator.java"
        )

        smell_types = [s["type"] for s in payment_validator["smells"]]
        assert "Duplicated Conditions" in smell_types
        assert "Duplicated Code" in smell_types


class TestSonarQubeRules:
    """Test SonarQube rule mappings."""

    def test_all_rules_documented(self, smells_manifest):
        """Test all SonarQube rules are documented."""
        rules = smells_manifest["sonarqube_rules"]

        expected_rules = {
            "java:S138",  # Long Method
            "java:S1541",  # Complex Method
            "java:S1067",  # Conditional Complexity
            "java:S107",  # Long Parameter List
            "java:S1200",  # God Class
            "java:S110",  # Large Class
            "java:S1871",  # Duplicated Conditions
            "java:S106",  # Print Statements
        }

        assert set(rules.keys()) == expected_rules

    def test_rule_descriptions_present(self, smells_manifest):
        """Test all rules have descriptions."""
        rules = smells_manifest["sonarqube_rules"]

        for rule_id, description in rules.items():
            assert description, f"Missing description for {rule_id}"
            assert len(description) > 10


class TestDependencyGraph:
    """Test that smell dependencies form valid graphs."""

    def test_no_circular_positive_dependencies(self, smells_manifest):
        """Test positive dependencies don't form cycles."""
        positive = smells_manifest["smell_dependencies"]["positive"]

        # Build adjacency list
        graph = {smell: set(deps) for smell, deps in positive.items()}

        # Check for cycles using DFS
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)

            if node in graph:
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        if has_cycle(neighbor, visited, rec_stack):
                            return True
                    elif neighbor in rec_stack:
                        return True

            rec_stack.remove(node)
            return False

        visited = set()
        for smell in graph:
            if smell not in visited:
                assert not has_cycle(smell, visited, set()), \
                    "Circular dependency detected in positive dependencies"

    def test_negative_dependencies_valid(self, smells_manifest):
        """Test negative dependencies have required fields."""
        negative = smells_manifest["smell_dependencies"]["negative"]

        for smell, deps in negative.items():
            assert isinstance(deps, list)
            for dep in deps:
                assert isinstance(dep, dict)
                assert "refactoring" in dep
                assert "creates" in dep


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
