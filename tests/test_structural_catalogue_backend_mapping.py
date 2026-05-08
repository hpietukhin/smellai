"""Coverage tests for refactoring operations declared in :mod:`domain.rules`.

The goal is to ensure every operation in ``REFACTORING_CATALOGUE`` has a
stable backend suggestion and that known special-cases remain in the expected
bucket.
"""

from __future__ import annotations

import unittest

from domain.rules import REFACTORING_CATALOGUE
from agents.tools.edit_tools import suggest_structural_backend, structural_backend_advice


def _all_refactoring_ops() -> set[str]:
    """Return all distinct operation names declared in the refactoring catalogue."""

    ops: set[str] = set()
    for refs in REFACTORING_CATALOGUE.values():
        for op, _ in refs:
            ops.add(op)
    return ops


class TestStructuralBackendMapping(unittest.TestCase):
    """Back-end routing should remain stable for every operation in rules.py."""

    def test_catalogue_operations_have_expected_backend(self) -> None:
        # Inline/explicit fixtures for the operations that are not plain textual edits.
        spoon_ops: set[str] = {
            "Rename Method",
            "Move Method",
            "Extract Class",
            "Move Class",
            "Pull Up Method",
            "Pull Up Field",
            "Pull Up Attribute",
            "Push Down Method",
            "Push Down Field",
            "Push Down Attribute",
            "Replace Conditional with Polymorphism",
        }

        ast_grep_ops: set[str] = {
            "Rename Local Variable",
            "Rename Variable",
            "Replace with Logger",
            "Extract Method",
            "Inline Method",
        }

        known_catalogue_ops = _all_refactoring_ops()
        for op in sorted(known_catalogue_ops):
            with self.subTest(operation=op):
                expected = "text"
                if op in spoon_ops:
                    expected = "spoon"
                elif op in ast_grep_ops:
                    expected = "ast-grep"
                self.assertIn(op, known_catalogue_ops)
                self.assertEqual(structural_backend_advice(op), expected)
                self.assertEqual(suggest_structural_backend.invoke({"operation": op}), expected)

    def test_non_catalogue_operations_keep_default_policy(self) -> None:
        # Inlined fixture-style matrix for non-catalogue operations we still support.
        fixture: tuple[tuple[str, str], ...] = (
            ("Rename Local Variable", "ast-grep"),
            ("Rename Variable", "ast-grep"),
            ("Replace with Logger", "ast-grep"),
            ("Extract Method", "ast-grep"),
            ("Inline Method", "ast-grep"),
            ("Move Field", "text"),
            ("Rename Class", "text"),
            ("Move Attribute", "text"),
        )

        for op, expected_backend in fixture:
            with self.subTest(operation=op):
                self.assertEqual(suggest_structural_backend.invoke({"operation": op}), expected_backend)
