"""Coverage tests for dataset refactoring labels in REFACTORING_CATALOGUE."""
from __future__ import annotations

from domain.rules import REFACTORING_CATALOGUE, REFUSED_BEQUEST


COMPOSITE_REFACTORINGS_2020_LABELS = {
    # Observed in the local Composite Refactorings 2020 Neo4j graph.
    "Move Attribute",
    "Rename Method",
    "Extract Method",
    "Inline Method",
    "Move Class",
    "Move Method",
    "Pull Up Method",
    "Rename Class",
    "Pull Up Attribute",
    "Extract Superclass",
    "Extract Interface",
    "Push Down Method",
    "Push Down Attribute",
}


def _catalogue_labels() -> set[str]:
    return {
        ref_type
        for operations in REFACTORING_CATALOGUE.values()
        for ref_type, _rank in operations
    }


def test_composite_refactorings_2020_labels_are_in_catalogue() -> None:
    """Planner/dataset matching must not rely on eval-layer aliases."""
    missing = COMPOSITE_REFACTORINGS_2020_LABELS - _catalogue_labels()
    assert missing == set()


def test_attribute_variants_are_catalogue_entries_not_aliases() -> None:
    """Dataset uses Attribute where Markovič/Fowler usually use Field."""
    catalogue_labels = _catalogue_labels()
    assert "Move Field" in catalogue_labels
    assert "Move Attribute" in catalogue_labels
    assert "Pull Up Field" in catalogue_labels
    assert "Pull Up Attribute" in catalogue_labels
    assert "Push Down Field" in catalogue_labels
    assert "Push Down Attribute" in catalogue_labels


def test_dataset_refused_bequest_move_attribute_example_is_represented() -> None:
    """Concrete dataset example: Google I/O VendorDetailFragment.

    Before: RefusedBequest smells.
    Developer: Move Method + Move Attribute.
    After: zero smells.
    Therefore the catalogue must allow those operations for Refused Bequest.
    """
    ops = {op for op, _rank in REFACTORING_CATALOGUE[REFUSED_BEQUEST]}
    assert "Move Method" in ops
    assert "Move Attribute" in ops
