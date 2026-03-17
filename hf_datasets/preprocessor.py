"""Preprocessing utilities for HuggingFace Datasets.

Provides deduplication, train/val/test splitting, filtering, and
Arrow-format persistence.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


def deduplicate(ds: Dataset, key_cols: list[str]) -> Dataset:
    """Remove duplicate rows based on a combination of columns.

    Args:
        ds: Input dataset
        key_cols: Column names that together form the dedup key

    Returns:
        Dataset with duplicate rows removed (first occurrence kept)
    """
    seen: set[tuple] = set()
    keep_indices: list[int] = []

    for i, row in enumerate(ds):
        key = tuple(row.get(c) for c in key_cols)
        if key not in seen:
            seen.add(key)
            keep_indices.append(i)

    return ds.select(keep_indices)


def split(
    ds: Dataset,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    stratify_col: str | None = None,
    seed: int = 42,
) -> DatasetDict:
    """Split dataset into train / val / test subsets.

    Args:
        ds: Input dataset
        train: Fraction for training set
        val: Fraction for validation set
        test: Fraction for test set (remainder from train+val)
        stratify_col: Column to stratify on (optional)
        seed: Random seed for reproducibility

    Returns:
        DatasetDict with keys "train", "val", "test"
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Fractions must sum to 1.0"

    # First split: separate test set
    test_size = test
    train_val = ds.train_test_split(
        test_size=test_size,
        seed=seed,
        stratify_by_column=stratify_col,
    )

    # Second split: separate val from remaining train+val
    val_ratio_of_remaining = val / (train + val)
    train_val_split = train_val["train"].train_test_split(
        test_size=val_ratio_of_remaining,
        seed=seed,
        stratify_by_column=stratify_col,
    )

    return DatasetDict(
        {
            "train": train_val_split["train"],
            "val": train_val_split["test"],
            "test": train_val["test"],
        }
    )


def filter_by(ds: Dataset, **kwargs) -> Dataset:
    """Filter dataset rows by exact column value matches.

    Args:
        ds: Input dataset
        **kwargs: Column → value pairs (e.g. refactoring_type="Extract Method")

    Returns:
        Filtered dataset
    """
    def _predicate(row: dict) -> bool:
        return all(row.get(k) == v for k, v in kwargs.items())

    return ds.filter(_predicate)


def save(ds: Dataset | DatasetDict, path: str) -> None:
    """Save dataset to disk in Arrow format.

    Args:
        ds: Dataset or DatasetDict to save
        path: Output directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(path)


def load(path: str) -> Dataset | DatasetDict:
    """Load dataset from Arrow format on disk.

    Args:
        path: Directory path written by save()

    Returns:
        Dataset or DatasetDict depending on what was saved
    """
    return load_from_disk(path)
