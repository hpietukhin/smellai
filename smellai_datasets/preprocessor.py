"""Preprocessing utilities for pandas DataFrames.

Provides deduplication, train/val/test splitting, filtering, and
Parquet persistence.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def deduplicate(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Remove duplicate rows based on a combination of columns.

    Args:
        df: Input DataFrame
        key_cols: Column names that together form the dedup key

    Returns:
        DataFrame with duplicate rows removed (first occurrence kept)
    """
    return df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)


def split(
    df: pd.DataFrame,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    stratify_col: str | None = None,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split DataFrame into train / val / test subsets.

    Args:
        df: Input DataFrame
        train: Fraction for training set
        val: Fraction for validation set
        test: Fraction for test set (remainder from train+val)
        stratify_col: Column to stratify on (optional)
        seed: Random seed for reproducibility

    Returns:
        Dict with keys "train", "val", "test"
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Fractions must sum to 1.0"

    if stratify_col is not None:
        return _stratified_split(df, train, val, test, stratify_col, seed)

    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(n * train)
    val_end = train_end + int(n * val)

    return {
        "train": shuffled.iloc[:train_end].reset_index(drop=True),
        "val": shuffled.iloc[train_end:val_end].reset_index(drop=True),
        "test": shuffled.iloc[val_end:].reset_index(drop=True),
    }


def _stratified_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    stratify_col: str,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Stratified split preserving class proportions via groupby."""
    trains, vals, tests = [], [], []

    for _, group in df.groupby(stratify_col):
        shuffled = group.sample(frac=1, random_state=seed)
        n = len(shuffled)
        train_end = int(n * train_frac)
        val_end = train_end + int(n * val_frac)

        trains.append(shuffled.iloc[:train_end])
        vals.append(shuffled.iloc[train_end:val_end])
        tests.append(shuffled.iloc[val_end:])

    return {
        "train": pd.concat(trains, ignore_index=True),
        "val": pd.concat(vals, ignore_index=True),
        "test": pd.concat(tests, ignore_index=True),
    }


def filter_by(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Filter DataFrame rows by exact column value matches.

    Args:
        df: Input DataFrame
        **kwargs: Column → value pairs (e.g. refactoring_type="Extract Method")

    Returns:
        Filtered DataFrame
    """
    mask = pd.Series(True, index=df.index)
    for col, value in kwargs.items():
        mask &= df[col] == value
    return df.loc[mask].reset_index(drop=True)


def save(df: pd.DataFrame | dict[str, pd.DataFrame], path: str) -> None:
    """Save DataFrame(s) to Parquet format.

    Args:
        df: DataFrame or dict of split name → DataFrame
        path: Output directory path
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    if isinstance(df, dict):
        for name, split_df in df.items():
            split_df.to_parquet(out / f"{name}.parquet", index=False)
    else:
        df.to_parquet(out / "data.parquet", index=False)


def load(path: str) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Load DataFrame(s) from Parquet format on disk.

    Args:
        path: Directory path written by save()

    Returns:
        DataFrame (single file) or dict of split name → DataFrame (multi-split)
    """
    p = Path(path)
    single = p / "data.parquet"
    if single.exists():
        return pd.read_parquet(single)

    # Multi-split: find all .parquet files
    parquet_files = sorted(p.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {path}")

    if len(parquet_files) == 1:
        return pd.read_parquet(parquet_files[0])

    return {f.stem: pd.read_parquet(f) for f in parquet_files}
