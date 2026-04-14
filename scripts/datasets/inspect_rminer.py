#!/usr/bin/env python
"""Step-by-step inspection of RMiner oracle dataset pipeline.

Shows the DataFrame at each transformation stage:
  1. Raw oracle → DataFrame (rminer_to_df)
  2. Deduplication
  3. Train/val/test split
  4. MLflow GenAI format (hf_to_genai_records)

Usage:
    uv run scripts/datasets/inspect_rminer.py \\
        --oracle-path ~/Downloads/data.json

    # Limit rows for fast inspection
    uv run scripts/datasets/inspect_rminer.py \\
        --oracle-path ~/Downloads/data.json --limit 500

    # Include false positives
    uv run scripts/datasets/inspect_rminer.py \\
        --oracle-path ~/Downloads/data.json --include-fp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from smellai_datasets.config import DATASET_CONFIGS, MLFLOW_COLUMN_MAP
from smellai_datasets.converter import rminer_to_df
from smellai_datasets.mlflow_bridge import hf_to_genai_records
from smellai_datasets.preprocessor import deduplicate, split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP = "=" * 70


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def show_df(df: pd.DataFrame, n: int = 3) -> None:
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Columns: {list(df.columns)}\n")
    print(df.head(n).to_string())


def show_value_counts(df: pd.DataFrame, col: str, top: int = 10) -> None:
    vc = df[col].value_counts().head(top)
    print(f"\nTop {top} {col!r}:")
    for val, cnt in vc.items():
        print(f"  {cnt:6d}  {val}")


def show_mlflow_record(record: dict) -> None:
    """Pretty-print a single MLflow GenAI record."""
    print(json.dumps(record, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect RMiner oracle pipeline")
    parser.add_argument(
        "--oracle-path",
        default="/Users/havriil.pietukhin/uni/masterThesis/datasets/rminer_oracle_java1.json",
        help="Path to RMiner oracle data.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Row limit applied after loading (default: all)",
    )
    parser.add_argument(
        "--include-fp",
        action="store_true",
        help="Include false positives (default: TP only)",
    )
    parser.add_argument(
        "--split",
        default="0.8/0.1/0.1",
        metavar="TRAIN/VAL/TEST",
    )
    parser.add_argument(
        "--show-rows",
        type=int,
        default=3,
        help="Number of sample rows to print (default: 3)",
    )
    args = parser.parse_args()

    oracle_path = Path(args.oracle_path).expanduser()
    if not oracle_path.exists():
        print(f"Oracle not found: {oracle_path}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # STAGE 1: Raw oracle → DataFrame
    # ------------------------------------------------------------------
    section("STAGE 1: Raw oracle → DataFrame  (rminer_to_df)")
    print(f"Source: {oracle_path}")
    print(f"filter_tp={not args.include_fp}, limit={args.limit}\n")

    df = rminer_to_df(
        oracle_path=oracle_path,
        filter_tp=not args.include_fp,
        limit=args.limit,
    )
    show_df(df, n=args.show_rows)

    # Breakdown by refactoring type
    show_value_counts(df, "refactoring_type", top=15)

    # Validation distribution
    print("\nValidation distribution:")
    print(df["validation"].value_counts().to_string())

    # ------------------------------------------------------------------
    # STAGE 2: Deduplication
    # ------------------------------------------------------------------
    section("STAGE 2: Deduplication")
    dedup_keys = DATASET_CONFIGS["rminer"]["dedup_keys"]
    print(f"Dedup keys: {dedup_keys}\n")

    before = len(df)
    df_dedup = deduplicate(df, dedup_keys)
    after = len(df_dedup)
    print(f"Before: {before} rows")
    print(f"After:  {after} rows")
    print(f"Removed: {before - after} duplicates")
    show_df(df_dedup, n=args.show_rows)

    # ------------------------------------------------------------------
    # STAGE 3: Train/val/test split
    # ------------------------------------------------------------------
    section("STAGE 3: Train / val / test split")
    parts = [float(x) for x in args.split.split("/")]
    total = sum(parts)
    train_frac, val_frac, test_frac = parts[0] / total, parts[1] / total, parts[2] / total
    stratify_col = DATASET_CONFIGS["rminer"]["stratify_col"]
    print(f"Fractions: train={train_frac:.2f} val={val_frac:.2f} test={test_frac:.2f}")
    print(f"Stratify on: {stratify_col!r}\n")

    splits = split(
        df_dedup,
        train=train_frac,
        val=val_frac,
        test=test_frac,
        stratify_col=stratify_col,
    )
    for name, sdf in splits.items():
        print(f"  {name:5s}: {len(sdf):6d} rows")

    print("\nSample train row:")
    show_df(splits["train"], n=1)

    # ------------------------------------------------------------------
    # STAGE 4: MLflow GenAI format
    # ------------------------------------------------------------------
    section("STAGE 4: MLflow GenAI format  (hf_to_genai_records)")
    col_map = MLFLOW_COLUMN_MAP["rminer"]
    print("Column mapping:")
    print(f"  input_cols:       {col_map['input_cols']}")
    print(f"  expectation_cols: {col_map['expectation_cols']}")
    print(f"  tag_cols:         {col_map['tag_cols']}\n")

    # Convert test split to MLflow format
    records = hf_to_genai_records(splits["test"], source="rminer")
    print(f"Total MLflow records (test split): {len(records)}\n")
    print("Sample record:")
    show_mlflow_record(records[0] if records else {})

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section("SUMMARY")
    print(f"oracle path:          {oracle_path}")
    print(f"raw rows:             {before}")
    print(f"after dedup:          {after}")
    print(f"train:                {len(splits['train'])}")
    print(f"val:                  {len(splits['val'])}")
    print(f"test:                 {len(splits['test'])}")
    print(f"mlflow records (test):{len(records)}")
    print(f"unique ref types:     {df_dedup['refactoring_type'].nunique()}")
    print()


if __name__ == "__main__":
    main()
