#!/usr/bin/env python
"""Pre-processing pipeline CLI: dedup, filter, split, save to Parquet format.

Usage:
    uv run scripts/datasets/preprocess.py \\
        --source rminer \\
        --oracle-path /path/to/data.json \\
        --dedup \\
        --split 0.8/0.1/0.1 \\
        --filter-type "Extract Method" \\
        --output data/processed/

    uv run scripts/datasets/preprocess.py \\
        --source swe \\
        --dataset-path /path/to/SWE-Refactor.zip \\
        --split 0.8/0.1/0.1 \\
        --output data/processed/swe/

    uv run scripts/datasets/preprocess.py \\
        --source tdd \\
        --db-path /path/to/tdd.db \\
        --dedup \\
        --output data/processed/tdd/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root on path when run as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from smellai_datasets.config import DATASET_CONFIGS
from smellai_datasets.converter import rminer_to_df, swe_refactor_to_df, tdd_to_df
from smellai_datasets.preprocessor import deduplicate, filter_by, save, split


def _parse_split(split_str: str) -> tuple[float, float, float]:
    parts = [float(x) for x in split_str.split("/")]
    if len(parts) != 3:
        raise ValueError(f"--split must be 'train/val/test', got: {split_str!r}")
    total = sum(parts)
    return parts[0] / total, parts[1] / total, parts[2] / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-process research datasets")
    parser.add_argument(
        "--source",
        choices=["rminer", "swe", "tdd", "all"],
        required=True,
    )
    parser.add_argument("--oracle-path", default=os.environ.get("RMINER_ORACLE_PATH"))
    parser.add_argument("--dataset-path", default=os.environ.get("SWE_REFACTOR_PATH"))
    parser.add_argument("--db-path", default=os.environ.get("TDD_DB_PATH"))
    parser.add_argument("--dedup", action="store_true", help="Deduplicate rows")
    parser.add_argument(
        "--split",
        default=None,
        metavar="TRAIN/VAL/TEST",
        help="e.g. 0.8/0.1/0.1",
    )
    parser.add_argument(
        "--filter-type",
        default=None,
        metavar="TYPE",
        help="Keep only rows where refactoring_type == TYPE",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Row limit per source")
    args = parser.parse_args()

    sources_to_run = (
        ["rminer", "swe", "tdd"] if args.source == "all" else [args.source]
    )

    for source in sources_to_run:
        print(f"\n[{source}] Loading...")
        ds = _load(source, args)
        print(f"[{source}] Loaded {len(ds)} rows")

        if args.dedup:
            before = len(ds)
            ds = deduplicate(ds, DATASET_CONFIGS[source]["dedup_keys"])
            print(f"[{source}] After dedup: {len(ds)} rows (removed {before - len(ds)})")

        if args.filter_type:
            before = len(ds)
            ds = filter_by(ds, refactoring_type=args.filter_type)
            print(f"[{source}] After filter '{args.filter_type}': {len(ds)} rows (removed {before - len(ds)})")

        out_path = str(Path(args.output) / source)
        if args.split:
            train_frac, val_frac, test_frac = _parse_split(args.split)
            ds_dict = split(ds, train=train_frac, val=val_frac, test=test_frac)
            print(
                f"[{source}] Split: train={len(ds_dict['train'])} "
                f"val={len(ds_dict['val'])} test={len(ds_dict['test'])}"
            )
            save(ds_dict, out_path)
        else:
            save(ds, out_path)

        print(f"[{source}] Saved to {out_path}")


def _load(source: str, args: argparse.Namespace):
    if source == "rminer":
        if not args.oracle_path:
            raise ValueError("--oracle-path required for rminer source")
        return rminer_to_df(args.oracle_path, filter_tp=True, limit=args.limit)
    if source == "swe":
        if not args.dataset_path:
            raise ValueError("--dataset-path required for swe source")
        return swe_refactor_to_df(args.dataset_path, limit=args.limit)
    if source == "tdd":
        if not args.db_path:
            raise ValueError("--db-path required for tdd source")
        return tdd_to_df(db_path=args.db_path, limit=args.limit)
    raise ValueError(f"Unknown source: {source!r}")


if __name__ == "__main__":
    main()
