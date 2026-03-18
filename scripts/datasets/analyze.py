#!/usr/bin/env python
"""Dataset aggregation and statistics CLI.

Usage:
    uv run scripts/datasets/analyze.py --source rminer --oracle-path /path/to/data.json
    uv run scripts/datasets/analyze.py --source swe --dataset-path /path/to/SWE-Refactor.zip
    uv run scripts/datasets/analyze.py --source tdd --db-path /path/to/tdd.db
    uv run scripts/datasets/analyze.py --source all --output report.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

# Ensure project root on path when run as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from smellai_datasets.converter import rminer_to_df, swe_refactor_to_df, tdd_to_df


def _loc(text: str) -> int:
    """Count non-empty lines in a code string."""
    return sum(1 for line in text.splitlines() if line.strip())


def analyze_rminer(oracle_path: str) -> dict[str, Any]:
    """Compute stats for the RMiner oracle dataset."""
    ds = rminer_to_df(oracle_path, filter_tp=False)
    rows = ds.to_dict("records")

    type_counts: dict[str, int] = {}
    repo_counts: dict[str, int] = {}
    tp_count = 0
    fp_count = 0

    for row in rows:
        rtype = row.get("refactoring_type", "unknown")
        repo = row.get("repository", "unknown")
        validation = row.get("validation", "")

        type_counts[rtype] = type_counts.get(rtype, 0) + 1
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
        if validation == "TP":
            tp_count += 1
        elif validation == "FP":
            fp_count += 1

    total = len(rows)
    return {
        "source": "rminer",
        "total_rows": total,
        "tp_count": tp_count,
        "fp_count": fp_count,
        "tp_rate": round(tp_count / total, 4) if total else 0,
        "unique_repos": len(repo_counts),
        "refactoring_type_distribution": {
            k: {"count": v, "pct": round(v / total * 100, 2)}
            for k, v in sorted(type_counts.items(), key=lambda x: -x[1])
        },
        "top_repos": dict(
            sorted(repo_counts.items(), key=lambda x: -x[1])[:10]
        ),
    }


def analyze_swe(dataset_path: str) -> dict[str, Any]:
    """Compute stats for the SWE-Refactor dataset."""
    ds = swe_refactor_to_df(dataset_path)
    rows = ds.to_dict("records")

    type_counts: dict[str, int] = {}
    project_counts: dict[str, int] = {}
    compound_count = 0
    pure_count = 0
    loc_deltas: list[int] = []

    for row in rows:
        rtype = row.get("refactoring_type", "unknown")
        project = row.get("project_name", "unknown")
        is_compound = row.get("is_compound", False)
        is_pure = row.get("is_pure", False)
        src_before = row.get("source_before", "")
        src_after = row.get("source_after", "")

        type_counts[rtype] = type_counts.get(rtype, 0) + 1
        project_counts[project] = project_counts.get(project, 0) + 1
        if is_compound:
            compound_count += 1
        if is_pure:
            pure_count += 1
        if src_before or src_after:
            loc_deltas.append(_loc(src_after) - _loc(src_before))

    total = len(rows)
    return {
        "source": "swe_refactor",
        "total_rows": total,
        "compound_count": compound_count,
        "atomic_count": total - compound_count,
        "pure_count": pure_count,
        "unique_projects": len(project_counts),
        "loc_delta_stats": _distribution_stats(loc_deltas),
        "refactoring_type_distribution": {
            k: {"count": v, "pct": round(v / total * 100, 2)}
            for k, v in sorted(type_counts.items(), key=lambda x: -x[1])
        },
        "project_breakdown": dict(
            sorted(project_counts.items(), key=lambda x: -x[1])
        ),
    }


def analyze_tdd(db_path: str) -> dict[str, Any]:
    """Compute stats for the Technical Debt Dataset."""
    ds = tdd_to_df(db_path=db_path)
    rows = ds.to_dict("records")

    project_counts: dict[str, int] = {}
    smell_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    chain_lengths: dict[str, int] = {}  # project+smell_type+file → count

    for row in rows:
        project = row.get("project", "unknown")
        smell_type = row.get("smell_type", "unknown")
        status = row.get("status", "unknown")
        file_path = row.get("file_path", "")

        project_counts[project] = project_counts.get(project, 0) + 1
        smell_counts[smell_type] = smell_counts.get(smell_type, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
        chain_key = f"{project}|{smell_type}|{file_path}"
        chain_lengths[chain_key] = chain_lengths.get(chain_key, 0) + 1

    lengths = list(chain_lengths.values())
    total = len(rows)

    return {
        "source": "tdd",
        "total_rows": total,
        "unique_projects": len(project_counts),
        "smell_type_distribution": {
            k: {"count": v, "pct": round(v / total * 100, 2)}
            for k, v in sorted(smell_counts.items(), key=lambda x: -x[1])
        },
        "status_distribution": status_counts,
        "project_breakdown": dict(
            sorted(project_counts.items(), key=lambda x: -x[1])
        ),
        "chain_length_stats": _distribution_stats(lengths),
    }


def _distribution_stats(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "avg": 0, "p50": 0, "p95": 0, "min": 0, "max": 0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    p50_idx = int(n * 0.50)
    p95_idx = min(int(n * 0.95), n - 1)
    return {
        "count": n,
        "avg": round(statistics.mean(values), 2),
        "p50": sorted_vals[p50_idx],
        "p95": sorted_vals[p95_idx],
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze research datasets")
    parser.add_argument(
        "--source",
        choices=["rminer", "swe", "tdd", "all"],
        required=True,
        help="Which dataset to analyze",
    )
    parser.add_argument("--oracle-path", default=os.environ.get("RMINER_ORACLE_PATH"))
    parser.add_argument("--dataset-path", default=os.environ.get("SWE_REFACTOR_PATH"))
    parser.add_argument("--db-path", default=os.environ.get("TDD_DB_PATH"))
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSON report to this file (default: stdout)",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {}

    if args.source in ("rminer", "all"):
        if not args.oracle_path:
            parser.error("--oracle-path required for rminer source")
        report["rminer"] = analyze_rminer(args.oracle_path)

    if args.source in ("swe", "all"):
        if not args.dataset_path:
            parser.error("--dataset-path required for swe source")
        report["swe_refactor"] = analyze_swe(args.dataset_path)

    if args.source in ("tdd", "all"):
        if not args.db_path:
            parser.error("--db-path required for tdd source")
        report["tdd"] = analyze_tdd(args.db_path)

    json_out = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(json_out)
        print(f"Report written to {args.output}")
    else:
        print(json_out)


if __name__ == "__main__":
    main()
