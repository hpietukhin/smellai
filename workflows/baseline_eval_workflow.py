#!/usr/bin/env python3
"""Baseline evaluation workflow — single LLM call, no framework.

Uses the same scorers and data as rminer_eval_workflow for direct comparison.

Usage:
    uv run workflows/baseline_eval_workflow.py --manifest ~/uni/masterThesis/datasets/rminer_data/manifest.json --limit 5
    uv run workflows/baseline_eval_workflow.py --manifest ~/uni/masterThesis/datasets/rminer_data/manifest.json --model gpt-4o-mini
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from agents.baseline import invoke_baseline_agent
from workflows.common import (
    make_rminer_eval_sample,
    load_rminer_records,
    run_rminer_evaluation,
    _get_rminer_scorers,
)

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline evaluation (single LLM call)")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--experiment", default="baseline-evaluation")
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, help="Limit number of pairs")
    args = parser.parse_args()

    records, err = load_rminer_records(args)
    if records is None:
        return err

    print(f"Model: {args.model}")
    print(f"Records: {len(records)}")

    def predict_fn(
        pair_id: str,
        before_code: str,
        file_path: str,
        refactoring_types: list,
        refactoring_descriptions: list,
        diff_hunks: list,
        sonar_issues: list | None = None,
    ) -> dict:
        sample = make_rminer_eval_sample(
            pair_id, before_code, file_path,
            refactoring_types, refactoring_descriptions, diff_hunks, sonar_issues,
        )
        return invoke_baseline_agent(sample, model_name=args.model)

    return run_rminer_evaluation(records, predict_fn, _get_rminer_scorers(), args.tracking_uri)


if __name__ == "__main__":
    raise SystemExit(main())
