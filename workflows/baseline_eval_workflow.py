#!/usr/bin/env python3
"""Baseline evaluation workflow — single LLM call, no framework.

Uses the same scorers and data as rminer_eval_workflow for direct comparison.

Usage:
    uv run workflows/baseline_eval_workflow.py --manifest ~/uni/masterThesis/datasets/rminer_data/manifest.json --limit 5
    uv run workflows/baseline_eval_workflow.py --manifest ~/uni/masterThesis/datasets/rminer_data/manifest.json --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from agents.baseline import invoke_baseline_agent
from agents.rminer_eval import mapping_accuracy, hunk_coverage, prediction_completeness
from rminer.create_rminer_dataset import build_genai_records
from workflows.common import setup_workflow_mlflow, print_eval_results

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline evaluation (single LLM call)")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--experiment", default="baseline-evaluation")
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, help="Limit number of pairs")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    setup_workflow_mlflow(args.tracking_uri, args.experiment)

    records = build_genai_records(manifest_path, limit=args.limit)
    if not records:
        print("No records to evaluate", file=sys.stderr)
        return 1

    print(f"Model: {args.model}")
    print(f"Records: {len(records)}")

    def predict_fn(pair_id: str, sonar_issues: list[dict] | None = None) -> dict:
        return invoke_baseline_agent(pair_id, str(manifest_path), model_name=args.model)

    results = mlflow.genai.evaluate(
        data=records,
        predict_fn=predict_fn,
        scorers=[mapping_accuracy, hunk_coverage, prediction_completeness],
    )

    print_eval_results(results, args.tracking_uri)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
