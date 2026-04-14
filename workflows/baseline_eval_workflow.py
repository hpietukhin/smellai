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
from smellai_datasets import load_eval_samples, samples_to_mlflow_records, EvalSample
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

    samples = load_eval_samples(
        ["rminer"],
        rminer_manifest_path=manifest_path,
        limit=args.limit,
    )
    if not samples:
        print("No records to evaluate", file=sys.stderr)
        return 1

    records = samples_to_mlflow_records(samples)
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
        sample = EvalSample(
            source="rminer",
            sample_id=f"rminer:{pair_id}",
            inputs={
                "pair_id": pair_id,
                "before_code": before_code,
                "file_path": file_path,
                "refactoring_types": refactoring_types,
                "refactoring_descriptions": refactoring_descriptions,
                "diff_hunks": diff_hunks,
                "sonar_issues": sonar_issues or [],
            },
            expectations={},
            tags={},
        )
        return invoke_baseline_agent(sample, model_name=args.model)

    results = mlflow.genai.evaluate(
        data=records,
        predict_fn=predict_fn,
        scorers=[mapping_accuracy, hunk_coverage, prediction_completeness],
    )

    print_eval_results(results, args.tracking_uri)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
