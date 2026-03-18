#!/usr/bin/env python3
"""MLflow GenAI evaluation workflow for the refactoring mapping agent.

Evaluates the agent's ability to correctly map refactorings to diff hunks.

Scorers:
- mapping_accuracy: fraction of predictions that overlap with ground truth hunks
- hunk_coverage: fraction of ground truth hunks covered by predictions
- prediction_completeness: whether agent made expected number of predictions

Usage:
    # Evaluate using inline data (from manifest)
uv run workflows/rminer_eval_workflow.py --manifest rminer_data/manifest.json --limit 5

    # Evaluate using saved dataset
    uv run workflows/rminer_eval_workflow.py --dataset-name rminer-eval-dataset

    # Evaluate using dataset ID
    uv run workflows/rminer_eval_workflow.py --dataset-id <dataset-id>

    # Use different model
    uv run workflows/rminer_eval_workflow.py --manifest rminer_data/manifest.json --model claude-sonnet-4-5-20250929

    # Draw agent graph
    uv run workflows/rminer_eval_workflow.py --draw-graph
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from agents.rminer_eval import (
    create_rminer_eval_agent,
    invoke_agent,
    mapping_accuracy,
    hunk_coverage,
    prediction_completeness,
)
from rminer.create_rminer_dataset import build_genai_records
from workflows.common import setup_workflow_mlflow, save_agent_graph, print_eval_results

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate refactoring mapping agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--manifest", help="Path to manifest.json")
    parser.add_argument("--dataset-name", help="MLflow dataset name to use")
    parser.add_argument(
        "--dataset-id", help="MLflow dataset ID to use (alternative to --dataset-name)"
    )
    parser.add_argument("--experiment", default="rminer-evaluation")
    parser.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking URI (server will auto-start if needed)",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, help="Limit number of pairs")
    parser.add_argument("--run-name", help="MLflow run name")
    parser.add_argument(
        "--draw-graph",
        action="store_true",
        help="Draw the agent graph to a PNG file",
    )
    args = parser.parse_args()

    if args.draw_graph:
        print("Generating agent graph...")
        agent = create_rminer_eval_agent(model_name=args.model)
        save_agent_graph(agent, "rminer_agent_graph.png")
        return 0

    if not args.manifest and not args.dataset_name and not args.dataset_id:
        print(
            "Either --manifest, --dataset-name, or --dataset-id is required",
            file=sys.stderr,
        )
        return 1

    setup_workflow_mlflow(args.tracking_uri, args.experiment)

    print(f"Model: {args.model}")

    # Load data
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"Manifest not found: {manifest_path}", file=sys.stderr)
            return 1
        records = build_genai_records(manifest_path, limit=args.limit)
        print(f"Loaded {len(records)} records from manifest")
    elif args.dataset_id:
        from mlflow.genai.datasets import get_dataset

        dataset = get_dataset(dataset_id=args.dataset_id)
        records = dataset.records
        print(f"Loaded {len(records)} records from dataset ID: {args.dataset_id}")
        manifest_path = Path(os.environ.get("RMINER_MANIFEST_PATH", "rminer_data/manifest.json"))
    else:
        from mlflow.genai.datasets import search_datasets

        all_datasets = search_datasets()
        matching = [ds for ds in all_datasets if ds.name == args.dataset_name]
        if not matching:
            print(f"Dataset not found: {args.dataset_name}", file=sys.stderr)
            print("\nAvailable datasets:", file=sys.stderr)
            for ds in all_datasets:
                print(f"  - {ds.name} (ID: {ds.dataset_id})", file=sys.stderr)
            return 1
        dataset = matching[0]
        records = dataset.records
        print(
            f"Loaded {len(records)} records from dataset {args.dataset_name} (ID: {dataset.dataset_id})"
        )
        manifest_path = Path(os.environ.get("RMINER_MANIFEST_PATH", "rminer_data/manifest.json"))

    if not records:
        print("No records to evaluate", file=sys.stderr)
        return 1

    print("Creating agent...")
    agent = create_rminer_eval_agent(model_name=args.model)

    def predict_fn(pair_id: str, sonar_issues: list[dict] | None = None) -> dict:
        return invoke_agent(agent, pair_id, str(manifest_path), sonar_issues)

    print(f"Running evaluation on {len(records)} records...")

    results = mlflow.genai.evaluate(
        data=records,
        predict_fn=predict_fn,
        scorers=[mapping_accuracy, hunk_coverage, prediction_completeness],
    )

    print_eval_results(results, args.tracking_uri)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
