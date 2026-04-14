#!/usr/bin/env python3
"""MLflow GenAI evaluation workflow for the refactoring mapping agent.

Evaluates the agent's ability to correctly map refactorings to diff hunks.

Scorers:
- mapping_accuracy: fraction of predictions that overlap with ground truth hunks
- hunk_coverage: fraction of ground truth hunks covered by predictions
- prediction_completeness: whether agent made expected number of predictions

Usage:
    # Evaluate using manifest
    uv run workflows/rminer_eval_workflow.py --manifest rminer_data/manifest.json --limit 5

    # Use different model
    uv run workflows/rminer_eval_workflow.py --manifest rminer_data/manifest.json --model claude-sonnet-4-5-20250929

    # Draw agent graph
    uv run workflows/rminer_eval_workflow.py --draw-graph
"""

from __future__ import annotations

import argparse
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
from smellai_datasets import load_eval_samples, samples_to_mlflow_records, EvalSample
from workflows.common import setup_workflow_mlflow, save_agent_graph, print_eval_results

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate refactoring mapping agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--manifest", help="Path to manifest.json", required=True)
    parser.add_argument("--experiment", default="rminer-evaluation")
    parser.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking URI",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, help="Limit number of pairs")
    parser.add_argument("--run-name", help="MLflow run name")
    parser.add_argument("--draw-graph", action="store_true", help="Draw agent graph to PNG")
    args = parser.parse_args()

    if args.draw_graph:
        print("Generating agent graph...")
        agent = create_rminer_eval_agent(model_name=args.model)
        save_agent_graph(agent, "rminer_agent_graph.png")
        return 0

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
    print(f"Loaded {len(records)} RMiner EvalSamples from manifest")
    print(f"Model: {args.model}")

    agent = create_rminer_eval_agent(model_name=args.model)

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
        return invoke_agent(agent, sample)

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
