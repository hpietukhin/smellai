#!/usr/bin/env python3
"""MLflow GenAI evaluation workflow for SWE-Refactor agent.

Evaluates the agent's ability to generate correct refactorings.

Scorers:
- compile_success_rate: fraction of generated code that compiles
- test_pass_rate: fraction of compilable code that passes tests
- overall_success_rate: fraction that both compiles and passes tests

Usage:
    # Evaluate single commit
    uv run workflows/swe_eval_workflow.py --commit 65655da4 --project checkstyle

    # Evaluate using dataset
    uv run workflows/swe_eval_workflow.py --dataset /tmp/SWE-Refactor/pure_refactoring_data.json --limit 10

    # Use different model
    uv run workflows/swe_eval_workflow.py --dataset <path> --model gpt-4o

    # Draw agent graph
    uv run workflows/swe_eval_workflow.py --draw-graph
"""

import argparse
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from agents.swe_eval import create_swe_eval_agent, invoke_agent
from swe_refactor.dataset import load_swe_refactor_dataset, RefactoringRecord
from mlflow_utils import setup_mlflow_tracking

load_dotenv()


def compile_success_scorer(outputs: dict, inputs: dict) -> float:
    """Score: 1.0 if compilation succeeded, 0.0 otherwise."""
    return 1.0 if outputs.get("compile_success", False) else 0.0


def test_pass_scorer(outputs: dict, inputs: dict) -> float:
    """Score: 1.0 if tests passed, 0.0 otherwise (NA if no compilation)."""
    if not outputs.get("compile_success", False):
        return 0.0
    return 1.0 if outputs.get("test_success", False) else 0.0


def overall_success_scorer(outputs: dict, inputs: dict) -> float:
    """Score: 1.0 if both compile and tests pass, 0.0 otherwise."""
    compile_ok = outputs.get("compile_success", False)
    test_ok = outputs.get("test_success", False)
    return 1.0 if (compile_ok and test_ok) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate SWE-Refactor agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        help="Path to pure_refactoring_data.json",
        default="/tmp/SWE-Refactor/pure_refactoring_data.json",
    )
    parser.add_argument("--commit", help="Specific commit to evaluate")
    parser.add_argument("--project", help="Project name (with --commit)")
    parser.add_argument("--experiment", default="swe-refactor-evaluation")
    parser.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking URI",
    )
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929")
    parser.add_argument("--limit", type=int, help="Limit number of records")
    parser.add_argument(
        "--workspace",
        default="/tmp/swe-eval-workspace",
        help="Workspace directory for cloned repos",
    )
    parser.add_argument(
        "--draw-graph",
        action="store_true",
        help="Draw agent graph to PNG",
    )
    args = parser.parse_args()

    if args.draw_graph:
        print("Generating agent graph...")
        agent = create_swe_eval_agent(model_name=args.model)
        try:
            png_bytes = agent.get_graph().draw_mermaid_png()
            output_path = "swe_eval_agent_graph.png"
            with open(output_path, "wb") as f:
                f.write(png_bytes)
            print(f"Graph saved to {output_path}")
        except Exception as e:
            print(f"Failed to draw graph: {e}")
        return 0

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    records = load_swe_refactor_dataset(dataset_path)

    if args.commit:
        if not args.project:
            print("--project required with --commit", file=sys.stderr)
            return 1
        records = [
            r
            for r in records
            if r.commitId.startswith(args.commit) and r.projectName == args.project
        ]
        print(f"Filtered to {len(records)} records from commit {args.commit}")

    if args.limit:
        records = records[: args.limit]

    if not records:
        print("No records to evaluate", file=sys.stderr)
        return 1

    setup_mlflow_tracking(
        tracking_uri=args.tracking_uri,
        backend_uri="sqlite:///mlflow.db",
        experiment_name=args.experiment,
        auto_start_server=True,
    )

    print(f"Model: {args.model}")
    print(f"Records: {len(records)}")
    print(f"Workspace: {args.workspace}")

    print("Creating agent...")
    agent = create_swe_eval_agent(model_name=args.model)

    genai_records = [
        {
            "inputs": {
                "project_name": r.projectName,
                "commit_id": r.commitId,
                "type": r.type,
            },
            "outputs": {},
            "metadata": {"record": r.model_dump()},
        }
        for r in records
    ]

    def predict_fn(
        project_name: str,
        commit_id: str,
        type: str,
        **metadata,
    ) -> dict:
        """Prediction function for MLflow evaluation."""
        record_dict = metadata.get("record", {})
        record = RefactoringRecord(**record_dict)
        return invoke_agent(agent, record, args.workspace)

    print(f"Running evaluation on {len(genai_records)} records...")

    results = mlflow.genai.evaluate(
        data=genai_records,
        predict_fn=predict_fn,
        scorers=[
            compile_success_scorer,
            test_pass_scorer,
            overall_success_scorer,
        ],
    )

    run_id = results.run_id

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for metric_name, metric_value in results.metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}: {metric_value}")

    print("=" * 60)
    print(f"MLflow run ID: {run_id}")

    if run_id != "N/A" and args.tracking_uri.startswith("http://"):
        exp_id = getattr(results, "experiment_id", "?")
        print(f"View results: {args.tracking_uri}/#/experiments/{exp_id}/runs/{run_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
