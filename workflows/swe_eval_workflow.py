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
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

from agents.swe_eval import create_swe_eval_agent, invoke_agent
from swe_refactor.dataset import load_swe_refactor_dataset, RefactoringRecord
from workflows.common import setup_workflow_mlflow, save_agent_graph, print_eval_results

load_dotenv()


@scorer
def compile_success_scorer(outputs: dict) -> Feedback:
    """Score: 1.0 if compilation succeeded, 0.0 otherwise."""
    success = outputs.get("compile_success", False)
    return Feedback(
        value=1.0 if success else 0.0,
        rationale="Compilation succeeded" if success else "Compilation failed"
    )


@scorer
def test_pass_scorer(outputs: dict) -> Feedback:
    """Score: 1.0 if tests passed, 0.0 otherwise (NA if no compilation)."""
    compile_ok = outputs.get("compile_success", False)
    test_ok = outputs.get("test_success", False)

    if not compile_ok:
        return Feedback(value=0.0, rationale="Compilation failed, tests not run")

    return Feedback(
        value=1.0 if test_ok else 0.0,
        rationale="Tests passed" if test_ok else "Tests failed"
    )


@scorer
def overall_success_scorer(outputs: dict) -> Feedback:
    """Score: 1.0 if both compile and tests pass, 0.0 otherwise."""
    compile_ok = outputs.get("compile_success", False)
    test_ok = outputs.get("test_success", False)
    success = compile_ok and test_ok

    if success:
        rationale = "Both compilation and tests passed"
    elif not compile_ok:
        rationale = "Compilation failed"
    else:
        rationale = "Compilation passed but tests failed"

    return Feedback(value=1.0 if success else 0.0, rationale=rationale)


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
    parser.add_argument(
        "--enable-composite",
        action="store_true",
        help="Enable composite refactoring mode (A1-A6 loop)",
    )
    parser.add_argument(
        "--max-refactorings",
        type=int,
        default=5,
        help="Max refactoring iterations (N-action limit)",
    )
    parser.add_argument(
        "--analytics-db",
        type=str,
        default="analytics.db",
        help="Analytics database path",
    )
    parser.add_argument(
        "--sonar-url",
        type=str,
        default="http://localhost:9000",
        help="SonarQube server URL",
    )
    parser.add_argument(
        "--sonar-cache-dir",
        type=str,
        default="./sonar_cache",
        help="SonarQube cache directory",
    )
    args = parser.parse_args()

    if args.draw_graph:
        print("Generating agent graph...")
        agent = create_swe_eval_agent(
            model_name=args.model, enable_composite=args.enable_composite
        )
        save_agent_graph(agent, "swe_eval_agent_graph.png")
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

    setup_workflow_mlflow(args.tracking_uri, args.experiment)

    # Initialize analytics DB for composite mode
    analytics_db = None
    if args.enable_composite:
        from swe_refactor.persistence.database import AnalyticsDB

        analytics_db = AnalyticsDB(args.analytics_db)
        print(f"Analytics DB: {args.analytics_db}")

    print(f"Model: {args.model}")
    print(f"Mode: {'Composite' if args.enable_composite else 'Basic'}")
    print(f"Records: {len(records)}")
    print(f"Workspace: {args.workspace}")

    print("Creating agent...")
    agent = create_swe_eval_agent(
        model_name=args.model, enable_composite=args.enable_composite
    )

    genai_records = [
        {
            "inputs": {
                "record_dict": r.model_dump(),
            },
            "outputs": {},
            "tags": {
                "project": r.projectName,
                "commit": r.commitId[:8],
                "type": r.type,
            },
        }
        for r in records
    ]

    def predict_fn(record_dict: dict) -> dict:
        """Prediction function for MLflow evaluation."""
        record = RefactoringRecord(**record_dict)
        return invoke_agent(
            agent,
            record,
            args.workspace,
            analytics_db=analytics_db,
            max_refactorings=args.max_refactorings,
            sonar_url=args.sonar_url,
            sonar_cache_dir=args.sonar_cache_dir,
        )

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

    print_eval_results(results, args.tracking_uri)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
