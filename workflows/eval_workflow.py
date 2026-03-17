#!/usr/bin/env python3
"""Unified MLflow GenAI evaluation workflow.

Loads data from preprocessed HF datasets (preferred) or raw sources,
converts to MLflow GenAI format, and runs agent evaluation with
source-appropriate scorers.

Usage:
    # Evaluate from preprocessed HF dataset
    uv run workflows/eval_workflow.py --source swe --hf-dataset-path data/processed/swe --limit 5

    # Evaluate from raw data (fallback)
    uv run workflows/eval_workflow.py --source swe --raw-data-path /tmp/SWE-Refactor/pure_refactoring_data.json

    # RMiner evaluation
    uv run workflows/eval_workflow.py --source rminer --hf-dataset-path data/processed/rminer --limit 10

    # Draw agent graph
    uv run workflows/eval_workflow.py --source swe --draw-graph
"""

from __future__ import annotations

import argparse
import sys

import mlflow
from dotenv import load_dotenv

from hf_datasets.mlflow_bridge import hf_to_genai_records, load_for_evaluation
from swe_refactor.dataset import RefactoringRecord
from workflows.common import setup_workflow_mlflow, save_agent_graph, print_eval_results

load_dotenv()

# ---------------------------------------------------------------------------
# Source-specific agent/scorer factories
# ---------------------------------------------------------------------------

def _create_rminer_agent(model: str):
    from agents.rminer_eval import create_rminer_eval_agent
    return create_rminer_eval_agent(model_name=model)


def _create_swe_agent(model: str, *, enable_composite: bool = False):
    from agents.swe_eval import create_swe_eval_agent
    return create_swe_eval_agent(model_name=model, enable_composite=enable_composite)


def _get_rminer_scorers():
    from agents.rminer_eval import mapping_accuracy, hunk_coverage, prediction_completeness
    return [mapping_accuracy, hunk_coverage, prediction_completeness]


def _get_swe_scorers():
    from workflows.swe_eval_workflow import (
        compile_success_scorer,
        test_pass_scorer,
        overall_success_scorer,
    )
    return [compile_success_scorer, test_pass_scorer, overall_success_scorer]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_records(args) -> list[dict]:
    """Load evaluation records from HF dataset or raw fallback."""
    if args.hf_dataset_path:
        records = load_for_evaluation(args.hf_dataset_path, args.source)
        print(f"Loaded {len(records)} records from HF dataset: {args.hf_dataset_path}")
    elif args.source == "rminer" and args.raw_data_path:
        from hf_datasets.converter import rminer_to_hf
        ds = rminer_to_hf(args.raw_data_path, limit=args.limit)
        records = hf_to_genai_records(ds, "rminer")
        print(f"Loaded {len(records)} records from raw RMiner data")
    elif args.source == "swe" and args.raw_data_path:
        from hf_datasets.converter import swe_refactor_to_hf
        ds = swe_refactor_to_hf(args.raw_data_path, limit=args.limit)
        records = hf_to_genai_records(ds, "swe")
        print(f"Loaded {len(records)} records from raw SWE data")
    else:
        print(
            "Either --hf-dataset-path or --raw-data-path is required",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.limit and len(records) > args.limit:
        records = records[: args.limit]

    return records


# ---------------------------------------------------------------------------
# Predict functions
# ---------------------------------------------------------------------------

def _make_rminer_predict_fn(agent, manifest_path: str | None):
    from agents.rminer_eval import invoke_agent as rminer_invoke

    def predict_fn(pair_id: str, sonar_issues: list[dict] | None = None) -> dict:
        return rminer_invoke(agent, pair_id, manifest_path or "", sonar_issues)

    return predict_fn


def _make_swe_predict_fn(agent, args):
    from agents.swe_eval import invoke_agent as swe_invoke

    analytics_db = None
    if args.enable_composite:
        from swe_refactor.persistence.database import AnalyticsDB
        analytics_db = AnalyticsDB(args.analytics_db)

    def predict_fn(
        project_name: str,
        commit_id: str,
        refactoring_type: str,
        source_before: str,
        class_before: str,
        file_path_before: str,
        file_path_after: str,
        jdk_version: int,
        compile_command: str,
    ) -> dict:
        """Reconstruct RefactoringRecord from flat HF row inputs."""
        record = RefactoringRecord(
            projectName=project_name,
            commitId=commit_id,
            type=refactoring_type,
            filePathBefore=file_path_before,
            filePathAfter=file_path_after,
            sourceCodeBeforeForWhole=class_before,
            sourceCodeAfterForWhole="",  # ground truth not given to agent
            compileJDK=jdk_version,
            compileCommand=compile_command,
            compileResultBefore=True,
            compileResultCurrent=True,
        )
        return swe_invoke(
            agent,
            record,
            args.workspace,
            analytics_db=analytics_db,
            max_refactorings=args.max_refactorings,
            sonar_url=args.sonar_url,
            sonar_cache_dir=args.sonar_cache_dir,
        )

    return predict_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified evaluation workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        choices=["rminer", "swe"],
        required=True,
        help="Dataset source",
    )
    parser.add_argument("--hf-dataset-path", help="Path to preprocessed HF dataset on disk")
    parser.add_argument("--raw-data-path", help="Path to raw data (fallback)")
    parser.add_argument("--manifest", help="RMiner manifest path (for predict_fn)")
    parser.add_argument("--experiment", help="MLflow experiment name")
    parser.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking URI",
    )
    parser.add_argument("--model", help="LLM model name")
    parser.add_argument("--limit", type=int, help="Limit number of records")
    parser.add_argument(
        "--draw-graph",
        action="store_true",
        help="Draw agent graph to PNG",
    )
    # SWE-specific options
    parser.add_argument(
        "--workspace",
        default="/tmp/swe-eval-workspace",
        help="Workspace directory for cloned repos",
    )
    parser.add_argument(
        "--enable-composite",
        action="store_true",
        help="Enable composite refactoring mode",
    )
    parser.add_argument("--max-refactorings", type=int, default=5)
    parser.add_argument("--analytics-db", default="analytics.db")
    parser.add_argument("--sonar-url", default="http://localhost:9000")
    parser.add_argument("--sonar-cache-dir", default="./sonar_cache")
    args = parser.parse_args()

    # Defaults per source
    if not args.model:
        args.model = "gpt-4o-mini" if args.source == "rminer" else "claude-sonnet-4-5-20250929"
    if not args.experiment:
        args.experiment = f"{args.source}-evaluation"

    if args.draw_graph:
        print("Generating agent graph...")
        if args.source == "rminer":
            agent = _create_rminer_agent(args.model)
        else:
            agent = _create_swe_agent(args.model, enable_composite=args.enable_composite)
        save_agent_graph(agent, f"{args.source}_agent_graph.png")
        return 0

    records = _load_records(args)
    if not records:
        print("No records to evaluate", file=sys.stderr)
        return 1

    setup_workflow_mlflow(args.tracking_uri, args.experiment)

    print(f"Source: {args.source}")
    print(f"Model: {args.model}")
    print(f"Records: {len(records)}")

    if args.source == "rminer":
        agent = _create_rminer_agent(args.model)
        predict_fn = _make_rminer_predict_fn(agent, args.manifest)
        scorers = _get_rminer_scorers()
    else:
        agent = _create_swe_agent(args.model, enable_composite=args.enable_composite)
        predict_fn = _make_swe_predict_fn(agent, args)
        scorers = _get_swe_scorers()

    print(f"Running evaluation on {len(records)} records...")

    results = mlflow.genai.evaluate(
        data=records,
        predict_fn=predict_fn,
        scorers=scorers,
    )

    print_eval_results(results, args.tracking_uri)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
