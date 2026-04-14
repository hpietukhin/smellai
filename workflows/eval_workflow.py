#!/usr/bin/env python3
"""Unified MLflow GenAI evaluation workflow.

Loads EvalSamples from raw dataset sources, converts to MLflow records,
and runs agent evaluation with source-appropriate scorers.

Usage:
    # SWE evaluation
    uv run workflows/eval_workflow.py --source swe --raw-path /path/to/pure_refactoring_data.json --limit 5

    # RMiner evaluation
    uv run workflows/eval_workflow.py --source rminer --manifest rminer_data/manifest.json --limit 10

    # Draw agent graph
    uv run workflows/eval_workflow.py --source swe --draw-graph
"""

from __future__ import annotations

import argparse
import sys

import mlflow
from dotenv import load_dotenv

from smellai_datasets import load_eval_samples, samples_to_mlflow_records
from workflows.common import (
    setup_workflow_mlflow,
    save_agent_graph,
    print_eval_results,
    make_rminer_eval_sample,
    make_swe_eval_sample,
    invoke_swe_agent,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Agent / scorer factories
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
# Predict functions — reconstruct EvalSample from MLflow inputs kwargs
# ---------------------------------------------------------------------------

def _make_rminer_predict_fn(agent):
    from agents.rminer_eval import invoke_agent as rminer_invoke

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
        return rminer_invoke(agent, sample)

    return predict_fn


def _make_swe_predict_fn(agent, args):
    from agents.swe_eval import invoke_agent as swe_invoke

    analytics_db = None
    if getattr(args, "enable_composite", False):
        from swe_refactor.persistence.database import AnalyticsDB
        analytics_db = AnalyticsDB(args.analytics_db)

    def predict_fn(
        project_name: str,
        commit_id: str,
        refactoring_type: str,
        file_path_before: str,
        file_path_after: str,
        class_before: str,
        source_before: str,
        jdk_version: int,
        compile_command: str,
    ) -> dict:
        sample = make_swe_eval_sample(
            project_name, commit_id, refactoring_type,
            file_path_before, file_path_after, class_before,
            source_before, jdk_version, compile_command,
        )
        return invoke_swe_agent(swe_invoke, agent, sample, args, analytics_db)

    return predict_fn


def _make_mini_swe_predict_fn(handle, args):
    from evals.ablation.mini_swe_agent import invoke_agent as mini_invoke

    def predict_fn(
        project_name: str,
        commit_id: str,
        refactoring_type: str,
        file_path_before: str,
        file_path_after: str,
        class_before: str,
        source_before: str,
        jdk_version: int,
        compile_command: str,
    ) -> dict:
        sample = make_swe_eval_sample(
            project_name, commit_id, refactoring_type,
            file_path_before, file_path_after, class_before,
            source_before, jdk_version, compile_command,
        )
        return mini_invoke(handle, sample, args.workspace)

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
    parser.add_argument("--raw-path", help="Path to raw dataset file (SWE JSON or directory)")
    parser.add_argument("--manifest", help="RMiner manifest.json path")
    parser.add_argument("--experiment", help="MLflow experiment name")
    parser.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking URI",
    )
    parser.add_argument("--model", help="LLM model name")
    parser.add_argument("--limit", type=int, help="Limit number of records per source")
    parser.add_argument(
        "--agent",
        choices=["swe", "swe-composite", "mini-swe"],
        default="swe",
        help="Agent scaffold (default: swe)",
    )
    parser.add_argument("--mini-step-limit", type=int, default=80, help="mini-swe-agent step limit")
    parser.add_argument("--mini-cost-limit", type=float, default=2.0, help="mini-swe-agent cost limit (USD)")
    parser.add_argument("--draw-graph", action="store_true", help="Draw agent graph to PNG")
    # SWE-specific options
    parser.add_argument("--workspace", default="/tmp/swe-eval-workspace")
    parser.add_argument("--enable-composite", action="store_true")
    parser.add_argument("--max-refactorings", type=int, default=5)
    parser.add_argument("--analytics-db", default="analytics.db")
    parser.add_argument("--sonar-url", default="http://localhost:9000")
    parser.add_argument("--sonar-cache-dir", default="./sonar_cache")
    args = parser.parse_args()

    # --agent swe-composite is sugar for --agent swe --enable-composite
    if args.agent == "swe-composite":
        args.enable_composite = True

    if not args.model:
        args.model = "gpt-4o-mini" if args.source == "rminer" else "claude-sonnet-4-5-20250929"
    if not args.experiment:
        args.experiment = f"{args.source}-{args.agent}-evaluation"

    if args.draw_graph:
        agent = _create_rminer_agent(args.model) if args.source == "rminer" \
            else _create_swe_agent(args.model, enable_composite=args.enable_composite)
        save_agent_graph(agent, f"{args.source}_agent_graph.png")
        return 0

    # Load EvalSamples
    from pathlib import Path
    swe_path = Path(args.raw_path) if args.raw_path and args.source == "swe" else None
    rminer_manifest = Path(args.manifest) if args.manifest and args.source == "rminer" else None

    samples = load_eval_samples(
        [args.source],
        swe_path=swe_path,
        rminer_manifest_path=rminer_manifest,
        limit=args.limit,
    )
    if not samples:
        print("No records to evaluate", file=sys.stderr)
        return 1

    records = samples_to_mlflow_records(samples)
    print(f"Loaded {len(records)} {args.source} EvalSamples")

    setup_workflow_mlflow(args.tracking_uri, args.experiment)

    if args.source == "rminer":
        agent = _create_rminer_agent(args.model)
        predict_fn = _make_rminer_predict_fn(agent)
        scorers = _get_rminer_scorers()
    elif args.agent == "mini-swe":
        from evals.ablation.mini_swe_agent import create_agent as _create_mini
        from evals.ablation.mini_swe_agent.scorers import (
            mini_cost_scorer,
            mini_step_count_scorer,
            mini_exit_status_scorer,
        )
        handle = _create_mini(
            args.model,
            step_limit=args.mini_step_limit,
            cost_limit=args.mini_cost_limit,
        )
        predict_fn = _make_mini_swe_predict_fn(handle, args)
        scorers = _get_swe_scorers() + [mini_cost_scorer, mini_step_count_scorer, mini_exit_status_scorer]
        print(f"Agent: mini-swe | step_limit={args.mini_step_limit} | cost_limit={args.mini_cost_limit}")
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
