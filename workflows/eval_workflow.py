#!/usr/bin/env python3
"""Unified MLflow GenAI evaluation workflow.

Loads EvalSamples from raw dataset sources, converts to MLflow records,
and runs agent evaluation with source-appropriate scorers.

Usage:
    uv run workflows/eval_workflow.py --source swe --raw-path /path/to/data.json --limit 5
    uv run workflows/eval_workflow.py --source rminer --manifest path/to/manifest.json --limit 10
    uv run workflows/eval_workflow.py --source swe --draw-graph
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import mlflow
from dotenv import load_dotenv
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

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
    from agents.rminer_eval import (
        mapping_accuracy,
        hunk_coverage,
        prediction_completeness,
    )

    return [mapping_accuracy, hunk_coverage, prediction_completeness]


@scorer
def _scorer(outputs: dict, key: str, label: str) -> Feedback:
    """Generic scorer for boolean output keys."""
    value = outputs.get(key, False)
    return Feedback(
        value=1.0 if value else 0.0, rationale=f"{label}: {'yes' if value else 'no'}"
    )


def _compile_scorer(outputs: dict) -> Feedback:
    return _scorer(outputs, "compile_success", "Compilation")


def _test_scorer(outputs: dict) -> Feedback:
    compile_ok = outputs.get("compile_success", False)
    if not compile_ok:
        return Feedback(value=0.0, rationale="Compilation failed, tests not run")
    return _scorer(outputs, "test_success", "Tests")


def _overall_scorer(outputs: dict) -> Feedback:
    compile_ok = outputs.get("compile_success", False)
    test_ok = outputs.get("test_success", False)
    success = compile_ok and test_ok
    rationale = (
        "Both compilation and tests passed"
        if success
        else (
            "Compilation failed"
            if not compile_ok
            else "Compilation passed but tests failed"
        )
    )
    return Feedback(value=1.0 if success else 0.0, rationale=rationale)


SCOURERS = {
    "compile": _compile_scorer,
    "test": _test_scorer,
    "overall": _overall_scorer,
}

def _get_swe_scorers():
    return list(SCOURERS.values())


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
            pair_id,
            before_code,
            file_path,
            refactoring_types,
            refactoring_descriptions,
            diff_hunks,
            sonar_issues,
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
            project_name,
            commit_id,
            refactoring_type,
            file_path_before,
            file_path_after,
            class_before,
            source_before,
            jdk_version,
            compile_command,
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
            project_name,
            commit_id,
            refactoring_type,
            file_path_before,
            file_path_after,
            class_before,
            source_before,
            jdk_version,
            compile_command,
        )
        return mini_invoke(handle, sample, args.workspace)

    return predict_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    source: str,
    raw_path: str | None = None,
    manifest: str | None = None,
    run_name: str | None = None,
    experiment: str | None = None,
    tracking_uri: str = "http://localhost:5000",
    model: str | None = None,
    limit: int | None = None,
    agent: str = "swe",
    mini_step_limit: int = 80,
    mini_cost_limit: float = 2.0,
    draw_graph: bool = False,
    workspace: str = "/tmp/swe-eval-workspace",
    enable_composite: bool = False,
    max_refactorings: int = 5,
    analytics_db: str = "analytics.db",
    sonar_url: str = "http://localhost:9000",
    sonar_cache_dir: str = "./sonar_cache",
    with_sonar: bool = False,
) -> int:
    if source not in {"rminer", "swe"}:
        print("source must be one of: rminer, swe", file=sys.stderr)
        return 1
    if agent not in {"swe", "swe-composite", "mini-swe"}:
        print("agent must be one of: swe, swe-composite, mini-swe", file=sys.stderr)
        return 1

    args = SimpleNamespace(
        source=source,
        raw_path=raw_path,
        manifest=manifest,
        run_name=run_name,
        experiment=experiment,
        tracking_uri=tracking_uri,
        model=model,
        limit=limit,
        agent=agent,
        mini_step_limit=mini_step_limit,
        mini_cost_limit=mini_cost_limit,
        draw_graph=draw_graph,
        workspace=workspace,
        enable_composite=enable_composite,
        max_refactorings=max_refactorings,
        analytics_db=analytics_db,
        sonar_url=sonar_url,
        sonar_cache_dir=sonar_cache_dir,
        with_sonar=with_sonar,
    )

    # --agent swe-composite is sugar for --agent swe --enable-composite
    if args.agent == "swe-composite":
        args.enable_composite = True

    if not args.model:
        args.model = (
            "gpt-4o-mini" if args.source == "rminer" else "claude-sonnet-4-5-20250929"
        )
    if not args.experiment:
        args.experiment = f"{args.source}-{args.agent}-evaluation"

    if args.run_name:
        print("Note: --run-name is accepted for compatibility but currently ignored")

    if args.source == "swe" and not args.draw_graph and not args.raw_path:
        print("--raw-path is required for --source swe", file=sys.stderr)
        return 1

    if args.source == "rminer" and not args.draw_graph and not args.manifest:
        print("--manifest is required for --source rminer", file=sys.stderr)
        return 1

    if args.draw_graph:
        agent = (
            _create_rminer_agent(args.model)
            if args.source == "rminer"
            else _create_swe_agent(args.model, enable_composite=args.enable_composite)
        )
        save_agent_graph(agent, f"{args.source}_agent_graph.png")
        return 0

    # Load EvalSamples
    from pathlib import Path

    swe_path = Path(args.raw_path) if args.raw_path and args.source == "swe" else None
    rminer_manifest = (
        Path(args.manifest) if args.manifest and args.source == "rminer" else None
    )

    samples = load_eval_samples(
        [args.source],
        swe_path=swe_path,
        rminer_manifest_path=rminer_manifest,
        limit=args.limit,
    )
    if not samples:
        print("No records to evaluate", file=sys.stderr)
        return 1

    if args.source == "swe" and args.with_sonar:
        from smellai_datasets import enrich_swe_with_sonar
        import os

        sonar_token = os.environ.get("SONAR_TOKEN", "")
        if not sonar_token:
            print(
                "WARNING: SONAR_TOKEN not set — skipping sonar enrichment",
                file=sys.stderr,
            )
        else:
            print(f"Enriching {len(samples)} samples with SonarQube scan...")
            samples = enrich_swe_with_sonar(
                samples,
                sonar_url=args.sonar_url,
                sonar_token=sonar_token,
                cache_dir=args.sonar_cache_dir,
            )
            print("SonarQube enrichment complete")

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
        scorers = _get_swe_scorers() + [
            mini_cost_scorer,
            mini_step_count_scorer,
            mini_exit_status_scorer,
        ]
        print(
            f"Agent: mini-swe | step_limit={args.mini_step_limit} | cost_limit={args.mini_cost_limit}"
        )
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
    import fire

    raise SystemExit(fire.Fire(main))
