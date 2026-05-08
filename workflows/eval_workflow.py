#!/usr/bin/env python3
"""DEPRECATED unified MLflow GenAI evaluation workflow for SWE/RMiner.

Redirect to Composite Refactorings 2020 flow instead:
- dataset.neo4j_graph.DatasetGraph.composite_refactoring(...)
- workflows/planner_eval_workflow.py

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
import mlflow.langchain
from dotenv import load_dotenv

from agents.swe_eval.scorers import get_swe_scorers
from smellai_datasets import load_eval_samples, samples_to_mlflow_records
from workflows.common import (
    setup_workflow_mlflow,
    save_agent_graph,
    print_eval_results,
)
from workflows.predict_adapters import (
    make_mini_swe_predict_fn,
    make_rminer_predict_fn,
    make_swe_predict_fn,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------


def _create_rminer_agent(model: str):
    from agents.rminer_eval import create_rminer_eval_agent

    return create_rminer_eval_agent(model_name=model)


def _create_swe_agent(model: str, *, enable_composite: bool = False):
    from agents.swe_eval import create_swe_eval_agent

    return create_swe_eval_agent(model_name=model, enable_composite=enable_composite)


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
    mlflow.langchain.autolog(silent=True)

    if args.source == "rminer":
        from agents.rminer_eval import (
            hunk_coverage,
            mapping_accuracy,
            prediction_completeness,
        )

        agent = _create_rminer_agent(args.model)
        predict_fn = make_rminer_predict_fn(agent)
        scorers = [mapping_accuracy, hunk_coverage, prediction_completeness]
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
        predict_fn = make_mini_swe_predict_fn(handle, args)
        scorers = get_swe_scorers() + [
            mini_cost_scorer,
            mini_step_count_scorer,
            mini_exit_status_scorer,
        ]
        print(
            f"Agent: mini-swe | step_limit={args.mini_step_limit} | cost_limit={args.mini_cost_limit}"
        )
    else:
        agent = _create_swe_agent(args.model, enable_composite=args.enable_composite)
        predict_fn = make_swe_predict_fn(agent, args)
        scorers = get_swe_scorers()

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
