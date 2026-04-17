#!/usr/bin/env python3
"""Baseline evaluation workflow — single LLM call, no framework.

Uses the same scorers and data as rminer_eval_workflow for direct comparison.

Usage:
    uv run workflows/baseline_eval_workflow.py --manifest ~/uni/masterThesis/datasets/rminer_data/manifest.json --limit 5
    uv run workflows/baseline_eval_workflow.py --manifest ~/uni/masterThesis/datasets/rminer_data/manifest.json --model gpt-4o-mini
"""

from __future__ import annotations

from types import SimpleNamespace

from dotenv import load_dotenv

from agents.baseline import invoke_baseline_agent
from workflows.common import (
    make_rminer_eval_sample,
    load_rminer_records,
    run_rminer_evaluation,
    _get_rminer_scorers,
)

load_dotenv()


def main(
    manifest: str,
    experiment: str = "baseline-evaluation",
    tracking_uri: str = "http://localhost:5000",
    model: str = "gpt-4o-mini",
    limit: int | None = None,
) -> int:
    args = SimpleNamespace(
        manifest=manifest,
        experiment=experiment,
        tracking_uri=tracking_uri,
        model=model,
        limit=limit,
    )

    records, err = load_rminer_records(args)
    if records is None:
        return err

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
        sample = make_rminer_eval_sample(
            pair_id, before_code, file_path,
            refactoring_types, refactoring_descriptions, diff_hunks, sonar_issues,
        )
        return invoke_baseline_agent(sample, model_name=args.model)

    return run_rminer_evaluation(records, predict_fn, _get_rminer_scorers(), args.tracking_uri)


if __name__ == "__main__":
    import fire

    raise SystemExit(fire.Fire(main))
