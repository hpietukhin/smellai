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

from dotenv import load_dotenv

from agents.rminer_eval import create_rminer_eval_agent, invoke_agent
from workflows.common import (
    save_agent_graph,
    make_rminer_eval_sample,
    load_rminer_records,
    run_rminer_evaluation,
    _get_rminer_scorers,
)

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

    records, err = load_rminer_records(args)
    if records is None:
        return err

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
        sample = make_rminer_eval_sample(
            pair_id, before_code, file_path,
            refactoring_types, refactoring_descriptions, diff_hunks, sonar_issues,
        )
        return invoke_agent(agent, sample)

    print(f"Running evaluation on {len(records)} records...")
    return run_rminer_evaluation(records, predict_fn, _get_rminer_scorers(), args.tracking_uri)


if __name__ == "__main__":
    raise SystemExit(main())
