#!/usr/bin/env python3
"""Example: Using the RMiner evaluation agent.

This script demonstrates how to use the rminer_eval agent to map
refactorings to diff hunks in code changes.
"""

from pathlib import Path

from agents.rminer_eval import (
    create_rminer_eval_agent,
    invoke_agent,
    mapping_accuracy,
    hunk_coverage,
    prediction_completeness,
)
from rminer.create_rminer_dataset import build_genai_records


def basic_example():
    """Basic usage example."""
    print("=== Basic Example ===\n")

    agent = create_rminer_eval_agent(model_name="gpt-4o-mini")

    result = invoke_agent(
        agent=agent,
        pair_id="example-pair-id",
        manifest_path="rminer_data/manifest.json",
    )

    print(f"Pair ID: {result['pair_id']}")
    print(f"File: {result['filename']}")
    print(f"Predictions: {len(result['predictions'])}")

    for pred in result["predictions"]:
        print(f"\n  Refactoring {pred['refactoring_index']}: {pred['refactoring_type']}")
        print(f"  Maps to hunk {pred['predicted_hunk_index']}")
        print(f"  Lines: {pred['line_start']}-{pred['line_end']}")
        print(f"  Reasoning: {pred['reasoning']}")


def evaluation_example():
    """Example running full evaluation with MLflow."""
    print("\n=== Evaluation Example ===\n")

    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("rminer-evaluation-example")

    manifest_path = Path("rminer_data/manifest.json")
    records = build_genai_records(manifest_path, limit=3)

    print(f"Loaded {len(records)} records")

    agent = create_rminer_eval_agent(model_name="gpt-4o-mini")

    def predict_fn(pair_id: str, sonar_issues: list[dict] | None = None) -> dict:
        return invoke_agent(agent, pair_id, str(manifest_path), sonar_issues)

    results = mlflow.genai.evaluate(
        data=records,
        predict_fn=predict_fn,
        scorers=[mapping_accuracy, hunk_coverage, prediction_completeness],
    )

    print("\nResults:")
    for metric_name, metric_value in results.metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")


def anthropic_example():
    """Example using Anthropic Claude via LiteLLM."""
    print("\n=== Anthropic Example ===\n")

    agent = create_rminer_eval_agent(model_name="claude-3-5-sonnet-20241022")

    result = invoke_agent(
        agent=agent,
        pair_id="example-pair-id",
        manifest_path="rminer_data/manifest.json",
    )

    print(f"Predictions: {len(result['predictions'])}")


if __name__ == "__main__":
    print("\nTo run examples, edit this file and uncomment the example you want.")
    print("Make sure to replace 'example-pair-id' with an actual pair ID from your manifest.")
