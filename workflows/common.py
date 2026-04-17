"""Shared helpers for MLflow-based evaluation workflows."""

from __future__ import annotations

from smellai_datasets import EvalSample
from smellai_datasets.schema import rminer_sample


def make_rminer_eval_sample(
    pair_id: str,
    before_code: str,
    file_path: str,
    refactoring_types: list,
    refactoring_descriptions: list,
    diff_hunks: list,
    sonar_issues: list | None = None,
) -> EvalSample:
    """Build an EvalSample for an RMiner predict_fn call."""
    return rminer_sample(
        pair_id=pair_id,
        before_code=before_code,
        file_path=file_path,
        refactoring_types=refactoring_types,
        refactoring_descriptions=refactoring_descriptions,
        diff_hunks=diff_hunks,
        sonar_issues=sonar_issues,
    )


def make_swe_eval_sample(
    project_name: str,
    commit_id: str,
    refactoring_type: str,
    file_path_before: str,
    file_path_after: str,
    class_before: str,
    source_before: str,
    jdk_version: int,
    compile_command: str,
) -> EvalSample:
    """Build an EvalSample for a SWE predict_fn call."""
    return EvalSample(
        source="swe",
        sample_id=f"swe:{commit_id}",
        inputs={
            "project_name": project_name,
            "commit_id": commit_id,
            "refactoring_type": refactoring_type,
            "file_path_before": file_path_before,
            "file_path_after": file_path_after,
            "class_before": class_before,
            "source_before": source_before,
            "jdk_version": jdk_version,
            "compile_command": compile_command,
        },
        expectations={},
        tags={},
    )


def _get_rminer_scorers():
    """Return the standard RMiner scorer triple."""
    from agents.rminer_eval import (
        mapping_accuracy,
        hunk_coverage,
        prediction_completeness,
    )

    return [mapping_accuracy, hunk_coverage, prediction_completeness]


def run_rminer_evaluation(records, predict_fn, scorers, tracking_uri: str) -> int:
    """Run mlflow.genai.evaluate with rminer scorers, print results, and return 0."""
    from mlflow.genai import evaluate as genai_evaluate

    results = genai_evaluate(
        data=records,
        predict_fn=predict_fn,
        scorers=scorers,
    )
    print_eval_results(results, tracking_uri)
    return 0


def invoke_swe_agent(invoke_fn, agent, sample, args, analytics_db=None) -> dict:
    """Call a SWE invoke function with the standard args from CLI namespace."""
    return invoke_fn(
        agent,
        sample,
        args.workspace,
        analytics_db=analytics_db,
        max_refactorings=args.max_refactorings,
        sonar_url=args.sonar_url,
        sonar_cache_dir=args.sonar_cache_dir,
    )


def load_rminer_records(args):
    """Validate manifest, configure MLflow, load RMiner samples → MLflow records.

    Returns (records, None) on success, or (None, error_code) on failure.
    """
    import sys
    from pathlib import Path
    from smellai_datasets import load_eval_samples, samples_to_mlflow_records

    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return None, 1

    setup_workflow_mlflow(args.tracking_uri, args.experiment)

    samples = load_eval_samples(
        ["rminer"],
        rminer_manifest_path=manifest_path,
        limit=args.limit,
    )
    if not samples:
        print("No records to evaluate", file=sys.stderr)
        return None, 1

    records = samples_to_mlflow_records(samples)
    return records, 0


def setup_workflow_mlflow(
    tracking_uri: str,
    experiment: str,
    backend_uri: str = "sqlite:///mlflow.db",
) -> None:
    """Configure MLflow tracking with auto server start."""
    from mlflow_utils import setup_mlflow_tracking

    setup_mlflow_tracking(
        tracking_uri=tracking_uri,
        backend_uri=backend_uri,
        experiment_name=experiment,
        auto_start_server=True,
    )


def save_agent_graph(agent, output_path: str) -> None:
    """Render and save a LangGraph agent graph to PNG."""
    try:
        png_bytes = agent.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        print(f"Graph saved to {output_path}")
    except Exception as e:
        print(f"Failed to draw graph: {e}")
        print(
            "Ensure langgraph is installed. `pip install grandalf` may be needed for visualization."
        )


def print_eval_results(results, tracking_uri: str | None = None) -> None:
    """Print MLflow evaluation results to stdout."""
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

    if tracking_uri and run_id != "N/A" and tracking_uri.startswith("http://"):
        exp_id = getattr(results, "experiment_id", "?")
        print(f"View results: {tracking_uri}/#/experiments/{exp_id}/runs/{run_id}")
