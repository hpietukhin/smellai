"""Shared helpers for MLflow-based evaluation workflows."""

from __future__ import annotations


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
        print("Ensure langgraph is installed. `pip install grandalf` may be needed for visualization.")


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
