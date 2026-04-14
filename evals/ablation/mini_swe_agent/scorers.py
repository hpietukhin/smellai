"""Ablation-specific MLflow scorers for mini-swe-agent runs.

These supplement the baseline swe scorers (compile_success, test_pass, overall_success)
with mini-swe-agent-specific efficiency metrics.
"""

from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer


@scorer
def mini_cost_scorer(outputs: dict) -> Feedback:
    """LLM cost in USD accumulated by mini-swe-agent for this task."""
    cost = outputs.get("mini_cost", 0.0) or 0.0
    return Feedback(value=float(cost), rationale=f"${cost:.4f} LLM cost")


@scorer
def mini_step_count_scorer(outputs: dict) -> Feedback:
    """Number of LLM calls made by mini-swe-agent for this task."""
    steps = outputs.get("mini_n_calls", 0) or 0
    return Feedback(value=float(steps), rationale=f"{steps} LLM calls")


@scorer
def mini_exit_status_scorer(outputs: dict) -> Feedback:
    """1.0 if mini-swe-agent reached Submitted exit, 0.0 otherwise."""
    status = outputs.get("mini_exit", "")
    submitted = status == "submitted"
    return Feedback(
        value=1.0 if submitted else 0.0,
        rationale=f"exit_status={status!r}",
    )
