"""DEPRECATED: RMiner Evaluation Agent.

Redirect to Composite Refactorings 2020 flow instead:
- dataset.neo4j_graph.DatasetGraph.composite_refactoring(...)
- workflows/planner_eval_workflow.py

LangGraph-based agent for mapping refactorings to diff hunks in code changes.
Kept only for backward compatibility.
"""

from agents.rminer_eval.agent import (
    create_rminer_eval_agent,
    invoke_agent,
    RMinerEvalState,
)
from agents.rminer_eval.config import (
    RMinerEvalAgentConfig,
    DEFAULT_CONFIG,
)
from agents.rminer_eval.scorers import (
    mapping_accuracy,
    hunk_coverage,
    prediction_completeness,
)

__all__ = [
    "create_rminer_eval_agent",
    "invoke_agent",
    "RMinerEvalState",
    "RMinerEvalAgentConfig",
    "DEFAULT_CONFIG",
    "mapping_accuracy",
    "hunk_coverage",
    "prediction_completeness",
]
