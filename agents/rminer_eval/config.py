"""DEPRECATED: Configuration for RMiner Evaluation Agent.

Use Composite Refactorings 2020 flow instead (DatasetGraph.composite_refactoring + planner_eval_workflow).
"""

from dataclasses import dataclass


@dataclass
class RMinerEvalAgentConfig:
    """Configuration keys for RMiner Evaluation Agent."""

    MODEL_NAME = "model_name"


DEFAULT_CONFIG = {
    RMinerEvalAgentConfig.MODEL_NAME: "gpt-4o-mini",
}
