"""DEPRECATED: LangGraph agent for SWE-Refactor evaluation.

Use Composite Refactorings 2020 flow instead:
- dataset.neo4j_graph.DatasetGraph.composite_refactoring(...)
- workflows/planner_eval_workflow.py

This module is kept for backward compatibility and will be removed in a future cleanup.
"""

from .config import SWEEvalAgentConfig, DEFAULT_CONFIG
from .prompts import get_refactoring_prompt, SYSTEM_PROMPT
from .agent import create_swe_eval_agent, invoke_agent
from .scorers import get_swe_scorers

__all__ = [
    "SWEEvalAgentConfig",
    "DEFAULT_CONFIG",
    "get_refactoring_prompt",
    "SYSTEM_PROMPT",
    "create_swe_eval_agent",
    "invoke_agent",
    "get_swe_scorers",
]
