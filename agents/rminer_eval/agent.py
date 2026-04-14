"""LangGraph agent for refactoring mapping evaluation.

This module provides a LangGraph agent that maps refactorings to diff hunks
in code changes. The agent analyzes before/after code, refactoring metadata,
and diff hunks to determine which code changes correspond to which refactorings.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, List

from langchain_core.messages import BaseMessage
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from agents.rminer_eval.config import DEFAULT_CONFIG, RMinerEvalAgentConfig
from agents.dependency_analysis.agent import analyze_dependencies, DependencyAnalysis
from smellai_datasets.schema import EvalSample

LOGGER = logging.getLogger(__name__)


class RefactoringMapping(BaseModel):
    """A mapping between a refactoring and a diff hunk."""

    refactoring_index: int = Field(description="Index of the refactoring (0-based)")
    hunk_index: int = Field(description="Index of the diff hunk (0-based)")
    reasoning: str = Field(description="Why this refactoring maps to this hunk")


class RefactoringMappingOutput(BaseModel):
    """Complete output from the refactoring mapping agent."""

    analysis: str = Field(description="Overall analysis")
    mappings: List[RefactoringMapping] = Field(description="Mappings")


class RMinerEvalState(dict):
    """State for RMiner evaluation agent.

    # TODO SPEC-003: Add simple persistence mechanism for long-running workflows.
    # Currently state is in-memory only and cleared after each refactoring operation.
    # Low priority - not critical for current evaluation-based workflow.
    # (See TECHNICAL_SPECIFICATION.md §3.4)
    """

    messages: Annotated[List[BaseMessage], add_messages]
    before_code: str
    filename: str
    refactoring_types: List[str]
    refactoring_descriptions: List[str]
    diff_hunks: List[dict]
    sonar_issues: List[dict]
    dependency_analysis: List[DependencyAnalysis]
    predictions: List[dict]


SYSTEM_PROMPT = """You are an expert code refactoring assistant.

Map each refactoring to the diff hunk where it occurred.

Return JSON:
{
  "analysis": "your analysis",
  "mappings": [{"refactoring_index": 0, "hunk_index": 0, "reasoning": "..."}]
}
"""


def create_rminer_eval_agent(model_name: str | None = None) -> StateGraph:
    """Create a LangGraph agent for refactoring mapping evaluation.

    Args:
        model_name: Name of the LLM model to use. If None, uses default from config.

    Returns:
        Compiled LangGraph StateGraph
    """
    if model_name is None:
        model_name = DEFAULT_CONFIG[RMinerEvalAgentConfig.MODEL_NAME]

    model = ChatLiteLLM(model=model_name)

    try:
        structured_model = model.with_structured_output(RefactoringMappingOutput)
        use_structured = True
    except (NotImplementedError, AttributeError) as e:
        LOGGER.info("Model %s does not support structured output: %s", model_name, e)
        structured_model = model
        use_structured = False

    def map_refactorings(state: RMinerEvalState) -> dict:
        """Map refactorings to diff hunks.

        # TODO SPEC-007: Implement token counting and truncation strategy for large files.
        # Large files (e.g., "God Classes") may exceed LLM context windows.
        # Need to implement token counting and truncation before prompt construction.
        # HIGH priority.
        # (See TECHNICAL_SPECIFICATION.md §4.3)
        """
        refactorings_str = "\n".join(
            f"{i}. Type: {rt}\n   Description: {rd}"
            for i, (rt, rd) in enumerate(
                zip(state["refactoring_types"], state["refactoring_descriptions"])
            )
        )

        hunks_str = "\n".join(
            f"{i}. Lines {h['old_start']}-{h['old_start'] + h['old_count'] - 1}"
            for i, h in enumerate(state["diff_hunks"])
        )

        # TODO SPEC-004: Document when sonar_issues context is included vs excluded.
        # Currently included if state contains sonar_issues, but decision criteria not documented.
        # (See TECHNICAL_SPECIFICATION.md §4.3)

        # TODO SPEC-005: Document when dependency_analysis context is included vs excluded.
        # Currently computed from sonar_issues, but inclusion criteria not documented.
        # (See TECHNICAL_SPECIFICATION.md §4.3)

        # TODO SPEC-006: Add reference link to exact prompt structure datamodels in code.
        # Prompt construction should reference the specific datamodels being used.
        # (See TECHNICAL_SPECIFICATION.md §4.3)

        sonar_str = ""
        dep_str = ""
        if state.get("sonar_issues"):
            sonar_str = "\n## SonarQube Issues\n" + "\n".join(
                f"- {issue.get('message')} (Line {issue.get('line')}, {issue.get('severity')})"
                for issue in state["sonar_issues"]
            )

            # Analyze dependencies
            deps = analyze_dependencies(state["sonar_issues"])
            if deps:
                dep_str = "\n## Dependency Analysis\n"
                for d in deps:
                    dep_str += f"- Smell: {d.smell_type}\n"
                    dep_str += f"  - Positive Dependencies (Solve): {', '.join(d.positive_dependencies)}\n"
                    dep_str += f"  - Negative Dependencies (Cause): {', '.join(d.negative_dependencies)}\n"

        prompt = f"""## File: {state["filename"]}

## Refactorings
{refactorings_str}

## Diff Hunks
{hunks_str}
{sonar_str}
{dep_str}
## BEFORE Code
```java
{state["before_code"]}
```
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = structured_model.invoke(messages)

        if use_structured:
            mappings_data = response.mappings
        else:
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            try:
                parsed = json.loads(response_text)
                mappings_data = [
                    RefactoringMapping(**m) for m in parsed.get("mappings", [])
                ]
            except json.JSONDecodeError as e:
                LOGGER.warning("Failed to parse LLM response as JSON: %s", e)
                mappings_data = []

        predictions = []
        for mapping in mappings_data:
            ref_idx = mapping.refactoring_index
            hunk_idx = mapping.hunk_index

            if 0 <= ref_idx < len(state["refactoring_types"]) and 0 <= hunk_idx < len(
                state["diff_hunks"]
            ):
                hunk = state["diff_hunks"][hunk_idx]
                predictions.append(
                    {
                        "refactoring_index": ref_idx,
                        "predicted_hunk_index": hunk_idx,
                        "refactoring_type": state["refactoring_types"][ref_idx],
                        "line_start": hunk["old_start"],
                        "line_end": hunk["old_start"] + hunk["old_count"] - 1,
                        "reasoning": mapping.reasoning,
                    }
                )

        return {"predictions": predictions}

    workflow = StateGraph(RMinerEvalState)
    workflow.add_node("map_refactorings", map_refactorings)
    workflow.set_entry_point("map_refactorings")
    workflow.add_edge("map_refactorings", END)

    return workflow.compile()


def invoke_agent(
    agent,
    sample: EvalSample,
) -> dict:
    """Invoke agent for a single refactoring pair from an EvalSample.

    The EvalSample must have source="rminer" and carry all required data in
    inputs (pair_id, before_code, file_path, refactoring_types,
    refactoring_descriptions, diff_hunks, sonar_issues).  These are produced
    by smellai_datasets.loaders._rminer_samples() at load time.

    Args:
        agent: Compiled LangGraph agent
        sample: EvalSample with source="rminer"

    Returns:
        Dictionary with pair_id, filename, and predictions
    """
    if sample.source != "rminer":
        raise ValueError(
            f"RMiner agent expects source='rminer', got {sample.source!r}"
        )

    inputs = sample.inputs
    pair_id: str = inputs["pair_id"]

    result = agent.invoke(
        {
            "messages": [],
            "before_code": inputs["before_code"],
            "filename": inputs["file_path"],
            "refactoring_types": inputs["refactoring_types"],
            "refactoring_descriptions": inputs["refactoring_descriptions"],
            "diff_hunks": inputs["diff_hunks"],
            "sonar_issues": inputs.get("sonar_issues") or [],
            "dependency_analysis": [],
            "predictions": [],
        }
    )

    return {
        "pair_id": pair_id,
        "filename": inputs["file_path"],
        "predictions": result.get("predictions", []),
    }
