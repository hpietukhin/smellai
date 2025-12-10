"""LangGraph agent for refactoring mapping evaluation.

This module provides a LangGraph agent that maps refactorings to diff hunks
in code changes. The agent analyzes before/after code, refactoring metadata,
and diff hunks to determine which code changes correspond to which refactorings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, List

from langchain_core.messages import BaseMessage
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from rminer.create_rminer_dataset import parse_diff_hunks, parse_refactoring_info
from agents.rminer_eval.config import DEFAULT_CONFIG, RMinerEvalAgentConfig


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
    """State for RMiner evaluation agent."""

    messages: Annotated[List[BaseMessage], add_messages]
    before_code: str
    filename: str
    refactoring_types: List[str]
    refactoring_descriptions: List[str]
    diff_hunks: List[dict]
    sonar_issues: List[dict]
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
    except Exception:
        structured_model = model
        use_structured = False

    def map_refactorings(state: RMinerEvalState) -> dict:
        """Map refactorings to diff hunks."""
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

        sonar_str = ""
        if state.get("sonar_issues"):
            sonar_str = "\n## SonarQube Issues\n" + "\n".join(
                f"- {issue.get('message')} (Line {issue.get('line')}, {issue.get('severity')})"
                for issue in state["sonar_issues"]
            )

        prompt = f"""## File: {state['filename']}

## Refactorings
{refactorings_str}

## Diff Hunks
{hunks_str}
{sonar_str}
## BEFORE Code
```java
{state['before_code']}
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
            except Exception:
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
    pair_id: str,
    manifest_path: str | Path,
    sonar_issues: List[dict] | None = None,
) -> dict:
    """Invoke agent for a single refactoring pair.

    Args:
        agent: Compiled LangGraph agent
        pair_id: ID of the refactoring pair to evaluate
        manifest_path: Path to the manifest.json file
        sonar_issues: Optional SonarQube issues for the file

    Returns:
        Dictionary with pair_id, filename, and predictions
    """
    manifest_file = Path(manifest_path)
    base_dir = manifest_file.parent

    with open(manifest_file) as f:
        manifest = json.load(f)

    pair = None
    for p in manifest.get("pairs", []):
        if p["id"] == pair_id:
            pair = p
            break

    if not pair:
        return {"pair_id": pair_id, "predictions": [], "error": "Pair not found"}

    before_path = base_dir / pair["before_file"]
    after_path = base_dir / pair["after_file"]

    with open(before_path) as f:
        before_code = f.read()

    diff_hunks = parse_diff_hunks(before_path, after_path)
    types, descriptions = parse_refactoring_info(pair)

    result = agent.invoke(
        {
            "messages": [],
            "before_code": before_code,
            "filename": pair["file_path"],
            "refactoring_types": types,
            "refactoring_descriptions": descriptions,
            "diff_hunks": [h.to_dict() for h in diff_hunks],
            "sonar_issues": sonar_issues or [],
            "predictions": [],
        }
    )

    return {
        "pair_id": pair_id,
        "filename": pair["file_path"],
        "predictions": result.get("predictions", []),
    }
