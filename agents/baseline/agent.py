"""Baseline refactoring mapping agent.

Single LLM call — no LangGraph, no SonarQube context, no dependency analysis.
Serves as a lower-bound comparison for the full rminer_eval agent.
"""

from __future__ import annotations

import json
import logging

from langchain_litellm import ChatLiteLLM

from smellai_datasets.schema import EvalSample

LOGGER = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Map each refactoring to the diff hunk where it occurred.

## File: {filename}

## Refactorings
{refactorings}

## Diff Hunks
{hunks}

## BEFORE Code
```java
{before_code}
```

Return ONLY JSON:
{{"mappings": [{{"refactoring_index": 0, "hunk_index": 0, "reasoning": "..."}}]}}
"""


def invoke_baseline_agent(
    sample: EvalSample,
    model_name: str = "gpt-4o-mini",
) -> dict:
    """Single LLM call to map refactorings to diff hunks.

    Args:
        sample: EvalSample with source="rminer", containing before_code,
                file_path, refactoring_types, refactoring_descriptions,
                diff_hunks in inputs.
        model_name: LiteLLM model identifier.

    Returns:
        Dictionary with pair_id, filename, and predictions.
    """
    if sample.source != "rminer":
        raise ValueError(
            f"Baseline agent expects source='rminer', got {sample.source!r}"
        )

    inputs = sample.inputs
    pair_id: str = inputs["pair_id"]
    before_code: str = inputs["before_code"]
    file_path: str = inputs["file_path"]
    types: list[str] = inputs["refactoring_types"]
    descriptions: list[str] = inputs["refactoring_descriptions"]
    hunk_dicts: list[dict] = inputs["diff_hunks"]

    refactorings_str = "\n".join(
        f"{i}. Type: {rt}\n   Description: {rd}"
        for i, (rt, rd) in enumerate(zip(types, descriptions))
    )
    hunks_str = "\n".join(
        f"{i}. Lines {h['old_start']}-{h['old_start'] + h['old_count'] - 1}"
        for i, h in enumerate(hunk_dicts)
    )

    prompt = PROMPT_TEMPLATE.format(
        filename=file_path,
        refactorings=refactorings_str,
        hunks=hunks_str,
        before_code=before_code,
    )

    llm = ChatLiteLLM(model=model_name)
    response = llm.invoke([{"role": "user", "content": prompt}])

    response_text = response.content if hasattr(response, "content") else str(response)

    # Parse JSON from response
    if "```json" in response_text:
        json_start = response_text.find("```json") + 7
        json_end = response_text.find("```", json_start)
        response_text = response_text[json_start:json_end].strip()
    elif "```" in response_text:
        json_start = response_text.find("```") + 3
        json_end = response_text.find("```", json_start)
        response_text = response_text[json_start:json_end].strip()

    try:
        parsed = json.loads(response_text)
        mappings = parsed.get("mappings", [])
    except json.JSONDecodeError as e:
        LOGGER.warning("Failed to parse baseline response: %s", e)
        mappings = []

    predictions = []
    for m in mappings:
        ref_idx = m.get("refactoring_index", -1)
        hunk_idx = m.get("hunk_index", -1)
        if 0 <= ref_idx < len(types) and 0 <= hunk_idx < len(hunk_dicts):
            hunk = hunk_dicts[hunk_idx]
            predictions.append(
                {
                    "refactoring_index": ref_idx,
                    "predicted_hunk_index": hunk_idx,
                    "refactoring_type": types[ref_idx],
                    "line_start": hunk["old_start"],
                    "line_end": hunk["old_start"] + hunk["old_count"] - 1,
                    "reasoning": m.get("reasoning", ""),
                }
            )

    return {
        "pair_id": pair_id,
        "filename": file_path,
        "predictions": predictions,
    }
