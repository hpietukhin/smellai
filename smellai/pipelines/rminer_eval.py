#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlflow>=3.3",
#     "langchain-core",
#     "langchain-litellm",
#     "langgraph",
#     "pydantic",
#     "litellm",
#     "python-dotenv"
# ]
# ///
"""MLflow GenAI evaluation pipeline for the refactoring mapping agent.

Evaluates the agent's ability to correctly map refactorings to diff hunks.

Scorers:
- mapping_accuracy: fraction of predictions that overlap with ground truth hunks
- hunk_coverage: fraction of ground truth hunks covered by predictions
- prediction_count: whether agent made expected number of predictions

Usage:
    # Evaluate using inline data (from manifest)
    uv run infra/mlflow/rminer_evaluate.py --manifest rminer_data/manifest.json --limit 5

    # Evaluate using saved dataset
    uv run infra/mlflow/rminer_evaluate.py --dataset-name rminer-eval-dataset

    # Use different model
    uv run infra/mlflow/rminer_evaluate.py --manifest rminer_data/manifest.json --model claude-sonnet-4-5-20250929
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Annotated

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

load_dotenv()


# -----------------------------------------------------------------------------
# Data structures (duplicated to avoid import issues with uv run)
# -----------------------------------------------------------------------------


@dataclass
class DiffHunk:
    """A hunk from git diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    removed_lines: List[str] = field(default_factory=list)
    added_lines: List[str] = field(default_factory=list)
    context_lines: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def parse_diff_hunks(before_file: Path, after_file: Path) -> List[DiffHunk]:
    """Compute diff hunks between before and after files."""
    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--no-index",
                "--unified=3",
                str(before_file),
                str(after_file),
            ],
            capture_output=True,
            text=True,
        )

        hunk_pattern = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        hunks = []
        current_hunk = None

        for line in result.stdout.split("\n"):
            match = hunk_pattern.match(line)
            if match:
                if current_hunk:
                    hunks.append(current_hunk)

                current_hunk = DiffHunk(
                    old_start=int(match.group(1)),
                    old_count=int(match.group(2)) if match.group(2) else 1,
                    new_start=int(match.group(3)),
                    new_count=int(match.group(4)) if match.group(4) else 1,
                )
            elif current_hunk:
                if line.startswith("---") or line.startswith("+++"):
                    continue
                elif line.startswith("-"):
                    current_hunk.removed_lines.append(line[1:])
                elif line.startswith("+"):
                    current_hunk.added_lines.append(line[1:])
                elif line.startswith(" "):
                    current_hunk.context_lines.append(line[1:])

        if current_hunk:
            hunks.append(current_hunk)

        return hunks
    except Exception:
        return []


def parse_refactoring_info(pair: dict) -> Tuple[List[str], List[str]]:
    """Extract refactoring types and descriptions."""
    ref_type = pair.get("refactoring_type", "")
    ref_desc = pair.get("refactoring_description", "")

    types = [t.strip() for t in ref_type.split("|")] if ref_type else []
    descriptions = [d.strip() for d in ref_desc.split("\n")] if ref_desc else []

    return types, descriptions


def build_genai_records(manifest_path: Path, limit: int | None = None) -> list[dict]:
    """Build GenAI evaluation records from manifest."""
    base_dir = manifest_path.parent

    with open(manifest_path) as f:
        manifest = json.load(f)

    pairs = manifest.get("pairs", [])
    if limit:
        pairs = pairs[:limit]

    records = []

    for pair in pairs:
        before_path = base_dir / pair["before_file"]
        after_path = base_dir / pair["after_file"]

        if not before_path.exists() or not after_path.exists():
            continue

        diff_hunks = parse_diff_hunks(before_path, after_path)
        types, descriptions = parse_refactoring_info(pair)

        if not diff_hunks:
            continue

        record = {
            "inputs": {
                "pair_id": pair["id"],
            },
            "expectations": {
                "num_refactorings": len(types),
                "num_hunks": len(diff_hunks),
                "diff_hunks": [h.to_dict() for h in diff_hunks],
                "refactoring_types": types,
                "refactoring_descriptions": descriptions,
            },
            "tags": {
                "repository": pair.get("repository", ""),
                "commit_sha": pair.get("commit_sha", ""),
            },
        }
        records.append(record)

    return records


# -----------------------------------------------------------------------------
# Scorers
# -----------------------------------------------------------------------------


def _calculate_line_overlap(start1: int, end1: int, start2: int, end2: int) -> int:
    """Calculate overlapping lines between two ranges."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0, overlap_end - overlap_start + 1)


@scorer
def mapping_accuracy(outputs: dict, expectations: dict) -> Feedback:
    """
    Compute accuracy: fraction of predictions that overlap with ground truth hunks.

    A prediction is correct if its line range overlaps with any ground truth hunk.
    """
    predictions = outputs.get("predictions", [])
    hunks = expectations.get("diff_hunks", [])

    if not predictions:
        return Feedback(value=0.0, rationale="No predictions made by agent")

    correct = 0
    details = []

    for pred in predictions:
        pred_start = pred.get("line_start", 0)
        pred_end = pred.get("line_end", 0)
        pred_hunk_idx = pred.get("predicted_hunk_index", -1)

        # Check if prediction overlaps with any hunk
        has_overlap = False
        for i, hunk in enumerate(hunks):
            hunk_start = hunk.get("old_start", 0)
            hunk_end = hunk_start + hunk.get("old_count", 1) - 1

            overlap = _calculate_line_overlap(
                pred_start, pred_end, hunk_start, hunk_end
            )
            if overlap > 0:
                has_overlap = True
                details.append(f"Pred {pred_hunk_idx}→Hunk {i}: ✓ (overlap={overlap})")
                break

        if has_overlap:
            correct += 1
        else:
            details.append(f"Pred {pred_hunk_idx}: ✗ (no overlap)")

    accuracy = correct / len(predictions)

    return Feedback(
        value=accuracy,
        rationale=f"{correct}/{len(predictions)} predictions correct. "
        + "; ".join(details[:5]),
    )


@scorer
def hunk_coverage(outputs: dict, expectations: dict) -> Feedback:
    """
    Compute coverage: fraction of ground truth hunks covered by at least one prediction.
    """
    predictions = outputs.get("predictions", [])
    hunks = expectations.get("diff_hunks", [])

    if not hunks:
        return Feedback(value=1.0, rationale="No hunks to cover")

    covered_hunks = set()

    for pred in predictions:
        pred_start = pred.get("line_start", 0)
        pred_end = pred.get("line_end", 0)

        for i, hunk in enumerate(hunks):
            hunk_start = hunk.get("old_start", 0)
            hunk_end = hunk_start + hunk.get("old_count", 1) - 1

            overlap = _calculate_line_overlap(
                pred_start, pred_end, hunk_start, hunk_end
            )
            if overlap > 0:
                covered_hunks.add(i)

    coverage = len(covered_hunks) / len(hunks)

    return Feedback(
        value=coverage,
        rationale=f"{len(covered_hunks)}/{len(hunks)} hunks covered by predictions",
    )


@scorer
def prediction_completeness(outputs: dict, expectations: dict) -> Feedback:
    """
    Check if agent made predictions for all refactorings.
    """
    predictions = outputs.get("predictions", [])
    expected_count = expectations.get("num_refactorings", 0)

    if expected_count == 0:
        return Feedback(value=1.0, rationale="No refactorings expected")

    actual_count = len(predictions)
    ratio = min(actual_count / expected_count, 1.0)

    if actual_count == expected_count:
        rationale = f"Agent made exactly {expected_count} predictions as expected"
    elif actual_count < expected_count:
        rationale = (
            f"Agent made {actual_count}/{expected_count} predictions (missing some)"
        )
    else:
        rationale = (
            f"Agent made {actual_count} predictions but only {expected_count} expected"
        )

    return Feedback(value=ratio, rationale=rationale)


# -----------------------------------------------------------------------------
# Agent setup (inline to avoid import issues)
# -----------------------------------------------------------------------------


def create_prediction_agent(model_name: str):
    """Create the refactoring mapping agent."""
    from langchain_litellm import ChatLiteLLM
    from langgraph.graph import StateGraph, END

    class RefactoringMapping(BaseModel):
        refactoring_index: int = Field(description="Index of the refactoring (0-based)")
        hunk_index: int = Field(description="Index of the diff hunk (0-based)")
        reasoning: str = Field(description="Why this refactoring maps to this hunk")

    class RefactoringMappingOutput(BaseModel):
        analysis: str = Field(description="Overall analysis")
        mappings: List[RefactoringMapping] = Field(description="Mappings")

    class PredictionState(dict):
        messages: Annotated[List[BaseMessage], add_messages]
        before_code: str
        filename: str
        refactoring_types: List[str]
        refactoring_descriptions: List[str]
        diff_hunks: List[dict]
        sonar_issues: List[dict]
        predictions: List[dict]

    model = ChatLiteLLM(model=model_name)

    try:
        structured_model = model.with_structured_output(RefactoringMappingOutput)
        use_structured = True
    except Exception:
        structured_model = model
        use_structured = False

    SYSTEM_PROMPT = """You are an expert code refactoring assistant.

Map each refactoring to the diff hunk where it occurred.

Return JSON:
{
  "analysis": "your analysis",
  "mappings": [{"refactoring_index": 0, "hunk_index": 0, "reasoning": "..."}]
}
"""

    def map_refactorings(state: PredictionState) -> dict:
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
            except:
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

    workflow = StateGraph(PredictionState)
    workflow.add_node("map_refactorings", map_refactorings)
    workflow.set_entry_point("map_refactorings")
    workflow.add_edge("map_refactorings", END)

    return workflow.compile()


def invoke_agent(agent, pair_id: str, manifest_path: str, sonar_issues: List[dict] = None) -> dict:
    """Invoke agent for a single pair."""
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate refactoring mapping agent")
    parser.add_argument("--manifest", help="Path to manifest.json")
    parser.add_argument("--dataset-name", help="MLflow dataset name to use")
    parser.add_argument(
        "--dataset-id", help="MLflow dataset ID to use (alternative to --dataset-name)"
    )
    parser.add_argument("--experiment", default="rminer-evaluation")
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, help="Limit number of pairs")
    parser.add_argument("--run-name", help="MLflow run name")
    parser.add_argument("--draw-graph", action="store_true", help="Draw the agent graph to a PNG file")
    args = parser.parse_args()

    if args.draw_graph:
        print("Generating agent graph...")
        agent = create_prediction_agent(model_name=args.model)
        try:
            png_bytes = agent.get_graph().draw_mermaid_png()
            output_path = "agent_graph.png"
            with open(output_path, "wb") as f:
                f.write(png_bytes)
            print(f"Graph saved to {output_path}")
        except Exception as e:
            print(f"Failed to draw graph: {e}")
            print("Ensure you have `langgraph` installed. `pip install grandalf` might be required for some visualization features.")
        return 0

    if not args.manifest and not args.dataset_name and not args.dataset_id:
        print(
            "Either --manifest, --dataset-name, or --dataset-id is required",
            file=sys.stderr,
        )
        return 1

    # Setup MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    print(f"Model: {args.model}")
    print(f"Experiment: {args.experiment}")

    # Load data
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"Manifest not found: {manifest_path}", file=sys.stderr)
            return 1
        records = build_genai_records(manifest_path, limit=args.limit)
        print(f"Loaded {len(records)} records from manifest")
    elif args.dataset_id:
        # Load by ID (works in OSS MLflow)
        from mlflow.genai.datasets import get_dataset

        dataset = get_dataset(dataset_id=args.dataset_id)
        records = dataset.records
        print(f"Loaded {len(records)} records from dataset ID: {args.dataset_id}")
        manifest_path = Path("rminer_data/manifest.json")  # Fallback
    else:
        # Load from MLflow dataset by name
        from mlflow.genai.datasets import get_dataset, search_datasets

        # OSS MLflow doesn't support get by name - search and filter
        all_datasets = search_datasets()
        matching = [ds for ds in all_datasets if ds.name == args.dataset_name]
        if not matching:
            print(f"Dataset not found: {args.dataset_name}", file=sys.stderr)
            print("\nAvailable datasets:", file=sys.stderr)
            for ds in all_datasets:
                print(f"  - {ds.name} (ID: {ds.dataset_id})", file=sys.stderr)
            return 1
        dataset = matching[0]
        records = dataset.records
        print(
            f"Loaded {len(records)} records from dataset {args.dataset_name} (ID: {dataset.dataset_id})"
        )
        manifest_path = Path("rminer_data/manifest.json")  # Fallback

    if not records:
        print("No records to evaluate", file=sys.stderr)
        return 1

    # Create agent
    print("Creating agent...")
    agent = create_prediction_agent(model_name=args.model)

    # Create predict function
    def predict_fn(pair_id: str, sonar_issues: List[dict] = None) -> dict:
        return invoke_agent(agent, pair_id, str(manifest_path), sonar_issues)

    # Run evaluation
    print(f"Running evaluation on {len(records)} records...")

    results = mlflow.genai.evaluate(
        data=records,
        predict_fn=predict_fn,
        scorers=[mapping_accuracy, hunk_coverage, prediction_completeness],
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for metric_name, metric_value in results.metrics.items():
        print(
            f"{metric_name}: {metric_value:.4f}"
            if isinstance(metric_value, float)
            else f"{metric_name}: {metric_value}"
        )

    print("=" * 60)
    print(
        f"MLflow run ID: {mlflow.active_run().info.run_id if mlflow.active_run() else 'N/A'}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
