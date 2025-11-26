#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlflow>=3.3",
#     "python-dotenv"
# ]
# ///
"""Create MLflow GenAI evaluation dataset from RefactoringMiner pairs.

This script creates a dataset with:
- inputs: pair_id (passed to predict_fn)
- expectations: ground truth (diff_hunks, refactoring metadata)
- tags: repository, commit info

Usage:
    # Create dataset
    uv run infra/mlflow/rminer_dataset.py --manifest rminer_data/manifest.json

    # Dry run (preview records)
    uv run infra/mlflow/rminer_dataset.py --manifest rminer_data/manifest.json --dry-run

    # Limit number of pairs
    uv run infra/mlflow/rminer_dataset.py --manifest rminer_data/manifest.json --limit 10
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()


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
    except Exception as e:
        print(f"Warning: Failed to parse diff: {e}", file=sys.stderr)
        return []


def parse_refactoring_info(pair: dict) -> Tuple[List[str], List[str]]:
    """Extract refactoring types and descriptions."""
    ref_type = pair.get("refactoring_type", "")
    ref_desc = pair.get("refactoring_description", "")

    types = [t.strip() for t in ref_type.split("|")] if ref_type else []
    descriptions = [d.strip() for d in ref_desc.split("\n")] if ref_desc else []

    return types, descriptions


def build_genai_records(manifest_path: Path, limit: int | None = None) -> list[dict]:
    """
    Build GenAI evaluation records from manifest.

    Each record has:
    - inputs: {"pair_id": "..."}
    - expectations: ground truth data
    - tags: metadata
    """
    base_dir = manifest_path.parent

    with open(manifest_path) as f:
        manifest = json.load(f)

    pairs = manifest.get("pairs", [])
    if limit:
        pairs = pairs[:limit]

    records = []
    skipped = 0

    for pair in pairs:
        before_path = base_dir / pair["before_file"]
        after_path = base_dir / pair["after_file"]

        if not before_path.exists() or not after_path.exists():
            skipped += 1
            continue

        diff_hunks = parse_diff_hunks(before_path, after_path)
        types, descriptions = parse_refactoring_info(pair)

        if not diff_hunks:
            skipped += 1
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
                "file_path": pair["file_path"],
            },
            "tags": {
                "repository": pair.get("repository", ""),
                "commit_sha": pair.get("commit_sha", ""),
                "status": pair.get("status", "modified"),
            },
        }
        records.append(record)

    print(f"Built {len(records)} records ({skipped} skipped)")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Create MLflow GenAI dataset")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--experiment", default="rminer-evaluation")
    parser.add_argument("--dataset-name", default="rminer-eval-dataset")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--limit", type=int, help="Limit number of pairs")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument(
        "--output-json", help="Save records to JSON file (for debugging)"
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    # Build records
    records = build_genai_records(manifest_path, limit=args.limit)

    if not records:
        print("No valid records found", file=sys.stderr)
        return 1

    # Stats
    total_hunks = sum(r["expectations"]["num_hunks"] for r in records)
    total_refactorings = sum(r["expectations"]["num_refactorings"] for r in records)
    print(f"Total hunks: {total_hunks}")
    print(f"Total refactorings: {total_refactorings}")

    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Saved records to {args.output_json}")

    if args.dry_run:
        print("\n--- Sample Record ---")
        sample = records[0].copy()
        if sample["expectations"]["diff_hunks"]:
            sample["expectations"]["diff_hunks"] = [
                {
                    **sample["expectations"]["diff_hunks"][0],
                    "removed_lines": ["..."],
                    "added_lines": ["..."],
                }
            ]
        print(json.dumps(sample, indent=2))
        return 0

    # Create MLflow dataset
    import mlflow
    from mlflow.genai.datasets import create_dataset

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    experiment = mlflow.get_experiment_by_name(args.experiment)
    if not experiment:
        print(f"Creating experiment: {args.experiment}")
        experiment_id = mlflow.create_experiment(args.experiment)
    else:
        experiment_id = experiment.experiment_id

    print(f"Experiment ID: {experiment_id}")

    # Create dataset
    dataset = create_dataset(
        name=args.dataset_name,
        experiment_id=[experiment_id],
        tags={
            "source": "RefactoringMiner",
            "total_pairs": str(len(records)),
            "total_hunks": str(total_hunks),
            "total_refactorings": str(total_refactorings),
        },
    )

    # Merge records
    dataset.merge_records(records)

    # Verify persistence
    actual_count = len(dataset.records)
    if actual_count != len(records):
        print(
            f"Warning: Expected {len(records)} records but dataset has {actual_count}",
            file=sys.stderr,
        )

    print(f"\n{'='*50}")
    print(f"Dataset created: {args.dataset_name}")
    print(f"Dataset ID: {dataset.dataset_id}")
    print(f"Records: {actual_count}")
    print(f"{'='*50}")
    print("\nTo inspect:")
    print(
        f"  uv run infra/rminer_dataset_cli.py get --id {dataset.dataset_id} --show-records"
    )
    print("\nTo evaluate:")
    print(f"  uv run src/pipelines/rminer_eval.py --dataset-id {dataset.dataset_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
