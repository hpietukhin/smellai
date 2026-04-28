"""Create MLflow GenAI evaluation dataset from RefactoringMiner pairs.

This script creates a dataset with:
- inputs: pair_id (passed to predict_fn)
- expectations: ground truth (diff_hunks, refactoring metadata)
- tags: repository, commit info

Usage:
    # Create dataset
    uv run rminer/rminer_dataset.py --manifest data/rminer/manifest.json

    # Dry run (preview records)
    uv run rminer/rminer_dataset.py --manifest data/rminer/manifest.json --dry-run

    # Limit number of pairs
    uv run rminer/rminer_dataset.py --manifest data/rminer/manifest.json --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from rminer.diff_hunk import DiffHunk
from rminer.rminer_utils import compute_diff_hunks_from_files, parse_refactoring_info
from smellai_datasets.loaders import _iter_valid_rminer_pairs

try:
    from smellai.sonarqube.commit_scan import scan_commit
except ImportError:
    scan_commit = None

load_dotenv()


def parse_diff_hunks(before_file: Path, after_file: Path) -> List[DiffHunk]:
    """Compute diff hunks between before and after files.

    This is a convenience wrapper around compute_diff_hunks_from_files.
    """
    return compute_diff_hunks_from_files(before_file, after_file)


def build_genai_records(
    manifest_path: Path,
    limit: int | None = None,
    sonar_url: str | None = None,
    sonar_token: str | None = None,
    sonar_cache_dir: Path | None = None,
) -> list[dict]:
    """
    Build GenAI evaluation records from manifest.

    Each record has:
    - inputs: {"pair_id": "...", "sonar_issues": [...]}
    - expectations: ground truth data
    - tags: metadata
    """
    records = []

    for pair, _before_path, _after_path, diff_hunks in _iter_valid_rminer_pairs(
        manifest_path, limit
    ):
        types, descriptions = parse_refactoring_info(pair)

        # SonarQube scan
        sonar_issues = []
        if scan_commit and sonar_url and sonar_token:
            try:
                repo_url = pair.get("repository")
                commit_sha = pair.get("commit_sha")
                file_path = pair.get("file_path")

                if repo_url and commit_sha and file_path:
                    issues_map = scan_commit(
                        repo_url=repo_url,
                        commit_sha=commit_sha,
                        sonar_url=sonar_url,
                        sonar_token=sonar_token,
                        cache_dir=sonar_cache_dir,
                    )
                    sonar_issues = issues_map.get(file_path, [])
            except Exception as e:
                print(
                    f"Warning: SonarQube scan failed for {pair['id']}: {e}",
                    file=sys.stderr,
                )

        record = {
            "inputs": {
                "pair_id": pair["id"],
                "sonar_issues": sonar_issues,
            },
            "expectations": {
                "num_refactorings": len(types),
                "num_hunks": len(diff_hunks),
                "diff_hunks": [h.model_dump() for h in diff_hunks],
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

    print(f"Built {len(records)} records")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Create MLflow GenAI dataset")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--experiment", default="rminer-evaluation")
    parser.add_argument("--dataset-name", default="rminer-eval-dataset")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--limit", type=int, help="Limit number of pairs")
    parser.add_argument("--sonar-url", help="SonarQube server URL")
    parser.add_argument("--sonar-token", help="SonarQube authentication token")
    parser.add_argument("--sonar-cache-dir", help="Directory for SonarQube scan cache")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument(
        "--output-json", help="Save records to JSON file (for debugging)"
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    # Resolve SonarQube config
    sonar_url = args.sonar_url or os.getenv("SONAR_HOST_URL")
    sonar_token = args.sonar_token or os.getenv("SONAR_TOKEN")
    sonar_cache_dir = (
        Path(args.sonar_cache_dir)
        if args.sonar_cache_dir
        else Path(os.environ.get("RMINER_MANIFEST_PATH", "rminer_data/manifest.json")).parent / "sonar_cache"
    )

    # Build records
    records = build_genai_records(
        manifest_path,
        limit=args.limit,
        sonar_url=sonar_url,
        sonar_token=sonar_token,
        sonar_cache_dir=sonar_cache_dir,
    )

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

    print(f"\n{'=' * 50}")
    print(f"Dataset created: {args.dataset_name}")
    print(f"Dataset ID: {dataset.dataset_id}")
    print(f"Records: {actual_count}")
    print(f"{'=' * 50}")
    print("\nTo inspect:")
    print(
        f"  uv run cli/datasets/rminer_dataset_cli.py get --id {dataset.dataset_id} --show-records"
    )
    print("\nTo evaluate:")
    print(f"  uv run src/pipelines/rminer_eval.py --dataset-id {dataset.dataset_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
