#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlflow>=3.3",
#     "python-dotenv",
#     "requests"
# ]
# ///
"""Create MLflow GenAI evaluation dataset from RefactoringMiner pairs with SonarQube smell enrichment.

This script creates a dataset with:
- inputs: pair_id (passed to predict_fn)
- expectations: ground truth (diff_hunks, refactoring metadata, code smells)
- tags: repository, commit info

The key enhancement is SonarQube integration:
1. For each pair, checkout the parent commit
2. Run SonarQube analysis on the "before" state
3. Map detected smells to the lines that will be refactored
4. Add smell metadata to dataset records

Usage:
    # Create dataset with smell enrichment
    uv run infra/mlflow/rminer_dataset.py --manifest rminer_data/manifest.json --enable-sonar

    # Dry run (preview records)
    uv run infra/mlflow/rminer_dataset.py --manifest rminer_data/manifest.json --dry-run

    # Skip SonarQube scanning
    uv run infra/mlflow/rminer_dataset.py --manifest rminer_data/manifest.json

    # Limit number of pairs
    uv run infra/mlflow/rminer_dataset.py --manifest rminer_data/manifest.json --limit 10 --enable-sonar
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

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


def map_smells_to_hunks(
    issues: List[Dict[str, Any]], hunks: List[DiffHunk]
) -> List[Dict[str, Any]]:
    """
    Map SonarQube issues to diff hunks based on line numbers.

    Returns list of smells that occur in lines that will be refactored.
    """
    mapped_smells = []

    for issue in issues:
        issue_line = issue.get("line")
        if not issue_line:
            continue

        # Check if issue line falls within any hunk's "before" range
        for hunk_idx, hunk in enumerate(hunks):
            hunk_start = hunk.old_start
            hunk_end = hunk.old_start + hunk.old_count

            if hunk_start <= issue_line < hunk_end:
                mapped_smells.append(
                    {
                        "smell_type": issue.get("smell_type"),
                        "line": issue_line,
                        "severity": issue.get("severity"),
                        "message": issue.get("message"),
                        "rule": issue.get("rule"),
                        "hunk_index": hunk_idx,
                        "hunk_old_start": hunk.old_start,
                        "hunk_old_count": hunk.old_count,
                    }
                )
                break

    return mapped_smells


def enrich_with_sonarqube(
    pair: dict,
    hunks: List[DiffHunk],
    sonar_url: str,
    sonar_token: str,
    cache_dir: Optional[Path] = None,
    use_docker: bool = True,
) -> Optional[List[Dict[str, Any]]]:
    """
    Run SonarQube analysis on parent commit and map smells to hunks.

    Returns list of mapped smells or None if scanning fails/disabled.
    """
    try:
        # Import here to avoid dependency if not using SonarQube
        from infra.sonarqube.commit_scan import scan_commit

        repo_url = pair.get("repository")
        parent_sha = pair.get("parent_sha")
        file_path = pair.get("file_path")

        if not repo_url or not parent_sha or not file_path:
            return None

        # Scan entire commit (cached)
        issues_by_file = scan_commit(
            repo_url=repo_url,
            commit_sha=parent_sha,
            sonar_url=sonar_url,
            sonar_token=sonar_token,
            cache_dir=cache_dir,
            use_docker=use_docker,
        )

        # Get issues for this specific file
        file_issues = issues_by_file.get(file_path, [])

        if not file_issues:
            return []

        # Map issues to hunks
        mapped_smells = map_smells_to_hunks(file_issues, hunks)

        return mapped_smells

    except Exception as e:
        print(
            f"Warning: SonarQube enrichment failed for {pair.get('id')}: {e}",
            file=sys.stderr,
        )
        return None


def build_genai_records(
    manifest_path: Path,
    limit: int | None = None,
    enable_sonar: bool = False,
    sonar_url: Optional[str] = None,
    sonar_token: Optional[str] = None,
    sonar_cache_dir: Optional[Path] = None,
    use_docker: bool = True,
) -> list[dict]:
    """
    Build GenAI evaluation records from manifest.

    Each record has:
    - inputs: {"pair_id": "..."}
    - expectations: ground truth data (including optional smell mappings)
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
    sonar_enriched = 0

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

        # Build base record
        expectations = {
            "num_refactorings": len(types),
            "num_hunks": len(diff_hunks),
            "diff_hunks": [h.to_dict() for h in diff_hunks],
            "refactoring_types": types,
            "refactoring_descriptions": descriptions,
            "file_path": pair["file_path"],
        }

        # Enrich with SonarQube if enabled
        if enable_sonar and sonar_url and sonar_token:
            mapped_smells = enrich_with_sonarqube(
                pair=pair,
                hunks=diff_hunks,
                sonar_url=sonar_url,
                sonar_token=sonar_token,
                cache_dir=sonar_cache_dir,
                use_docker=use_docker,
            )

            if mapped_smells is not None:
                expectations["code_smells"] = mapped_smells
                expectations["num_smells"] = len(mapped_smells)
                sonar_enriched += 1
            else:
                expectations["code_smells"] = []
                expectations["num_smells"] = 0
        else:
            expectations["code_smells"] = []
            expectations["num_smells"] = 0

        record = {
            "inputs": {
                "pair_id": pair["id"],
            },
            "expectations": expectations,
            "tags": {
                "repository": pair.get("repository", ""),
                "commit_sha": pair.get("commit_sha", ""),
                "parent_sha": pair.get("parent_sha", ""),
                "status": pair.get("status", "modified"),
                "sonar_enriched": str(
                    enable_sonar and expectations.get("num_smells", 0) > 0
                ),
            },
        }
        records.append(record)

    print(f"Built {len(records)} records ({skipped} skipped)")
    if enable_sonar:
        print(
            f"SonarQube enrichment: {sonar_enriched}/{len(records)} records have smell data"
        )

    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create MLflow GenAI dataset with SonarQube enrichment"
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--experiment", default="rminer-evaluation")
    parser.add_argument("--dataset-name", default="rminer-eval-dataset")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--limit", type=int, help="Limit number of pairs")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument(
        "--output-json", help="Save records to JSON file (for debugging)"
    )

    # SonarQube options
    parser.add_argument(
        "--enable-sonar", action="store_true", help="Enable SonarQube smell enrichment"
    )
    parser.add_argument(
        "--sonar-url", default=None, help="SonarQube URL (default: SONAR_URL env)"
    )
    parser.add_argument(
        "--sonar-token", default=None, help="SonarQube token (default: SONAR_TOKEN env)"
    )
    parser.add_argument(
        "--sonar-cache-dir",
        default=".sonar_cache",
        help="Cache directory for SonarQube results",
    )
    parser.add_argument(
        "--local-scanner",
        action="store_true",
        help="Use local sonar-scanner instead of Docker",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    # SonarQube configuration
    sonar_url = args.sonar_url or os.environ.get("SONAR_URL", "http://localhost:9000")
    sonar_token = args.sonar_token or os.environ.get("SONAR_TOKEN")

    if args.enable_sonar and not sonar_token:
        print(
            "Error: --enable-sonar requires SONAR_TOKEN environment variable or --sonar-token",
            file=sys.stderr,
        )
        return 1

    sonar_cache_dir = Path(args.sonar_cache_dir) if args.enable_sonar else None

    # Build records
    records = build_genai_records(
        manifest_path,
        limit=args.limit,
        enable_sonar=args.enable_sonar,
        sonar_url=sonar_url,
        sonar_token=sonar_token,
        sonar_cache_dir=sonar_cache_dir,
        use_docker=not args.local_scanner,
    )

    if not records:
        print("No valid records found", file=sys.stderr)
        return 1

    # Stats
    total_hunks = sum(r["expectations"]["num_hunks"] for r in records)
    total_refactorings = sum(r["expectations"]["num_refactorings"] for r in records)
    total_smells = sum(r["expectations"]["num_smells"] for r in records)

    print(f"Total hunks: {total_hunks}")
    print(f"Total refactorings: {total_refactorings}")
    if args.enable_sonar:
        print(f"Total code smells mapped: {total_smells}")

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
        if sample["expectations"]["code_smells"]:
            sample["expectations"]["code_smells"] = sample["expectations"][
                "code_smells"
            ][:2] + ["..."]
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
    dataset_tags = {
        "source": "RefactoringMiner",
        "total_pairs": str(len(records)),
        "total_hunks": str(total_hunks),
        "total_refactorings": str(total_refactorings),
    }

    if args.enable_sonar:
        dataset_tags["sonar_enriched"] = "true"
        dataset_tags["total_smells"] = str(total_smells)

    dataset = create_dataset(
        name=args.dataset_name,
        experiment_id=[experiment_id],
        tags=dataset_tags,
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
    if args.enable_sonar:
        print(f"SonarQube enriched: {total_smells} smells mapped")
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
