#!/usr/bin/env python3
"""Extract compound refactorings from SWE-Refactor dataset.

Compound refactorings = multiple operations in one refactoring:
- Extract And Move Method (142 records, 12.9%)
- Move And Rename Method (21 records, 1.9%)
- Move And Inline Method (14 records, 1.3%)

Total: 177 compound refactorings out of 1,099 (16.1%)

Usage:
    # Extract all compound refactorings
    uv run python scripts/extract_compound_refactorings.py

    # Filter by specific type
    uv run python scripts/extract_compound_refactorings.py --type "Extract And Move Method"

    # Filter by project
    uv run python scripts/extract_compound_refactorings.py --project checkstyle

    # Filter by JDK version
    uv run python scripts/extract_compound_refactorings.py --jdk 17

    # Limit output size
    uv run python scripts/extract_compound_refactorings.py --limit 20

    # Custom output path
    uv run python scripts/extract_compound_refactorings.py --output /tmp/compound_subset.json

    # Create MLflow dataset
    uv run python scripts/extract_compound_refactorings.py --create-mlflow-dataset
"""

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

from swe_refactor.dataset import RefactoringRecord, load_swe_refactor_dataset

# Compound refactoring types (contain "And")
COMPOUND_TYPES = [
    "Extract And Move Method",
    "Move And Rename Method",
    "Move And Inline Method",
]


def is_compound(record: RefactoringRecord) -> bool:
    """Check if refactoring is compound (multiple operations)."""
    return "And" in record.type


def extract_from_zip(zip_path: Path) -> list[dict]:
    """Extract records directly from ZIP file."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the JSON file in the ZIP
        json_files = [f for f in zf.namelist() if f.endswith(".json")]
        if not json_files:
            raise ValueError(f"No JSON files found in {zip_path}")

        # Use the first JSON file (should be pure_refactoring_data.json)
        with zf.open(json_files[0]) as f:
            data = json.load(f)
            print(f"Loaded {len(data)} records from {json_files[0]}")
            return data


def print_statistics(records: list[RefactoringRecord]):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    # Type distribution
    type_counts = Counter(r.type for r in records)
    print("\nRefactoring Type Distribution:")
    for refactor_type, count in sorted(
        type_counts.items(), key=lambda x: x[1], reverse=True
    ):
        pct = count / len(records) * 100
        compound_marker = " [COMPOUND]" if "And" in refactor_type else ""
        print(f"  {refactor_type}: {count} ({pct:.1f}%){compound_marker}")

    # Project distribution
    project_counts = Counter(r.projectName for r in records)
    print("\nProject Distribution (top 10):")
    for project, count in project_counts.most_common(10):
        pct = count / len(records) * 100
        print(f"  {project}: {count} ({pct:.1f}%)")

    # JDK distribution
    jdk_counts = Counter(r.compileJDK for r in records)
    print("\nJDK Version Distribution:")
    for jdk, count in sorted(jdk_counts.items()):
        pct = count / len(records) * 100
        print(f"  Java {jdk}: {count} ({pct:.1f}%)")

    # Compound vs atomic
    compound_count = sum(1 for r in records if is_compound(r))
    atomic_count = len(records) - compound_count
    print("\nRefactoring Complexity:")
    print(f"  Compound: {compound_count} ({compound_count/len(records)*100:.1f}%)")
    print(f"  Atomic: {atomic_count} ({atomic_count/len(records)*100:.1f}%)")

    # Compilation stats
    compile_before = sum(1 for r in records if r.compileResultBefore)
    compile_after = sum(1 for r in records if r.compileResultCurrent)
    print("\nCompilation Success Rates:")
    print(f"  Before: {compile_before}/{len(records)} ({compile_before/len(records)*100:.1f}%)")
    print(f"  After: {compile_after}/{len(records)} ({compile_after/len(records)*100:.1f}%)")

    print("=" * 60 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract compound refactorings from SWE-Refactor dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to pure_refactoring_data.json or SWE-Refactor.zip",
    )
    parser.add_argument(
        "--type",
        choices=COMPOUND_TYPES,
        help="Filter by specific compound refactoring type",
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Filter by project name (e.g., checkstyle, guava)",
    )
    parser.add_argument(
        "--jdk",
        type=int,
        choices=[8, 11, 17, 21],
        help="Filter by JDK version",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of records in output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="compound_refactorings.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--create-mlflow-dataset",
        action="store_true",
        help="Create MLflow dataset instead of JSON file",
    )
    parser.add_argument(
        "--mlflow-dataset-name",
        type=str,
        default="compound-refactorings",
        help="MLflow dataset name (used with --create-mlflow-dataset)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics, don't create output",
    )
    args = parser.parse_args()

    # Determine dataset path
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        # Try default locations
        zip_path = Path(__file__).parent.parent / "swe_refactor" / "SWE-Refactor.zip"
        json_path = Path("/tmp/SWE-Refactor/pure_refactoring_data.json")

        if zip_path.exists():
            dataset_path = zip_path
            print(f"Using dataset from: {zip_path}")
        elif json_path.exists():
            dataset_path = json_path
            print(f"Using dataset from: {json_path}")
        else:
            print(
                "Error: Dataset not found. Please extract SWE-Refactor.zip or use --dataset-path",
                file=sys.stderr,
            )
            return 1

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    try:
        if dataset_path.suffix == ".zip":
            # Extract from ZIP
            data = extract_from_zip(dataset_path)
            all_records = [RefactoringRecord.model_validate(r) for r in data]
        else:
            # Load from JSON
            all_records = load_swe_refactor_dataset(str(dataset_path))
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return 1

    print(f"Loaded {len(all_records)} total records")

    # Filter for compound refactorings
    compound_records = [r for r in all_records if is_compound(r)]
    print(f"Found {len(compound_records)} compound refactorings ({len(compound_records)/len(all_records)*100:.1f}%)")

    # Apply additional filters
    filtered_records = compound_records

    if args.type:
        filtered_records = [r for r in filtered_records if r.type == args.type]
        print(f"Filtered by type '{args.type}': {len(filtered_records)} records")

    if args.project:
        filtered_records = [r for r in filtered_records if r.projectName == args.project]
        print(f"Filtered by project '{args.project}': {len(filtered_records)} records")

    if args.jdk:
        filtered_records = [r for r in filtered_records if r.compileJDK == args.jdk]
        print(f"Filtered by JDK {args.jdk}: {len(filtered_records)} records")

    if args.limit:
        filtered_records = filtered_records[: args.limit]
        print(f"Limited to {len(filtered_records)} records")

    if not filtered_records:
        print("No records match the filter criteria", file=sys.stderr)
        return 1

    # Print statistics
    print_statistics(filtered_records)

    if args.stats_only:
        print("Stats-only mode: not creating output file")
        return 0

    # Create output
    if args.create_mlflow_dataset:
        # Save filtered records to temp JSON first
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            output_data = [r.model_dump(mode="json") for r in filtered_records]
            json.dump(output_data, tmp)
            tmp_path = tmp.name

        # Create MLflow dataset
        print(f"\nCreating MLflow dataset '{args.mlflow_dataset_name}'...")
        try:
            from smellai_datasets.converter import swe_refactor_to_df
            from smellai_datasets.mlflow_bridge import hf_to_genai_records
            from mlflow.genai.datasets import create_dataset

            ds = swe_refactor_to_df(tmp_path)
            genai_records = hf_to_genai_records(ds, "swe")
            dataset = create_dataset(
                name=args.mlflow_dataset_name,
                description=f"SWE-Refactor compound: {len(genai_records)} records",
                records=genai_records,
            )
            dataset_id = dataset.dataset_id
            print(f"✅ MLflow dataset created: {args.mlflow_dataset_name}")
            print(f"   Dataset ID: {dataset_id}")
            print(f"   Records: {len(filtered_records)}")
        except Exception as e:
            print(f"Error creating MLflow dataset: {e}", file=sys.stderr)
            return 1
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
    else:
        # Save to JSON file
        output_path = Path(args.output)
        print(f"\nSaving {len(filtered_records)} records to {output_path}...")

        output_data = [r.model_dump(mode="json") for r in filtered_records]

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"✅ Saved to {output_path}")
        print("\nUsage:")
        print(f"  uv run workflows/swe_eval_workflow.py --dataset {output_path} --enable-composite")

    return 0


if __name__ == "__main__":
    sys.exit(main())
