#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["mlflow>=3.3", "python-dotenv", "tabulate"]
# ///
"""List and inspect MLflow GenAI datasets.

Since MLflow CLI doesn't have dataset commands, this script provides
utilities to manage GenAI datasets.

Usage:
    # List all datasets
    uv run infra/mlflow/rminer_datasets_cli.py list

    # List datasets for specific experiment
    uv run infra/mlflow/rminer_datasets_cli.py list --experiment rminer-evaluation

    # Get dataset details
    uv run infra/mlflow/rminer_datasets_cli.py get --name rminer-eval-dataset

    # Get dataset by ID
    uv run infra/mlflow/rminer_datasets_cli.py get --id d-abc123

    # Export dataset to JSON
    uv run infra/mlflow/rminer_datasets_cli.py export --name rminer-eval-dataset -o dataset.json

    # Delete dataset
    uv run infra/mlflow/rminer_datasets_cli.py delete --name rminer-eval-dataset
"""

from __future__ import annotations

import argparse
import json
import sys

import mlflow
from dotenv import load_dotenv

load_dotenv()


def cmd_list(args):
    """List all datasets."""
    from mlflow.genai.datasets import search_datasets

    experiment_ids = None
    if args.experiment:
        exp = mlflow.get_experiment_by_name(args.experiment)
        if exp:
            experiment_ids = [exp.experiment_id]
        else:
            print(f"Experiment not found: {args.experiment}", file=sys.stderr)
            return 1

    datasets = search_datasets(experiment_ids=experiment_ids)

    if not datasets:
        print("No datasets found.")
        return 0

    if args.json:
        output = []
        for ds in datasets:
            try:
                record_count = len(ds.records)
            except Exception:
                record_count = "N/A"
            output.append(
                {
                    "dataset_id": ds.dataset_id,
                    "name": ds.name,
                    "experiment_ids": ds.experiment_ids,
                    "tags": ds.tags,
                    "record_count": record_count,
                }
            )
        print(json.dumps(output, indent=2))
    else:
        try:
            from tabulate import tabulate

            rows = []
            for ds in datasets:
                try:
                    record_count = len(ds.records)
                except Exception:
                    record_count = "?"
                rows.append(
                    [
                        ds.dataset_id,
                        ds.name,
                        record_count,
                        ", ".join(ds.experiment_ids) if ds.experiment_ids else "",
                    ]
                )
            print(
                tabulate(
                    rows,
                    headers=["ID", "Name", "Records", "Experiments"],
                    tablefmt="simple",
                )
            )
        except ImportError:
            for ds in datasets:
                try:
                    record_count = len(ds.records)
                except Exception:
                    record_count = "?"
                print(f"ID: {ds.dataset_id}")
                print(f"  Name: {ds.name}")
                print(f"  Records: {record_count}")
                print(f"  Experiments: {ds.experiment_ids}")
                print(f"  Tags: {ds.tags}")
                print()

    return 0


def cmd_get(args):
    """Get dataset details."""
    from mlflow.genai.datasets import get_dataset, search_datasets

    try:
        if args.id:
            dataset = get_dataset(dataset_id=args.id)
        elif args.name:
            # OSS MLflow doesn't support get by name - search and filter instead
            all_datasets = search_datasets()
            matching = [ds for ds in all_datasets if ds.name == args.name]
            if not matching:
                print(f"Dataset not found with name: {args.name}", file=sys.stderr)
                print("\nAvailable datasets:", file=sys.stderr)
                for ds in all_datasets:
                    print(f"  - {ds.name} (ID: {ds.dataset_id})", file=sys.stderr)
                return 1
            dataset = matching[0]
            if len(matching) > 1:
                print(
                    f"Warning: Multiple datasets with name '{args.name}', using first one.",
                    file=sys.stderr,
                )
        else:
            print("Either --name or --id is required", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"Dataset not found: {e}", file=sys.stderr)
        return 1

    print(f"Dataset ID: {dataset.dataset_id}")
    print(f"Name: {dataset.name}")
    print(f"Experiment IDs: {dataset.experiment_ids}")
    print(f"Tags: {dataset.tags}")
    print(f"Records: {len(dataset.records)}")

    if args.show_records:
        print("\n--- Records ---")
        for i, record in enumerate(dataset.records[: args.limit]):
            # DatasetRecord objects have inputs, expectations, and tags as attributes
            inputs = record.inputs if hasattr(record, "inputs") else {}
            expectations = (
                record.expectations if hasattr(record, "expectations") else {}
            )

            print(f"\n[{i}] inputs: {json.dumps(inputs, indent=2)}")

            # Truncate large fields for display
            display_exp = {}
            for k, v in expectations.items():
                if isinstance(v, str) and len(v) > 100:
                    display_exp[k] = v[:100] + "..."
                elif isinstance(v, list) and len(v) > 3:
                    display_exp[k] = v[:3] + ["..."]
                else:
                    display_exp[k] = v
            print(f"    expectations: {json.dumps(display_exp, indent=2)}")

    if args.schema:
        print("\n--- Schema ---")
        print(json.dumps(dataset.schema, indent=2))

    return 0


def cmd_export(args):
    """Export dataset to JSON."""
    from mlflow.genai.datasets import get_dataset, search_datasets

    try:
        if args.id:
            dataset = get_dataset(dataset_id=args.id)
        elif args.name:
            # OSS MLflow doesn't support get by name - search and filter
            all_datasets = search_datasets()
            matching = [ds for ds in all_datasets if ds.name == args.name]
            if not matching:
                print(f"Dataset not found with name: {args.name}", file=sys.stderr)
                return 1
            dataset = matching[0]
        else:
            print("Either --name or --id is required", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"Dataset not found: {e}", file=sys.stderr)
        return 1

    output = {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "experiment_ids": dataset.experiment_ids,
        "tags": dataset.tags,
        "records": dataset.records,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Exported {len(dataset.records)} records to {args.output}")
    else:
        print(json.dumps(output, indent=2))

    return 0


def cmd_delete(args):
    """Delete a dataset."""
    from mlflow.genai.datasets import get_dataset, search_datasets

    try:
        if args.id:
            dataset = get_dataset(dataset_id=args.id)
        elif args.name:
            # OSS MLflow doesn't support get by name - search and filter
            all_datasets = search_datasets()
            matching = [ds for ds in all_datasets if ds.name == args.name]
            if not matching:
                print(f"Dataset not found with name: {args.name}", file=sys.stderr)
                return 1
            dataset = matching[0]
        else:
            print("Either --name or --id is required", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"Dataset not found: {e}", file=sys.stderr)
        return 1

    if not args.force:
        confirm = input(
            f"Delete dataset '{dataset.name}' ({dataset.dataset_id})? [y/N] "
        )
        if confirm.lower() != "y":
            print("Aborted.")
            return 0

    # Note: MLflow GenAI datasets API might not have delete - check documentation
    try:
        # This might not exist - MLflow may not support dataset deletion
        dataset.delete()
        print(f"Deleted dataset: {dataset.name}")
    except AttributeError:
        print(
            "Dataset deletion is not supported by MLflow GenAI datasets API.",
            file=sys.stderr,
        )
        print("You may need to delete directly from the database.", file=sys.stderr)
        return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="MLflow GenAI Datasets CLI")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    list_parser = subparsers.add_parser("list", help="List datasets")
    list_parser.add_argument("--experiment", help="Filter by experiment name")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # get
    get_parser = subparsers.add_parser("get", help="Get dataset details")
    get_parser.add_argument("--name", help="Dataset name")
    get_parser.add_argument("--id", help="Dataset ID")
    get_parser.add_argument("--show-records", action="store_true", help="Show records")
    get_parser.add_argument("--limit", type=int, default=5, help="Limit records shown")
    get_parser.add_argument("--schema", action="store_true", help="Show schema")

    # export
    export_parser = subparsers.add_parser("export", help="Export dataset to JSON")
    export_parser.add_argument("--name", help="Dataset name")
    export_parser.add_argument("--id", help="Dataset ID")
    export_parser.add_argument("-o", "--output", help="Output file")

    # delete
    delete_parser = subparsers.add_parser("delete", help="Delete dataset")
    delete_parser.add_argument("--name", help="Dataset name")
    delete_parser.add_argument("--id", help="Dataset ID")
    delete_parser.add_argument(
        "--force", "-f", action="store_true", help="Skip confirmation"
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    commands = {
        "list": cmd_list,
        "get": cmd_get,
        "export": cmd_export,
        "delete": cmd_delete,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
