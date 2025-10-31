#!/usr/bin/env python3
"""CLI utility that materialises a DACOS-backed MLflow eval dataset.

The script connects to the DACOS MySQL catalogue, converts the selected samples
into the schema expected by ``mlflow.genai.datasets`` and appends the resulting
records to a named dataset. It replicates the prompt/expectation layout used by
``src.pipelines.react_agent_mlflow`` so the generated dataset can be consumed by
that evaluation pipeline without further massaging.

Typical usage (with uv):

    uv run python scripts/create_dacos_mlflow_dataset.py --limit 20 \
        --dataset-name dacos-evals --experiment-name react-agent-mlflow

Make sure your ``.env`` file contains both the MySQL DACOS credentials and the
MLflow tracking configuration. See ``src/data/mysql_connector.py`` for the
required database environment variables.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.genai.datasets import create_dataset

from src.data.mysql_connector import (
    fetch_balanced_smell_samples,
    fetch_sample_by_id,
    fetch_samples,
    fetch_samples_dataframe,
    get_connection_pool,
)
from src.models.entities import DACOSSample

LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET_TAGS: Dict[str, str] = {
    "source": "dacos",
    "dataset_type": "mlflow-genai",
}


def _parse_tag(tag_pair: str) -> Tuple[str, str]:
    """Split ``KEY=VALUE`` pairs supplied via the CLI."""

    if "=" not in tag_pair:
        raise argparse.ArgumentTypeError("Tags must be provided as KEY=VALUE")

    key, value = tag_pair.split("=", 1)
    key = key.strip()
    value = value.strip()

    if not key:
        raise argparse.ArgumentTypeError("Tag key cannot be empty")

    return key, value


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _resolve_smell_name(sample: DACOSSample) -> str:
    smell_name = sample.smell_name
    if smell_name:
        return smell_name

    smells = sample.ground_truth_smells()
    if smells:
        return smells[0]

    return "Unknown smell"


def _sample_to_prompt(sample: DACOSSample, *, smell_override: Optional[str] = None) -> str:
    smell_name = smell_override or _resolve_smell_name(sample)
    smell_description = sample.smell_description or ""
    constraints = sample.sample_constraints or ""

    prompt_lines = [
        f"We are reviewing DACOS sample {sample.id}.",
        f"Repository slug: {sample.project_name}.",
        f"File: {sample.path_to_file}.",
        f"Annotated smell: {smell_name}.",
    ]

    if smell_description:
        prompt_lines.append(f"Smell description: {smell_description}.")
    if constraints:
        prompt_lines.append(f"Additional notes: {constraints}.")

    prompt_lines.extend(
        [
            "Use the available DACOS tools to ground your reasoning before answering.",
            "Provide a concise refactoring plan that directly addresses the annotated smell.",
        ]
    )

    return " \n".join(prompt_lines)


def _build_expectations(sample: DACOSSample, *, smell_override: Optional[str] = None) -> Dict[str, str]:
    smell_name = smell_override or _resolve_smell_name(sample)
    return {
        "sample_id": str(sample.id),
        "smell_name": smell_name,
        "smell_description": sample.smell_description or "",
    }


def _sample_to_record(sample: DACOSSample, *, smell_override: Optional[str] = None) -> Dict[str, str]:
    smell_name = smell_override or _resolve_smell_name(sample)
    expectations = _build_expectations(sample, smell_override=smell_override)
    ground_truth = sample.ground_truth_smells()

    record: Dict[str, str] = {
        "inputs.inputs": _sample_to_prompt(sample, smell_override=smell_override),
        "expectations.sample_id": expectations["sample_id"],
        "expectations.smell_name": expectations["smell_name"],
        "expectations.smell_description": expectations["smell_description"],
        "tags.smell_name": smell_name,
        "metadata.project_name": sample.project_name,
        "metadata.path_to_file": sample.path_to_file,
        "metadata.has_smell": str(int(sample.has_smell)),
        "metadata.is_class": str(int(sample.is_class)),
        "metadata.sample_id": str(sample.id),
        "metadata.designite_id": (
            str(sample.designite_id) if sample.designite_id is not None else ""
        ),
        "metadata.ground_truth_smells": ", ".join(ground_truth) if ground_truth else "",
    }

    if sample.sample_constraints:
        record["metadata.sample_constraints"] = str(sample.sample_constraints)
    if sample.smells:
        record["metadata.smell_ids"] = str(sample.smells)
    if sample.repo_url:
        record["metadata.repo_url"] = sample.repo_url
    if sample.commit_sha:
        record["metadata.commit_sha"] = sample.commit_sha

    return record


def _load_samples_from_ids(sample_ids: Iterable[int]) -> List[DACOSSample]:
    loaded: List[DACOSSample] = []
    for sample_id in sample_ids:
        sample = fetch_sample_by_id(int(sample_id))
        if sample is None:
            LOGGER.warning("Sample %s not found; skipping", sample_id)
            continue
        loaded.append(sample)
    return loaded


def _gather_samples(
    *,
    sample_ids: Optional[Sequence[int]] = None,
    smell_ids: Optional[Sequence[int]] = None,
    project_name: Optional[str] = None,
    include_non_smelly: bool = False,
    limit: int = 25,
    balanced_per_smell: Optional[int] = None,
) -> List[Tuple[DACOSSample, Optional[str]]]:
    get_connection_pool()  # validates environment early

    entries: List[Tuple[DACOSSample, Optional[str]]]

    if sample_ids:
        samples = _load_samples_from_ids(sample_ids)
        entries = [(sample, None) for sample in samples]
    elif smell_ids:
        df = fetch_samples_dataframe(smell_ids=list(smell_ids), limit=limit)
        if df.empty:
            LOGGER.warning("No DACOS rows matched smell IDs %s", smell_ids)
            return []
        sampled_ids = df["id"].dropna().astype(int).tolist()
        samples = _load_samples_from_ids(sampled_ids)
        entries = [(sample, None) for sample in samples]
    elif balanced_per_smell:
        entries = fetch_balanced_smell_samples(per_smell=balanced_per_smell)
    else:
        samples = fetch_samples(
            project_name=project_name,
            has_smell=None if include_non_smelly else True,
            limit=limit,
        )
        entries = [(sample, None) for sample in samples]

    if project_name:
        entries = [
            (sample, smell_label)
            for sample, smell_label in entries
            if sample.project_name == project_name
        ]

    if limit and not balanced_per_smell and len(entries) > limit:
        entries = entries[:limit]

    return entries


def _samples_to_dataframe(entries: Sequence[Tuple[DACOSSample, Optional[str]]]) -> pd.DataFrame:
    records = [
        _sample_to_record(sample, smell_override=smell_label)
        for sample, smell_label in entries
    ]
    return pd.DataFrame.from_records(records)


def _materialise_dataset(
    *,
    dataset_name: str,
    experiment_identifier: str,
    dataset_tags: Dict[str, str],
    dataset_description: Optional[str],
    records_df: pd.DataFrame,
    dry_run: bool,
) -> None:
    if records_df.empty:
        raise ValueError("No records to merge; dataset creation aborted.")

    LOGGER.info("Preparing dataset '%s' with %s rows", dataset_name, len(records_df))

    if dry_run:
        LOGGER.info("Dry run enabled; dataset will not be modified.")
        LOGGER.info("Preview:\n%s", records_df.head())
        return

    dataset = create_dataset(
        name=dataset_name,
        experiment_id=[experiment_identifier],
        tags=dataset_tags,
        description=dataset_description,
    )

    dataset.merge_records(records_df)

    LOGGER.info(
        "Dataset '%s' updated with %s records (experiment_id=%s)",
        dataset.name,
        len(records_df),
        experiment_identifier,
    )


def _resolve_experiment_id(
    *,
    experiment_id: Optional[str],
    experiment_name: Optional[str],
) -> str:
    if experiment_id:
        return str(experiment_id)

    if not experiment_name:
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "react-agent-mlflow")

    experiment = mlflow.set_experiment(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Unable to set MLflow experiment '{experiment_name}'")

    return experiment.experiment_id


def _configure_mlflow(tracking_uri: Optional[str]) -> None:
    effective_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if effective_uri:
        mlflow.set_tracking_uri(effective_uri)
        LOGGER.info("Using MLflow tracking URI: %s", effective_uri)
    else:
        LOGGER.info("MLFLOW_TRACKING_URI not set; falling back to MLflow defaults")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--dataset-name", default="dacos-eval-dataset", help="MLflow dataset name")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI override")
    parser.add_argument("--experiment-id", default=None, help="Target MLflow experiment ID")
    parser.add_argument("--experiment-name", default=None, help="Target MLflow experiment name (ignored if --experiment-id provided)")
    parser.add_argument("--description", default=None, help="Optional dataset description")
    parser.add_argument("--limit", type=int, default=25, help="Maximum number of samples")
    parser.add_argument("--project-name", default=None, help="Filter samples by DACOS project slug")
    parser.add_argument("--include-non-smelly", action="store_true", help="Allow samples without smell annotations")
    parser.add_argument("--sample-id", dest="sample_ids", type=int, nargs="*", help="Explicit DACOS sample IDs")
    parser.add_argument("--smell-id", dest="smell_ids", type=int, nargs="*", help="Filter by DACOS smell IDs")
    parser.add_argument("--balanced", dest="balanced_per_smell", type=int, default=None, help="Fetch a balanced batch per smell (overrides --limit)")
    parser.add_argument("--tag", dest="tags", type=_parse_tag, action="append", default=[], help="Additional dataset tag (KEY=VALUE). Repeatable.")
    parser.add_argument("--dry-run", action="store_true", help="Show the planned dataset without writing to MLflow")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    _configure_logging(args.verbose)
    load_dotenv()

    _configure_mlflow(args.tracking_uri)
    experiment_id = _resolve_experiment_id(
        experiment_id=args.experiment_id,
        experiment_name=args.experiment_name,
    )

    dataset_tags = DEFAULT_DATASET_TAGS.copy()
    dataset_tags.update(dict(args.tags))

    if args.project_name:
        dataset_tags.setdefault("project", args.project_name)

    entries = _gather_samples(
        sample_ids=args.sample_ids,
        smell_ids=args.smell_ids,
        project_name=args.project_name,
        include_non_smelly=args.include_non_smelly,
        limit=args.limit,
        balanced_per_smell=args.balanced_per_smell,
    )

    if not entries:
        raise SystemExit("No DACOS samples matched the provided filters.")

    records_df = _samples_to_dataframe(entries)

    if args.verbose:
        LOGGER.debug("Generated dataframe:\n%s", records_df)

    _materialise_dataset(
        dataset_name=args.dataset_name,
        experiment_identifier=experiment_id,
        dataset_tags=dataset_tags,
        dataset_description=args.description,
        records_df=records_df,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
