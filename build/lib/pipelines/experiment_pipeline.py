from __future__ import annotations

import os
from typing import Any, Dict, Tuple
import subprocess

import pandas as pd

from src.connectors import mysql_connector
from src.tracking import init_run, log_metrics, log_artifact_file, finish_run


def load_dataset(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    classes_csv = config.get("classes_csv") or os.getenv("CLASSES_CSV_PATH")
    refactorings_csv = config.get("refactorings_csv") or os.getenv("REFACTORINGS_CSV_PATH")
    df_classes, df_refactorings = mysql_connector.fetch(
        classes_csv_path=classes_csv, refactorings_csv_path=refactorings_csv
    )
    return df_classes, df_refactorings


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def run_experiment(
    df_classes: pd.DataFrame,
    df_refactorings: pd.DataFrame,
    *,
    project: str = None,
    dataset_name: str = "dataset",
    dataset_version: str = "v0",
    connector_name: str = "mysql",
    llm_model: str | None = None,
    temperature: float | None = None,
) -> None:
    config = {
        "git_sha": _git_sha(),
        "dataset_id": dataset_name,
        "dataset_version": dataset_version,
        "connector": connector_name,
        "llm_model": llm_model,
        "temperature": temperature,
    }
    run = init_run(project=project, config=config, job_type="experiment")

    # Example metrics: counts and simple sizes
    log_metrics({
        "classes_rows": int(len(df_classes)),
        "refactorings_rows": int(len(df_refactorings)),
    })

    # Optionally log raw CSVs as artifacts for traceability
    classes_csv = os.getenv("CLASSES_CSV_PATH")
    refactorings_csv = os.getenv("REFACTORINGS_CSV_PATH")
    if classes_csv and os.path.exists(classes_csv):
        log_artifact_file(classes_csv, name=f"{dataset_name}_classes.csv", type="dataset")
    if refactorings_csv and os.path.exists(refactorings_csv):
        log_artifact_file(refactorings_csv, name=f"{dataset_name}_refactorings.csv", type="dataset")

    finish_run()


