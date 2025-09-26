from __future__ import annotations

from typing import Dict, Tuple, Optional
import os
import pandas as pd


def schema() -> Dict[str, Tuple[str, ...]]:
    """Return connector-specific DataFrame schemas.

    Keys:
      - classes: columns for the classes dataset
      - refactorings: columns for the refactorings dataset
    """
    classes_cols = (
        "class_id",
        "project_url",
        "commit_sha",
        "file_path",
        "code",
    )
    refactorings_cols = (
        "class_id",
        "project_url",
        "commit_sha",
        "file_path",
        "refactored_code",
    )
    return {"classes": classes_cols, "refactorings": refactorings_cols}


def fetch(
    *,
    classes_csv_path: Optional[str] = None,
    refactorings_csv_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch datasets from pre-edited CSV files.

    Paths can be provided directly or via env vars:
      - CLASSES_CSV_PATH
      - REFACTORINGS_CSV_PATH
    """
    classes_path = classes_csv_path or os.getenv("CLASSES_CSV_PATH")
    refactorings_path = refactorings_csv_path or os.getenv("REFACTORINGS_CSV_PATH")

    if not classes_path or not os.path.exists(classes_path):
        raise FileNotFoundError(
            "Missing classes CSV. Set CLASSES_CSV_PATH or pass classes_csv_path."
        )
    if not refactorings_path or not os.path.exists(refactorings_path):
        raise FileNotFoundError(
            "Missing refactorings CSV. Set REFACTORINGS_CSV_PATH or pass refactorings_csv_path."
        )

    df_classes = pd.read_csv(classes_path)
    df_refactorings = pd.read_csv(refactorings_path)

    # Ensure required columns exist; if missing, create empty columns
    sch = schema()
    for col in sch["classes"]:
        if col not in df_classes.columns:
            df_classes[col] = pd.Series(dtype="object")
    for col in sch["refactorings"]:
        if col not in df_refactorings.columns:
            df_refactorings[col] = pd.Series(dtype="object")

    return df_classes[sch["classes"]], df_refactorings[sch["refactorings"]]


