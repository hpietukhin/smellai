"""Bridge between pandas DataFrames and MLflow GenAI evaluation format.

Converts DataFrame rows into MLflow GenAI evaluate() records using
column mappings defined in config.py.
"""

from __future__ import annotations

import pandas as pd

from .config import MLFLOW_COLUMN_MAP
from .preprocessor import load


def hf_to_genai_records(df: pd.DataFrame, source: str) -> list[dict]:
    """Convert DataFrame rows to MLflow GenAI evaluate() format.

    Args:
        df: pandas DataFrame (flat rows from converter)
        source: Dataset source key ("swe" or "rminer")

    Returns:
        List of dicts with keys: inputs, expectations, tags
    """
    col_map = MLFLOW_COLUMN_MAP.get(source)
    if col_map is None:
        raise ValueError(
            f"Unknown source {source!r}. Available: {list(MLFLOW_COLUMN_MAP)}"
        )

    input_cols = col_map["input_cols"]
    expectation_cols = col_map["expectation_cols"]
    tag_cols = col_map["tag_cols"]

    records: list[dict] = []
    for row in df.to_dict("records"):
        record = {
            "inputs": {col: row.get(col) for col in input_cols if col in row},
            "expectations": {col: row.get(col) for col in expectation_cols if col in row},
            "tags": {col: str(row.get(col, "")) for col in tag_cols if col in row},
        }
        records.append(record)

    return records


def load_for_evaluation(path: str, source: str) -> list[dict]:
    """Load a preprocessed dataset from disk and convert to MLflow format.

    Args:
        path: Directory written by preprocessor.save()
        source: Dataset source key ("swe" or "rminer")

    Returns:
        List of MLflow GenAI records
    """
    result = load(path)
    if isinstance(result, dict):
        df = pd.concat(result.values(), ignore_index=True)
    else:
        df = result
    return hf_to_genai_records(df, source)
