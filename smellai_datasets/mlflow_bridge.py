"""Thin bridge between EvalSample objects and MLflow GenAI evaluate format."""

from __future__ import annotations

import pandas as pd

from .schema import EvalSample


def samples_to_mlflow_records(samples: list[EvalSample]) -> list[dict]:
    """Convert EvalSamples to the MLflow GenAI evaluate() record format.

    Each record is a dict with keys: source, sample_id, inputs, expectations, tags.
    This matches the shape consumed by mlflow.genai.evaluate(data=...).
    """
    return [s.model_dump() for s in samples]


def samples_to_mlflow_df(samples: list[EvalSample]) -> pd.DataFrame:
    """Convert EvalSamples to a MLflow-ready DataFrame.

    Columns: source, sample_id, inputs (dict), expectations (dict), tags (dict).
    """
    return pd.DataFrame([s.model_dump() for s in samples])


__all__ = ["samples_to_mlflow_records", "samples_to_mlflow_df"]
