"""Unified dataset integration for SmellAI research datasets.

Public API
----------
EvalSample, DatasetSource   — unified evaluation-sample schema
load_swe_raw_df             — raw SWE-Refactor DataFrame (for inspection)
load_rminer_raw_df          — raw RMiner oracle DataFrame (for inspection)
load_eval_samples           — list[EvalSample] from one or more sources
load_eval_df                — MLflow-ready DataFrame
enrich_swe_with_sonar       — optional SonarQube enrichment for SWE samples
samples_to_mlflow_records   — EvalSample → list[dict] for mlflow.genai.evaluate
samples_to_mlflow_df        — EvalSample → pandas DataFrame
"""

from .schema import DatasetSource, EvalSample
from .loaders import (
    load_rminer_raw_df,
    load_swe_raw_df,
    load_eval_samples,
    load_eval_df,
)
from .enrich_sonar import enrich_swe_with_sonar
from .mlflow_bridge import samples_to_mlflow_records, samples_to_mlflow_df

__all__ = [
    "DatasetSource",
    "EvalSample",
    "load_rminer_raw_df",
    "load_swe_raw_df",
    "load_eval_samples",
    "load_eval_df",
    "enrich_swe_with_sonar",
    "samples_to_mlflow_records",
    "samples_to_mlflow_df",
]
