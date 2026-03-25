# smellai_datasets

Pandas-based dataset pipeline for converting, preprocessing, and exporting research datasets to MLflow GenAI format.

## Pipeline

```
Raw source (JSON/SQLite/ZIP) --> converter --> pd.DataFrame --> preprocessor --> Parquet --> mlflow_bridge --> MLflow GenAI records
```

## Key Files

- **converter.py** - Raw-to-DataFrame converters: `rminer_to_df`, `swe_refactor_to_df`, `tdd_to_df`, `rminer_planner_to_df`
- **preprocessor.py** - Deduplication, train/val/test splitting, filtering, Parquet I/O
- **mlflow_bridge.py** - DataFrame-to-MLflow GenAI format (`hf_to_genai_records`, `load_for_evaluation`)
- **config.py** - `DATASET_CONFIGS` (dedup keys, filters) and `MLFLOW_COLUMN_MAP` (column mappings per source)
- **models.py** - Shared Pydantic models (`DiffHunk`)

## Usage

```python
from smellai_datasets import rminer_to_df, deduplicate, save, hf_to_genai_records

# Convert raw data to DataFrame
df = rminer_to_df("path/to/data.json")

# Preprocess
df = deduplicate(df, key_cols=["commit_sha", "refactoring_type", "description"])

# Save as Parquet
save(df, "data/processed/rminer")

# Export to MLflow GenAI format
records = hf_to_genai_records(df, source="rminer")
```

## Supported Datasets

| Source | Converter | Input Format |
|--------|-----------|--------------|
| RMiner 2.0 oracle | `rminer_to_df` | JSON (data.json) |
| RMiner planner | `rminer_planner_to_df` | JSON (data.json, per-commit) |
| SWE-Refactor | `swe_refactor_to_df` | ZIP, JSON, or directory |
| TDD v2 (Lenarduzzi) | `tdd_to_df` | SQLite (td_V2.db) |
