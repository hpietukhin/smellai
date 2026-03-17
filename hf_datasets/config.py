"""Dataset preprocessing and MLflow column mapping configuration."""

from __future__ import annotations

DATASET_CONFIGS = {
    "rminer": {
        "dedup_keys": ["commit_sha", "refactoring_type", "description"],
        "default_filters": {"validation": "TP"},
        "stratify_col": "refactoring_type",
    },
    "swe": {
        "dedup_keys": ["pair_id"],
        "default_filters": {},
        "stratify_col": "refactoring_type",
    },
    "tdd": {
        "dedup_keys": ["commit_sha", "smell_type", "file_path"],
        "default_filters": {},
        "stratify_col": "smell_type",
    },
}

MLFLOW_COLUMN_MAP = {
    "swe": {
        "input_cols": [
            "project_name",
            "commit_id",
            "refactoring_type",
            "source_before",
            "class_before",
            "file_path_before",
            "file_path_after",
            "jdk_version",
            "compile_command",
        ],
        "expectation_cols": ["source_after", "class_after"],
        "tag_cols": ["has_tests", "is_compound", "is_pure"],
    },
    "rminer": {
        "input_cols": [
            "commit_sha",
            "repository",
            "refactoring_type",
            "description",
        ],
        "expectation_cols": [],
        "tag_cols": ["validation", "detection_tools"],
    },
}
