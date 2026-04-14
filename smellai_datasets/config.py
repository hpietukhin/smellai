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
        "dedup_keys": ["project", "creation_commit", "rule", "component", "start_line"],
        "default_filters": {},
        "stratify_col": "rule",
    },
    "planner": {
        "dedup_keys": ["commit_sha"],
        "default_filters": {},
        "stratify_col": "first_refactoring_type",
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
    "planner": {
        "input_cols": ["repository", "commit_sha", "smell_set_s0"],
        "expectation_cols": [
            "refactorings_json",
            "first_refactoring_type",
            "first_refactoring_class",
            "refactoring_count",
        ],
        "tag_cols": ["smell_count_s0", "smell_relevant_types"],
    },
}
