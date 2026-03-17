"""Data access layer for code smell detection."""

# Lazy imports to avoid dependency issues
__all__ = [
    "get_connection_pool",
    "get_connection",
    "fetch_sample_by_id",
    "fetch_samples",
    "fetch_samples_dataframe",
    "derive_repo_url",
    "get_commit_before_date",
    "clone_and_read_file",
    "load_rminer_data",
    "group_by_repository",
    "find_consecutive_commits",
    "calculate_semantic_similarity",
    "compute_statistics",
    "parse_diff_hunks",
    "compute_diff_hunks_from_files",
    "DiffHunk",
    "fetch_full_refactoring_json",
    "build_dependency_graph",
]


def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies."""
    if name in [
        "get_connection_pool",
        "get_connection",
        "fetch_sample_by_id",
        "fetch_samples",
        "fetch_samples_dataframe",
    ]:
        from .mysql_connector import (  # noqa: F401
            fetch_sample_by_id,
            fetch_samples,
            fetch_samples_dataframe,
            get_connection,
            get_connection_pool,
        )

        return locals()[name]
    elif name in ["derive_repo_url", "get_commit_before_date", "clone_and_read_file"]:
        from .git_ops import (  # noqa: F401
            clone_and_read_file,
            derive_repo_url,
            get_commit_before_date,
        )

        return locals()[name]
    elif name == "DiffHunk":
        from smellai_datasets.models import DiffHunk  # noqa: F401

        return DiffHunk
    elif name in [
        "load_rminer_data",
        "group_by_repository",
        "find_consecutive_commits",
        "calculate_semantic_similarity",
        "compute_statistics",
        "parse_diff_hunks",
        "compute_diff_hunks_from_files",
        "fetch_full_refactoring_json",
        "build_dependency_graph",
    ]:
        from .rminer_utils import (  # noqa: F401
            build_dependency_graph,
            calculate_semantic_similarity,
            compute_diff_hunks_from_files,
            compute_statistics,
            fetch_full_refactoring_json,
            find_consecutive_commits,
            group_by_repository,
            load_rminer_data,
            parse_diff_hunks,
        )

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
