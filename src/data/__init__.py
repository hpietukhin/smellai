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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
