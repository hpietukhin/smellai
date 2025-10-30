"""Tooling used by the LangGraph ReAct agent.

Provides typed helpers that give the model access to the DACOS dataset via the
shared MySQL connection utilities and optional Git-based code retrieval.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.data.git_ops import (
    clone_and_read_file,
    derive_repo_url,
    get_commit_before_date,
)
from src.data.mysql_connector import (
    fetch_sample_by_id,
    fetch_samples_dataframe,
    get_connection_pool,
)
from src.models.entities import DACOSSample


class SamplesArgs(BaseModel):
    """Input schema for the `load_dacos_samples` tool."""

    limit: int = Field(
        ...,
        ge=1,
        le=200,
        description="Maximum number of DACOS samples to fetch (1-200).",
    )
    smell_ids: Optional[List[int]] = Field(
        default=None,
        description=(
            "Optional list of smell identifiers from `tagman5.smell` to filter by."
        ),
    )


@tool("load_dacos_samples", args_schema=SamplesArgs)
def load_dacos_samples(limit: int, smell_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """Return DACOS samples as a JSON-serialisable payload.

    The underlying query mirrors the exploratory DataFrame utility from
    `src/data/mysql_connector.py`. The response contains:
        - `records`: list of dictionaries suitable for grounding in prompts
        - `summary`: compact textual overview (row count, smell distribution)
    """

    # Ensure the pool is initialised before running pandas SQL helpers.
    get_connection_pool()

    df = fetch_samples_dataframe(smell_ids=smell_ids, limit=limit)

    if df.empty:
        return {
            "records": [],
            "summary": "No samples matched the provided criteria.",
        }

    smell_counts = df["smells"].value_counts().to_dict()
    summary = (
        f"Fetched {len(df)} sample(s). has_smell={df['has_smell'].sum()} flagged. "
        f"Smell ID breakdown: {smell_counts}."
    )

    return {
        "records": df.to_dict(orient="records"),
        "summary": summary,
    }


class SampleDetailArgs(BaseModel):
    """Input schema for the `fetch_dacos_sample` tool."""

    sample_id: int = Field(..., ge=1, description="DACOS sample identifier (tagman5.sample.id)")
    include_code: bool = Field(
        default=False,
        description=(
            "If true, attempt to retrieve the associated source file via sparse git checkout."
        ),
    )
    max_lines: int = Field(
        default=120,
        ge=10,
        le=400,
        description="Maximum number of lines to include when returning code snippets.",
    )


def _maybe_fetch_code(sample: DACOSSample, max_lines: int) -> Optional[str]:
    """Best effort attempt to pull the referenced file content for a sample.

    The dataset does not store raw code, so we clone the upstream repository and
    checkout the commit immediately prior to the DACOS publication cutoff. This is
    expensive; callers should guard behind an explicit signal.
    """

    try:
        repo_url = sample.repo_url or derive_repo_url(sample.project_name)
    except Exception as exc:  # pragma: no cover - defensive path
        return f"Unable to derive repository URL: {exc}."

    commit_sha = sample.commit_sha
    if not commit_sha:
        try:
            commit_sha = get_commit_before_date(repo_url)
        except Exception as exc:
            return f"Unable to resolve commit for {repo_url}: {exc}."

    if not commit_sha:
        return (
            "Commit lookup returned no result. Provide `commit_sha` in the database or set it "
            "via environment overrides to enable code retrieval."
        )

    try:
        raw_content = clone_and_read_file(repo_url, commit_sha, sample.path_to_file)
    except Exception as exc:  # pragma: no cover - network/IO errors
        return f"Failed to clone or read file: {exc}."

    lines = raw_content.splitlines()
    snippet = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        snippet += "\n..."

    return snippet


@tool("fetch_dacos_sample", args_schema=SampleDetailArgs)
def fetch_dacos_sample(
    sample_id: int, include_code: bool = False, max_lines: int = 120
) -> Dict[str, Any]:
    """Load a single DACOS sample along with optional source code snippet."""

    sample = fetch_sample_by_id(sample_id)

    if sample is None:
        return {
            "sample_id": sample_id,
            "error": "Sample not found in DACOS dataset.",
        }

    payload: Dict[str, Any] = {
        "sample": sample.model_dump(),
        "ground_truth_smells": sample.ground_truth_smells(),
    }

    if include_code:
        if os.environ.get("DACOS_ENABLE_GIT_FETCH", "false").lower() in {
            "1",
            "true",
            "yes",
        }:
            payload["code_fragment"] = _maybe_fetch_code(sample, max_lines)
        else:
            payload["code_fragment"] = (
                "Set DACOS_ENABLE_GIT_FETCH=1 to allow sparse git checkout for code retrieval."
            )

    return payload


TOOLS: List[Callable[..., Any]] = [load_dacos_samples, fetch_dacos_sample]
