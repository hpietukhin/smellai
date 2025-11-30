"""Tooling used by the LangGraph ReAct agent.

Provides typed helpers that give the model access to the DACOS dataset via the
shared MySQL connection utilities and optional Git-based code retrieval.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from smellai.data.git_ops import (
    clone_and_read_file,
    derive_repo_url,
    get_commit_before_date,
)
from smellai.data.mysql_connector import (
    fetch_sample_by_id_async,
    fetch_samples_dataframe_async,
    get_connection_pool,
)
from smellai.models.entities import DACOSSample


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
async def load_dacos_samples(
    limit: int, smell_ids: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Return DACOS samples as a JSON-serialisable payload.

    The underlying query mirrors the exploratory DataFrame utility from
    `src/data/mysql_connector.py`. The response contains:
        - `records`: list of dictionaries suitable for grounding in prompts
        - `summary`: compact textual overview (row count, smell distribution)
    """

    # Ensure the pool is initialised before running pandas SQL helpers.
    get_connection_pool()

    df = await fetch_samples_dataframe_async(smell_ids=smell_ids, limit=limit)

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

    sample_id: int = Field(
        ..., ge=1, description="DACOS sample identifier (tagman5.sample.id)"
    )
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


_DATASET_ROOT_ENV_KEYS: Tuple[str, ...] = (
    "DACOS_FILES_ROOT",
    "DACOS_DATA_ROOT",
    "DACOS_DATASET_ROOT",
)


def _expand_dataset_roots() -> List[Path]:
    """Collect candidate dataset roots from environment variables and defaults."""

    roots: List[Path] = []
    seen: set[Path] = set()

    for key in _DATASET_ROOT_ENV_KEYS:
        value = os.getenv(key)
        if not value:
            continue

        candidate = Path(value).expanduser()
        if candidate in seen:
            continue

        seen.add(candidate)
        if candidate.is_dir():
            roots.append(candidate)

    # Fall back to a sibling "files" directory if it exists alongside the repo.
    project_root = Path(__file__).resolve().parents[3]
    default_roots = [project_root / "files", project_root.parent / "files"]
    for candidate in default_roots:
        if candidate.is_dir() and candidate not in seen:
            seen.add(candidate)
            roots.append(candidate)

    return roots


def _read_text_preview(file_path: Path, max_lines: int) -> str:
    """Read a text file and return at most ``max_lines`` lines."""

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = file_path.read_text(encoding="latin-1")

    lines = content.splitlines()
    snippet = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        snippet += "\n..."
    return snippet


def _try_local_code_lookup(path_to_file: str, max_lines: int) -> Tuple[bool, str]:
    """Attempt to resolve DACOS `.code` files from the extracted dataset."""

    normalized = Path(path_to_file.replace("\\", "/").lstrip("/"))
    if normalized.suffix.lower() != ".code":
        return False, ""

    roots = _expand_dataset_roots()
    if not roots:
        return False, (
            "Local DACOS dataset directory not configured. Set DACOS_FILES_ROOT to the extracted "
            "`files` folder to enable direct `.code` retrieval."
        )

    for root in roots:
        candidate = (root / normalized).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue

        if candidate.is_file():
            try:
                return True, _read_text_preview(candidate, max_lines)
            except (OSError, UnicodeDecodeError) as exc:
                return False, f"Failed to read local code file {candidate}: {exc}."

    return False, (
        f"Code file {normalized} not found under configured DACOS dataset roots. "
        "Check DACOS_FILES_ROOT or ensure the dataset is extracted."
    )


def _maybe_fetch_code(sample: DACOSSample, max_lines: int) -> Optional[str]:
    """Best effort attempt to pull the referenced file content for a sample.

    The dataset does not store raw code, so we clone the upstream repository and
    checkout the commit immediately prior to the DACOS publication cutoff. This is
    expensive; callers should guard behind an explicit signal.
    """

    success, local_payload = _try_local_code_lookup(sample.path_to_file, max_lines)
    if success:
        return local_payload
    if local_payload:
        return local_payload

    allow_git = os.environ.get("DACOS_ENABLE_GIT_FETCH", "false").lower() in {
        "1",
        "true",
        "yes",
    }

    if not allow_git:
        return "Set DACOS_ENABLE_GIT_FETCH=1 to allow sparse git checkout for code retrieval."

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
async def fetch_dacos_sample(
    sample_id: int, include_code: bool = False, max_lines: int = 120
) -> Dict[str, Any]:
    """Load a single DACOS sample along with optional source code snippet."""

    sample = await fetch_sample_by_id_async(sample_id)

    if sample is None:
        return {
            "sample_id": sample_id,
            "error": "Sample not found in DACOS dataset.",
        }

    payload: Dict[str, Any] = {
        "sample": sample.model_dump(),
        "ground_truth_smells": sample.ground_truth_smells(),
    }

    code_details: Optional[Dict[str, Any]] = None
    if sample.path_to_file.lower().endswith(".code"):
        # Run synchronous file I/O in thread pool
        loop = asyncio.get_event_loop()
        success, code_result = await loop.run_in_executor(
            None, _try_local_code_lookup, sample.path_to_file, max_lines
        )
        code_details = {
            "path": sample.path_to_file,
            "source": "local_dataset",
        }
        if success:
            code_details["content"] = code_result
        else:
            code_details["error"] = code_result

    if code_details:
        payload["code_sample"] = code_details

    if include_code:
        if code_details and "content" in code_details:
            payload["code_fragment"] = code_details["content"]
        else:
            # Run potentially slow git operation in thread pool
            loop = asyncio.get_event_loop()
            payload["code_fragment"] = await loop.run_in_executor(
                None, _maybe_fetch_code, sample, max_lines
            )
    elif code_details and "content" in code_details:
        payload.setdefault("code_fragment_preview", code_details["content"])

    return payload


TOOLS: List[Callable[..., Any]] = [load_dacos_samples, fetch_dacos_sample]
