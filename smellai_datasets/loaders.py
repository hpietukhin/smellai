"""Dataset loaders: raw JSON/manifest → EvalSample.

Replaces the old converter.py + config.py + mlflow_bridge pipeline.

Public API
----------
load_swe_raw_df     : SWE-Refactor JSON → flat pandas DataFrame (for inspection)
load_rminer_raw_df  : RMiner oracle JSON → flat pandas DataFrame (for inspection)
load_eval_samples   : load list[EvalSample] from one or more sources
load_eval_df        : MLflow-ready DataFrame (source/sample_id/inputs/expectations/tags)
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from pydantic import TypeAdapter

from .schema import DatasetSource, EvalSample

# ---------------------------------------------------------------------------
# Default dataset path resolution
# ---------------------------------------------------------------------------

_RMINER_MANIFEST_DEFAULT = Path(
    "/Users/havriil.pietukhin/uni/masterThesis/datasets/rminer_data/manifest.json"
)
_SWE_JSON_CANDIDATES = (
    Path("/Users/havriil.pietukhin/uni/masterThesis/datasets/pure_refactoring_data.json"),
    Path(
        "/Users/havriil.pietukhin/uni/masterThesis/datasets/SWE-Refactor/pure_refactoring_data.json"
    ),
    Path(
        "/Users/havriil.pietukhin/uni/masterThesis/SWE-Refactor/SWE-Refactor/pure_refactoring_data.json"
    ),
)
def _first_existing(*paths: Path | None) -> Path | None:
    for p in paths:
        if p and p.exists():
            return p
    return None


def _resolve_swe_path(path: Path | None) -> Path | None:
    """Accept a file path or a directory containing pure_refactoring_data.json."""
    if path is None:
        return _first_existing(*_SWE_JSON_CANDIDATES)
    if path.is_file():
        return path
    for candidate in (
        path / "pure_refactoring_data.json",
        path / "SWE-Refactor" / "pure_refactoring_data.json",
    ):
        if candidate.exists():
            return candidate
    return None


def _require(path: Path | None, label: str) -> Path:
    if path is None:
        raise FileNotFoundError(
            f"Could not resolve {label}. "
            "Set the matching env var or place the dataset at the expected default location."
        )
    return path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_jdk(value: Any) -> int:
    if value is None:
        return 11
    if isinstance(value, float) and value == 1.8:
        return 8
    return int(round(float(value)))


def _load_swe_raw_jsons(path: Path) -> list[dict]:
    """Load raw SWE-Refactor JSON objects from .zip, .json file, or directory."""
    if path.suffix == ".zip":
        results: list[dict] = []
        with zipfile.ZipFile(path, "r") as zf:
            namelist = zf.namelist()
            pure = next(
                (n for n in namelist if n.endswith("pure_refactoring_data.json")), None
            )
            candidates = [pure] if pure else [
                n for n in namelist
                if n.endswith(".json") and not n.startswith("__MACOSX/")
            ]
            for name in candidates:
                try:
                    with zf.open(name) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        results.extend(data)
                    else:
                        results.append(data)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
        return results

    if path.suffix == ".json":
        with path.open() as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    # Directory
    results = []
    for json_file in path.rglob("*.json"):
        with json_file.open() as f:
            data = json.load(f)
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
    return results


# ---------------------------------------------------------------------------
# Raw DataFrames (for inspection / debugging)
# ---------------------------------------------------------------------------

def load_swe_raw_df(path: Path | None = None) -> pd.DataFrame:
    """Load the raw SWE-Refactor JSON into a normalised DataFrame.

    One row per refactoring pair with all relevant columns.
    """
    resolved = _require(_resolve_swe_path(path), "SWE-Refactor JSON")
    records = _load_swe_raw_jsons(resolved)

    rows: list[dict[str, Any]] = []
    for rec in records:
        refactoring_type = rec.get("type", "")
        rows.append(
            {
                "pair_id": rec.get("uniqueId") or rec.get("commitId", ""),
                "project_name": rec.get("projectName", ""),
                "commit_id": rec.get("commitId", ""),
                "refactoring_type": refactoring_type,
                "file_path_before": rec.get("filePathBefore", ""),
                "file_path_after": rec.get("filePathAfter", ""),
                "class_before": rec.get("sourceCodeBeforeForWhole", ""),
                "class_after": rec.get("sourceCodeAfterForWhole", ""),
                "source_before": rec.get("sourceCodeBeforeRefactoring", ""),
                "source_after": rec.get("sourceCodeAfterRefactoring", ""),
                "compile_command": rec.get("compileCommand", ""),
                "is_compound": "+" in refactoring_type,
                "is_pure": bool(rec.get("isPureRefactoring", False)),
                "has_tests": bool(rec.get("hasTestC", False)),
                "jdk_version": _coerce_jdk(rec.get("compileJDK")),
            }
        )
    return pd.DataFrame(rows)


def load_rminer_raw_df(path: Path | None = None, *, tp_only: bool = True) -> pd.DataFrame:
    """Load the raw RMiner oracle JSON into a flat DataFrame.

    One row per refactoring. Uses the oracle for inspection/analysis only —
    evaluation pairs come from the manifest (see load_eval_samples).
    """
    resolved = _require(
        path or _first_existing(
            Path(os.environ.get("RMINER_ORACLE_PATH", "")),
            Path("/Users/havriil.pietukhin/uni/masterThesis/datasets/rminer_oracle_java1.json"),
        ),
        "RMiner oracle JSON",
    )
    with resolved.open() as f:
        commits = json.load(f)

    rows: list[dict[str, Any]] = []
    for commit in commits:
        for ref in commit.get("refactorings", []):
            if tp_only and ref.get("validation") != "TP":
                continue
            tools = ref.get("detectionTools", [])
            rows.append(
                {
                    "repository": commit.get("repository", ""),
                    "commit_sha": commit.get("sha1", ""),
                    "author": commit.get("author", ""),
                    "time": commit.get("time", ""),
                    "refactoring_type": ref.get("type", ""),
                    "description": ref.get("description", ""),
                    "validation": ref.get("validation", ""),
                    "detection_tools": ",".join(tools)
                    if isinstance(tools, list)
                    else str(tools),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EvalSample projections
# ---------------------------------------------------------------------------

def _swe_samples(df: pd.DataFrame) -> list[EvalSample]:
    """Project a SWE raw DataFrame into EvalSample objects."""
    rows = df.to_dict("records")
    return [
        EvalSample(
            source="swe",
            sample_id=f"swe:{row['pair_id']}",
            inputs={
                "project_name": row["project_name"],
                "commit_id": row["commit_id"],
                "refactoring_type": row["refactoring_type"],
                "file_path_before": row["file_path_before"],
                "file_path_after": row["file_path_after"],
                "class_before": row["class_before"],
                "source_before": row["source_before"],
                "jdk_version": int(row["jdk_version"]),
                "compile_command": row["compile_command"],
            },
            expectations={
                "class_after": row["class_after"],
                "source_after": row["source_after"],
            },
            tags={
                "is_pure": bool(row["is_pure"]),
                "is_compound": bool(row["is_compound"]),
                "has_tests": bool(row["has_tests"]),
            },
        )
        for row in rows
    ]


def _rminer_samples(manifest_path: Path, limit: int | None = None) -> list[EvalSample]:
    """Build self-contained EvalSample objects from a RefactoringMiner manifest.

    Each EvalSample embeds before_code, diff_hunks, and refactoring metadata in
    inputs so that inference requires no external file access.
    """
    from rminer.rminer_utils import compute_diff_hunks_from_files, parse_refactoring_info

    base_dir = manifest_path.parent
    with manifest_path.open() as f:
        manifest = json.load(f)

    pairs = manifest.get("pairs", [])
    if limit is not None:
        pairs = pairs[:limit]

    samples: list[EvalSample] = []
    skipped = 0

    for pair in pairs:
        before_path = base_dir / pair["before_file"]
        after_path = base_dir / pair["after_file"]

        if not before_path.exists() or not after_path.exists():
            skipped += 1
            continue

        diff_hunks = compute_diff_hunks_from_files(before_path, after_path)
        if not diff_hunks:
            skipped += 1
            continue

        types, descriptions = parse_refactoring_info(pair)
        pair_id: str = pair["id"]
        before_code = before_path.read_text(errors="replace")
        hunks_dicts = [h.model_dump() for h in diff_hunks]

        samples.append(
            EvalSample(
                source="rminer",
                sample_id=f"rminer:{pair_id}",
                inputs={
                    "pair_id": pair_id,
                    "before_code": before_code,
                    "file_path": pair["file_path"],
                    "refactoring_types": types,
                    "refactoring_descriptions": descriptions,
                    "diff_hunks": hunks_dicts,
                    "sonar_issues": [],
                },
                expectations={
                    "num_hunks": len(diff_hunks),
                    "num_refactorings": len(types),
                    "diff_hunks": hunks_dicts,
                    "refactoring_types": types,
                    "refactoring_descriptions": descriptions,
                    "file_path": pair["file_path"],
                },
                tags={
                    "repository": pair.get("repository", ""),
                    "commit_sha": pair.get("commit_sha", ""),
                    "status": pair.get("status", "modified"),
                },
            )
        )

    if skipped:
        import logging
        logging.getLogger(__name__).warning(
            "Skipped %d manifest pairs (missing files or empty diff)", skipped
        )

    return samples


# ---------------------------------------------------------------------------
# Public unified loaders
# ---------------------------------------------------------------------------

def load_eval_samples(
    sources: Sequence[DatasetSource] = ("swe",),
    *,
    swe_path: Path | None = None,
    rminer_manifest_path: Path | None = None,
    limit: int | None = None,
) -> list[EvalSample]:
    """Load EvalSample objects from one or more dataset sources.

    Args:
        sources: Which sources to load. Defaults to ("swe",).
        swe_path: Path to SWE-Refactor JSON (or dir). Falls back to env/defaults.
        rminer_manifest_path: Path to RMiner manifest.json. Falls back to defaults.
        limit: Optional cap applied *per source*.

    Returns:
        Validated list[EvalSample] — the unified evaluation contract.
    """
    samples: list[EvalSample] = []

    for source in sources:
        if source == "swe":
            df = load_swe_raw_df(swe_path)
            if limit is not None:
                df = df.head(limit)
            samples.extend(_swe_samples(df))

        elif source == "rminer":
            resolved = rminer_manifest_path or _first_existing(
                Path(os.environ.get("RMINER_MANIFEST_PATH", "")),
                _RMINER_MANIFEST_DEFAULT,
            )
            resolved = _require(resolved, "RMiner manifest.json")
            samples.extend(_rminer_samples(resolved, limit=limit))

        elif source == "tdd":
            raise NotImplementedError(
                "TDD source is not yet implemented. "
                "Load TDD data directly via load_tdd_raw_df() for inspection."
            )
        else:
            raise ValueError(f"Unsupported source: {source!r}")

    # Belt-and-suspenders re-validation
    return TypeAdapter(list[EvalSample]).validate_python(
        [s.model_dump() for s in samples]
    )


def load_eval_df(
    sources: Sequence[DatasetSource] = ("swe",),
    *,
    swe_path: Path | None = None,
    rminer_manifest_path: Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Return a MLflow-ready DataFrame from unified EvalSamples.

    Columns: source, sample_id, inputs (dict), expectations (dict), tags (dict).
    """
    samples = load_eval_samples(
        sources,
        swe_path=swe_path,
        rminer_manifest_path=rminer_manifest_path,
        limit=limit,
    )
    return pd.DataFrame([s.model_dump() for s in samples])


__all__ = [
    "load_swe_raw_df",
    "load_rminer_raw_df",
    "load_eval_samples",
    "load_eval_df",
]
