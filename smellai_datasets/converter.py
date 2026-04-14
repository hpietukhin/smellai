"""Convert raw dataset sources to pandas DataFrames.

Four converters:
- rminer_to_df: RMiner 2.0 oracle data.json → flat DataFrame (one row per refactoring)
- rminer_planner_to_df: RMiner oracle → per-commit DataFrame for planner evaluation
- swe_refactor_to_df: SWE-Refactor ZIP/dir/JSON → flat DataFrame (one row per record)
- tdd_to_df: Technical Debt Dataset SQLite → flat DataFrame (one row per smell event)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

SMELL_RELEVANT_TYPES = {
    "Extract Method",
    "Move Method",
    "Extract Class",
    "Move Class",
    "Extract And Move Method",
    "Extract Superclass",
    "Inline Method",
}


def rminer_to_df(
    oracle_path: str | Path,
    filter_tp: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    """Convert RMiner 2.0 oracle data.json to a DataFrame.

    One row per refactoring (flattened from commit objects).

    Args:
        oracle_path: Path to data.json (RMiner oracle)
        filter_tp: If True, keep only validation == "TP" rows
        limit: Optional row limit (applied after filtering)

    Returns:
        DataFrame with columns:
            commit_id, repository, commit_sha, author, time,
            refactoring_type, description, validation, detection_tools
    """
    oracle_path = Path(os.path.expandvars(str(oracle_path)))
    with open(oracle_path) as f:
        raw = json.load(f)

    # oracle is a list of commit objects
    rows: list[dict[str, Any]] = []
    for idx, commit in enumerate(raw):
        repo = commit.get("repository", "")
        commit_sha = commit.get("sha1", "")
        author = commit.get("author", "")
        time = commit.get("time", "")

        for ref in commit.get("refactorings", []):
            validation = ref.get("validation", None)
            if filter_tp and validation != "TP":
                continue

            tools = ref.get("detectionTools", [])
            detection_tools = ",".join(tools) if isinstance(tools, list) else str(tools)

            rows.append(
                {
                    "commit_id": idx,
                    "repository": repo,
                    "commit_sha": commit_sha,
                    "author": author,
                    "time": time,
                    "refactoring_type": ref.get("type", ""),
                    "description": ref.get("description", ""),
                    "validation": validation or "",
                    "detection_tools": detection_tools,
                }
            )

    if limit is not None:
        rows = rows[:limit]

    return pd.DataFrame(rows)


def rminer_planner_to_df(
    oracle_path: str | Path,
    filter_tp: bool = True,
    smell_relevant_only: bool = True,
    min_refactorings: int = 1,
    limit: int | None = None,
    max_per_repo: int | None = None,
) -> pd.DataFrame:
    """Convert RMiner oracle to a per-commit DataFrame for planner evaluation.

    Unlike rminer_to_df (one row per refactoring), this produces one row per
    commit — aggregating all smell-relevant refactorings.

    Args:
        oracle_path: Path to data.json (RMiner oracle)
        filter_tp: If True, keep only validation == "TP" refactorings
        smell_relevant_only: If True, keep only smell-relevant refactoring types
        min_refactorings: Minimum number of (filtered) refactorings per commit
        limit: Optional commit limit (applied after all filtering)
        max_per_repo: Optional cap per repository

    Returns:
        DataFrame with one row per commit.
    """
    oracle_path = Path(os.path.expandvars(str(oracle_path)))
    with open(oracle_path) as f:
        raw = json.load(f)

    rows: list[dict[str, Any]] = []
    for commit in raw:
        repo = commit.get("repository", "")
        sha = commit.get("sha1", "")

        refs = commit.get("refactorings", [])
        if filter_tp:
            refs = [r for r in refs if r.get("validation") == "TP"]
        if smell_relevant_only:
            refs = [r for r in refs if r.get("type") in SMELL_RELEVANT_TYPES]
        if len(refs) < min_refactorings:
            continue

        first = refs[0]
        first_desc = first.get("description", "")
        first_class = _parse_class_from_description(first_desc)
        unique_types = sorted({r.get("type", "") for r in refs})

        refs_json = json.dumps(
            [{"type": r.get("type", ""), "description": r.get("description", "")} for r in refs],
            ensure_ascii=False,
        )

        rows.append({
            "commit_sha": sha,
            "repository": repo,
            "author": commit.get("author", ""),
            "time": commit.get("time", ""),
            "refactoring_count": len(refs),
            "refactorings_json": refs_json,
            "first_refactoring_type": first.get("type", ""),
            "first_refactoring_class": first_class,
            "smell_relevant_types": "|".join(unique_types),
        })

    if max_per_repo is not None:
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.groupby("repository").head(max_per_repo).reset_index(drop=True)
            rows = df.to_dict("records")

    if limit is not None:
        rows = rows[:limit]

    return pd.DataFrame(rows)


def _parse_class_from_description(desc: str) -> str:
    """Extract class name from refactoring description."""
    m = re.search(r"(?:in|from) class (\S+)", desc)
    return m.group(1) if m else ""


def swe_refactor_to_df(
    dataset_path: str,
    limit: int | None = None,
    **filters: Any,
) -> pd.DataFrame:
    """Convert SWE-Refactor dataset to a DataFrame.

    Supports three input formats:
    - .zip: SWE-Refactor ZIP archive with JSON files inside
    - .json: pure_refactoring_data.json (flat list of records)
    - directory: extracted SWE-Refactor dir with JSON files

    Args:
        dataset_path: Path to SWE-Refactor.zip, .json, or extracted directory
        limit: Optional row limit
        **filters: Keyword filters applied to flat row fields (e.g. is_compound=False)

    Returns:
        DataFrame with columns:
            pair_id, project_name, commit_id, refactoring_type,
            is_compound, is_pure, source_before, source_after,
            class_before, class_after, jdk_version, compile_command,
            has_tests, file_path_before, file_path_after
    """
    path = Path(dataset_path)
    raw_jsons = _load_swe_raw_jsons(path)

    rows: list[dict[str, Any]] = []
    for data in raw_jsons:
        row = _flatten_swe_record(data)
        if row is None:
            continue
        if all(row.get(k) == v for k, v in filters.items()):
            rows.append(row)

    if limit is not None:
        rows = rows[:limit]

    return pd.DataFrame(rows)


def _load_swe_raw_jsons(path: Path) -> list[dict]:
    """Load raw SWE-Refactor JSON objects from zip, json file, or directory."""
    if path.suffix == ".zip":
        results = []
        with zipfile.ZipFile(path, "r") as zf:
            namelist = zf.namelist()
            # Prefer the canonical pure_refactoring_data.json if present
            pure_json = next(
                (n for n in namelist if n.endswith("pure_refactoring_data.json")),
                None,
            )
            candidates = [pure_json] if pure_json else [
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
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    # Directory: find all JSON files
    results = []
    for json_file in path.rglob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
    return results


def _flatten_swe_record(data: dict) -> dict[str, Any] | None:
    """Flatten a raw SWE-Refactor JSON object to a flat row dict."""
    try:
        refactoring_type = data.get("type", "")
        jdk = data.get("compileJDK", 11)
        if isinstance(jdk, float) and jdk == 1.8:
            jdk = 8

        return {
            "pair_id": data.get("uniqueId", data.get("commitId", "")),
            "project_name": data.get("projectName", ""),
            "commit_id": data.get("commitId", ""),
            "refactoring_type": refactoring_type,
            "is_compound": "+" in refactoring_type,
            "is_pure": bool(data.get("isPureRefactoring", False)),
            "source_before": data.get("sourceCodeBeforeRefactoring", ""),
            "source_after": data.get("sourceCodeAfterRefactoring", ""),
            "class_before": data.get("sourceCodeBeforeForWhole", ""),
            "class_after": data.get("sourceCodeAfterForWhole", ""),
            "jdk_version": int(jdk),
            "compile_command": data.get("compileCommand", ""),
            "has_tests": bool(data.get("hasTestC", False)),
            "file_path_before": data.get("filePathBefore", ""),
            "file_path_after": data.get("filePathAfter", ""),
        }
    except (KeyError, TypeError):
        return None


def tdd_to_df(
    db_path: str | None = None,
    project: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Convert Technical Debt Dataset v2 SQLite to a DataFrame.

    One row per SonarQube issue, with creation/close commit hashes resolved
    via SONAR_ANALYSIS.

    Args:
        db_path: Path to TDD SQLite file (td_V2.db).  Falls back to TDD_DB_PATH env var.
        project: Filter to a single PROJECT_ID (e.g. "org.apache:cayenne")
        limit: Optional row limit

    Returns:
        DataFrame with columns:
            project, creation_commit, close_commit, issue_type, rule,
            severity, status, resolution, component, message,
            start_line, end_line, creation_date, close_date, effort, debt
    """
    resolved_path = db_path or os.environ.get("TDD_DB_PATH")
    if not resolved_path:
        raise ValueError(
            "TDD SQLite path required: pass db_path or set TDD_DB_PATH env var"
        )

    con = sqlite3.connect(resolved_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Probe schema to identify available tables/columns
    tables = {
        row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }

    rows: list[dict[str, Any]] = _query_tdd(cur, tables, project, limit)
    con.close()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _query_tdd(
    cur: sqlite3.Cursor,
    tables: set[str],
    project: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Query TDD DB with best-effort schema detection.

    TDD v2.0.1 schema (Lenarduzzi et al.):
        SONAR_ISSUES  — one row per SonarQube issue with CREATION/CLOSE_ANALYSIS_KEY
        SONAR_ANALYSIS — maps ANALYSIS_KEY → git REVISION (commit hash)
        REFACTORING_MINER — refactoring type + detail per commit
        PROJECTS — project metadata
    """
    upper = {t.upper() for t in tables}
    if "SONAR_ISSUES" in upper and "SONAR_ANALYSIS" in upper:
        return _query_tdd_v2(cur, project, limit)
    raise ValueError(
        f"Unrecognised TDD schema. Expected SONAR_ISSUES + SONAR_ANALYSIS tables, "
        f"got: {sorted(tables)}"
    )


def _query_tdd_v2(
    cur: sqlite3.Cursor,
    project: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """TDD v2.0.1 schema: SONAR_ISSUES + SONAR_ANALYSIS → commit hashes."""
    sql = """
        SELECT
            si.PROJECT_ID            AS project,
            sa_create.REVISION       AS creation_commit,
            sa_close.REVISION        AS close_commit,
            si.TYPE                  AS issue_type,
            si.RULE                  AS rule,
            si.SEVERITY              AS severity,
            si.STATUS                AS status,
            si.RESOLUTION            AS resolution,
            si.COMPONENT             AS component,
            si.MESSAGE               AS message,
            si.START_LINE            AS start_line,
            si.END_LINE              AS end_line,
            si.CREATION_DATE         AS creation_date,
            si.CLOSE_DATE            AS close_date,
            si.EFFORT                AS effort,
            si.DEBT                  AS debt
        FROM SONAR_ISSUES si
        LEFT JOIN SONAR_ANALYSIS sa_create
            ON sa_create.ANALYSIS_KEY = si.CREATION_ANALYSIS_KEY
        LEFT JOIN SONAR_ANALYSIS sa_close
            ON sa_close.ANALYSIS_KEY = si.CLOSE_ANALYSIS_KEY
            AND si.CLOSE_ANALYSIS_KEY != ''
    """
    params: list[Any] = []
    if project:
        sql += " WHERE si.PROJECT_ID = ?"
        params.append(project)
    if limit:
        sql += f" LIMIT {limit}"

    cur.execute(sql, params)
    return [dict(row) for row in cur.fetchall()]
