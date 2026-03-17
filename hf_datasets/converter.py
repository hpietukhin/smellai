"""Convert raw dataset sources to HuggingFace Dataset objects.

Three converters:
- rminer_to_hf: RMiner 2.0 oracle data.json → flat HF Dataset (one row per refactoring)
- swe_refactor_to_hf: SWE-Refactor ZIP/dir/JSON → flat HF Dataset (one row per record)
- tdd_to_hf: Technical Debt Dataset SQLite → flat HF Dataset (one row per smell event)
"""

from __future__ import annotations

import json
import os
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

from datasets import Dataset


def rminer_to_hf(
    oracle_path: str | Path,
    filter_tp: bool = True,
    limit: int | None = None,
) -> Dataset:
    """Convert RMiner 2.0 oracle data.json to a HF Dataset.

    One row per refactoring (flattened from commit objects).

    Args:
        oracle_path: Path to data.json (RMiner oracle)
        filter_tp: If True, keep only validation == "TP" rows
        limit: Optional row limit (applied after filtering)

    Returns:
        Dataset with schema:
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

    return Dataset.from_list(rows)


def swe_refactor_to_hf(
    dataset_path: str,
    limit: int | None = None,
    **filters: Any,
) -> Dataset:
    """Convert SWE-Refactor dataset to a HF Dataset.

    Supports three input formats:
    - .zip: SWE-Refactor ZIP archive with JSON files inside
    - .json: pure_refactoring_data.json (flat list of records)
    - directory: extracted SWE-Refactor dir with JSON files

    Args:
        dataset_path: Path to SWE-Refactor.zip, .json, or extracted directory
        limit: Optional row limit
        **filters: Keyword filters applied to flat row fields (e.g. is_compound=False)

    Returns:
        Dataset with schema:
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

    return Dataset.from_list(rows)


def _load_swe_raw_jsons(path: Path) -> list[dict]:
    """Load raw SWE-Refactor JSON objects from zip, json file, or directory."""
    if path.suffix == ".zip":
        results = []
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".json"):
                    with zf.open(name) as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            results.extend(data)
                        else:
                            results.append(data)
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


def tdd_to_hf(
    db_path: str | None = None,
    project: str | None = None,
    limit: int | None = None,
) -> Dataset:
    """Convert Technical Debt Dataset SQLite to a HF Dataset.

    One row per smell event (introduced / resolved / persistent).

    Args:
        db_path: Path to TDD SQLite file.  Falls back to TDD_DB_PATH env var.
        project: Filter to a single project name (optional)
        limit: Optional row limit

    Returns:
        Dataset with schema:
            project, commit_sha, parent_sha, smell_type, severity,
            file_path, rule_id, status
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

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _query_tdd(
    cur: sqlite3.Cursor,
    tables: set[str],
    project: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Query TDD DB with best-effort schema detection."""
    # Try common table/column naming conventions used by TDD releases
    if "REFACTORING_MINER" in tables or "refactoring_miner" in tables:
        return _query_tdd_v2(cur, project, limit)
    if "SMELL" in tables or "smell" in tables:
        return _query_tdd_smell_table(cur, project, limit)
    # Fallback: try to read any table that looks smell-like
    return _query_tdd_fallback(cur, tables, project, limit)


def _query_tdd_v2(
    cur: sqlite3.Cursor,
    project: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """TDD v2 schema: COMMIT + SMELL_METRICS tables."""
    sql = """
        SELECT
            c.project         AS project,
            c.commit_hash     AS commit_sha,
            c.parent_hash     AS parent_sha,
            s.smell_type      AS smell_type,
            s.severity        AS severity,
            s.file_path       AS file_path,
            s.rule_id         AS rule_id,
            s.status          AS status
        FROM SMELL s
        JOIN COMMIT c ON c.id = s.commit_id
    """
    params: list[Any] = []
    if project:
        sql += " WHERE c.project = ?"
        params.append(project)
    if limit:
        sql += f" LIMIT {limit}"

    cur.execute(sql, params)
    return [dict(row) for row in cur.fetchall()]


def _query_tdd_smell_table(
    cur: sqlite3.Cursor,
    project: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """TDD schema with single smell table containing all fields."""
    # Detect actual column names
    cur.execute("PRAGMA table_info(smell)")
    cols = {row[1].lower() for row in cur.fetchall()}

    col_map = {
        "project": next((c for c in cols if "project" in c), "project"),
        "commit_sha": next((c for c in cols if "commit" in c and "hash" in c or c == "commit_sha"), "commit_sha"),
        "parent_sha": next((c for c in cols if "parent" in c), "parent_sha"),
        "smell_type": next((c for c in cols if "type" in c), "smell_type"),
        "severity": next((c for c in cols if "severity" in c), "severity"),
        "file_path": next((c for c in cols if "file" in c), "file_path"),
        "rule_id": next((c for c in cols if "rule" in c), "rule_id"),
        "status": next((c for c in cols if "status" in c), "status"),
    }

    sel = ", ".join(f"{v} AS {k}" for k, v in col_map.items())
    sql = f"SELECT {sel} FROM smell"
    params: list[Any] = []
    if project:
        sql += f" WHERE {col_map['project']} = ?"
        params.append(project)
    if limit:
        sql += f" LIMIT {limit}"

    cur.execute(sql, params)
    return [dict(row) for row in cur.fetchall()]


def _query_tdd_fallback(
    cur: sqlite3.Cursor,
    tables: set[str],
    project: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Last-resort: dump first smell-like table as-is."""
    candidates = [t for t in tables if any(k in t.lower() for k in ("smell", "issue", "metric"))]
    if not candidates:
        candidates = list(tables)[:1]
    if not candidates:
        return []

    table = candidates[0]
    sql = f"SELECT * FROM {table}"  # noqa: S608 — internal helper, no user input
    if limit:
        sql += f" LIMIT {limit}"
    cur.execute(sql)
    rows = cur.fetchall()
    if not rows:
        return []

    # Normalise to expected schema with empty defaults for missing keys
    schema_defaults = {
        "project": "",
        "commit_sha": "",
        "parent_sha": "",
        "smell_type": "",
        "severity": "",
        "file_path": "",
        "rule_id": "",
        "status": "",
    }
    result = []
    for row in rows:
        d = dict(row)
        entry = {**schema_defaults, **d}
        if project and entry.get("project", "") != project:
            continue
        result.append(entry)
    return result
