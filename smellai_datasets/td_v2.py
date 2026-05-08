"""Technical Debt Dataset v2 extraction helpers.

This module supports two related use cases:

1. load the published ``td_V2.db`` SQLite file into flat issue-event DataFrames
   for inspection and EvalSample projection;
2. build refactoring-transition records of the form
   ``CodeState_before(parent) -> CodeState_after(child)``.

The transition pipeline is intentionally DB-first: it uses the published TDD v2
SQLite artifact as the source of truth and reconstructs active smell state from
``SONAR_ANALYSIS`` + ``SONAR_ISSUES`` lifecycle data.
"""

from __future__ import annotations

import ast
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from sonarqube.constants import RULE_NAME_MAP, SEVERITY_MAP

RELEVANT_TABLES = (
    "PROJECTS",
    "GIT_COMMITS",
    "GIT_COMMITS_CHANGES",
    "REFACTORING_MINER",
    "SONAR_ANALYSIS",
    "SONAR_ISSUES",
    "SONAR_MEASURES",
    "SONAR_RULES",
)


def connect_td_v2(path: Path) -> sqlite3.Connection:
    """Open the TDD v2 SQLite DB with row access by column name."""
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def parse_git_parents(raw: str | None) -> list[str]:
    """Parse ``GIT_COMMITS.PARENTS`` which is stored like a Python list string."""
    if raw is None:
        return []
    text = raw.strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def normalize_component_path(component: str | None) -> str:
    """Strip the Sonar component prefix (``project_key:path``) when present."""
    if not component:
        return ""
    return component.split(":", 1)[1] if ":" in component else component


def class_name_from_path(file_path: str | None) -> str:
    """Best-effort Java class name extraction from a file path."""
    if not file_path:
        return ""
    name = Path(file_path).name
    return name[:-5] if name.endswith(".java") else name


def parse_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def make_fingerprint(
    *,
    rule_id: str,
    file_path: str,
    line_start: int | None,
    line_end: int | None,
    message: str,
    hash_value: str | None,
) -> str:
    """Build a stable-ish smell fingerprint when ``ISSUE_KEY`` alone is not enough."""
    if hash_value:
        return hash_value
    return "|".join(
        [
            rule_id,
            file_path,
            str(line_start or ""),
            str(line_end or ""),
            normalize_text(message),
        ]
    )


def load_tdd_raw_df(
    path: Path,
    *,
    limit: int | None = None,
    issue_type: str | None = None,
) -> pd.DataFrame:
    """Load the published TDD v2 issue lifecycle rows into a flat DataFrame.

    This mirrors the exploratory notebook shape: one row per SonarQube issue
    event, joined to creation and close commit SHAs through ``SONAR_ANALYSIS``.
    """
    sql = """
    SELECT
        si.PROJECT_ID        AS project,
        sa_c.REVISION        AS creation_commit,
        sa_x.REVISION        AS close_commit,
        si.ISSUE_KEY         AS issue_key,
        si.TYPE              AS issue_type,
        si.RULE              AS rule,
        si.SEVERITY          AS severity,
        si.STATUS            AS status,
        si.RESOLUTION        AS resolution,
        si.COMPONENT         AS component,
        si.MESSAGE           AS message,
        si.START_LINE        AS start_line,
        si.END_LINE          AS end_line,
        si.CREATION_DATE     AS creation_date,
        si.CLOSE_DATE        AS close_date,
        si.EFFORT            AS effort,
        si.DEBT              AS debt,
        si.HASH              AS hash
    FROM SONAR_ISSUES si
    LEFT JOIN SONAR_ANALYSIS sa_c
        ON sa_c.PROJECT_ID = si.PROJECT_ID
       AND sa_c.ANALYSIS_KEY = si.CREATION_ANALYSIS_KEY
    LEFT JOIN SONAR_ANALYSIS sa_x
        ON sa_x.PROJECT_ID = si.PROJECT_ID
       AND sa_x.ANALYSIS_KEY = si.CLOSE_ANALYSIS_KEY
       AND si.CLOSE_ANALYSIS_KEY != ''
    """
    clauses: list[str] = []
    params: list[Any] = []
    if issue_type:
        clauses.append("si.TYPE = ?")
        params.append(issue_type)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    with connect_td_v2(path) as con:
        df = pd.read_sql_query(sql, con, params=params)

    if "start_line" in df.columns:
        df["start_line"] = df["start_line"].map(parse_optional_int)
    if "end_line" in df.columns:
        df["end_line"] = df["end_line"].map(parse_optional_int)
    return df


def _load_analysis_index(con: sqlite3.Connection) -> dict[tuple[str, str], dict[str, str]]:
    """Return latest known analysis row for each (project_id, revision)."""
    rows = con.execute(
        """
        SELECT PROJECT_ID, REVISION, ANALYSIS_KEY, DATE
        FROM SONAR_ANALYSIS
        WHERE COALESCE(REVISION, '') != ''
        ORDER BY DATE DESC
        """
    ).fetchall()
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["PROJECT_ID"], row["REVISION"])
        index.setdefault(
            key,
            {
                "analysis_key": row["ANALYSIS_KEY"],
                "date": row["DATE"],
                "revision": row["REVISION"],
            },
        )
    return index


def _load_rule_name_index(con: sqlite3.Connection) -> dict[str, str]:
    rows = con.execute(
        "SELECT PLUGIN_NAME, PLUGIN_RULE_KEY, NAME FROM SONAR_RULES"
    ).fetchall()
    index: dict[str, str] = {}
    for row in rows:
        full_rule = f"{row['PLUGIN_NAME']}:{row['PLUGIN_RULE_KEY']}"
        if full_rule not in index:
            index[full_rule] = row["NAME"]
    return index


def load_project_registry(con: sqlite3.Connection) -> dict[str, dict[str, str]]:
    rows = con.execute(
        """
        SELECT PROJECT_ID, PROJECT_KEY, GIT_LINK, SONAR_PROJECT_KEY
        FROM PROJECTS
        """
    ).fetchall()
    return {
        row["PROJECT_ID"]: {
            "project_id": row["PROJECT_ID"],
            "project_name": row["PROJECT_KEY"],
            "repository_url": row["GIT_LINK"],
            "sonar_project_key": row["SONAR_PROJECT_KEY"],
            "source_dataset": "technical-debt-dataset-v2.0.1",
            "language": "Java",
        }
        for row in rows
    }


def get_table_counts(
    con: sqlite3.Connection,
    tables: Iterable[str] = RELEVANT_TABLES,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in tables:
        counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return counts


def get_table_schema(
    con: sqlite3.Connection,
    table: str,
) -> list[dict[str, Any]]:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return [
        {
            "cid": row[0],
            "name": row[1],
            "type": row[2],
            "notnull": row[3],
            "default": row[4],
            "pk": row[5],
        }
        for row in rows
    ]


def render_schema_markdown(
    con: sqlite3.Connection,
    tables: Iterable[str] = RELEVANT_TABLES,
) -> str:
    counts = get_table_counts(con, tables)
    lines = ["# TDD v2 schema snapshot", ""]
    lines.append("## Row counts")
    lines.append("")
    lines.append("| Table | Rows |")
    lines.append("|---|---:|")
    for table in tables:
        lines.append(f"| `{table}` | {counts[table]} |")
    lines.append("")
    lines.append("## Table schemas")
    lines.append("")
    for table in tables:
        lines.append(f"### `{table}`")
        lines.append("")
        lines.append("| cid | name | type | notnull | pk | default |")
        lines.append("|---:|---|---|---:|---:|---|")
        for col in get_table_schema(con, table):
            lines.append(
                "| {cid} | `{name}` | `{type}` | {notnull} | {pk} | {default} |".format(
                    **col
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def load_refactoring_events(
    con: sqlite3.Connection,
    project_id: str,
    commit_sha: str,
) -> list[dict[str, str]]:
    rows = con.execute(
        """
        SELECT REFACTORING_TYPE, REFACTORING_DETAIL
        FROM REFACTORING_MINER
        WHERE PROJECT_ID = ? AND COMMIT_HASH = ?
        ORDER BY REFACTORING_TYPE, REFACTORING_DETAIL
        """,
        (project_id, commit_sha),
    ).fetchall()
    return [
        {
            "type": row["REFACTORING_TYPE"],
            "detail": row["REFACTORING_DETAIL"],
        }
        for row in rows
    ]


def _chunked(items: list[str], size: int = 900) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def load_refactoring_events_batch(
    con: sqlite3.Connection,
    candidates: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Batch-load refactoring rows for candidate child commits."""
    grouped_commits: dict[str, set[str]] = {}
    for candidate in candidates:
        grouped_commits.setdefault(candidate["project_id"], set()).add(
            candidate["child_sha"]
        )

    results: dict[tuple[str, str], list[dict[str, str]]] = {
        (candidate["project_id"], candidate["child_sha"]): []
        for candidate in candidates
    }
    for project_id, commit_shas in grouped_commits.items():
        commit_list = sorted(commit_shas)
        for chunk in _chunked(commit_list):
            placeholders = ", ".join("?" for _ in chunk)
            rows = con.execute(
                f"""
                SELECT PROJECT_ID, COMMIT_HASH, REFACTORING_TYPE, REFACTORING_DETAIL
                FROM REFACTORING_MINER
                WHERE PROJECT_ID = ?
                  AND COMMIT_HASH IN ({placeholders})
                ORDER BY PROJECT_ID, COMMIT_HASH, REFACTORING_TYPE, REFACTORING_DETAIL
                """,
                [project_id, *chunk],
            ).fetchall()
            for row in rows:
                results[(row["PROJECT_ID"], row["COMMIT_HASH"])].append(
                    {
                        "type": row["REFACTORING_TYPE"],
                        "detail": row["REFACTORING_DETAIL"],
                    }
                )
    return results


def load_changed_files(
    con: sqlite3.Connection,
    project_id: str,
    commit_sha: str,
) -> list[str]:
    rows = con.execute(
        """
        SELECT DISTINCT FILE
        FROM GIT_COMMITS_CHANGES
        WHERE PROJECT_ID = ? AND COMMIT_HASH = ?
        ORDER BY FILE
        """,
        (project_id, commit_sha),
    ).fetchall()
    return [row["FILE"] for row in rows if row["FILE"]]


def load_changed_files_batch(
    con: sqlite3.Connection,
    candidates: list[dict[str, Any]],
) -> dict[tuple[str, str], list[str]]:
    """Batch-load changed files for candidate child commits."""
    grouped_commits: dict[str, set[str]] = {}
    for candidate in candidates:
        grouped_commits.setdefault(candidate["project_id"], set()).add(
            candidate["child_sha"]
        )

    results: dict[tuple[str, str], set[str]] = {
        (candidate["project_id"], candidate["child_sha"]): set()
        for candidate in candidates
    }
    for project_id, commit_shas in grouped_commits.items():
        commit_list = sorted(commit_shas)
        for chunk in _chunked(commit_list):
            placeholders = ", ".join("?" for _ in chunk)
            rows = con.execute(
                f"""
                SELECT PROJECT_ID, COMMIT_HASH, FILE
                FROM GIT_COMMITS_CHANGES
                WHERE PROJECT_ID = ?
                  AND COMMIT_HASH IN ({placeholders})
                ORDER BY PROJECT_ID, COMMIT_HASH, FILE
                """,
                [project_id, *chunk],
            ).fetchall()
            for row in rows:
                file_path = row["FILE"]
                if file_path:
                    results[(row["PROJECT_ID"], row["COMMIT_HASH"])].add(file_path)

    return {
        key: sorted(files)
        for key, files in results.items()
    }


def extract_class_names_from_refactoring_detail(detail: str | None) -> list[str]:
    """Heuristic class extraction from textual RefactoringMiner details."""
    if not detail:
        return []

    classes: set[str] = set()
    for pattern in (
        r"\bclass\s+([A-Za-z_][\w.$]+)",
        r"\binterface\s+([A-Za-z_][\w.$]+)",
        r"\benum\s+([A-Za-z_][\w.$]+)",
    ):
        for match in re.findall(pattern, detail):
            classes.add(match.split(".")[-1])
    return sorted(classes)


def _normalize_smell_row(
    row: sqlite3.Row,
    *,
    rule_name_index: dict[str, str],
) -> dict[str, Any]:
    rule_id = row["RULE"] or ""
    file_path = normalize_component_path(row["COMPONENT"])
    line_start = parse_optional_int(row["START_LINE"])
    line_end = parse_optional_int(row["END_LINE"])
    message = row["MESSAGE"] or ""
    hash_value = row["HASH"] or ""
    severity_raw = row["SEVERITY"] or ""

    return {
        "issue_key": row["ISSUE_KEY"],
        "smell_id": row["ISSUE_KEY"],
        "rule_id": rule_id,
        "smell_type": rule_name_index.get(rule_id) or RULE_NAME_MAP.get(rule_id) or rule_id,
        "severity": SEVERITY_MAP.get(severity_raw, severity_raw or "LOW"),
        "severity_raw": severity_raw,
        "status": row["STATUS"],
        "resolution": row["RESOLUTION"] or "",
        "effort": parse_optional_float(row["EFFORT"]),
        "debt": parse_optional_float(row["DEBT"]),
        "component": row["COMPONENT"] or "",
        "file_path": file_path,
        "class_name": class_name_from_path(file_path),
        "line_start": line_start,
        "line_end": line_end,
        "message": message,
        "hash": hash_value,
        "fingerprint": make_fingerprint(
            rule_id=rule_id,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            message=message,
            hash_value=hash_value,
        ),
    }


def load_active_code_smells(
    con: sqlite3.Connection,
    project_id: str,
    commit_sha: str,
    *,
    analysis_index: dict[tuple[str, str], dict[str, str]],
    rule_name_index: dict[str, str],
) -> list[dict[str, Any]]:
    analysis = analysis_index.get((project_id, commit_sha))
    if analysis is None:
        return []

    rows = con.execute(
        """
        SELECT
            si.ISSUE_KEY,
            si.RULE,
            si.SEVERITY,
            si.STATUS,
            si.RESOLUTION,
            si.EFFORT,
            si.DEBT,
            si.MESSAGE,
            si.COMPONENT,
            si.START_LINE,
            si.END_LINE,
            si.HASH,
            si.CREATION_ANALYSIS_KEY,
            si.CLOSE_ANALYSIS_KEY
        FROM SONAR_ISSUES si
        JOIN SONAR_ANALYSIS sa_creation
          ON sa_creation.PROJECT_ID = si.PROJECT_ID
         AND sa_creation.ANALYSIS_KEY = si.CREATION_ANALYSIS_KEY
        LEFT JOIN SONAR_ANALYSIS sa_close
          ON sa_close.PROJECT_ID = si.PROJECT_ID
         AND sa_close.ANALYSIS_KEY = si.CLOSE_ANALYSIS_KEY
         AND si.CLOSE_ANALYSIS_KEY != ''
        WHERE si.PROJECT_ID = ?
          AND si.TYPE = 'CODE_SMELL'
          AND sa_creation.DATE <= ?
          AND (
            COALESCE(si.CLOSE_ANALYSIS_KEY, '') = ''
            OR sa_close.DATE > ?
          )
        ORDER BY si.ISSUE_KEY
        """,
        (project_id, analysis["date"], analysis["date"]),
    ).fetchall()
    return [
        _normalize_smell_row(row, rule_name_index=rule_name_index)
        for row in rows
    ]


def load_project_code_smell_lifecycle(
    con: sqlite3.Connection,
    project_id: str,
    *,
    rule_name_index: dict[str, str],
) -> list[dict[str, Any]]:
    """Load and normalize all CODE_SMELL lifecycle rows for one project once."""
    rows = con.execute(
        """
        SELECT
            si.ISSUE_KEY,
            si.RULE,
            si.SEVERITY,
            si.STATUS,
            si.RESOLUTION,
            si.EFFORT,
            si.DEBT,
            si.MESSAGE,
            si.COMPONENT,
            si.START_LINE,
            si.END_LINE,
            si.HASH,
            sa_creation.DATE AS OPEN_DATE,
            sa_close.DATE AS CLOSE_DATE
        FROM SONAR_ISSUES si
        JOIN SONAR_ANALYSIS sa_creation
          ON sa_creation.PROJECT_ID = si.PROJECT_ID
         AND sa_creation.ANALYSIS_KEY = si.CREATION_ANALYSIS_KEY
        LEFT JOIN SONAR_ANALYSIS sa_close
          ON sa_close.PROJECT_ID = si.PROJECT_ID
         AND sa_close.ANALYSIS_KEY = si.CLOSE_ANALYSIS_KEY
         AND si.CLOSE_ANALYSIS_KEY != ''
        WHERE si.PROJECT_ID = ?
          AND si.TYPE = 'CODE_SMELL'
        ORDER BY si.ISSUE_KEY
        """,
        (project_id,),
    ).fetchall()

    lifecycle_rows: list[dict[str, Any]] = []
    for row in rows:
        smell = _normalize_smell_row(row, rule_name_index=rule_name_index)
        lifecycle_rows.append(
            {
                "open_date": row["OPEN_DATE"],
                "close_date": row["CLOSE_DATE"],
                "smell": smell,
            }
        )
    return lifecycle_rows


def build_project_state_cache(
    *,
    project_id: str,
    candidates: list[dict[str, Any]],
    analysis_index: dict[tuple[str, str], dict[str, str]],
    lifecycle_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Build all needed commit states for one project in memory.

    Semantics intentionally match ``load_active_code_smells``:
    open_date <= point_date and (close_date is null or close_date > point_date).
    """
    needed_commits = {
        candidate["parent_sha"] for candidate in candidates if candidate["project_id"] == project_id
    } | {
        candidate["child_sha"] for candidate in candidates if candidate["project_id"] == project_id
    }

    state_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for commit_sha in needed_commits:
        analysis = analysis_index.get((project_id, commit_sha))
        if analysis is None:
            continue
        point_date = analysis["date"]
        active_smells = [
            entry["smell"]
            for entry in lifecycle_rows
            if entry["open_date"] <= point_date
            and (entry["close_date"] is None or entry["close_date"] > point_date)
        ]
        state_cache[(project_id, commit_sha)] = active_smells
    return state_cache


def build_state_cache_bulk(
    con: sqlite3.Connection,
    candidates: list[dict[str, Any]],
    *,
    analysis_index: dict[tuple[str, str], dict[str, str]],
    rule_name_index: dict[str, str],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Build a commit->active-smells cache by bulk-loading project histories."""
    grouped_candidates: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped_candidates.setdefault(candidate["project_id"], []).append(candidate)

    state_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for project_id, project_candidates in grouped_candidates.items():
        lifecycle_rows = load_project_code_smell_lifecycle(
            con,
            project_id,
            rule_name_index=rule_name_index,
        )
        state_cache.update(
            build_project_state_cache(
                project_id=project_id,
                candidates=project_candidates,
                analysis_index=analysis_index,
                lifecycle_rows=lifecycle_rows,
            )
        )
    return state_cache


def get_active_code_smells_cached(
    con: sqlite3.Connection,
    project_id: str,
    commit_sha: str,
    *,
    analysis_index: dict[tuple[str, str], dict[str, str]],
    rule_name_index: dict[str, str],
    state_cache: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Return active smells for a commit, memoizing by ``(project_id, commit_sha)``."""
    cache_key = (project_id, commit_sha)
    if state_cache is not None and cache_key in state_cache:
        return state_cache[cache_key]

    smells = load_active_code_smells(
        con,
        project_id,
        commit_sha,
        analysis_index=analysis_index,
        rule_name_index=rule_name_index,
    )
    if state_cache is not None:
        state_cache[cache_key] = smells
    return smells


def build_smell_delta(
    before_smells: list[dict[str, Any]],
    after_smells: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build exact-``ISSUE_KEY`` smell delta statistics."""
    before_index = {smell["issue_key"]: smell for smell in before_smells}
    after_index = {smell["issue_key"]: smell for smell in after_smells}

    before_keys = set(before_index)
    after_keys = set(after_index)
    persisted = sorted(before_keys & after_keys)
    resolved = sorted(before_keys - after_keys)
    created = sorted(after_keys - before_keys)

    return {
        "resolved": resolved,
        "created": created,
        "persisted": persisted,
        "counts": {
            "before": len(before_smells),
            "after": len(after_smells),
            "resolved": len(resolved),
            "created": len(created),
            "persisted": len(persisted),
        },
    }


def validate_transition(transition: dict[str, Any]) -> list[str]:
    """Return validation errors for a transition record."""
    counts = transition.get("smell_delta", {}).get("counts", {})
    before_total = counts.get("before")
    after_total = counts.get("after")
    resolved = counts.get("resolved")
    created = counts.get("created")
    persisted = counts.get("persisted")

    errors: list[str] = []
    if before_total != persisted + resolved:
        errors.append("before_total != persisted + resolved")
    if after_total != persisted + created:
        errors.append("after_total != persisted + created")
    return errors


def iter_candidate_transitions(
    con: sqlite3.Connection,
    *,
    project_id: str | None = None,
    analysis_index: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    """Return v1 candidate commits: main-branch, non-merge, single-parent, both analyses."""
    sql = """
    SELECT
        gc.PROJECT_ID,
        gc.COMMIT_HASH,
        gc.AUTHOR_DATE,
        gc.COMMIT_MESSAGE,
        gc.PARENTS
    FROM GIT_COMMITS gc
    JOIN REFACTORING_MINER rm
      ON rm.PROJECT_ID = gc.PROJECT_ID
     AND rm.COMMIT_HASH = gc.COMMIT_HASH
    WHERE gc.IN_MAIN_BRANCH = 'True'
      AND gc.MERGE = 'False'
    """
    params: list[Any] = []
    if project_id is not None:
        sql += " AND gc.PROJECT_ID = ?"
        params.append(project_id)
    sql += " GROUP BY gc.PROJECT_ID, gc.COMMIT_HASH ORDER BY gc.PROJECT_ID, gc.AUTHOR_DATE, gc.COMMIT_HASH"

    rows = con.execute(sql, params).fetchall()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        parents = parse_git_parents(row["PARENTS"])
        if len(parents) != 1:
            continue
        parent_sha = parents[0]
        child_sha = row["COMMIT_HASH"]
        child_analysis = analysis_index.get((row["PROJECT_ID"], child_sha))
        parent_analysis = analysis_index.get((row["PROJECT_ID"], parent_sha))
        if child_analysis is None or parent_analysis is None:
            continue
        candidates.append(
            {
                "project_id": row["PROJECT_ID"],
                "parent_sha": parent_sha,
                "child_sha": child_sha,
                "author_date": row["AUTHOR_DATE"],
                "commit_message": row["COMMIT_MESSAGE"],
                "analysis_before": parent_analysis,
                "analysis_after": child_analysis,
            }
        )
    return candidates


def build_focus_slice(
    *,
    smells_before: list[dict[str, Any]],
    refactorings: list[dict[str, str]],
    changed_files: list[str],
) -> dict[str, Any]:
    smelly_classes_before = sorted(
        {smell["class_name"] for smell in smells_before if smell.get("class_name")}
    )
    touched_from_files = {
        class_name_from_path(path) for path in changed_files if path.endswith(".java")
    }
    touched_from_details: set[str] = set()
    for ref in refactorings:
        touched_from_details.update(
            extract_class_names_from_refactoring_detail(ref.get("detail"))
        )
    touched_classes = sorted(
        {name for name in touched_from_files | touched_from_details if name}
    )
    return {
        "seed_classes": smelly_classes_before,
        "refactoring_touched_classes": touched_classes,
        "refactoring_touched_files": changed_files,
        "expanded_classes": sorted(set(smelly_classes_before) | set(touched_classes)),
        "policy": "smelly_before + refactoring_touched",
    }


def build_transition(
    con: sqlite3.Connection,
    candidate: dict[str, Any],
    *,
    project_registry: dict[str, dict[str, str]],
    analysis_index: dict[tuple[str, str], dict[str, str]],
    rule_name_index: dict[str, str],
    batched_refactorings: dict[tuple[str, str], list[dict[str, str]]] | None = None,
    batched_changed_files: dict[tuple[str, str], list[str]] | None = None,
    state_cache: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    project_meta = project_registry[candidate["project_id"]]
    before_smells = get_active_code_smells_cached(
        con,
        candidate["project_id"],
        candidate["parent_sha"],
        analysis_index=analysis_index,
        rule_name_index=rule_name_index,
        state_cache=state_cache,
    )
    after_smells = get_active_code_smells_cached(
        con,
        candidate["project_id"],
        candidate["child_sha"],
        analysis_index=analysis_index,
        rule_name_index=rule_name_index,
        state_cache=state_cache,
    )
    refactoring_key = (candidate["project_id"], candidate["child_sha"])
    refactorings = (
        batched_refactorings.get(refactoring_key, [])
        if batched_refactorings is not None
        else load_refactoring_events(
            con,
            candidate["project_id"],
            candidate["child_sha"],
        )
    )
    changed_files = (
        batched_changed_files.get(refactoring_key, [])
        if batched_changed_files is not None
        else load_changed_files(
            con,
            candidate["project_id"],
            candidate["child_sha"],
        )
    )
    smell_delta = build_smell_delta(before_smells, after_smells)

    transition = {
        "transition_id": (
            f"{candidate['project_id']}:{candidate['parent_sha']}->{candidate['child_sha']}"
        ),
        "project_id": candidate["project_id"],
        "project_name": project_meta["project_name"],
        "repository_url": project_meta["repository_url"],
        "commit_before": candidate["parent_sha"],
        "commit_after": candidate["child_sha"],
        "refactoring_commit_sha": candidate["child_sha"],
        "commit_message": candidate["commit_message"],
        "author_date": candidate["author_date"],
        "analysis_before": candidate["analysis_before"],
        "analysis_after": candidate["analysis_after"],
        "refactorings": refactorings,
        "state_before": {
            "analysis_key": candidate["analysis_before"]["analysis_key"],
            "smells": before_smells,
            "smells_total": len(before_smells),
            "smelly_classes": sorted(
                {smell["class_name"] for smell in before_smells if smell.get("class_name")}
            ),
        },
        "state_after": {
            "analysis_key": candidate["analysis_after"]["analysis_key"],
            "smells": after_smells,
            "smells_total": len(after_smells),
            "smelly_classes": sorted(
                {smell["class_name"] for smell in after_smells if smell.get("class_name")}
            ),
        },
        "focus": build_focus_slice(
            smells_before=before_smells,
            refactorings=refactorings,
            changed_files=changed_files,
        ),
        "smell_delta": smell_delta,
        "provenance": {
            "dataset": "technical-debt-dataset-v2.0.1",
            "pipeline": "smellai.td_v2.v1",
            "delta_strategy": "exact_issue_key",
        },
    }
    transition["validation_errors"] = validate_transition(transition)
    return transition


def count_candidate_oracles(
    con: sqlite3.Connection,
) -> dict[str, int]:
    """Compute the key DB-level counts used as replication oracles."""
    main_nonmerge_refactoring_commits = con.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT rm.PROJECT_ID, rm.COMMIT_HASH
          FROM REFACTORING_MINER rm
          JOIN GIT_COMMITS gc
            ON gc.PROJECT_ID = rm.PROJECT_ID
           AND gc.COMMIT_HASH = rm.COMMIT_HASH
          WHERE gc.IN_MAIN_BRANCH = 'True'
            AND gc.MERGE = 'False'
          GROUP BY rm.PROJECT_ID, rm.COMMIT_HASH
        )
        """
    ).fetchone()[0]

    main_nonmerge_refactoring_commits_with_child_analysis = con.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT rm.PROJECT_ID, rm.COMMIT_HASH
          FROM REFACTORING_MINER rm
          JOIN GIT_COMMITS gc
            ON gc.PROJECT_ID = rm.PROJECT_ID
           AND gc.COMMIT_HASH = rm.COMMIT_HASH
          JOIN SONAR_ANALYSIS sa
            ON sa.PROJECT_ID = rm.PROJECT_ID
           AND sa.REVISION = rm.COMMIT_HASH
          WHERE gc.IN_MAIN_BRANCH = 'True'
            AND gc.MERGE = 'False'
          GROUP BY rm.PROJECT_ID, rm.COMMIT_HASH
        )
        """
    ).fetchone()[0]

    analysis_index = _load_analysis_index(con)
    single_parent = len(
        iter_candidate_transitions(
            con,
            analysis_index=analysis_index,
        )
    )

    return {
        "main_nonmerge_refactoring_commits": main_nonmerge_refactoring_commits,
        "main_nonmerge_refactoring_commits_with_child_analysis": (
            main_nonmerge_refactoring_commits_with_child_analysis
        ),
        "single_parent_refactoring_commits_with_parent_child_analysis": single_parent,
    }


def extract_transitions(
    db_path: Path,
    *,
    project_id: str | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract transition records plus a summary dictionary."""
    with connect_td_v2(db_path) as con:
        project_registry = load_project_registry(con)
        analysis_index = _load_analysis_index(con)
        rule_name_index = _load_rule_name_index(con)
        all_candidates = iter_candidate_transitions(
            con,
            project_id=project_id,
            analysis_index=analysis_index,
        )
        selected_candidates = all_candidates[:limit] if limit is not None else all_candidates

        batched_refactorings = load_refactoring_events_batch(con, selected_candidates)
        batched_changed_files = load_changed_files_batch(con, selected_candidates)
        state_cache = build_state_cache_bulk(
            con,
            selected_candidates,
            analysis_index=analysis_index,
            rule_name_index=rule_name_index,
        )

        transitions = [
            build_transition(
                con,
                candidate,
                project_registry=project_registry,
                analysis_index=analysis_index,
                rule_name_index=rule_name_index,
                batched_refactorings=batched_refactorings,
                batched_changed_files=batched_changed_files,
                state_cache=state_cache,
            )
            for candidate in selected_candidates
        ]
        validation_error_count = sum(bool(t["validation_errors"]) for t in transitions)
        summary = {
            "db_path": str(db_path),
            "project_filter": project_id,
            "candidate_count_total": len(all_candidates),
            "candidate_count_selected": len(selected_candidates),
            "transition_count": len(transitions),
            "validation_error_count": validation_error_count,
            "smell_state_cache_entries": len(state_cache),
            "smell_state_references": 2 * len(selected_candidates),
            "counts": get_table_counts(con),
            "oracle_counts": count_candidate_oracles(con),
        }
        return transitions, summary


def write_transitions_jsonl(path: Path, transitions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for transition in transitions:
            f.write(json.dumps(transition, ensure_ascii=False) + "\n")


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def write_schema_markdown(path: Path, db_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect_td_v2(db_path) as con:
        path.write_text(render_schema_markdown(con))


__all__ = [
    "RELEVANT_TABLES",
    "build_project_state_cache",
    "build_smell_delta",
    "build_state_cache_bulk",
    "build_transition",
    "class_name_from_path",
    "connect_td_v2",
    "count_candidate_oracles",
    "extract_transitions",
    "get_active_code_smells_cached",
    "get_table_counts",
    "iter_candidate_transitions",
    "load_active_code_smells",
    "load_changed_files",
    "load_changed_files_batch",
    "load_project_code_smell_lifecycle",
    "load_project_registry",
    "load_refactoring_events",
    "load_refactoring_events_batch",
    "load_tdd_raw_df",
    "normalize_component_path",
    "parse_git_parents",
    "render_schema_markdown",
    "validate_transition",
    "write_schema_markdown",
    "write_summary_json",
    "write_transitions_jsonl",
]
