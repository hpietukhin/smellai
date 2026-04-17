import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Data preparation — SmellAI (stage 1)

        Three research datasets feed the SmellAI pipeline:

        | Source     | What it is                                      | Unit of a row                |
        |------------|--------------------------------------------------|------------------------------|
        | **rminer** | RMiner 2.0 oracle of mined refactorings          | one refactoring occurrence   |
        | **swe**    | SWE-Refactor benchmark (before/after Java source)| one refactoring pair         |
        | **tdd**    | Technical Debt Dataset v2 (SonarQube issues)     | one code-smell event         |

        The notebook is organised so **every single conversion step is a
        separate, labelled cell** — load → peek → explode → filter → rename →
        coerce → project to `inputs` / `expectations` / `tags` → assemble
        `sample_id` → wrap in `EvalSample` → validate → unify.

        Read top-to-bottom; each cell does one small thing and hands its
        result to the next.
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Setup — imports, paths, schema""")
    return


@app.cell
def _():
    import json
    import sqlite3
    import sys
    from pathlib import Path
    from typing import Any

    import pandas as pd
    from pydantic import TypeAdapter

    # Make the smellai package importable when the notebook runs from notebooks/
    _REPO_ROOT = Path(
        "/Users/havriil.pietukhin/uni/masterThesis/code/code_cleaned/smellai"
    )
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from smellai.schema import EvalSample  # noqa: E402

    print("imports ok")
    return Any, EvalSample, TypeAdapter, json, pd, sqlite3


@app.cell
def _():
    from pathlib import Path as _P

    # Absolute paths to raw dataset files on this host.
    RMINER_ORACLE_PATH = _P(
        "/Users/havriil.pietukhin/uni/masterThesis/datasets/rminer_oracle_java1.json"
    )
    SWE_REFACTOR_JSON = _P(
        "/Users/havriil.pietukhin/uni/masterThesis/SWE-Refactor/SWE-Refactor/pure_refactoring_data.json"
    )
    TDD_DB_PATH = _P(
        "/Users/havriil.pietukhin/uni/masterThesis/datasets/td_V2.db"
    )

    for _p in (RMINER_ORACLE_PATH, SWE_REFACTOR_JSON, TDD_DB_PATH):
        assert _p.exists(), f"missing: {_p}"
    print("raw sources ok")
    return RMINER_ORACLE_PATH, SWE_REFACTOR_JSON, TDD_DB_PATH


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Group A — raw frames

        Load each source into a *flat* pandas DataFrame with source-specific
        columns. No harmonisation, no pydantic. Every atomic operation
        (load, peek, explode, filter, rename, coerce, build DataFrame) gets
        its own cell so nothing is hidden inside a one-shot list-comprehension.
        """
    )
    return


# =============================================================================
# A.1 · RMiner oracle — raw frame
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### A.1 · RMiner oracle""")
    return


@app.cell
def _(RMINER_ORACLE_PATH, json):
    # A.1.1 — Load the raw JSON file. Top-level is a list of commit objects.
    with open(RMINER_ORACLE_PATH) as _f:
        rminer_commits = json.load(_f)

    print(f"rminer: {len(rminer_commits)} commits loaded")
    return (rminer_commits,)


@app.cell
def _(rminer_commits):
    # A.1.2 — Peek one commit to confirm the nested shape we have to flatten.
    _first = rminer_commits[0]
    print("top-level keys:", list(_first.keys()))
    print("refactorings on first commit:", len(_first.get("refactorings", [])))
    _first.get("refactorings", [{}])[0]
    return


@app.cell
def _(Any, rminer_commits):
    # A.1.3 — EXPLODE: one row per (commit, refactoring) pair, still carrying
    # the raw nested refactoring dict alongside commit-level metadata.
    # No field renaming yet, no filtering.
    rminer_exploded: list[dict[str, Any]] = []
    for _commit in rminer_commits:
        for _ref in _commit.get("refactorings", []):
            rminer_exploded.append(
                {
                    "repository": _commit.get("repository", ""),
                    "commit_sha": _commit.get("sha1", ""),
                    "author": _commit.get("author", ""),
                    "time": _commit.get("time", ""),
                    "_ref": _ref,
                }
            )
    print(f"exploded: {len(rminer_exploded)} (commit, refactoring) pairs")
    return (rminer_exploded,)


@app.cell
def _(rminer_exploded):
    # A.1.4 — FILTER: keep only TP (true-positive) rows. The oracle also
    # contains FP rows we do not want in training/eval data.
    rminer_tp = [r for r in rminer_exploded if r["_ref"].get("validation") == "TP"]
    print(
        f"filter TP: {len(rminer_tp)} / {len(rminer_exploded)} rows kept "
        f"({100 * len(rminer_tp) / max(len(rminer_exploded), 1):.1f}%)"
    )
    return (rminer_tp,)


@app.cell
def _(Any, rminer_tp):
    # A.1.5 — RENAME / project nested refactoring fields onto the flat row and
    # coerce the `detectionTools` list into a comma-separated string so the
    # resulting column has a single scalar dtype.
    rminer_flat: list[dict[str, Any]] = []
    for _r in rminer_tp:
        _ref = _r["_ref"]
        _tools = _ref.get("detectionTools", [])
        rminer_flat.append(
            {
                "repository": _r["repository"],
                "commit_sha": _r["commit_sha"],
                "author": _r["author"],
                "time": _r["time"],
                "refactoring_type": _ref.get("type", ""),
                "description": _ref.get("description", ""),
                "validation": _ref.get("validation", ""),
                "detection_tools": ",".join(_tools)
                if isinstance(_tools, list)
                else str(_tools),
            }
        )
    print(f"flat rows: {len(rminer_flat)}, {len(rminer_flat[0])} columns each")
    return (rminer_flat,)


@app.cell
def _(pd, rminer_flat):
    # A.1.6 — Materialise the flat list of dicts into a pandas DataFrame.
    df_rminer_raw = pd.DataFrame(rminer_flat)
    print(
        f"df_rminer_raw: {df_rminer_raw.shape[0]} rows x {df_rminer_raw.shape[1]} cols"
    )
    print(df_rminer_raw.dtypes.to_dict())
    df_rminer_raw.head(3)
    return (df_rminer_raw,)


# =============================================================================
# A.2 · SWE-Refactor — raw frame
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### A.2 · SWE-Refactor pure refactorings""")
    return


@app.cell
def _(SWE_REFACTOR_JSON, json):
    # A.2.1 — Load the raw JSON. Top-level is a flat list of refactoring
    # records, each carrying before/after source of a whole Java class.
    with open(SWE_REFACTOR_JSON) as _f:
        swe_records = json.load(_f)

    print(f"swe: {len(swe_records)} records loaded")
    return (swe_records,)


@app.cell
def _(swe_records):
    # A.2.2 — Peek the keys of the first record to confirm field names before
    # we hand-map them.
    print("keys on first record:")
    for _k in sorted(swe_records[0].keys()):
        print(f"  {_k}")
    return


@app.cell
def _(Any):
    # A.2.3 — Define the `compileJDK` coercion helper.
    # The raw field is often a float (e.g. 1.8) — normalise to the int major
    # version the build system expects (8).
    def coerce_jdk(v: Any) -> int:
        if v is None:
            return 11
        if isinstance(v, float) and v == 1.8:
            return 8
        return int(round(float(v)))

    # quick sanity checks
    assert coerce_jdk(1.8) == 8
    assert coerce_jdk(11) == 11
    assert coerce_jdk(None) == 11
    print("coerce_jdk ok")
    return (coerce_jdk,)


@app.cell
def _(Any, swe_records):
    # A.2.4 — RENAME raw camelCase fields to snake_case, keeping the full
    # before/after source payloads 1:1. No type coercion here yet.
    swe_renamed: list[dict[str, Any]] = []
    for _rec in swe_records:
        swe_renamed.append(
            {
                "pair_id": _rec.get("uniqueId") or _rec.get("commitId", ""),
                "project_name": _rec.get("projectName", ""),
                "commit_id": _rec.get("commitId", ""),
                "refactoring_type": _rec.get("type", ""),
                "file_path_before": _rec.get("filePathBefore", ""),
                "file_path_after": _rec.get("filePathAfter", ""),
                "class_before": _rec.get("sourceCodeBeforeForWhole", ""),
                "class_after": _rec.get("sourceCodeAfterForWhole", ""),
                "source_before": _rec.get("sourceCodeBeforeRefactoring", ""),
                "source_after": _rec.get("sourceCodeAfterRefactoring", ""),
                "compile_command": _rec.get("compileCommand", ""),
                "_raw_is_pure": _rec.get("isPureRefactoring", False),
                "_raw_has_tests": _rec.get("hasTestC", False),
                "_raw_compile_jdk": _rec.get("compileJDK"),
            }
        )
    print(f"renamed: {len(swe_renamed)} rows, {len(swe_renamed[0])} cols")
    return (swe_renamed,)


@app.cell
def _(Any, coerce_jdk, swe_renamed):
    # A.2.5 — COERCE types: booleans, compound-type detection, JDK normalisation.
    # The coerced row is the final per-record raw shape we expose as a DataFrame.
    swe_coerced: list[dict[str, Any]] = []
    for _r in swe_renamed:
        _type = _r["refactoring_type"]
        swe_coerced.append(
            {
                **{k: v for k, v in _r.items() if not k.startswith("_raw_")},
                "is_compound": "+" in _type,
                "is_pure": bool(_r["_raw_is_pure"]),
                "has_tests": bool(_r["_raw_has_tests"]),
                "jdk_version": coerce_jdk(_r["_raw_compile_jdk"]),
            }
        )
    print(
        f"coerced: is_pure={sum(r['is_pure'] for r in swe_coerced)}, "
        f"is_compound={sum(r['is_compound'] for r in swe_coerced)}"
    )
    return (swe_coerced,)


@app.cell
def _(pd, swe_coerced):
    # A.2.6 — Materialise the coerced rows into a DataFrame.
    df_swe_raw = pd.DataFrame(swe_coerced)
    print(
        f"df_swe_raw: {df_swe_raw.shape[0]} rows x {df_swe_raw.shape[1]} cols"
    )
    df_swe_raw.head(3)
    return (df_swe_raw,)


# =============================================================================
# A.3 · Technical Debt Dataset v2 — raw frame
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### A.3 · Technical Debt Dataset v2 (SonarQube issues)""")
    return


@app.cell
def _():
    # A.3.1 — Define the SQL: SONAR_ISSUES joined to SONAR_ANALYSIS twice
    # (creation + close commit SHAs). One row per SonarQube issue event.
    # LIMIT 5000 keeps the notebook interactive — production converter will
    # stream/chunk instead (see TODO stage 2).
    tdd_sql = """
    SELECT
        si.PROJECT_ID        AS project,
        sa_c.REVISION        AS creation_commit,
        sa_x.REVISION        AS close_commit,
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
        si.DEBT              AS debt
    FROM SONAR_ISSUES si
    LEFT JOIN SONAR_ANALYSIS sa_c
        ON sa_c.ANALYSIS_KEY = si.CREATION_ANALYSIS_KEY
    LEFT JOIN SONAR_ANALYSIS sa_x
        ON sa_x.ANALYSIS_KEY = si.CLOSE_ANALYSIS_KEY
        AND si.CLOSE_ANALYSIS_KEY != ''
    LIMIT 5000
    """
    print(f"tdd_sql: {len(tdd_sql.splitlines())} lines")
    return (tdd_sql,)


@app.cell
def _(TDD_DB_PATH, pd, sqlite3, tdd_sql):
    # A.3.2 — Open the SQLite connection and execute the query. pandas handles
    # cursor → DataFrame mapping; columns come from the SELECT aliases.
    with sqlite3.connect(TDD_DB_PATH) as _con:
        df_tdd_raw_untyped = pd.read_sql_query(tdd_sql, _con)
    print(
        f"df_tdd_raw_untyped: {df_tdd_raw_untyped.shape[0]} rows "
        f"x {df_tdd_raw_untyped.shape[1]} cols (LIMIT 5000)"
    )
    return (df_tdd_raw_untyped,)


@app.cell
def _(df_tdd_raw_untyped):
    # A.3.3 — Inspect raw dtypes. `start_line` / `end_line` land as *object*
    # because some rules emit empty strings; we fix that in the next cell.
    print(df_tdd_raw_untyped.dtypes.to_dict())
    df_tdd_raw_untyped.head(3)
    return


@app.cell
def _(Any):
    # A.3.4 — Define the "int-or-None" coercer used for line numbers and any
    # other numeric-ish column that may arrive as an empty string.
    def as_int_or_none(v: Any) -> int | None:
        if v is None or v == "":
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    assert as_int_or_none("") is None
    assert as_int_or_none(None) is None
    assert as_int_or_none("42") == 42
    assert as_int_or_none(42) == 42
    print("as_int_or_none ok")
    return (as_int_or_none,)


@app.cell
def _(as_int_or_none, df_tdd_raw_untyped):
    # A.3.5 — COERCE `start_line` / `end_line` via the helper. The resulting
    # column is `object` (nullable ints) — scorers must treat it as optional.
    df_tdd_raw = df_tdd_raw_untyped.copy()
    df_tdd_raw["start_line"] = df_tdd_raw["start_line"].map(as_int_or_none)
    df_tdd_raw["end_line"] = df_tdd_raw["end_line"].map(as_int_or_none)
    print(
        f"df_tdd_raw: {df_tdd_raw.shape[0]} rows; "
        f"null start_line = {df_tdd_raw['start_line'].isna().sum()}"
    )
    df_tdd_raw.head(3)
    return (df_tdd_raw,)


# =============================================================================
# Group B — conform each raw frame into EvalSample
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Group B — conforming operations

        Every source is projected into the *generalised* shape:

        ```
        EvalSample(source, sample_id, inputs: dict, expectations: dict, tags: dict)
        ```

        The three dicts mirror MLflow GenAI's `evaluate()` contract:
        `inputs` is what the agent sees, `expectations` is ground truth for
        scorers, `tags` is metadata for filtering & stratification.

        Per source, **every projection is a separate cell**: build `inputs`,
        build `expectations`, build `tags`, compose `sample_id`, then zip
        them into `EvalSample` objects. Each intermediate step is inspectable.
        """
    )
    return


# -----------------------------------------------------------------------------
# B.1 · RMiner → EvalSample
# -----------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### B.1 · RMiner → `EvalSample`

        RMiner is a **detection** oracle — there is no post-refactoring source
        code. `expectations` is therefore empty; the generative scorer will
        need a separate oracle-to-diff join later (see TODO stage 2 caveat).
        """
    )
    return


@app.cell
def _(Any, df_rminer_raw):
    # B.1.1 — PROJECT: `inputs` dict per row. This is what the agent will see
    # at inference time for an RMiner sample.
    rminer_inputs: list[dict[str, Any]] = [
        {
            "repository": r["repository"],
            "commit_sha": r["commit_sha"],
            "refactoring_type": r["refactoring_type"],
            "description": r["description"],
        }
        for r in df_rminer_raw.to_dict("records")
    ]
    print(f"rminer inputs: {len(rminer_inputs)}")
    rminer_inputs[0]
    return (rminer_inputs,)


@app.cell
def _(Any, df_rminer_raw):
    # B.1.2 — PROJECT: `expectations` dict. Empty for all RMiner rows — the
    # oracle carries no ground-truth post-refactoring source.
    rminer_expectations: list[dict[str, Any]] = [
        {} for _ in range(len(df_rminer_raw))
    ]
    print(f"rminer expectations: {len(rminer_expectations)} (all empty)")
    return (rminer_expectations,)


@app.cell
def _(Any, df_rminer_raw):
    # B.1.3 — PROJECT: `tags` dict. Not shown to the agent — used for
    # stratification, filtering, and MLflow run tagging.
    rminer_tags: list[dict[str, Any]] = [
        {
            "validation": r["validation"],
            "detection_tools": r["detection_tools"],
            "author": r["author"],
            "time": r["time"],
        }
        for r in df_rminer_raw.to_dict("records")
    ]
    print(f"rminer tags: {len(rminer_tags)}")
    rminer_tags[0]
    return (rminer_tags,)


@app.cell
def _(df_rminer_raw):
    # B.1.4 — Compose stable `sample_id` strings. A commit can contribute
    # multiple refactorings, so we disambiguate with the row index.
    rminer_sample_ids: list[str] = [
        f"rminer:{r['commit_sha']}:{i}"
        for i, r in enumerate(df_rminer_raw.to_dict("records"))
    ]
    print(
        f"rminer sample_ids: {len(rminer_sample_ids)} "
        f"(unique: {len(set(rminer_sample_ids))})"
    )
    return (rminer_sample_ids,)


@app.cell
def _(
    EvalSample,
    rminer_expectations,
    rminer_inputs,
    rminer_sample_ids,
    rminer_tags,
):
    # B.1.5 — WRAP: zip all four per-row vectors into `EvalSample` objects.
    samples_rminer: list[EvalSample] = [
        EvalSample(
            source="rminer",
            sample_id=sid,
            inputs=inp,
            expectations=exp,
            tags=tg,
        )
        for sid, inp, exp, tg in zip(
            rminer_sample_ids, rminer_inputs, rminer_expectations, rminer_tags
        )
    ]
    print(f"samples_rminer: {len(samples_rminer)}")
    samples_rminer[0]
    return (samples_rminer,)


# -----------------------------------------------------------------------------
# B.2 · SWE-Refactor → EvalSample
# -----------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### B.2 · SWE-Refactor → `EvalSample`

        SWE-Refactor is the only source with real post-refactoring source
        code, so `expectations` carries `class_after` / `source_after` —
        everything a compile-and-test scorer needs.
        """
    )
    return


@app.cell
def _(Any, df_swe_raw):
    # B.2.1 — PROJECT: `inputs`. Everything the agent needs to *reproduce* a
    # refactoring: project, commit, type, source BEFORE, and build metadata.
    swe_inputs: list[dict[str, Any]] = [
        {
            "project_name": r["project_name"],
            "commit_id": r["commit_id"],
            "refactoring_type": r["refactoring_type"],
            "file_path_before": r["file_path_before"],
            "file_path_after": r["file_path_after"],
            "class_before": r["class_before"],
            "source_before": r["source_before"],
            "jdk_version": int(r["jdk_version"]),
            "compile_command": r["compile_command"],
        }
        for r in df_swe_raw.to_dict("records")
    ]
    print(f"swe inputs: {len(swe_inputs)}")
    return (swe_inputs,)


@app.cell
def _(Any, df_swe_raw):
    # B.2.2 — PROJECT: `expectations`. Ground-truth source code AFTER the
    # refactoring, at both class and fragment granularity.
    swe_expectations: list[dict[str, Any]] = [
        {
            "class_after": r["class_after"],
            "source_after": r["source_after"],
        }
        for r in df_swe_raw.to_dict("records")
    ]
    print(f"swe expectations: {len(swe_expectations)}")
    return (swe_expectations,)


@app.cell
def _(Any, df_swe_raw):
    # B.2.3 — PROJECT: `tags`. Purity / compound / has_tests flags for slicing
    # the benchmark at eval time.
    swe_tags: list[dict[str, Any]] = [
        {
            "is_pure": bool(r["is_pure"]),
            "is_compound": bool(r["is_compound"]),
            "has_tests": bool(r["has_tests"]),
        }
        for r in df_swe_raw.to_dict("records")
    ]
    print(f"swe tags: {len(swe_tags)}")
    return (swe_tags,)


@app.cell
def _(df_swe_raw):
    # B.2.4 — Compose `sample_id`. `pair_id` is already unique per refactoring.
    swe_sample_ids: list[str] = [
        f"swe:{r['pair_id']}" for r in df_swe_raw.to_dict("records")
    ]
    print(
        f"swe sample_ids: {len(swe_sample_ids)} "
        f"(unique: {len(set(swe_sample_ids))})"
    )
    return (swe_sample_ids,)


@app.cell
def _(EvalSample, swe_expectations, swe_inputs, swe_sample_ids, swe_tags):
    # B.2.5 — WRAP: build `EvalSample` list from the four per-row vectors.
    samples_swe: list[EvalSample] = [
        EvalSample(
            source="swe",
            sample_id=sid,
            inputs=inp,
            expectations=exp,
            tags=tg,
        )
        for sid, inp, exp, tg in zip(
            swe_sample_ids, swe_inputs, swe_expectations, swe_tags
        )
    ]
    print(f"samples_swe: {len(samples_swe)}")
    samples_swe[0]
    return (samples_swe,)


# -----------------------------------------------------------------------------
# B.3 · Technical Debt Dataset → EvalSample
# -----------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### B.3 · Technical Debt Dataset → `EvalSample`

        TDD rows are SonarQube issue events. `expectations` carries the
        `close_commit` (empty if still open); scorers that evaluate "did the
        agent actually fix the smell?" need to diff that commit.
        """
    )
    return


@app.cell
def _(Any, df_tdd_raw):
    # B.3.1 — PROJECT: `inputs`. Project, creation commit, the SonarQube rule
    # that fired, the component file, the line range, and the raw message.
    tdd_inputs: list[dict[str, Any]] = [
        {
            "project": r["project"],
            "creation_commit": r["creation_commit"] or "",
            "rule": r["rule"],
            "component": r["component"],
            "message": r["message"],
            "start_line": r["start_line"],
            "end_line": r["end_line"],
            "issue_type": r["issue_type"],
        }
        for r in df_tdd_raw.to_dict("records")
    ]
    print(f"tdd inputs: {len(tdd_inputs)}")
    return (tdd_inputs,)


@app.cell
def _(Any, df_tdd_raw):
    # B.3.2 — PROJECT: `expectations`. Only the resolving commit SHA.
    tdd_expectations: list[dict[str, Any]] = [
        {"close_commit": r["close_commit"] or ""}
        for r in df_tdd_raw.to_dict("records")
    ]
    _n_closed = sum(1 for e in tdd_expectations if e["close_commit"])
    print(
        f"tdd expectations: {len(tdd_expectations)} "
        f"({_n_closed} with a close_commit)"
    )
    return (tdd_expectations,)


@app.cell
def _(Any, df_tdd_raw):
    # B.3.3 — PROJECT: `tags`. Priority / effort metadata used for slicing.
    tdd_tags: list[dict[str, Any]] = [
        {
            "severity": r["severity"],
            "status": r["status"],
            "resolution": r["resolution"] or "",
            "effort": r["effort"],
            "debt": r["debt"],
        }
        for r in df_tdd_raw.to_dict("records")
    ]
    print(f"tdd tags: {len(tdd_tags)}")
    return (tdd_tags,)


@app.cell
def _(df_tdd_raw):
    # B.3.4 — Compose `sample_id`. Project + creation commit + rule + component
    # + start_line — stable within the source, disambiguates repeats per file.
    tdd_sample_ids: list[str] = [
        f"tdd:{r['project']}:{r['creation_commit']}:{r['rule']}:{r['component']}:{r['start_line']}"
        for r in df_tdd_raw.to_dict("records")
    ]
    print(
        f"tdd sample_ids: {len(tdd_sample_ids)} "
        f"(unique: {len(set(tdd_sample_ids))})"
    )
    return (tdd_sample_ids,)


@app.cell
def _(EvalSample, tdd_expectations, tdd_inputs, tdd_sample_ids, tdd_tags):
    # B.3.5 — WRAP: build `EvalSample` list.
    samples_tdd: list[EvalSample] = [
        EvalSample(
            source="tdd",
            sample_id=sid,
            inputs=inp,
            expectations=exp,
            tags=tg,
        )
        for sid, inp, exp, tg in zip(
            tdd_sample_ids, tdd_inputs, tdd_expectations, tdd_tags
        )
    ]
    print(f"samples_tdd: {len(samples_tdd)}")
    samples_tdd[0]
    return (samples_tdd,)


# =============================================================================
# Final — unified DataFrame[EvalSample]
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Final — unified `DataFrame[EvalSample]`

        One long-format frame, one row per validated `EvalSample`. `source`
        is the discriminator; downstream code should branch on it and read
        `inputs` / `expectations` / `tags`.

        Handed to a coding agent via `df_eval.to_dict("records")` — each
        record is already a dict in MLflow-GenAI-ready shape.
        """
    )
    return


@app.cell
def _(samples_rminer, samples_swe, samples_tdd):
    # Final.1 — Concatenate the three per-source sample lists.
    all_samples = samples_rminer + samples_swe + samples_tdd
    print(
        f"all_samples: {len(all_samples)} "
        f"(rminer={len(samples_rminer)}, swe={len(samples_swe)}, tdd={len(samples_tdd)})"
    )
    return (all_samples,)


@app.cell
def _(EvalSample, TypeAdapter, all_samples):
    # Final.2 — One-shot validation through a pydantic TypeAdapter. Fails fast
    # if any row violates the `EvalSample` contract.
    validated_samples: list[EvalSample] = TypeAdapter(
        list[EvalSample]
    ).validate_python([s.model_dump() for s in all_samples])
    print(f"validated: {len(validated_samples)} EvalSample rows")
    return (validated_samples,)


@app.cell
def _(pd, validated_samples):
    # Final.3 — Build the unified DataFrame. Each row carries a `source`
    # discriminator plus the three dict columns.
    df_eval = pd.DataFrame([s.model_dump() for s in validated_samples])
    print(
        f"df_eval: {len(df_eval)} rows  |  "
        f"sources: {df_eval['source'].value_counts().to_dict()}"
    )
    df_eval.head(3)
    return (df_eval,)


@app.cell
def _(df_eval):
    # Final.4 — Smoke check: round-trip to agent-ready records and compare
    # the per-source projection keys so the reader can see at a glance what
    # the agent will actually receive.
    _records = df_eval.to_dict("records")
    assert len(_records) == len(df_eval)
    _first_by_source = {
        s: next(r for r in _records if r["source"] == s)
        for s in ("rminer", "swe", "tdd")
    }
    print("keys per source:")
    for _src, _rec in _first_by_source.items():
        print(
            f"  {_src:6s} "
            f"inputs={sorted(_rec['inputs'])[:4]}... "
            f"expectations={sorted(_rec['expectations'])} "
            f"tags={sorted(_rec['tags'])[:3]}..."
        )
    _first_by_source["swe"]
    return


if __name__ == "__main__":
    app.run()
