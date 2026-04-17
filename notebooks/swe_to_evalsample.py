import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # SWE-Refactor → EvalSample — playbook for junior engineers

        This notebook is a **step-by-step guide** for preparing one SWE-Refactor
        dataset record into a complete `EvalSample` that can feed the SmellAI
        evaluation pipeline.

        It mirrors **stages A and F** of the system described in `conf.tex`:

        | Step | What happens | conf.tex stage |
        |------|-------------|----------------|
        | 1. Pick record | Select one SWE-Refactor row with known repo URL | dataset prep |
        | 2. Clone repo | `git clone` + `checkout` to the target commit | stage A |
        | 3. Scan with SonarQube | Run sonar-scanner, poll until complete | stage F |
        | 4. Map rules | `RULE_NAME_MAP` turns SQ rule IDs → human smell names | stage F |
        | 5. Check coverage | Are the dataset's expected rules actually detected? | stage F |
        | 6. Assemble `EvalSample` | Build the typed object; validate with pydantic | output |

        Read **top-to-bottom**. Each cell does one small thing and hands its
        result to the next.

        > **Prerequisites**: SonarQube running (`docker compose … up -d`),
        > `SONAR_TOKEN` and `SWE_REFACTOR_PATH` set in `.env`.
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup — imports, paths, credentials
    """)
    return


@app.cell
def _():
    import sys
    import tempfile
    from pathlib import Path

    import git
    import pandas as pd
    from pydantic import TypeAdapter

    # Make smellai importable when the notebook runs from notebooks/
    _REPO_ROOT = Path(
        "/Users/havriil.pietukhin/uni/masterThesis/code/code_cleaned/smellai"
    )
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from smellai import config
    from smellai.datasets import load_swe_raw_df
    from smellai.schema import EvalSample
    from smellai.sonarqube.constants import RULE_NAME_MAP
    from smellai.sonarqube.scanner import scan_commit

    if not config.SONAR_TOKEN:
        print(
            "WARNING: SONAR_TOKEN is not set — add it to .env and restart the kernel."
        )
        print("  Example:  SONAR_TOKEN=sqa_xxxx")
    else:
        print(f"SONAR_URL  = {config.SONAR_URL}")
        print("SONAR_TOKEN = [set]")
    print("imports ok")
    return (
        EvalSample,
        Path,
        RULE_NAME_MAP,
        TypeAdapter,
        config,
        git,
        load_swe_raw_df,
        pd,
        scan_commit,
        tempfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Config constants
    """)
    return


@app.cell
def _(Path, tempfile):
    # ── Project name → GitHub repo URL ────────────────────────────────────────
    # SWE-Refactor records carry `project_name` but no URL. We maintain a small
    # lookup for the projects referenced in conf.tex §III-A. Extend this dict
    # when working with other projects in the dataset.
    PROJECT_TO_REPO_URL: dict[str, str] = {
        "checkstyle": "https://github.com/checkstyle/checkstyle.git",
        "guava": "https://github.com/google/guava.git",
        "junit5": "https://github.com/junit-team/junit5.git",
        "hibernate-orm": "https://github.com/hibernate/hibernate-orm.git",
        "mockito": "https://github.com/mockito/mockito.git",
        "spring-framework": "https://github.com/spring-projects/spring-framework.git",
        "RxJava": "https://github.com/ReactiveX/RxJava.git",
        "commons-lang": "https://github.com/apache/commons-lang.git",
        "elasticsearch": "https://github.com/elastic/elasticsearch.git",
        "flink": "https://github.com/apache/flink.git",
    }

    # ── Refactoring type → SonarQube rules it should eliminate ────────────────
    # Derived from Table I of conf.tex (§II, tab:smells). These are the rules we
    # *expect* to see violated in the pre-refactoring version of the code.
    # Used in step 5 (coverage check) to verify the scanner actually found them.
    REFACTORING_TO_RULES: dict[str, set[str]] = {
        "Extract Method": {"java:S138", "java:S1541", "java:S1067"},
        "Extract Class": {"java:S1200", "java:S110"},
        "Move Method": {"java:S1200"},
        "Extract and Move Method": {"java:S1200", "java:S138", "java:S1541"},
        "Move Attribute": {"java:S1200"},
        "Introduce Parameter Object": {"java:S107"},
        "Consolidate Conditional Expression": {"java:S1871"},
    }

    # ── Working directories ────────────────────────────────────────────────────
    # sonar_cache: cache scan results to disk — re-runs are near-instant.
    # work_dir: temporary directory for the git clone (cleaned up by the OS).
    CACHE_DIR = Path("sonar_cache")
    WORK_DIR = Path(tempfile.mkdtemp(prefix="smellai_nb_"))

    print(f"PROJECT_TO_REPO_URL: {len(PROJECT_TO_REPO_URL)} projects")
    print(f"REFACTORING_TO_RULES covers: {sorted(REFACTORING_TO_RULES)}")
    print(f"CACHE_DIR = {CACHE_DIR.resolve()}")
    print(f"WORK_DIR  = {WORK_DIR}")
    return CACHE_DIR, PROJECT_TO_REPO_URL, REFACTORING_TO_RULES, WORK_DIR


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 1 — Load SWE-Refactor raw frame

    `load_swe_raw_df()` reads the raw JSON and normalises it into a flat
    pandas DataFrame. Every row is one refactoring pair (before/after source
    of a Java class, plus metadata).
    """)
    return


@app.cell
def _(load_swe_raw_df):
    # 1.1 — Load. `load_swe_raw_df()` resolves the path from SWE_REFACTOR_PATH
    # env var or from known default locations on this host.
    df_swe = load_swe_raw_df()
    print(f"swe raw: {df_swe.shape[0]} rows × {df_swe.shape[1]} cols")
    print("columns:", list(df_swe.columns))
    df_swe.head(3)
    return (df_swe,)


@app.cell
def _(df_swe):
    # 1.2 — Peek: distribution of refactoring types.
    _counts = df_swe["refactoring_type"].value_counts()
    print("top refactoring types:")
    print(_counts.head(10).to_string())
    return


@app.cell
def _(df_swe):
    # 1.3 — Peek: which project names appear (needed to build PROJECT_TO_REPO_URL).
    print("project_name values in dataset:")
    print(sorted(df_swe["project_name"].dropna().unique().tolist()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 2 — Pick one record

    We filter for:
    - `is_pure=True` — no mixed-purpose commits
    - `has_tests=True` — test suite available (needed for stage D of the pipeline)
    - `refactoring_type == "Extract Method"` — well-studied; expected rules: `S138`, `S1541`, `S1067`
    - `project_name` in our `PROJECT_TO_REPO_URL` map

    Then take `.iloc[0]` to get a deterministic first match.
    """)
    return


@app.cell
def _(PROJECT_TO_REPO_URL: dict[str, str], df_swe):
    # 2.1 — Filter to candidates we can actually clone.
    # Note: has_tests is almost always False in SWE-Refactor — we drop that filter.
    # is_pure=True is already the whole dataset (all 1099 rows).
    _mask = (df_swe["refactoring_type"] == "Extract Method") & (
        df_swe["project_name"].isin(PROJECT_TO_REPO_URL)
    )
    df_candidates = df_swe[_mask].copy()
    print(f"candidates matching all filters: {len(df_candidates)}")
    df_candidates[
        ["pair_id", "project_name", "commit_id", "refactoring_type"]
    ].head(5)
    return (df_candidates,)


@app.cell
def _(df_candidates):
    # 2.2 — Pick the first match and cast to a plain dict.
    # In production you would pin a specific pair_id for reproducibility.
    assert len(df_candidates) > 0, (
        "No candidates found — expand PROJECT_TO_REPO_URL or relax the filter above."
    )
    rec = df_candidates.iloc[0].to_dict()
    print(f"selected record:")
    print(f"  pair_id:          {rec['pair_id']}")
    print(f"  project_name:     {rec['project_name']}")
    print(f"  commit_id:        {rec['commit_id']}")
    print(f"  refactoring_type: {rec['refactoring_type']}")
    print(f"  file_before:      {rec['file_path_before']}")
    print(f"  jdk_version:      {rec['jdk_version']}")
    return (rec,)


@app.cell
def _(PROJECT_TO_REPO_URL: dict[str, str], rec):
    # 2.3 — Derive the repo URL from the project name.
    repo_url = PROJECT_TO_REPO_URL[rec["project_name"]]
    print(f"repo_url = {repo_url}")
    return (repo_url,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 3 — Clone the repo and checkout the target commit

    This is **stage A** of the conf.tex pipeline: load source code and detect
    the build system. We clone once into `WORK_DIR` and then pass `repo_path`
    to `scan_commit` so it doesn't clone again.

    > `scan_commit` can do this internally — but separating clone from scan
    > lets us inspect the working tree before committing to a full analysis.
    """)
    return


@app.cell
def _(WORK_DIR, git, repo_url):
    # 3.1 — Clone the repo.
    _clone_dir = WORK_DIR / "repo"
    print(f"cloning {repo_url} → {_clone_dir} …")
    _repo = git.Repo.clone_from(repo_url, _clone_dir)
    print(f"clone done. HEAD = {_repo.head.commit.hexsha[:8]}")
    repo_path = _clone_dir
    return (repo_path,)


@app.cell
def _(git, rec, repo_path):
    # 3.2 — Checkout the exact commit from the record.
    _repo = git.Repo(repo_path)
    _repo.git.checkout(rec["commit_id"])
    _head = _repo.head.commit.hexsha
    print(f"checked out: {_head[:8]} (expected: {rec['commit_id'][:8]})")

    # How many Java files are in scope?
    _java_files = list(repo_path.rglob("*.java"))
    print(f"Java files in working tree: {len(_java_files)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 4 — SonarQube scan

    This is **stage F**: scan the target commit via the SonarQube REST API and
    normalise output into typed smell instances with severity labels.

    `scan_commit` will:
    1. Write `sonar-project.properties` to the clone
    2. Invoke `sonar-scanner` CLI (must be on `$PATH`)
    3. Poll `/api/ce/task` until analysis is complete
    4. Fetch all `CODE_SMELL` issues for the 8 rules in `RULE_NAME_MAP`
    5. Return `{file_path: [normalised_issue, ...]}` — normalised via
       `normalize_issue()` which maps `rule → smell_type` using `RULE_NAME_MAP`

    > **`skip_compile=True`** is used here for speed. This means bytecode-
    > based rules (mainly `java:S110 Large Class`) may miss some violations.
    > For production runs, set `skip_compile=False` (requires Maven/Gradle).

    Results are **cached** to `sonar_cache/{commit_sha}_full.json` — re-runs
    complete in milliseconds.
    """)
    return


@app.cell
def _(CACHE_DIR, config, rec, repo_path, repo_url, scan_commit):
    # 4.1 — Run (or load from cache) the full scan.
    print(f"scanning {rec['commit_id'][:8]} … (this may take 1–3 min on first run)")
    smells_by_file = scan_commit(
        repo_url=repo_url,
        commit_sha=rec["commit_id"],
        sonar_url=config.SONAR_URL,
        sonar_token=config.SONAR_TOKEN,
        cache_dir=CACHE_DIR,
        repo_path=repo_path,
        skip_compile=True,
    )

    _total = sum(len(v) for v in smells_by_file.values())
    print(f"scan complete: {_total} issues across {len(smells_by_file)} files")
    return (smells_by_file,)


@app.cell
def _(pd, smells_by_file):
    # 4.2 — Flatten into a DataFrame for easy inspection.
    _rows = []
    for _file, _issues in smells_by_file.items():
        for _i in _issues:
            _rows.append({"file": _file, **_i})

    df_smells = pd.DataFrame(_rows)
    print(f"df_smells: {df_smells.shape[0]} rows")
    if not df_smells.empty:
        print("smell_type distribution:")
        print(df_smells["smell_type"].value_counts().to_string())
    df_smells.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 5 — Rule map + coverage check

    `RULE_NAME_MAP` (from `smellai/sonarqube/constants.py`) defines the 8
    SonarQube rules the scanner looks for — they match Table I of the paper.

    The **coverage check** asks: does the scanner detect at least one issue
    whose rule is among the rules we *expect* for this refactoring type?

    - `expected_rules` — from `REFACTORING_TO_RULES` (defined in step 0)
    - `found_rules` — all distinct rule IDs in `smells_by_file`
    - `covered = bool(expected_rules & found_rules)`

    If `covered=False` it does NOT mean the dataset record is wrong — it may
    mean the smell was already partially fixed, or `skip_compile` missed it.
    The notebook surface this so you can investigate rather than silently skip.
    """)
    return


@app.cell
def _(RULE_NAME_MAP):
    # 5.1 — Display the 8-rule taxonomy (= Table I of conf.tex).
    print("RULE_NAME_MAP (SonarQube rule → smell name from conf.tex Table I):")
    for _rule, _name in RULE_NAME_MAP.items():
        print(f"  {_rule:15s}  →  {_name}")
    return


@app.cell
def _(REFACTORING_TO_RULES: dict[str, set[str]], rec, smells_by_file):
    # 5.2 — Coverage check.
    _rtype = rec["refactoring_type"]
    expected_rules = REFACTORING_TO_RULES.get(_rtype, set())
    found_rules = {
        issue["rule"]
        for issues in smells_by_file.values()
        for issue in issues
    }

    _covered = bool(expected_rules & found_rules)

    print(f"refactoring_type : {_rtype}")
    print(f"expected_rules   : {sorted(expected_rules)}")
    print(f"found_rules      : {sorted(found_rules)}")
    print(f"overlap          : {sorted(expected_rules & found_rules)}")
    print(f"dataset_rules_covered = {_covered}")

    if not _covered and expected_rules:
        print(
            "\n[INFO] No overlap found. Possible reasons:\n"
            "  • skip_compile=True degraded rule S110 (Large Class)\n"
            "  • The smell was already partially addressed in this commit\n"
            "  • project is not in our rule scope (e.g. Kotlin, not Java)\n"
            "  Consider re-running with skip_compile=False or a different record."
        )
    return expected_rules, found_rules


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 6 — Assemble the `EvalSample`

    `EvalSample` is the typed output unit of all dataset loaders in SmellAI
    (see `smellai/schema.py`). It carries three dicts:

    - `inputs` — what the agent sees: source code, repo, commit, file paths
    - `expectations` — ground truth: what the refactored code should look like
    - `tags` — metadata: purity flag, test coverage, scanner output, rule coverage

    We extend the standard SWE projection (from `smellai/datasets.py:_swe_samples`)
    with three extra `tags` keys: `sonar_smells_count`, `dataset_rules_covered`,
    and `expected_rules` / `found_rules`.
    """)
    return


@app.cell
def _(EvalSample, expected_rules, found_rules, rec, repo_url, smells_by_file):
    # 6.1 — Build the EvalSample, mirroring _swe_samples() in datasets.py:243
    # but with repo_url and scanner output added to inputs/tags.
    _total_smells = sum(len(v) for v in smells_by_file.values())
    _covered = bool(expected_rules & found_rules)

    sample = EvalSample.model_validate(
        {
            "source": "swe",
            "sample_id": f"swe:{rec['pair_id']}",
            "inputs": {
                # ── from SWE record ──
                "project_name": rec["project_name"],
                "commit_id": rec["commit_id"],
                "refactoring_type": rec["refactoring_type"],
                "file_path_before": rec["file_path_before"],
                "file_path_after": rec["file_path_after"],
                "class_before": rec["class_before"],
                "source_before": rec["source_before"],
                "jdk_version": rec["jdk_version"],
                "compile_command": rec["compile_command"],
                # ── added in this playbook ──
                "repo_url": repo_url,
            },
            "expectations": {
                "class_after": rec["class_after"],
                "source_after": rec["source_after"],
            },
            "tags": {
                # ── from SWE record ──
                "is_pure": rec["is_pure"],
                "is_compound": rec["is_compound"],
                "has_tests": rec["has_tests"],
                # ── from SonarQube scan ──
                "sonar_smells_count": _total_smells,
                "dataset_rules_covered": _covered,
                "expected_rules": sorted(expected_rules),
                "found_rules": sorted(found_rules),
            },
        }
    )

    print(f"EvalSample created:  source={sample.source}  sample_id={sample.sample_id}")
    print(f"  inputs keys:       {sorted(sample.inputs)}")
    print(f"  expectations keys: {sorted(sample.expectations)}")
    print(f"  tags keys:         {sorted(sample.tags)}")
    return (sample,)


@app.cell
def _(EvalSample, TypeAdapter, sample):
    # 6.2 — Final validation: pydantic confirms the object is well-formed.
    # This is what evaluation/dataset_manager.py does for the full dataset.
    _adapter = TypeAdapter(list[EvalSample])
    _validated = _adapter.validate_python([sample])
    print(f"TypeAdapter validated {len(_validated)} sample(s) — no errors")
    print(f"\nFull EvalSample:\n{sample.model_dump_json(indent=2)[:800]} …")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What we just built — recap

    | Cell group | Stage (conf.tex) | What happened |
    |---|---|---|
    | 1 | dataset prep | Loaded SWE-Refactor, filtered to clonable records |
    | 2 | dataset prep | Derived repo URL from project_name |
    | 3 | **Stage A** | Cloned repo, checked out exact commit |
    | 4 | **Stage F** | SonarQube scan → normalised smell issues |
    | 5 | **Stage F** | Mapped rules to Table I taxonomy; coverage check |
    | 6 | output | Assembled and validated `EvalSample` |

    ---

    ### What is NOT shown here (remaining pipeline stages)

    | Stage | What it does | Code |
    |---|---|---|
    | B | Check every function has a test | `smellai/agents/` |
    | C | LLM-generate missing tests | `smellai/agents/` |
    | D | Run test suite — baseline | `smellai/utils/` |
    | E | Build inter-smell dependency graph (NetworkX) | `smellai/agents/dependency_analysis` |
    | G | Developer smell selection | UI / agent turn |
    | H | BeFS on dependency graph → ordered refactoring plan π | `smellai/agents/` |
    | I | LLM executes each planned refactoring | `smellai/agents/swe_eval` |
    | J | Run tests after refactoring; rollback on failure | `smellai/agents/` |

    The `EvalSample` produced here (`inputs`, `expectations`, `tags`) is the
    **entry point** for all downstream stages — it is what you pass to
    `smellai/evaluation/` for MLflow tracking.
    """)
    return


if __name__ == "__main__":
    app.run()
