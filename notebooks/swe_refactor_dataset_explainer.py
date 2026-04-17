import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # SWE-Refactor dataset — complete field-by-field explanation, transformation, and PDF-spec mapping

        This notebook is intentionally about **one dataset only**: **SWE-Refactor**.

        It answers three concrete questions:

        1. **What fields are literally present in the real JSON file?**
        2. **How exactly does SmellAI combine and transform those fields?**
        3. **How does that concrete payload relate to the paper spec in `docs/conf_Pietukhin_10_3_rev2-2.pdf`?**

        We do **not** discuss RMiner or TDD here.
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup
    """)
    return


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import pandas as pd
    from pydantic import TypeAdapter

    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))

    from smellai_datasets import EvalSample, load_eval_samples, load_swe_raw_df
    from smellai_datasets.loaders import _resolve_swe_path
    from swe_refactor.dataset import RefactoringRecord

    print("imports ok")
    return (
        EvalSample,
        RefactoringRecord,
        TypeAdapter,
        json,
        load_eval_samples,
        load_swe_raw_df,
        pd,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Resolve and load the real raw dataset
    """)
    return


@app.cell
def _():
    resolved_path = _resolve_swe_path(None)
    assert resolved_path is not None and resolved_path.exists(), (
        "SWE-Refactor dataset not found at the configured path."
    )

    print(f"resolved_path = {resolved_path}")
    return (resolved_path,)


@app.cell
def _(json, resolved_path):
    with resolved_path.open() as f:
        raw_records = json.load(f)

    print(f"raw records loaded: {len(raw_records)}")
    raw_records[0]
    return (raw_records,)


@app.cell
def _(pd, raw_records):
    all_raw_keys = sorted({key for rec in raw_records for key in rec.keys()})
    df_raw_keys = pd.DataFrame({"raw_key": all_raw_keys})
    print(f"distinct keys in the real dataset: {len(df_raw_keys)}")
    df_raw_keys
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. What one SWE-Refactor record is

    One raw row is one **refactoring pair** from a real Java project.

    The key idea is that the row contains **both sides of the change**:

    - the **before** code
    - the **after** code

    That is why this dataset is so important for SmellAI:
    it can be turned into an evaluation sample where the system sees the **before** state
    and is evaluated against the known **after** state.
    """)
    return


@app.cell
def _(raw_records):
    sample_raw_record = raw_records[0]
    print("first raw record:")
    sample_raw_record
    return (sample_raw_record,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Literally every field in the real SWE-Refactor JSON

    This section is exhaustive.

    For each field we show:
    - presence across the 1,099 records
    - runtime value types that actually occur
    - what the field means
    - how SmellAI uses it (or drops it)
    """)
    return


@app.cell
def _(pd, raw_records):
    field_explanations = {
        "callGraph": {
            "meaning": "Optional structured call-graph fragment for the example.",
            "role": "Present only in a subset of rows; currently dropped by the unified loader.",
        },
        "callInfo": {
            "meaning": "Auxiliary call-related metadata, often just 'N/A'.",
            "role": "Raw metadata only; currently dropped.",
        },
        "classNameBefore": {
            "meaning": "Fully qualified class name before refactoring.",
            "role": "Raw structural metadata; currently dropped.",
        },
        "classNameBeforeSet": {
            "meaning": "List/set of involved class names before refactoring.",
            "role": "Raw structural metadata; currently dropped.",
        },
        "classSignatureBefore": {
            "meaning": "Class declaration/signature before refactoring.",
            "role": "Raw structural metadata; currently dropped.",
        },
        "classSignatureBeforeSet": {
            "meaning": "List/set of class signatures before refactoring.",
            "role": "Raw structural metadata; currently dropped.",
        },
        "commitId": {
            "meaning": "Commit SHA anchoring the example.",
            "role": "Projected to `inputs.commit_id`.",
        },
        "compileCommand": {
            "meaning": "Build command for the project, e.g. Maven or Gradle invocation.",
            "role": "Projected to `inputs.compile_command`.",
        },
        "compileJDK": {
            "meaning": "JDK version needed for compilation. In this file it appears as ints and also float `1.8`.",
            "role": "Validated/coerced by `RefactoringRecord`; becomes `inputs.jdk_version`.",
        },
        "compileResultBefore": {
            "meaning": "Whether the before-version compiled successfully during dataset preparation.",
            "role": "Raw quality metadata; currently dropped.",
        },
        "compileResultCurrent": {
            "meaning": "Whether the after/current version compiled successfully during dataset preparation.",
            "role": "Raw quality metadata; currently dropped.",
        },
        "coverageInfo": {
            "meaning": "Coverage-related metadata/counters attached to the record.",
            "role": "Raw QA metadata; currently dropped.",
        },
        "description": {
            "meaning": "Human-readable description of the refactoring instance.",
            "role": "Present in raw data but currently not projected into the final `EvalSample`.",
        },
        "diffLocations": {
            "meaning": "Structured changed spans: file path + start/end line/column ranges.",
            "role": "Useful localization metadata; currently dropped.",
        },
        "diffSourceCode": {
            "meaning": "Textual diff snippet representing the change.",
            "role": "Raw change representation; currently dropped.",
        },
        "diffSourceCodeSet": {
            "meaning": "List-form diff snippet container.",
            "role": "Raw change representation; currently dropped.",
        },
        "filePathAfter": {
            "meaning": "Source file path after refactoring.",
            "role": "Projected to `inputs.file_path_after`.",
        },
        "filePathBefore": {
            "meaning": "Source file path before refactoring.",
            "role": "Projected to `inputs.file_path_before`.",
        },
        "hasTestC": {
            "meaning": "Dataset flag related to test availability/coverage. In this concrete file it is almost absent.",
            "role": "Mapped to `tags.has_tests`.",
        },
        "invokedMethodSet": {
            "meaning": "List of invoked methods associated with the example.",
            "role": "Raw auxiliary metadata; currently dropped.",
        },
        "isPureRefactoring": {
            "meaning": "Whether the change is considered a pure refactoring rather than a mixed-purpose code change.",
            "role": "Mapped to `tags.is_pure`.",
        },
        "methodNameBefore": {
            "meaning": "Fully qualified method name before refactoring.",
            "role": "Raw structural metadata; currently dropped.",
        },
        "methodNameBeforeSet": {
            "meaning": "List/set of involved method names before refactoring.",
            "role": "Raw structural metadata; currently dropped.",
        },
        "moveFileExist": {
            "meaning": "Boolean flag indicating whether the target file exists for move-style refactorings.",
            "role": "Raw transformation metadata; currently dropped.",
        },
        "packageNameBefore": {
            "meaning": "Package name before refactoring.",
            "role": "Raw structural metadata; currently dropped.",
        },
        "projectName": {
            "meaning": "Repository/project name the record comes from.",
            "role": "Projected to `inputs.project_name`.",
        },
        "purityCheckResultList": {
            "meaning": "Detailed justification/evidence behind the purity decision.",
            "role": "Supports the `isPureRefactoring` flag, but only the final boolean is projected.",
        },
        "sourceCodeAfterForWhole": {
            "meaning": "Entire source file/class after refactoring.",
            "role": "Projected to `expectations.class_after`.",
        },
        "sourceCodeAfterRefactoring": {
            "meaning": "Local refactored fragment after the change.",
            "role": "Projected to `expectations.source_after`.",
        },
        "sourceCodeBeforeForWhole": {
            "meaning": "Entire source file/class before refactoring.",
            "role": "Projected to `inputs.class_before`.",
        },
        "sourceCodeBeforeRefactoring": {
            "meaning": "Local fragment before refactoring.",
            "role": "Projected to `inputs.source_before`.",
        },
        "testResult": {
            "meaning": "Whether tests passed in dataset preparation for that record.",
            "role": "Raw QA metadata; currently dropped.",
        },
        "type": {
            "meaning": "Refactoring label such as Extract Method, Move Method, Inline Method, etc.",
            "role": "Projected to `inputs.refactoring_type`; also used to derive `tags.is_compound`.",
        },
        "uniqueId": {
            "meaning": "Stable pair identifier for the refactoring example.",
            "role": "Becomes `pair_id`, then `sample_id = swe:<pair_id>`.",
        },
    }

    rows = []
    for key in sorted({k for rec in raw_records for k in rec.keys()}):
        values = [rec.get(key) for rec in raw_records if key in rec]
        non_null = [v for v in values if v is not None]
        types = sorted({type(v).__name__ for v in non_null})
        example = repr(non_null[0])[:140] if non_null else "None"
        meta = field_explanations[key]
        rows.append(
            {
                "field": key,
                "present_in_records": f"{len(values)}/{len(raw_records)}",
                "runtime_types": ", ".join(types),
                "meaning": meta["meaning"],
                "role_in_smellai": meta["role"],
                "example_value": example,
            }
        )

    df_all_fields = pd.DataFrame(rows)
    print(f"exhaustive raw fields: {len(df_all_fields)}")
    df_all_fields
    return


@app.cell
def _(pd, raw_records):
    dataset_stats = pd.DataFrame(
        {
            "metric": [
                "records",
                "distinct_projects",
                "distinct_refactoring_types",
                "raw_fields",
                "records_with_callGraph",
                "records_with_hasTestC_present",
                "records_with_testResult_present",
                "records_with_compileJDK_1.8",
            ],
            "value": [
                len(raw_records),
                len({r["projectName"] for r in raw_records}),
                len({r["type"] for r in raw_records}),
                len({k for r in raw_records for k in r.keys()}),
                sum(1 for r in raw_records if r.get("callGraph") is not None),
                sum(1 for r in raw_records if "hasTestC" in r and r.get("hasTestC") is not None),
                sum(1 for r in raw_records if "testResult" in r and r.get("testResult") is not None),
                sum(1 for r in raw_records if r.get("compileJDK") == 1.8),
            ],
        }
    )
    dataset_stats
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Validation and coercion with `RefactoringRecord`

    SmellAI does not directly trust the raw JSON.

    It first validates rows with `swe_refactor.dataset.RefactoringRecord`.
    The most visible coercion is:

    - raw `compileJDK = 1.8` → normalized `compileJDK = 8`

    So the loader is already doing a **semantic cleanup step** before building the unified dataset view.
    """)
    return


@app.cell
def _(RefactoringRecord, sample_raw_record):
    validated_record = RefactoringRecord.model_validate(sample_raw_record)
    print("validated via RefactoringRecord")
    print(f"projectName = {validated_record.projectName}")
    print(f"type        = {validated_record.type}")
    print(f"compileJDK  = {validated_record.compileJDK}")
    validated_record.model_dump()
    return


@app.cell
def _(pd, raw_records):
    compile_jdk_values = pd.Series([rec.get("compileJDK") for rec in raw_records], name="compileJDK")
    jdk_summary = pd.DataFrame(
        {
            "metric": [
                "records",
                "records_with_compileJDK_1.8",
                "distinct_raw_compileJDK_values",
            ],
            "value": [
                len(compile_jdk_values),
                int((compile_jdk_values == 1.8).sum()),
                str(sorted(compile_jdk_values.dropna().unique().tolist())),
            ],
        }
    )
    jdk_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Raw JSON → normalized SWE DataFrame via `load_swe_raw_df()`
    """)
    return


@app.cell
def _(load_swe_raw_df):
    df_swe = load_swe_raw_df()
    print(f"normalized DataFrame shape: {df_swe.shape[0]} rows × {df_swe.shape[1]} cols")
    print("normalized columns:")
    print(list(df_swe.columns))
    return (df_swe,)


@app.cell
def _(df_swe):
    df_swe.head(3)
    return


@app.cell
def _(pd):
    transform_map = pd.DataFrame(
        [
            {"raw": "uniqueId or commitId", "normalized": "pair_id", "why": "Stable pair identifier."},
            {"raw": "projectName", "normalized": "project_name", "why": "snake_case normalization."},
            {"raw": "commitId", "normalized": "commit_id", "why": "snake_case normalization."},
            {"raw": "type", "normalized": "refactoring_type", "why": "clear semantic label."},
            {"raw": "filePathBefore", "normalized": "file_path_before", "why": "snake_case normalization."},
            {"raw": "filePathAfter", "normalized": "file_path_after", "why": "snake_case normalization."},
            {"raw": "sourceCodeBeforeForWhole", "normalized": "class_before", "why": "full before-file/class."},
            {"raw": "sourceCodeAfterForWhole", "normalized": "class_after", "why": "full after-file/class."},
            {"raw": "sourceCodeBeforeRefactoring", "normalized": "source_before", "why": "local before-fragment."},
            {"raw": "sourceCodeAfterRefactoring", "normalized": "source_after", "why": "local after-fragment."},
            {"raw": "compileCommand", "normalized": "compile_command", "why": "build metadata."},
            {"raw": "compileJDK", "normalized": "jdk_version", "why": "coerced to int, e.g. 1.8 → 8."},
            {"raw": "type contains '+'", "normalized": "is_compound", "why": "derived boolean flag."},
            {"raw": "isPureRefactoring", "normalized": "is_pure", "why": "kept as experiment metadata."},
            {"raw": "hasTestC", "normalized": "has_tests", "why": "kept as experiment metadata."},
        ]
    )
    transform_map
    return


@app.cell
def _(df_swe, pd):
    derived_summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "pure_rows",
                "compound_rows",
                "rows_with_tests",
                "top_6_refactoring_types",
                "top_10_projects",
            ],
            "value": [
                len(df_swe),
                int(df_swe["is_pure"].sum()),
                int(df_swe["is_compound"].sum()),
                int(df_swe["has_tests"].sum()),
                str(df_swe["refactoring_type"].value_counts().head(6).to_dict()),
                str(df_swe["project_name"].value_counts().head(10).to_dict()),
            ],
        }
    )
    derived_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. How the fields are **combined** into the SmellAI data model

    This is the key modeling step.

    SmellAI does **not** keep the raw JSON as-is.
    Instead, it reorganizes the row into three semantic buckets:

    - **`inputs`** — what the agent sees before generating/refactoring
    - **`expectations`** — the known correct after-state
    - **`tags`** — metadata for filtering, slicing, and reporting

    So this is a **selection + coercion + regrouping** process, not a simple rename of every field.
    """)
    return


@app.cell
def _(pd):
    grouping = pd.DataFrame(
        [
            {
                "bucket": "inputs",
                "purpose": "Give the agent the pre-refactoring context and build instructions.",
                "raw_fields_used": [
                    "projectName",
                    "commitId",
                    "type",
                    "filePathBefore",
                    "filePathAfter",
                    "sourceCodeBeforeForWhole",
                    "sourceCodeBeforeRefactoring",
                    "compileJDK",
                    "compileCommand",
                ],
                "final_keys": [
                    "project_name",
                    "commit_id",
                    "refactoring_type",
                    "file_path_before",
                    "file_path_after",
                    "class_before",
                    "source_before",
                    "jdk_version",
                    "compile_command",
                ],
            },
            {
                "bucket": "expectations",
                "purpose": "Store the reference after-state used in evaluation.",
                "raw_fields_used": [
                    "sourceCodeAfterForWhole",
                    "sourceCodeAfterRefactoring",
                ],
                "final_keys": [
                    "class_after",
                    "source_after",
                ],
            },
            {
                "bucket": "tags",
                "purpose": "Store metadata used for filtering and analysis.",
                "raw_fields_used": [
                    "isPureRefactoring",
                    "hasTestC",
                    "type",
                ],
                "final_keys": [
                    "is_pure",
                    "has_tests",
                    "is_compound (derived from '+' in type)",
                ],
            },
            {
                "bucket": "dropped_in_current_projection",
                "purpose": "Rich raw metadata that exists in the dataset but is not carried into the current `EvalSample`.",
                "raw_fields_used": [
                    "description",
                    "compileResultBefore",
                    "compileResultCurrent",
                    "testResult",
                    "coverageInfo",
                    "purityCheckResultList",
                    "diffLocations",
                    "diffSourceCode",
                    "diffSourceCodeSet",
                    "callInfo",
                    "callGraph",
                    "invokedMethodSet",
                    "packageNameBefore",
                    "classNameBefore",
                    "classNameBeforeSet",
                    "classSignatureBefore",
                    "classSignatureBeforeSet",
                    "methodNameBefore",
                    "methodNameBeforeSet",
                    "moveFileExist",
                ],
                "final_keys": ["not projected by current loader"],
            },
        ]
    )
    grouping
    return


@app.cell
def _(df_swe):
    first_row = df_swe.iloc[0].to_dict()
    first_row
    return


@app.cell
def _(load_eval_samples):
    samples = load_eval_samples(["swe"], limit=3)
    print(f"EvalSample objects built: {len(samples)}")
    samples[0]
    return (samples,)


@app.cell
def _(samples):
    sample = samples[0]
    print(f"source    = {sample.source}")
    print(f"sample_id = {sample.sample_id}")
    print(f"inputs keys       = {sorted(sample.inputs.keys())}")
    print(f"expectations keys = {sorted(sample.expectations.keys())}")
    print(f"tags keys         = {sorted(sample.tags.keys())}")
    sample.model_dump()
    return


@app.cell
def _(EvalSample, TypeAdapter, samples):
    validated = TypeAdapter(list[EvalSample]).validate_python([s.model_dump() for s in samples])
    print(f"validated {len(validated)} EvalSample rows")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Link to the PDF spec: `docs/conf_Pietukhin_10_3_rev2-2.pdf`

    The paper describes SWE-Refactor at the **system/spec level**.
    This notebook shows the **actual implementation-level payload**.

    Important distinction:

    - the **PDF** explains what role the dataset plays in the pipeline
    - the **JSON** shows the concrete fields that make that role executable

    So the notebook bridges:

    **paper spec** ↔ **loader implementation** ↔ **actual dataset payload**
    """)
    return


@app.cell
def _(pd, raw_records):
    pdf_mapping = pd.DataFrame(
        [
            {
                "pdf anchor": "Abstract + Experiment section",
                "paper statement": "SWE-Refactor contributes 1,099 records.",
                "what the real JSON shows": f"The loaded file contains exactly {len(raw_records)} records.",
                "connection": "The notebook is inspecting the same benchmark scale described in the paper.",
            },
            {
                "pdf anchor": "Experiment section",
                "paper statement": "SWE-Refactor contributes 441 Extract Method and 410 Move Method instances.",
                "what the real JSON shows": str(pd.Series([r['type'] for r in raw_records]).value_counts().head(10).to_dict()),
                "connection": "The concrete source for those paper-level counts is the raw `type` field.",
            },
            {
                "pdf anchor": "Section III-A, stage A",
                "paper statement": "Stage A loads source code and detects the build system.",
                "what the real JSON shows": "`sourceCodeBeforeForWhole`, `sourceCodeAfterForWhole`, `compileCommand`, and `compileJDK` provide the concrete code/build context supporting that stage.",
                "connection": "These fields make the benchmark executable rather than just descriptive.",
            },
            {
                "pdf anchor": "Section III-A, stages D and J",
                "paper statement": "The system runs tests before and after refactoring.",
                "what the real JSON shows": "Fields such as `hasTestC`, `testResult`, and `coverageInfo` are related to testing/coverage viability, even though the current unified loader preserves only `has_tests`.",
                "connection": "The raw dataset is richer than the current final projection.",
            },
            {
                "pdf anchor": "Section III-A, stage F + Table I",
                "paper statement": "The system scans a commit with SonarQube and maps smells to 8 smell types.",
                "what the real JSON shows": "SWE-Refactor itself is not a SonarQube smell dump. It is a before/after refactoring oracle that can later be scanned and aligned with those smell categories.",
                "connection": "This explains why raw SWE fields talk about code pairs and build metadata rather than Sonar issues.",
            },
            {
                "pdf anchor": "Section III-A, stage I",
                "paper statement": "The LLM executes planned refactorings.",
                "what the real JSON shows": "`inputs` are built from BEFORE-side fields, while `expectations` are built from AFTER-side fields.",
                "connection": "That split is the exact data-model embodiment of the paper's evaluation scenario.",
            },
            {
                "pdf anchor": "Figure / coverage discussion",
                "paper statement": "Dataset coverage reflects where there are ground-truth refactoring examples.",
                "what the real JSON shows": "Fields like `type`, `diffLocations`, and the before/after source fields are the concrete ground-truth artifacts behind those claims.",
                "connection": "The JSON is the implementation-level evidence behind the paper's abstract coverage language.",
            },
        ]
    )
    pdf_mapping
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Are we analyzing the right file?

    After checking `swe_refactor/dataset_card.md` and the local `swe_refactor` package,
    the answer is:

    - **yes, this is the intended benchmark file for the current SmellAI integration**
    - but **there are a couple of dataset-card vs file-level inconsistencies that should be stated explicitly**

    Why "yes":

    - the dataset card describes **SWE-Refactor** as a 1,099-record benchmark
    - `pure_refactoring_data.json` contains exactly **1,099** rows
    - `swe_refactor/dataset.py` loads **that file directly**
    - `workflows/swe_eval_workflow.py` also expects **that file directly**

    So for this repository, `pure_refactoring_data.json` is the canonical SWE benchmark artifact.
    """)
    return


@app.cell
def _(pd, raw_records):
    validation_rows = pd.DataFrame(
        [
            {
                "check": "dataset card says 1099 records",
                "observed_in_file": len(raw_records),
                "interpretation": "matches the dataset card and the paper",
            },
            {
                "check": "all rows are pure refactorings",
                "observed_in_file": str(pd.Series([r.get('isPureRefactoring') for r in raw_records]).value_counts(dropna=False).to_dict()),
                "interpretation": "matches the dataset card claim about pure refactorings",
            },
            {
                "check": "pre-refactoring compile success",
                "observed_in_file": str(pd.Series([r.get('compileResultBefore') for r in raw_records]).value_counts(dropna=False).to_dict()),
                "interpretation": "all rows compile before refactoring",
            },
            {
                "check": "post-refactoring compile success",
                "observed_in_file": str(pd.Series([r.get('compileResultCurrent') for r in raw_records]).value_counts(dropna=False).to_dict()),
                "interpretation": "almost all pass, but 2 rows are False",
            },
            {
                "check": "test execution result",
                "observed_in_file": str(pd.Series([r.get('testResult', '__MISSING__') for r in raw_records]).value_counts(dropna=False).to_dict()),
                "interpretation": "1098 rows have True, 1 row is missing",
            },
            {
                "check": "hasTestC availability flag",
                "observed_in_file": str(pd.Series([r.get('hasTestC', '__MISSING__') for r in raw_records]).value_counts(dropna=False).to_dict()),
                "interpretation": "this is the main mismatch: almost entirely missing in this concrete file",
            },
        ]
    )
    validation_rows
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical implication for SmellAI

    The current loader is **pointing at the right benchmark file**.

    However, the current simplified projection has one weak spot:

    - it uses `hasTestC` → `has_tests`
    - but in this concrete JSON, `hasTestC` is missing for **1098/1099** rows
    - meanwhile `testResult` and `coverageInfo` are populated for almost all rows

    So:

    - for **dataset identity**, we are analyzing the correct file
    - for **test-availability semantics**, `hasTestC` is probably not the best field to rely on in this version of the dataset
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Final takeaway

    In the language of the paper:

    - the system needs a benchmark of **real refactoring examples**
    - SWE-Refactor provides those as **before/after code pairs plus build metadata**
    - the SmellAI loader then compresses that rich raw payload into the unified contract

    ```python
    EvalSample(source, sample_id, inputs, expectations, tags)
    ```

    So the chain is:

    1. **raw SWE JSON** — many concrete implementation fields
    2. **`RefactoringRecord`** — validation/coercion
    3. **`load_swe_raw_df()`** — normalized flat inspection layer
    4. **`load_eval_samples()`** — unified evaluation contract
    5. **paper stages A–J** — system semantics those transformed fields support

    Put differently:

    **actual payload** → **SmellAI data model** → **PDF spec**
    """)
    return


if __name__ == "__main__":
    app.run()
