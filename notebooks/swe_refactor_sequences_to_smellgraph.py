import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # SWE-Refactor: срез последовательностей рефакторингов → SmellGraph

        Цель ноутбука:

        1. загрузить локальный код/данные SWE-Refactor из
           `/Users/havriil.pietukhin/uni/masterThesis/SWE-Refactor/SWE-Refactor/code`;
        2. показать, как получить именно **sequence slice** — случаи, где один пример задаёт
           упорядоченную последовательность операций, например `Extract And Move Method`;
        3. показать, как этот срез можно встроить в текущий SmellAI contract:
           `EvalSample` → smell detection → `SmellEvent` → `SmellGraph` → priority sequence.

        По статье `SWE-Refactor: A Repository-Aware Benchmark...` датасет содержит pure
        refactorings, включая atomic (`Extract Method`, `Move Method`, `Inline Method`) и
        compound types. Compound-типы — это как раз рефакторинги-последовательности:

        - `Extract And Move Method` = `Extract Method` → `Move Method`
        - `Move And Rename Method` = `Move Method` → `Rename Method`
        - `Move And Inline Method` = `Move Method` → `Inline Method`
        """
    )
    return (mo,)


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import pandas as pd

    PROJECT_ROOT = Path.cwd()
    SWE_REFACTOR_CODE = Path(
        "/Users/havriil.pietukhin/uni/masterThesis/SWE-Refactor/SWE-Refactor/code"
    )
    SWE_REFACTOR_DATA = SWE_REFACTOR_CODE / "data"

    for p in [PROJECT_ROOT, SWE_REFACTOR_CODE]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from domain.graph import SmellGraph
    from domain.models import SmellEvent
    from smellai_datasets.schema import EvalSample

    print(f"SmellAI root: {PROJECT_ROOT}")
    print(f"SWE-Refactor code: {SWE_REFACTOR_CODE}")
    print(f"SWE-Refactor data exists: {SWE_REFACTOR_DATA.exists()}")
    return EvalSample, PROJECT_ROOT, SWE_REFACTOR_CODE, SWE_REFACTOR_DATA, SmellEvent, SmellGraph, json, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Load raw SWE-Refactor records from the dataset repository

        В текущей копии датасета данные лежат по проектам в файлах вида
        `data/<project>/<project>_pure_refactoring_data.json`.

        Код датасета (`pre_process_data.py`) строит эти файлы примерно так:

        - фильтрует `isPureRefactoring == True`;
        - проверяет target class для move-style refactorings;
        - проверяет compile/test/coverage;
        - сохраняет итоговый evaluation/pure JSON.
        """
    )
    return


@app.cell
def _(SWE_REFACTOR_DATA, json, pd):
    data_files = sorted(SWE_REFACTOR_DATA.glob("*/*_pure_refactoring_data.json"))
    rows = []
    for path in data_files:
        project_rows = json.loads(path.read_text())
        project_name = path.parent.name
        for record in project_rows:
            record = dict(record)
            # Per-project SWE-Refactor JSON files in the dataset repo do not
            # always carry `projectName`; the project is encoded by folder name.
            # SmellAI's merged/canonical loader may have this field already, so
            # keep it when present and fill it only when absent.
            record.setdefault("projectName", project_name)
            record["_data_file"] = str(path)
            rows.append(record)

    raw_df = pd.DataFrame(rows)
    print(f"files: {len(data_files)}")
    print(f"records: {len(raw_df)}")
    raw_df[["projectName", "commitId", "type", "uniqueId"]].head()
    return data_files, raw_df


@app.cell
def _(pd, raw_df):
    pd.DataFrame(
        {
            "metric": [
                "records",
                "projects",
                "refactoring_types",
                "pure_records",
                "compile_before_true",
                "compile_after_true",
                "test_result_true",
            ],
            "value": [
                len(raw_df),
                raw_df["projectName"].nunique(),
                raw_df["type"].nunique(),
                int(raw_df.get("isPureRefactoring", False).fillna(False).sum()),
                int(raw_df.get("compileResultBefore", False).fillna(False).sum()),
                int(raw_df.get("compileResultCurrent", False).fillna(False).sum()),
                int(raw_df.get("testResult", False).fillna(False).sum()),
            ],
        }
    )
    return


@app.cell
def _(raw_df):
    raw_df["type"].value_counts().rename_axis("refactoring_type").reset_index(name="count")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Define the sequence slice

        В статье compound refactoring types определены как последовательности операций.
        Поэтому основной фильтр для "последовательностей рефакторингов" — оставить типы,
        которые раскладываются в несколько шагов.

        Важно: в raw JSON порядок шагов не хранится отдельным массивом. Он восстанавливается
        из `type` согласно определениям статьи и RefactoringMiner naming convention.
        """
    )
    return


@app.cell
def _(pd, raw_df):
    REFACTORING_SEQUENCES = {
        "Extract And Move Method": ["Extract Method", "Move Method"],
        "Move And Rename Method": ["Move Method", "Rename Method"],
        "Move And Inline Method": ["Move Method", "Inline Method"],
    }

    def refactoring_sequence(refactoring_type: str) -> list[str]:
        return REFACTORING_SEQUENCES.get(refactoring_type, [refactoring_type])

    sequence_df = raw_df[raw_df["type"].isin(REFACTORING_SEQUENCES)].copy()
    sequence_df["refactoring_sequence"] = sequence_df["type"].map(refactoring_sequence)
    sequence_df["sequence_length"] = sequence_df["refactoring_sequence"].map(len)

    print(f"sequence rows: {len(sequence_df)}")
    sequence_df[
        [
            "projectName",
            "commitId",
            "type",
            "refactoring_sequence",
            "filePathBefore",
            "filePathAfter",
            "uniqueId",
        ]
    ].head(10)
    return REFACTORING_SEQUENCES, refactoring_sequence, sequence_df


@app.cell
def _(pd, sequence_df):
    pd.DataFrame(
        {
            "compound_type": sequence_df["type"].value_counts().index,
            "count": sequence_df["type"].value_counts().values,
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Optional: commit-level multi-refactoring groups

        Это другой полезный срез: один commit может содержать несколько raw records.
        Он не всегда означает *одну логическую последовательность*; это скорее batch/group
        refactoring context. Для sequence slice выше используем compound types, а этот блок
        оставляем как диагностический.
        """
    )
    return


@app.cell
def _(raw_df):
    commit_groups = (
        raw_df.groupby(["projectName", "commitId"])
        .agg(
            records=("uniqueId", "count"),
            types=("type", lambda s: list(s)),
            files_before=("filePathBefore", lambda s: sorted(set(s))),
        )
        .reset_index()
        .query("records > 1")
        .sort_values("records", ascending=False)
    )
    commit_groups.head(10)
    return (commit_groups,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Project sequence rows to SmellAI `EvalSample`

        Для текущего проекта полезно не тащить весь raw JSON дальше, а сохранить совместимый
        контракт: `EvalSample(source='swe', inputs, expectations, tags)`.

        Отличие от существующего `load_swe_raw_df()` — мы сохраняем sequence metadata:

        - `inputs.refactoring_sequence`
        - `inputs.sequence_length`
        - `tags.is_sequence = True`
        - raw quality flags (`compile/test`) в `tags`
        """
    )
    return


@app.cell
def _(EvalSample, sequence_df):
    def sequence_row_to_eval_sample(row) -> EvalSample:
        return EvalSample(
            source="swe",
            sample_id=f"swe-seq:{row['uniqueId']}",
            inputs={
                "project_name": row["projectName"],
                "commit_id": row["commitId"],
                "refactoring_type": row["type"],
                "refactoring_sequence": row["refactoring_sequence"],
                "sequence_length": int(row["sequence_length"]),
                "file_path_before": row["filePathBefore"],
                "file_path_after": row["filePathAfter"],
                "class_before": row.get("sourceCodeBeforeForWhole", ""),
                "source_before": row.get("sourceCodeBeforeRefactoring", ""),
                "description": row.get("description", ""),
                "diff_locations": row.get("diffLocations", []),
                "diff_source_code": row.get("diffSourceCode", ""),
                "jdk_version": 8 if row.get("compileJDK") == 1.8 else int(row.get("compileJDK", 0)),
                "compile_command": row.get("compileCommand", ""),
            },
            expectations={
                "class_after": row.get("sourceCodeAfterForWhole", ""),
                "source_after": row.get("sourceCodeAfterRefactoring", ""),
            },
            tags={
                "is_sequence": True,
                "is_pure": bool(row.get("isPureRefactoring", True)),
                "is_compound": True,
                "compile_before": bool(row.get("compileResultBefore", False)),
                "compile_after": bool(row.get("compileResultCurrent", False)),
                "test_result": bool(row.get("testResult", False)),
                "project_name": row["projectName"],
            },
        )

    sequence_samples = [sequence_row_to_eval_sample(row) for _, row in sequence_df.iterrows()]
    print(f"EvalSample sequence rows: {len(sequence_samples)}")
    sequence_samples[0].model_dump()
    return sequence_row_to_eval_sample, sequence_samples


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. How this enters SmellAI / `SmellGraph`

        `SWE-Refactor` сам по себе хранит before/after refactoring oracle, а не SonarQube smells.
        Поэтому в реальном workflow связь такая:

        1. взять `EvalSample.inputs.class_before` или checkout repo на `commit_id`;
        2. прогнать smell detector (`SonarQubeDetector` или тестовый `StaticDetector`);
        3. получить `list[SmellEvent]`;
        4. построить `SmellGraph.from_smells(smell_events)`;
        5. получить priority/refactoring order через `graph.calculate_priorities()`;
        6. сопоставить ожидаемый sequence из SWE-Refactor с тем, что планирует SmellAI.

        Ниже минимальный пример без SonarQube: строим synthetic smells из metadata строки,
        чтобы проверить именно контракт и graph construction. В production заменить функцию
        `mock_smells_for_sample()` на настоящий detector.
        """
    )
    return


@app.cell
def _(SmellEvent, SmellGraph, sequence_samples):
    def mock_smells_for_sample(sample) -> list[SmellEvent]:
        inputs = sample.inputs
        file_path = inputs["file_path_before"]
        refactoring_type = inputs["refactoring_type"]

        # Heuristic placeholders only for notebook wiring demo.
        # Real workflow should use SonarQubeDetector.detect(checkout_path).
        smells: list[SmellEvent] = []
        if "Extract" in refactoring_type:
            smells.append(
                SmellEvent(
                    smell_id=f"Long Method:{file_path}:1",
                    smell_type="Long Method",
                    severity="HIGH",
                    file_path=file_path,
                    line_number=1,
                )
            )
        if "Move" in refactoring_type:
            smells.append(
                SmellEvent(
                    smell_id=f"Feature Envy:{file_path}:1",
                    smell_type="Feature Envy",
                    severity="MEDIUM",
                    file_path=file_path,
                    line_number=1,
                )
            )
        if "Inline" in refactoring_type:
            smells.append(
                SmellEvent(
                    smell_id=f"Speculative Generality:{file_path}:1",
                    smell_type="Speculative Generality",
                    severity="LOW",
                    file_path=file_path,
                    line_number=1,
                )
            )
        return smells

    example_sample = sequence_samples[0]
    example_smells = mock_smells_for_sample(example_sample)
    example_graph = SmellGraph.from_smells(example_smells)

    print("expected refactoring sequence:", example_sample.inputs["refactoring_sequence"])
    print("smell nodes:", example_graph.to_dict()["nodes"])
    print("priority sequence:")
    example_graph.calculate_priorities()
    return example_graph, example_sample, example_smells, mock_smells_for_sample


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6. Production integration sketch

        ```python
        from pathlib import Path
        from domain.graph import SmellGraph
        from sonarqube.detector import SonarQubeDetector
        from repo_utils.operations import checkout_repo_at_commit  # or existing project checkout util

        sample = sequence_samples[0]
        checkout_path = checkout_repo_at_commit(
            project=sample.inputs["project_name"],
            commit=sample.inputs["commit_id"],
        )

        detector = SonarQubeDetector()
        smell_events = detector.detect(Path(checkout_path))
        graph = SmellGraph.from_smells(smell_events)
        planned_order = graph.calculate_priorities()

        expected_refactoring_sequence = sample.inputs["refactoring_sequence"]
        ```

        Главное архитектурное решение: **sequence slice остаётся dataset/evaluation metadata**,
        а `SmellGraph` строится только из detected smells. Так мы не смешиваем ground truth
        refactoring oracle с графом запахов, но можем оценивать, ведёт ли план SmellAI к нужной
        последовательности рефакторингов.
        """
    )
    return


if __name__ == "__main__":
    app.run()
