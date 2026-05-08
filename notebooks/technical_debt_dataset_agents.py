import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # Technical Debt Dataset v2 → SmellAI agents

        Этот notebook показывает, **что лежит в Technical Debt Dataset v2** и как превратить его
        в входы для существующих агентов из `agents/`.

        Ключевой вывод сразу: TDD v2 хранит **и smells, и refactorings**, но в разных таблицах:

        - `SONAR_ISSUES` — lifecycle SonarQube issues: в основном `CODE_SMELL`, плюс `BUG` и `VULNERABILITY`.
        - `REFACTORING_MINER` — refactoring events, найденные RefactoringMiner, привязанные к commit hash.
        - Это **не готовый before/after snapshot датасет**. Состояние smell'ов на commit надо реконструировать через
          `SONAR_ANALYSIS` + правило активности smell'а.
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 1. Контекст из PDF спецификации

        Спецификация описывает multi-agent pipeline A–J:

        | Stage | Смысл | Где в коде сейчас ближе всего |
        |---|---|---|
        | A–D | setup проекта, тесты, baseline | `agents/swe_eval/agent.py`, runtime helpers |
        | F | SonarQube scan → typed smell instances | `sonarqube/`, `domain.models.SmellEvent` |
        | G | выбор smell targets разработчиком | пока интерактивный/ручной слой |
        | E | dependency graph по smell'ам | `agents/dependency_analysis/agent.py`, `domain/graph.py` |
        | H | greedy / BeFS planner | `domain/graph.py`, scorer/rules |
        | I | LLM refactoring agent | `agents/swe_eval/agent.py` |
        | J | tests, rollback/replan | `agents/swe_eval/agent.py` |

        TDD v2 полезен как исторический oracle для переходов:
        `CodeState_before(parent commit) → CodeState_after(refactoring commit)`.
        """
    )
    return


@app.cell
def _():
    import json
    import sqlite3
    import sys
    from pathlib import Path
    from typing import Any

    import pandas as pd

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from agents.dependency_analysis.agent import (
        build_smell_graph_from_events,
        prioritize_smells,
    )
    from domain.models import SmellEvent
    from smellai_datasets.schema import EvalSample
    from smellai_datasets.td_v2 import (
        RELEVANT_TABLES,
        build_smell_delta,
        connect_td_v2,
        extract_transitions,
        get_table_counts,
        load_project_registry,
        load_tdd_raw_df,
    )

    TDD_DB_PATH = Path("/Users/havriil.pietukhin/uni/masterThesis/datasets/td_V2.db")
    return (
        Any,
        EvalSample,
        RELEVANT_TABLES,
        TDD_DB_PATH,
        build_smell_delta,
        build_smell_graph_from_events,
        connect_td_v2,
        extract_transitions,
        get_table_counts,
        json,
        load_project_registry,
        load_tdd_raw_df,
        mo,
        pd,
        prioritize_smells,
        sqlite3,
        SmellEvent,
    )


@app.cell
def _(TDD_DB_PATH, mo):
    exists = TDD_DB_PATH.exists()
    mo.md(
        f"""
        ## 2. Локальный DB artifact

        - DB path: `{TDD_DB_PATH}`
        - exists: **{exists}**

        Если DB нет локально, скачай/положи canonical artifact: `../datasets/td_V2.db`.
        """
    )
    return (exists,)


@app.cell
def _(TDD_DB_PATH, connect_td_v2, exists, get_table_counts, pd):
    if exists:
        with connect_td_v2(TDD_DB_PATH) as _con:
            counts = get_table_counts(_con)
        counts_df = pd.DataFrame(
            [{"table": k, "rows": v} for k, v in counts.items()]
        ).sort_values("rows", ascending=False)
    else:
        counts_df = pd.DataFrame(columns=["table", "rows"])
    counts_df
    return (counts_df,)


@app.cell
def _(TDD_DB_PATH, connect_td_v2, exists, pd):
    if exists:
        with connect_td_v2(TDD_DB_PATH) as _con:
            issue_types_df = pd.read_sql_query(
                """
                SELECT TYPE AS issue_type, COUNT(*) AS rows
                FROM SONAR_ISSUES
                GROUP BY TYPE
                ORDER BY rows DESC
                """,
                _con,
            )
            refactoring_types_df = pd.read_sql_query(
                """
                SELECT REFACTORING_TYPE AS refactoring_type, COUNT(*) AS rows
                FROM REFACTORING_MINER
                GROUP BY REFACTORING_TYPE
                ORDER BY rows DESC
                LIMIT 20
                """,
                _con,
            )
    else:
        issue_types_df = pd.DataFrame()
        refactoring_types_df = pd.DataFrame()
    return issue_types_df, refactoring_types_df


@app.cell(hide_code=True)
def _(issue_types_df, mo, refactoring_types_df):
    mo.vstack(
        [
            mo.md("### Что внутри `SONAR_ISSUES`"),
            issue_types_df,
            mo.md("### Top RefactoringMiner events"),
            refactoring_types_df,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 3. Главная семантика TDD v2

        `SONAR_ISSUES` — это **lifecycle rows**, а не snapshot на каждый commit.

        Smell считается активным в точке анализа commit'а, если:

        ```text
        open_date <= point_date AND (close_date IS NULL OR close_date > point_date)
        ```

        Поэтому transition строится так:

        1. Берём child commit из `REFACTORING_MINER`.
        2. Проверяем, что он main/master, non-merge, ровно один parent.
        3. Находим `SONAR_ANALYSIS` для parent и child.
        4. Реконструируем `state_before.smells` и `state_after.smells`.
        5. Считаем delta: `persisted`, `resolved`, `created`.
        """
    )
    return


@app.cell
def _(TDD_DB_PATH, connect_td_v2, exists, load_project_registry, mo):
    if exists:
        with connect_td_v2(TDD_DB_PATH) as _con:
            registry = load_project_registry(_con)
        project_ids = sorted(registry.keys())
    else:
        registry = {}
        project_ids = []

    project = mo.ui.dropdown(
        options=project_ids,
        value=("org.apache:commons-io" if "org.apache:commons-io" in project_ids else (project_ids[0] if project_ids else None)),
        label="Project",
    )
    limit = mo.ui.slider(1, 25, value=3, label="Transition sample size")
    run_extract = mo.ui.run_button(label="Extract transitions")
    mo.vstack([project, limit, run_extract])
    return limit, project, project_ids, registry, run_extract


@app.cell
def _(TDD_DB_PATH, extract_transitions, exists, limit, pd, project, run_extract):
    if exists and run_extract.value and project.value:
        transitions, report = extract_transitions(
            TDD_DB_PATH,
            project_id=project.value,
            limit=limit.value,
        )
        transition_summary_df = pd.DataFrame(
            [
                {
                    "transition_id": t["transition_id"],
                    "refactorings": len(t["refactorings"]),
                    "before_smells": t["smell_delta"]["counts"]["before"],
                    "resolved": t["smell_delta"]["counts"]["resolved"],
                    "created": t["smell_delta"]["counts"]["created"],
                    "after_smells": t["smell_delta"]["counts"]["after"],
                    "validation_errors": len(t["validation_errors"]),
                }
                for t in transitions
            ]
        )
    else:
        transitions, report = [], {}
        transition_summary_df = pd.DataFrame()
    transition_summary_df
    return report, transition_summary_df, transitions


@app.cell
def _(mo, report):
    mo.md(
        "### Extraction report\n\n```json\n"
        + __import__("json").dumps(report, indent=2, ensure_ascii=False)[:4000]
        + "\n```"
        if report
        else "Нажми **Extract transitions**, чтобы построить sample."
    )
    return


@app.cell
def _(transitions):
    transition = transitions[0] if transitions else None
    transition
    return (transition,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 4. Преобразование transition → вход для agents

        Для dependency/planning части агентам нужны Sonar-like issue dicts или `SmellEvent`:

        - `agents.dependency_analysis.prioritize_smells(sonar_issues)` принимает список issue dicts.
        - `build_smell_graph_from_events([...SmellEvent])` принимает нормализованные smell events.
        - `agents.swe_eval` в composite mode уже хранит `detected_smells`, `smell_graph`, `priority_queue`, `current_smell`.

        TDD transition даёт исторический контекст:

        - `state_before.smells` → smells перед refactoring commit.
        - `refactorings` → какие операции были сделаны в child commit.
        - `state_after.smells` и `smell_delta` → oracle эффекта: resolved/created/persisted.
        """
    )
    return


@app.cell
def _(SmellEvent, transition):
    def tdd_smell_to_event(smell: dict) -> SmellEvent:
        line = smell.get("line_start") or smell.get("line_end") or 0
        return SmellEvent(
            smell_id=smell.get("fingerprint") or smell.get("issue_key"),
            smell_type=smell.get("smell_type") or smell.get("rule_id"),
            severity=smell.get("severity") or "LOW",
            file_path=smell.get("file_path") or "",
            line_number=int(line or 0),
            action="detected",
        )

    before_events = (
        [tdd_smell_to_event(s) for s in transition["state_before"]["smells"]]
        if transition
        else []
    )
    before_events[:5]
    return before_events, tdd_smell_to_event


@app.cell
def _(before_events, build_smell_graph_from_events):
    smell_graph = build_smell_graph_from_events(before_events[:200]) if before_events else None
    graph_dict = smell_graph.to_dict() if smell_graph else {"nodes": [], "edges": []}
    {
        "nodes": len(graph_dict["nodes"]),
        "edges": len(graph_dict["edges"]),
        "example_nodes": graph_dict["nodes"][:3],
        "example_edges": graph_dict["edges"][:3],
    }
    return graph_dict, smell_graph


@app.cell
def _(pd, smell_graph):
    priority_df = pd.DataFrame(smell_graph.calculate_priorities()[:20]) if smell_graph else pd.DataFrame()
    priority_df
    return (priority_df,)


@app.cell
def _(EvalSample, transition):
    def transition_to_eval_sample(t: dict) -> EvalSample:
        return EvalSample(
            source="tdd",
            sample_id=f"tdd:{t['transition_id']}",
            inputs={
                "project_id": t["project_id"],
                "repository_url": t["repository_url"],
                "commit_before": t["commit_before"],
                "commit_after": t["commit_after"],
                "sonar_issues": t["state_before"]["smells"],
                "refactorings": t["refactorings"],
                "focus": t["focus"],
            },
            expectations={
                "smell_delta": t["smell_delta"],
                "state_after_smells": t["state_after"]["smells"],
            },
            tags={
                "project_name": t["project_name"],
                "source_dataset": t["provenance"]["dataset"],
                "pipeline": t["provenance"]["pipeline"],
            },
        )

    eval_sample = transition_to_eval_sample(transition) if transition else None
    eval_sample
    return eval_sample, transition_to_eval_sample


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 5. Как именно “запихивать” в agents

        Минимальный routing:

        ```python
        # 1) TDD transition -> before smell events
        before_events = [tdd_smell_to_event(s) for s in transition["state_before"]["smells"]]

        # 2) Stage E/H: dependency graph + priority plan
        graph = build_smell_graph_from_events(before_events)
        priority_queue = graph.calculate_priorities()

        # 3) Evaluation sample for experiments / MLflow
        sample = transition_to_eval_sample(transition)

        # 4) Historical oracle
        expected_resolved = sample.expectations["smell_delta"]["resolved"]
        expected_created = sample.expectations["smell_delta"]["created"]
        ```

        Caveat: `REFACTORING_MINER` in TDD v2 stores textual `REFACTORING_DETAIL`, not rich left/right code ranges.
        For exact patch localization Agent I may need repo checkout + diff, or rerun RefactoringMiner on selected commits.
        """
    )
    return


@app.cell
def _(transition):
    if transition:
        example_refactorings = transition["refactorings"][:10]
        example_delta = transition["smell_delta"]
    else:
        example_refactorings = []
        example_delta = {}
    {"refactorings": example_refactorings, "smell_delta": example_delta}
    return example_delta, example_refactorings


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 6. Практические ограничения

        - Dataset связывает smells и refactorings **на уровне commit/project/time**, а не всегда на уровне конкретного hunk.
        - `SONAR_ISSUES.ISSUE_KEY` может иметь дубликаты; для устойчивой идентификации loader добавляет `fingerprint`.
        - Есть несколько analyses на revision; код берёт latest по `DATE`.
        - Merge commits и commits без parent/child `SONAR_ANALYSIS` сейчас исключаются.
        - Для performance нельзя делать N+1 SQL на каждую transition: используем bulk project lifecycle cache.
        """
    )
    return


if __name__ == "__main__":
    app.run()
