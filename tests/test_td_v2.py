"""Tests for TDD v2 dataset helpers and loaders."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

from smellai_datasets import EvalSample, load_eval_samples
from smellai_datasets.td_v2 import (
    build_project_state_cache,
    build_smell_delta,
    build_transition,
    load_active_code_smells,
    load_changed_files,
    load_changed_files_batch,
    load_project_code_smell_lifecycle,
    load_refactoring_events,
    load_refactoring_events_batch,
    load_tdd_raw_df,
    normalize_component_path,
    parse_git_parents,
    validate_transition,
)


def _create_minimal_tdd_db(path: Path) -> Path:
    con = sqlite3.connect(path)
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE SONAR_ANALYSIS (
            PROJECT_ID TEXT,
            ANALYSIS_KEY TEXT,
            DATE TEXT,
            REVISION TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE SONAR_ISSUES (
            PROJECT_ID TEXT,
            CREATION_ANALYSIS_KEY TEXT,
            ISSUE_KEY TEXT,
            TYPE TEXT,
            RULE TEXT,
            SEVERITY TEXT,
            STATUS TEXT,
            RESOLUTION TEXT,
            EFFORT TEXT,
            DEBT TEXT,
            TAGS TEXT,
            CREATION_DATE TEXT,
            CLOSE_DATE TEXT,
            MESSAGE TEXT,
            COMPONENT TEXT,
            START_LINE REAL,
            END_LINE REAL,
            START_OFFSET REAL,
            END_OFFSET REAL,
            HASH TEXT,
            CLOSE_ANALYSIS_KEY TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE REFACTORING_MINER (
            PROJECT_ID TEXT,
            COMMIT_HASH TEXT,
            REFACTORING_TYPE TEXT,
            REFACTORING_DETAIL TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE GIT_COMMITS_CHANGES (
            PROJECT_ID TEXT,
            FILE TEXT,
            COMMIT_HASH TEXT,
            DATE TEXT,
            COMMITTER_ID TEXT,
            LINES_ADDED TEXT,
            LINES_REMOVED TEXT,
            NOTE TEXT
        )
        """
    )

    cur.executemany(
        "INSERT INTO SONAR_ANALYSIS VALUES (?, ?, ?, ?)",
        [
            ("org.apache:test", "A1", "2020-01-01 10:00:00", "sha-create"),
            ("org.apache:test", "A2", "2020-01-02 10:00:00", "sha-close"),
        ],
    )
    cur.executemany(
        "INSERT INTO SONAR_ISSUES VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "org.apache:test",
                "A1",
                "ISSUE-1",
                "CODE_SMELL",
                "java:S138",
                "CRITICAL",
                "OPEN",
                "",
                "10",
                "10",
                "",
                "2020-01-01 10:00:00",
                "",
                "Method too long",
                "proj:src/Foo.java",
                42.0,
                45.0,
                None,
                None,
                "hash-1",
                "",
            ),
            (
                "org.apache:test",
                "A1",
                "ISSUE-2",
                "CODE_SMELL",
                "java:S107",
                "MAJOR",
                "CLOSED",
                "FIXED",
                "5",
                "5",
                "",
                "2020-01-01 10:00:00",
                "2020-01-02 10:00:00",
                "Too many parameters",
                "proj:src/Bar.java",
                5.0,
                5.0,
                None,
                None,
                "hash-2",
                "A2",
            ),
            (
                "org.apache:test",
                "A2",
                "ISSUE-3",
                "CODE_SMELL",
                "java:S1200",
                "BLOCKER",
                "OPEN",
                "",
                "12",
                "12",
                "",
                "2020-01-02 10:00:00",
                "",
                "God class",
                "proj:src/Baz.java",
                9.0,
                9.0,
                None,
                None,
                "hash-3",
                "",
            ),
            (
                "org.apache:test",
                "A1",
                "ISSUE-4",
                "BUG",
                "java:S001",
                "MAJOR",
                "CLOSED",
                "FIXED",
                "5",
                "5",
                "",
                "2020-01-01 10:00:00",
                "2020-01-02 10:00:00",
                "Not used by smell loader",
                "proj:src/Bar.java",
                5.0,
                5.0,
                None,
                None,
                "hash-4",
                "A2",
            ),
        ],
    )
    cur.executemany(
        "INSERT INTO REFACTORING_MINER VALUES (?, ?, ?, ?)",
        [
            (
                "org.apache:test",
                "sha-close",
                "Extract Method",
                "Extract Method foo() extracted from class org.example.Foo",
            ),
            (
                "org.apache:test",
                "sha-close",
                "Rename Method",
                "Rename Method bar() renamed in class org.example.Foo",
            ),
        ],
    )
    cur.executemany(
        "INSERT INTO GIT_COMMITS_CHANGES VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "org.apache:test",
                "src/Foo.java",
                "sha-close",
                "2020-01-02 10:00:00",
                "dev",
                "10",
                "2",
                "note",
            ),
            (
                "org.apache:test",
                "src/Foo.java",
                "sha-close",
                "2020-01-02 10:00:00",
                "dev",
                "1",
                "0",
                "duplicate row to test dedup",
            ),
            (
                "org.apache:test",
                "src/Bar.java",
                "sha-close",
                "2020-01-02 10:00:00",
                "dev",
                "3",
                "1",
                "note",
            ),
        ],
    )

    con.commit()
    con.close()
    return path


class TestTdV2Helpers:
    def test_parse_git_parents_empty(self):
        assert parse_git_parents("[]") == []
        assert parse_git_parents("") == []
        assert parse_git_parents(None) == []

    def test_parse_git_parents_single_and_multi(self):
        assert parse_git_parents("['abc']") == ["abc"]
        assert parse_git_parents("['abc', 'def']") == ["abc", "def"]

    def test_parse_git_parents_malformed(self):
        assert parse_git_parents("not-a-list") == []
        assert parse_git_parents("{'abc'}") == []

    def test_normalize_component_path(self):
        assert normalize_component_path("proj:src/Foo.java") == "src/Foo.java"
        assert normalize_component_path("src/Foo.java") == "src/Foo.java"
        assert normalize_component_path("") == ""

    def test_build_smell_delta_and_validation(self):
        before = [{"issue_key": "A"}, {"issue_key": "B"}, {"issue_key": "C"}]
        after = [{"issue_key": "B"}, {"issue_key": "C"}, {"issue_key": "D"}]
        delta = build_smell_delta(before, after)

        assert delta["resolved"] == ["A"]
        assert delta["persisted"] == ["B", "C"]
        assert delta["created"] == ["D"]
        assert delta["counts"] == {
            "before": 3,
            "after": 3,
            "resolved": 1,
            "created": 1,
            "persisted": 2,
        }

        transition = {"smell_delta": delta}
        assert validate_transition(transition) == []

    def test_validate_transition_rejects_broken_accounting(self):
        transition = {
            "smell_delta": {
                "counts": {
                    "before": 3,
                    "after": 4,
                    "resolved": 1,
                    "created": 1,
                    "persisted": 1,
                }
            }
        }
        assert validate_transition(transition) == [
            "before_total != persisted + resolved",
            "after_total != persisted + created",
        ]


class TestTdV2BatchHelpers:
    def test_build_transition_reuses_state_cache(self):
        candidate = {
            "project_id": "org.apache:test",
            "parent_sha": "parent-sha",
            "child_sha": "child-sha",
            "author_date": "2020-01-02 10:00:00",
            "commit_message": "refactor",
            "analysis_before": {"analysis_key": "A1", "date": "2020-01-01", "revision": "parent-sha"},
            "analysis_after": {"analysis_key": "A2", "date": "2020-01-02", "revision": "child-sha"},
        }
        project_registry = {
            "org.apache:test": {
                "project_name": "test",
                "repository_url": "https://example.com/repo.git",
            }
        }
        analysis_index = {
            ("org.apache:test", "parent-sha"): candidate["analysis_before"],
            ("org.apache:test", "child-sha"): candidate["analysis_after"],
        }
        state_cache = {}

        with patch(
            "smellai_datasets.td_v2.load_active_code_smells",
            return_value=[],
        ) as mocked_loader:
            build_transition(
                con=None,  # type: ignore[arg-type]
                candidate=candidate,
                project_registry=project_registry,
                analysis_index=analysis_index,
                rule_name_index={},
                batched_refactorings={("org.apache:test", "child-sha"): []},
                batched_changed_files={("org.apache:test", "child-sha"): []},
                state_cache=state_cache,
            )
            build_transition(
                con=None,  # type: ignore[arg-type]
                candidate=candidate,
                project_registry=project_registry,
                analysis_index=analysis_index,
                rule_name_index={},
                batched_refactorings={("org.apache:test", "child-sha"): []},
                batched_changed_files={("org.apache:test", "child-sha"): []},
                state_cache=state_cache,
            )

        assert mocked_loader.call_count == 2
        assert set(state_cache) == {
            ("org.apache:test", "parent-sha"),
            ("org.apache:test", "child-sha"),
        }

    def test_load_refactoring_events_batch_matches_point_queries(self, tmp_path: Path):
        db_path = _create_minimal_tdd_db(tmp_path / "td_v2_test.db")
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        try:
            candidates = [
                {
                    "project_id": "org.apache:test",
                    "child_sha": "sha-close",
                }
            ]
            batched = load_refactoring_events_batch(con, candidates)
            point = load_refactoring_events(con, "org.apache:test", "sha-close")
            assert batched[("org.apache:test", "sha-close")] == point
        finally:
            con.close()

    def test_load_changed_files_batch_matches_point_queries(self, tmp_path: Path):
        db_path = _create_minimal_tdd_db(tmp_path / "td_v2_test.db")
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        try:
            candidates = [
                {
                    "project_id": "org.apache:test",
                    "child_sha": "sha-close",
                }
            ]
            batched = load_changed_files_batch(con, candidates)
            point = load_changed_files(con, "org.apache:test", "sha-close")
            assert batched[("org.apache:test", "sha-close")] == point
            assert batched[("org.apache:test", "sha-close")] == [
                "src/Bar.java",
                "src/Foo.java",
            ]
        finally:
            con.close()

    def test_bulk_project_state_cache_matches_point_queries(self, tmp_path: Path):
        db_path = _create_minimal_tdd_db(tmp_path / "td_v2_test.db")
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        try:
            analysis_index = {
                ("org.apache:test", "sha-create"): {
                    "analysis_key": "A1",
                    "date": "2020-01-01 10:00:00",
                    "revision": "sha-create",
                },
                ("org.apache:test", "sha-close"): {
                    "analysis_key": "A2",
                    "date": "2020-01-02 10:00:00",
                    "revision": "sha-close",
                },
            }
            candidates = [
                {
                    "project_id": "org.apache:test",
                    "parent_sha": "sha-create",
                    "child_sha": "sha-close",
                }
            ]
            lifecycle_rows = load_project_code_smell_lifecycle(
                con,
                "org.apache:test",
                rule_name_index={},
            )
            bulk_cache = build_project_state_cache(
                project_id="org.apache:test",
                candidates=candidates,
                analysis_index=analysis_index,
                lifecycle_rows=lifecycle_rows,
            )
            point_before = load_active_code_smells(
                con,
                "org.apache:test",
                "sha-create",
                analysis_index=analysis_index,
                rule_name_index={},
            )
            point_after = load_active_code_smells(
                con,
                "org.apache:test",
                "sha-close",
                analysis_index=analysis_index,
                rule_name_index={},
            )

            assert bulk_cache[("org.apache:test", "sha-create")] == point_before
            assert bulk_cache[("org.apache:test", "sha-close")] == point_after
            assert [s["issue_key"] for s in point_before] == ["ISSUE-1", "ISSUE-2"]
            assert [s["issue_key"] for s in point_after] == ["ISSUE-1", "ISSUE-3"]
        finally:
            con.close()


class TestTdV2Loaders:
    def test_load_tdd_raw_df(self, tmp_path: Path):
        db_path = _create_minimal_tdd_db(tmp_path / "td_v2_test.db")
        df = load_tdd_raw_df(db_path)

        assert len(df) == 4
        assert set(df.columns) >= {
            "project",
            "creation_commit",
            "close_commit",
            "issue_key",
            "issue_type",
            "rule",
            "component",
            "start_line",
            "end_line",
        }
        first = df[df["issue_key"] == "ISSUE-1"].iloc[0]
        assert first["creation_commit"] == "sha-create"
        assert first["close_commit"] in (None, "")
        assert first["start_line"] == 42

    def test_load_eval_samples_for_tdd(self, tmp_path: Path):
        db_path = _create_minimal_tdd_db(tmp_path / "td_v2_test.db")
        samples = load_eval_samples(["tdd"], tdd_db_path=db_path)

        assert len(samples) == 4
        assert all(isinstance(sample, EvalSample) for sample in samples)
        assert all(sample.source == "tdd" for sample in samples)

        sample = next(s for s in samples if s.inputs["rule"] == "java:S138")
        assert sample.sample_id.startswith("tdd:org.apache:test:sha-create:java:S138:")
        assert sample.inputs["component"] == "proj:src/Foo.java"
        assert sample.expectations["close_commit"] == ""
        assert sample.tags["severity"] == "CRITICAL"
