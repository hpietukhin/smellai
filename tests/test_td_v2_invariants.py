"""Higher-signal invariants and integration tests for TDD v2 transitions.

These tests focus on semantic safety for the bulk state reconstruction path:
- temporal boundary semantics
- candidate filtering correctness
- batching/chunking correctness
- duplicate issue-key hazards
- equivalence of old point-query path vs new bulk path on real data
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from smellai_datasets.td_v2 import (
    _load_analysis_index,
    _load_rule_name_index,
    build_project_state_cache,
    build_smell_delta,
    build_state_cache_bulk,
    build_transition,
    connect_td_v2,
    iter_candidate_transitions,
    load_active_code_smells,
    load_changed_files,
    load_changed_files_batch,
    load_project_code_smell_lifecycle,
    load_project_registry,
    load_refactoring_events,
    parse_git_parents,
    load_refactoring_events_batch,
    validate_transition,
)

REAL_TDD_DB = Path("/Users/havriil.pietukhin/uni/masterThesis/datasets/td_V2.db")


def _create_temporal_db(path: Path) -> Path:
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
    cur.executemany(
        "INSERT INTO SONAR_ANALYSIS VALUES (?, ?, ?, ?)",
        [
            ("org.apache:test", "A1", "2020-01-01 10:00:00", "sha-1"),
            ("org.apache:test", "A2", "2020-01-02 10:00:00", "sha-2"),
            ("org.apache:test", "A3", "2020-01-03 10:00:00", "sha-3"),
        ],
    )
    cur.executemany(
        "INSERT INTO SONAR_ISSUES VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "org.apache:test", "A1", "OPEN_AT_A1", "CODE_SMELL", "java:S138", "CRITICAL",
                "OPEN", "", "1", "1", "", "2020-01-01 10:00:00", "", "open at a1",
                "proj:src/A.java", 10.0, 10.0, None, None, "h1", "",
            ),
            (
                "org.apache:test", "A1", "CLOSE_AT_A2", "CODE_SMELL", "java:S107", "MAJOR",
                "CLOSED", "FIXED", "1", "1", "", "2020-01-01 10:00:00", "2020-01-02 10:00:00",
                "close exactly at a2", "proj:src/B.java", 11.0, 11.0, None, None, "h2", "A2",
            ),
            (
                "org.apache:test", "A2", "OPEN_AT_A2", "CODE_SMELL", "java:S1200", "BLOCKER",
                "OPEN", "", "1", "1", "", "2020-01-02 10:00:00", "", "open exactly at a2",
                "proj:src/C.java", 12.0, 12.0, None, None, "h3", "",
            ),
            (
                "org.apache:test", "A1", "CLOSE_AFTER_A2", "CODE_SMELL", "java:S1541", "MAJOR",
                "CLOSED", "FIXED", "1", "1", "", "2020-01-01 10:00:00", "2020-01-03 10:00:00",
                "close after a2", "proj:src/D.java", 13.0, 13.0, None, None, "h4", "A3",
            ),
        ],
    )
    con.commit()
    con.close()
    return path


def _create_candidate_filter_db(path: Path) -> Path:
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
        CREATE TABLE GIT_COMMITS (
            PROJECT_ID TEXT,
            COMMIT_HASH TEXT,
            AUTHOR_DATE TEXT,
            COMMIT_MESSAGE TEXT,
            IN_MAIN_BRANCH TEXT,
            MERGE TEXT,
            PARENTS TEXT
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

    cur.executemany(
        "INSERT INTO SONAR_ANALYSIS VALUES (?, ?, ?, ?)",
        [
            ("org.apache:test", "A_PARENT", "2020-01-01 00:00:00", "parent-1"),
            ("org.apache:test", "A_VALID", "2020-01-02 00:00:00", "valid-child"),
            ("org.apache:test", "A_MERGE", "2020-01-03 00:00:00", "merge-child"),
            ("org.apache:test", "A_MULTI", "2020-01-04 00:00:00", "multi-child"),
        ],
    )
    cur.executemany(
        "INSERT INTO GIT_COMMITS VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("org.apache:test", "valid-child", "2020-01-02 00:00:00", "valid", "True", "False", "['parent-1']"),
            ("org.apache:test", "merge-child", "2020-01-03 00:00:00", "merge", "True", "True", "['parent-1']"),
            ("org.apache:test", "multi-child", "2020-01-04 00:00:00", "multi", "True", "False", "['parent-1', 'parent-2']"),
            ("org.apache:test", "missing-parent-analysis", "2020-01-05 00:00:00", "no parent analysis", "True", "False", "['unknown-parent']"),
            ("org.apache:test", "missing-child-analysis", "2020-01-06 00:00:00", "no child analysis", "True", "False", "['parent-1']"),
        ],
    )
    cur.executemany(
        "INSERT INTO REFACTORING_MINER VALUES (?, ?, ?, ?)",
        [
            ("org.apache:test", "valid-child", "Extract Method", "valid"),
            ("org.apache:test", "merge-child", "Extract Method", "merge"),
            ("org.apache:test", "multi-child", "Extract Method", "multi"),
            ("org.apache:test", "missing-parent-analysis", "Extract Method", "no-parent"),
            ("org.apache:test", "missing-child-analysis", "Extract Method", "no-child"),
        ],
    )
    con.commit()
    con.close()
    return path


def _create_many_commit_db(path: Path, n: int = 905) -> tuple[Path, list[dict[str, str]]]:
    con = sqlite3.connect(path)
    cur = con.cursor()
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
    candidates: list[dict[str, str]] = []
    ref_rows = []
    change_rows = []
    for i in range(n):
        sha = f"c{i:04d}"
        candidates.append({"project_id": "org.apache:big", "child_sha": sha})
        ref_rows.append(("org.apache:big", sha, "Extract Method", f"detail-{i}"))
        change_rows.append(("org.apache:big", f"src/F{i}.java", sha, "2020-01-01", "dev", "1", "0", "n"))
    cur.executemany("INSERT INTO REFACTORING_MINER VALUES (?, ?, ?, ?)", ref_rows)
    cur.executemany("INSERT INTO GIT_COMMITS_CHANGES VALUES (?, ?, ?, ?, ?, ?, ?, ?)", change_rows)
    con.commit()
    con.close()
    return path, candidates


class TestTemporalSemantics:
    def test_point_query_temporal_boundaries(self, tmp_path: Path):
        db_path = _create_temporal_db(tmp_path / "temporal.db")
        with connect_td_v2(db_path) as con:
            analysis_index = _load_analysis_index(con)
            a1 = [s["issue_key"] for s in load_active_code_smells(
                con, "org.apache:test", "sha-1", analysis_index=analysis_index, rule_name_index={}
            )]
            a2 = [s["issue_key"] for s in load_active_code_smells(
                con, "org.apache:test", "sha-2", analysis_index=analysis_index, rule_name_index={}
            )]
            a3 = [s["issue_key"] for s in load_active_code_smells(
                con, "org.apache:test", "sha-3", analysis_index=analysis_index, rule_name_index={}
            )]

        assert a1 == ["CLOSE_AFTER_A2", "CLOSE_AT_A2", "OPEN_AT_A1"]
        assert a2 == ["CLOSE_AFTER_A2", "OPEN_AT_A1", "OPEN_AT_A2"]
        assert a3 == ["OPEN_AT_A1", "OPEN_AT_A2"]

    def test_bulk_state_cache_matches_temporal_boundaries(self, tmp_path: Path):
        db_path = _create_temporal_db(tmp_path / "temporal.db")
        with connect_td_v2(db_path) as con:
            analysis_index = _load_analysis_index(con)
            lifecycle = load_project_code_smell_lifecycle(
                con,
                "org.apache:test",
                rule_name_index={},
            )
            candidates = [{"project_id": "org.apache:test", "parent_sha": "sha-1", "child_sha": "sha-3"}]
            cache = build_project_state_cache(
                project_id="org.apache:test",
                candidates=candidates,
                analysis_index=analysis_index,
                lifecycle_rows=lifecycle,
            )

        assert [s["issue_key"] for s in cache[("org.apache:test", "sha-1")]] == [
            "CLOSE_AFTER_A2", "CLOSE_AT_A2", "OPEN_AT_A1"
        ]
        assert [s["issue_key"] for s in cache[("org.apache:test", "sha-3")]] == [
            "OPEN_AT_A1", "OPEN_AT_A2"
        ]


class TestAnalysisSelectionAndFiltering:
    def test_latest_analysis_per_revision_wins(self, tmp_path: Path):
        db_path = tmp_path / "analysis.db"
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("CREATE TABLE SONAR_ANALYSIS (PROJECT_ID TEXT, ANALYSIS_KEY TEXT, DATE TEXT, REVISION TEXT)")
        cur.executemany(
            "INSERT INTO SONAR_ANALYSIS VALUES (?, ?, ?, ?)",
            [
                ("org.apache:test", "OLD", "2020-01-01 00:00:00", "same-revision"),
                ("org.apache:test", "NEW", "2020-01-02 00:00:00", "same-revision"),
            ],
        )
        con.commit()
        con.row_factory = sqlite3.Row
        index = _load_analysis_index(con)
        con.close()

        assert index[("org.apache:test", "same-revision")]["analysis_key"] == "NEW"
        assert index[("org.apache:test", "same-revision")]["date"] == "2020-01-02 00:00:00"

    def test_iter_candidate_transitions_filters_invalid_commits(self, tmp_path: Path):
        db_path = _create_candidate_filter_db(tmp_path / "candidates.db")
        with connect_td_v2(db_path) as con:
            analysis_index = _load_analysis_index(con)
            candidates = iter_candidate_transitions(
                con,
                project_id="org.apache:test",
                analysis_index=analysis_index,
            )

        assert len(candidates) == 1
        assert candidates[0]["child_sha"] == "valid-child"
        assert candidates[0]["parent_sha"] == "parent-1"


class TestBatchAndChunking:
    @pytest.mark.parametrize("empty_candidates", [[], [{"project_id": "org.apache:empty", "child_sha": "missing"}]])
    def test_batch_helpers_handle_emptyish_inputs(self, tmp_path: Path, empty_candidates):
        db_path, _ = _create_many_commit_db(tmp_path / "emptyish.db", n=1)
        with connect_td_v2(db_path) as con:
            refs = load_refactoring_events_batch(con, empty_candidates)
            files = load_changed_files_batch(con, empty_candidates)
        if not empty_candidates:
            assert refs == {}
            assert files == {}
        else:
            assert refs == {("org.apache:empty", "missing"): []}
            assert files == {("org.apache:empty", "missing"): []}

    def test_batch_helpers_cross_chunk_boundary(self, tmp_path: Path):
        db_path, candidates = _create_many_commit_db(tmp_path / "many.db", n=905)
        with connect_td_v2(db_path) as con:
            batched_refs = load_refactoring_events_batch(con, candidates)
            batched_files = load_changed_files_batch(con, candidates)

            assert len(batched_refs) == 905
            assert len(batched_files) == 905
            for idx in (0, 450, 904):
                sha = f"c{idx:04d}"
                assert batched_refs[("org.apache:big", sha)] == load_refactoring_events(con, "org.apache:big", sha)
                assert batched_files[("org.apache:big", sha)] == load_changed_files(con, "org.apache:big", sha)


REALISTIC_ISSUE_KEYS = st.from_regex(r"[A-Z][A-Z0-9_]{0,7}-[0-9]{1,4}", fullmatch=True)
REALISTIC_HASHES = st.text(alphabet="0123456789abcdef", min_size=1, max_size=40)


class TestHypothesisProperties:
    @given(
        before_keys=st.lists(REALISTIC_ISSUE_KEYS, unique=True, max_size=20),
        after_keys=st.lists(REALISTIC_ISSUE_KEYS, unique=True, max_size=20),
    )
    def test_build_smell_delta_matches_set_semantics(self, before_keys: list[str], after_keys: list[str]):
        before = [{"issue_key": key} for key in before_keys]
        after = [{"issue_key": key} for key in after_keys]

        delta = build_smell_delta(before, after)
        before_set = set(before_keys)
        after_set = set(after_keys)

        assert delta["persisted"] == sorted(before_set & after_set)
        assert delta["resolved"] == sorted(before_set - after_set)
        assert delta["created"] == sorted(after_set - before_set)
        assert delta["counts"] == {
            "before": len(before_keys),
            "after": len(after_keys),
            "resolved": len(before_set - after_set),
            "created": len(after_set - before_set),
            "persisted": len(before_set & after_set),
        }
        assert validate_transition({"smell_delta": delta}) == []

    @given(parents=st.lists(REALISTIC_HASHES, max_size=6))
    def test_parse_git_parents_roundtrip_for_python_list_strings(self, parents: list[str]):
        assert parse_git_parents(repr(parents)) == parents

    @given(
        point=st.integers(min_value=0, max_value=20),
        lifecycle=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=8),
                st.integers(min_value=0, max_value=20),
                st.one_of(st.none(), st.integers(min_value=0, max_value=21)),
            ),
            unique_by=lambda item: item[0],
            max_size=15,
        ),
    )
    def test_build_project_state_cache_matches_temporal_predicate(
        self,
        point: int,
        lifecycle: list[tuple[str, int, int | None]],
    ):
        project_id = "org.apache:test"
        commit_sha = "sha-point"
        point_date = f"2020-01-{point:02d} 00:00:00"
        lifecycle_rows = []
        expected = []

        for issue_key, open_idx, close_idx in lifecycle:
            close_ok = close_idx is None or close_idx >= 0
            if not close_ok:
                continue
            open_date = f"2020-01-{open_idx:02d} 00:00:00"
            close_date = None if close_idx is None else f"2020-01-{close_idx:02d} 00:00:00"
            smell = {"issue_key": issue_key}
            lifecycle_rows.append(
                {
                    "open_date": open_date,
                    "close_date": close_date,
                    "smell": smell,
                }
            )
            if open_date <= point_date and (close_date is None or close_date > point_date):
                expected.append(smell)

        analysis_index = {
            (project_id, commit_sha): {
                "analysis_key": "A_POINT",
                "date": point_date,
                "revision": commit_sha,
            }
        }
        candidates = [{"project_id": project_id, "parent_sha": commit_sha, "child_sha": commit_sha}]
        cache = build_project_state_cache(
            project_id=project_id,
            candidates=candidates,
            analysis_index=analysis_index,
            lifecycle_rows=lifecycle_rows,
        )

        assert cache[(project_id, commit_sha)] == expected


class TestDuplicateIssueKeyHazards:
    def test_duplicate_issue_keys_break_accounting_invariant(self):
        before = [{"issue_key": "DUP"}, {"issue_key": "DUP"}]
        after = [{"issue_key": "DUP"}]
        delta = build_smell_delta(before, after)

        assert delta["counts"] == {
            "before": 2,
            "after": 1,
            "resolved": 0,
            "created": 0,
            "persisted": 1,
        }
        assert validate_transition({"smell_delta": delta}) == [
            "before_total != persisted + resolved"
        ]


@pytest.mark.skipif(not REAL_TDD_DB.exists(), reason="real td_V2.db not available on this host")
class TestRealDatasetEquivalence:
    @pytest.mark.parametrize(
        ("project_id", "limit"),
        [
            ("org.apache:commons-io", 5),
            ("org.apache:codec", 5),
            ("org.apache:collections", 5),
            ("org.apache:dbcp", 5),
        ],
    )
    def test_bulk_state_cache_matches_point_query_transitions(self, project_id: str, limit: int):
        with connect_td_v2(REAL_TDD_DB) as con:
            project_registry = load_project_registry(con)
            analysis_index = _load_analysis_index(con)
            rule_name_index = _load_rule_name_index(con)
            candidates = iter_candidate_transitions(
                con,
                project_id=project_id,
                analysis_index=analysis_index,
            )[:limit]
            batched_refactorings = load_refactoring_events_batch(con, candidates)
            batched_changed_files = load_changed_files_batch(con, candidates)
            bulk_state_cache = build_state_cache_bulk(
                con,
                candidates,
                analysis_index=analysis_index,
                rule_name_index=rule_name_index,
            )

            bulk = [
                build_transition(
                    con,
                    candidate,
                    project_registry=project_registry,
                    analysis_index=analysis_index,
                    rule_name_index=rule_name_index,
                    batched_refactorings=batched_refactorings,
                    batched_changed_files=batched_changed_files,
                    state_cache=bulk_state_cache,
                )
                for candidate in candidates
            ]
            point = [
                build_transition(
                    con,
                    candidate,
                    project_registry=project_registry,
                    analysis_index=analysis_index,
                    rule_name_index=rule_name_index,
                    batched_refactorings=batched_refactorings,
                    batched_changed_files=batched_changed_files,
                    state_cache=None,
                )
                for candidate in candidates
            ]

        assert bulk == point

    def test_real_transitions_are_deterministic_for_same_inputs(self):
        from smellai_datasets.td_v2 import extract_transitions

        first, first_summary = extract_transitions(REAL_TDD_DB, project_id="org.apache:commons-io", limit=5)
        second, second_summary = extract_transitions(REAL_TDD_DB, project_id="org.apache:commons-io", limit=5)

        assert first == second
        assert first_summary["transition_count"] == second_summary["transition_count"] == 5
        assert first_summary["validation_error_count"] == second_summary["validation_error_count"] == 0
