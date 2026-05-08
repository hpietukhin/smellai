
from agents.tools.java_test_tools import TestCounts, TestRunSummary
from evals.generate_batch_list import (
    _classify_baseline_failure,
    _load_repo_urls,
    _partition_requested_projects,
)


def test_load_repo_urls_reads_project_mapping(tmp_path):
    csv_path = tmp_path / "repos.csv"
    csv_path.write_text(
        "project,repo_url\nJUnit4,https://github.com/junit-team/junit4.git\n",
        encoding="utf-8",
    )

    mapping = _load_repo_urls(str(csv_path))

    assert mapping == {"JUnit4": "https://github.com/junit-team/junit4.git"}


def test_classify_baseline_failure_marks_missing_summary_as_toolchain_fail():
    status, details = _classify_baseline_failure({"summary": None, "error": "no build system"})

    assert status == "toolchain_fail"
    assert "no build system" in details


def test_classify_baseline_failure_marks_test_fail_when_tests_failed():
    summary = TestRunSummary(
        build_system="maven",
        exit_code=1,
        counts=TestCounts(total=10, failed=1, errors=0),
    )

    status, details = _classify_baseline_failure({"summary": summary})

    assert status == "test_fail"
    assert "failed=1" in details


def test_partition_requested_projects_excludes_okhttp_with_reason():
    active, excluded = _partition_requested_projects(["Lyra", "OkHttp", "Tap4j"])

    assert active == ["Lyra", "Tap4j"]
    assert excluded == {"OkHttp": "excluded from batch generation: historical compatibility/patch target"}


def test_classify_baseline_failure_marks_build_fail_when_command_failed_before_tests():
    summary = TestRunSummary(
        build_system="maven",
        exit_code=1,
        counts=TestCounts(total=0, failed=0, errors=0),
    )

    status, details = _classify_baseline_failure({"summary": summary})

    assert status == "build_fail"
    assert "code 1" in details
