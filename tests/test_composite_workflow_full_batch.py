import json
import sys
import types

import pytest
from typer.testing import CliRunner
from pathlib import Path
from eliot import FileDestination, Logger
import workflows.composite_workflow_full as workflow_module

from workflows.composite_workflow_full import (
    app,
    WorkflowRunArgs,
    _CURRENT_STEP,
    _CURRENT_TRACKER,
    _case_args_from_batch_case,
    _case_log_path,
    _profile_phase,
    _workflow_cli_args,
)


def test_case_args_from_batch_case_requires_explicit_batch_fields():
    cfg = {"planner": "befs"}
    case = {
        "case_id": "range:JUnit4:abc",
        "project": "JUnit4",
        "repo_url": "https://github.com/junit-team/junit4.git",
        "start_commit_hash": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "elements": ["org.example.Foo", "org.example.Bar"],
        "baseline_verification": {"status": "passed"},
    }

    args = _case_args_from_batch_case(cfg, case, "run-1")

    assert args.project == "JUnit4"
    assert args.repo_url == "https://github.com/junit-team/junit4.git"
    assert args.start_commit_hash == "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    assert args.elements == "org.example.Foo,org.example.Bar"
    assert args.worktree_suffix == "run-1"


def test_case_args_from_batch_case_supports_legacy_meta_payload():
    cfg = {"planner": "befs"}
    case = {
        "case_id": "range:JUnit4:abc",
        "project": "JUnit4",
        "meta": {
            "start_commit_hash": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            "elements": ["org.example.Foo", "org.example.Bar"],
        },
    }

    args = _case_args_from_batch_case(cfg, case, "run-1")

    assert args.repo_url == "https://github.com/junit-team/junit4.git"
    assert args.start_commit_hash == "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    assert args.elements == "org.example.Foo,org.example.Bar"


def test_case_args_from_batch_case_rejects_unverified_case():
    cfg = {"planner": "befs"}
    case = {
        "case_id": "range:JUnit4:abc",
        "project": "JUnit4",
        "repo_url": "https://github.com/junit-team/junit4.git",
        "start_commit_hash": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "elements": ["org.example.Foo"],
        "baseline_verification": {"status": "build_fail"},
    }

    with pytest.raises(AssertionError, match="not baseline-verified"):
        _case_args_from_batch_case(cfg, case, "run-1")


def test_workflow_cli_args_omits_false_and_none_values():
    args = WorkflowRunArgs(
        project="JUnit4",
        repo_url="https://github.com/junit-team/junit4.git",
        start_commit_hash="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        elements="org.example.Foo",
        skip_mlflow_healthcheck=False,
        verbose=True,
        run_name=None,
    )

    cli = _workflow_cli_args(args)

    assert "--project" in cli
    assert "JUnit4" in cli
    assert "--verbose" in cli
    assert "--skip-mlflow-healthcheck" not in cli
    assert "--run-name" not in cli
    assert "--targeted-testing" in cli



def test_workflow_cli_args_serializes_disabled_targeted_testing():
    args = WorkflowRunArgs(
        project="JUnit4",
        repo_url="https://github.com/junit-team/junit4.git",
        start_commit_hash="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        elements="org.example.Foo",
        targeted_testing=False,
    )

    cli = _workflow_cli_args(args)

    assert "--targeted-testing" not in cli
    assert "--no-targeted-testing" in cli


def test_case_log_path_uses_run_name_and_index():
    args = WorkflowRunArgs(
        project="JUnit4",
        repo_url="https://github.com/junit-team/junit4.git",
        start_commit_hash="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        elements="org.example.Foo",
        run_name="single junit4 run",
    )

    path = _case_log_path(Path("logs"), 7, "case-id", args)

    assert path == Path("logs/007-single_junit4_run.log")


def test_batch_help_exposes_num_cases_and_no_single_command():
    runner = CliRunner()

    batch_result = runner.invoke(app, ["batch", "--help"])
    root_result = runner.invoke(app, ["--help"])

    assert batch_result.exit_code == 0
    assert "--num-cases" in batch_result.stdout
    assert "--subset-manifest" not in batch_result.stdout
    assert root_result.exit_code == 0
    assert "--model" in root_result.stdout
    assert " single " not in root_result.stdout


def test_profile_phase_records_elapsed_time_with_step_context():
    events = []

    class FakeTracker:
        def log_timing(self, phase, elapsed_ms, step=None):
            events.append((phase, elapsed_ms, step))

    @_profile_phase("demo_phase")
    def work():
        return "ok"

    tracker_token = _CURRENT_TRACKER.set(FakeTracker())
    step_token = _CURRENT_STEP.set(3)
    try:
        assert work() == "ok"
    finally:
        _CURRENT_STEP.reset(step_token)
        _CURRENT_TRACKER.reset(tracker_token)

    assert len(events) == 1
    phase, elapsed_ms, step = events[0]
    assert phase == "demo_phase"
    assert step == 3
    assert elapsed_ms >= 0.0


def test_run_java_test_bool_forwards_target_files(monkeypatch):
    called = {}

    from agents.tools.java_test_tools import TestRunSummary

    def fake_run_java_test_analysis(
        project_path,
        clean=True,
        timeout=300,
        llm_repair_model=None,
        target_files=None,
        **kwargs,
    ):
        called["project_path"] = project_path
        called["clean"] = clean
        called["timeout"] = timeout
        called["llm_repair_model"] = llm_repair_model
        called["target_files"] = target_files
        called["kwargs"] = kwargs
        return {"summary": TestRunSummary(build_system="maven", exit_code=0)}

    monkeypatch.setattr(workflow_module, "run_java_test_analysis", fake_run_java_test_analysis)

    # import lazily here so the monkeypatch target resolves correctly
    from workflows.composite_workflow_full import _run_java_test_bool

    compile_ok, tests_ok = _run_java_test_bool(
        repo_path=Path('/tmp/project'),
        timeout=12,
        llm_repair_model='some-model',
        target_files=['a.java', 'b.java'],
    )

    assert compile_ok is True
    assert tests_ok is True
    assert called['clean'] is False
    assert called['timeout'] == 12
    assert called['llm_repair_model'] == 'some-model'
    assert called['target_files'] == ['a.java', 'b.java']
    assert called['project_path'] == '/tmp/project'



def test_execute_refactor_action_writes_eliot_subactions(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "src" / "main" / "java" / "com" / "example" / "A.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.example;\npublic class A { int x; }\n", encoding="utf-8")

    class FakeSmell:
        smell_id = "src/main/java/com/example/A.java:1:0"
        file_path = str(source)
        smell_type = "LongMethod"
        severity = 1
        detection_reason = "test"
        line_number = 1

    before_module = types.SimpleNamespace(
        ChatLiteLLM=lambda model: types.SimpleNamespace(
            invoke=lambda _messages: types.SimpleNamespace(
                content="package com.example;\npublic class A { int x; int y; }\n"
            )
        )
    )

    monkeypatch.setitem(sys.modules, "langchain_litellm", before_module)

    def fake_diff_run(cmd, *args, **kwargs):
        assert cmd[:2] == ["git", "diff"]
        return types.SimpleNamespace(returncode=0, stdout="diff --git a/A.java b/A.java\n")

    monkeypatch.setattr(workflow_module.subprocess, "run", fake_diff_run)

    log_path = tmp_path / "eliot_refactor.jsonl"
    with log_path.open("wb") as handle:
        destination = FileDestination(handle)
        Logger._destinations.add(destination)
        try:
            ok, modified = workflow_module._execute_refactor_action(repo, 0, FakeSmell(), "extract", "test-model")
        finally:
            Logger._destinations.remove(destination)

    assert ok is True
    assert modified == source

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    action_types = {event["action_type"] for event in events if "action_type" in event}

    assert "llm_refactor" in action_types
    assert "resolve_refactor_target" in action_types
    assert "import_llm_client" in action_types
    assert "read_refactor_source" in action_types
    assert "llm_refactor_call" in action_types
    assert "parse_refactor_llm_output" in action_types
    assert "write_refactor_output" in action_types
    assert "verify_refactor_diff" in action_types
