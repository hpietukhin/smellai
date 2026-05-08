import types
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner
from pathlib import Path
import workflows.composite_workflow_full as workflow_module

from workflows.composite_workflow_full import (
    app,
    WorkflowRunArgs,
    _CURRENT_STEP,
    _CURRENT_TRACKER,
    _case_args_from_batch_case,
    StepLog,
    _case_log_path,
    _complete_h_trace,
    _derive_run_name_prefix,
    _profile_phase,
    _resolve_smell_file,
    RefactorFilePatch,
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

    with pytest.raises(ValueError, match="not baseline-verified"):
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


def test_derive_run_name_prefix_uses_searchable_parameters():
    prefix = _derive_run_name_prefix(
        {
            "batch_list": "outputs/evals/safe_maven_range_batch_list.json",
            "planner": "befs",
            "detector_backend": "organic",
            "locality": "none",
            "model": "openrouter/minimax/minimax-m2.7",
            "max_steps": 5,
        },
        Path("evals/config/full_eval.json"),
        now=datetime(2026, 5, 5, tzinfo=UTC),
    )

    assert prefix == (
        "full-20260505-batch-safe_maven_range_batch_list-planner-befs-"
        "det-organic-loc-none-model-openrouter_minimax_minimax-m2.7-steps-5"
    )


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


def test_batch_help_exposes_limit_and_no_single_command():
    runner = CliRunner()

    batch_result = runner.invoke(app, ["batch", "--help"])
    root_result = runner.invoke(app, ["--help"])

    assert batch_result.exit_code == 0
    assert "--limit" in batch_result.stdout
    assert "--list-cases" in batch_result.stdout
    assert "--num-cases" not in batch_result.stdout
    assert "--subset-manifest" not in batch_result.stdout
    assert root_result.exit_code == 0
    assert "--model" in root_result.stdout
    assert " single " not in root_result.stdout


def test_complete_h_trace_includes_terminal_post_step_state():
    step_logs = [
        StepLog(
            step=0,
            smell_count_before=1,
            smell_count_after=1,
            h_before=1.0,
            h_after=1.0,
            action_smell_id="Lazy Class:A.java:1",
            action_ref_type="Inline Class",
            compile_passed=False,
            tests_passed=False,
            execution_ok=True,
            stop_reason="retry",
        ),
        StepLog(
            step=1,
            smell_count_before=1,
            smell_count_after=0,
            h_before=1.0,
            h_after=0.0,
            action_smell_id="Lazy Class:A.java:1",
            action_ref_type="Inline Class",
            compile_passed=True,
            tests_passed=True,
            execution_ok=True,
        ),
    ]

    assert _complete_h_trace([1.0, 1.0], step_logs) == [1.0, 1.0, 0.0]


def test_complete_h_trace_preserves_no_progress_transition():
    step_logs = [
        StepLog(
            step=0,
            smell_count_before=1,
            smell_count_after=1,
            h_before=1.0,
            h_after=1.0,
            action_smell_id="Lazy Class:A.java:1",
            action_ref_type="Inline Class",
            compile_passed=True,
            tests_passed=True,
            execution_ok=True,
        )
    ]

    assert _complete_h_trace([1.0], step_logs) == [1.0, 1.0]


def test_resolve_smell_file_handles_inner_class_detector_path(tmp_path):
    source = tmp_path / "src" / "test" / "java" / "ch" / "bind" / "philib" / "msg" / "vm" / "PubSubVMTest.java"
    source.parent.mkdir(parents=True)
    source.write_text("class PubSubVMTest { class FanOut {} }", encoding="utf-8")
    smell = SimpleNamespace(
        file_path="ch/bind/philib/msg/vm/PubSubVMTest/FanOut.java",
        smell_id="Feature Envy:ch/bind/philib/msg/vm/PubSubVMTest/FanOut.java:482",
        class_name="ch.bind.philib.msg.vm.PubSubVMTest.FanOut",
    )

    assert _resolve_smell_file(tmp_path, smell) == source.resolve()


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



def test_execute_refactor_action_records_mlflow_subactions(tmp_path, monkeypatch):
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

    class FakeStructuredModel:
        def invoke(self, _messages):
            return workflow_module.JavaRefactorOutput(
                java_source="package com.example;\npublic class A { int x; int y; }\n",
                refactoring_summary="Added a field for test coverage.",
            )

    class FakeModel:
        def with_structured_output(self, schema, *, method="json_schema"):
            assert schema is workflow_module.JavaRefactorOutput
            assert method == "function_calling"
            return FakeStructuredModel()

    workflow_module._structured_refactor_model.cache_clear()
    monkeypatch.setattr(workflow_module, "make_openrouter_chat_model", lambda _model: FakeModel())

    def fake_diff_run(cmd, *args, **kwargs):
        if cmd[:2] == ["sg", "-p"]:
            return types.SimpleNamespace(
                returncode=0,
                stdout='[{"range":{"start":{"line":0},"end":{"line":0}}}]',
                stderr="",
            )
        assert cmd[:2] == ["git", "diff"]
        return types.SimpleNamespace(returncode=0, stdout="diff --git a/A.java b/A.java\n", stderr="")

    monkeypatch.setattr(workflow_module.subprocess, "run", fake_diff_run)

    action_types: list[str] = []

    class FakeAction:
        def addSuccessFields(self, **_kwargs):
            return None

    @contextmanager
    def fake_start_action(action_type, **kwargs):
        action_types.append(action_type)
        yield FakeAction()

    monkeypatch.setattr(workflow_module, "start_action", fake_start_action)
    ok, modified_files, reason = workflow_module._execute_refactor_action(repo, 0, FakeSmell(), "extract", "test-model", "file")

    assert ok is True
    assert modified_files == [source]
    assert reason == "execution_ok"

    assert "llm_refactor" in action_types
    assert "resolve_refactor_target" in action_types
    assert "preflight_refactor_target" in action_types
    assert "import_llm_client" in action_types
    assert "read_refactor_source" in action_types
    assert "llm_refactor_call" in action_types
    assert "validate_refactor_output" in action_types
    assert "write_refactor_output" in action_types
    assert "verify_refactor_diff" in action_types


def test_invoke_refactor_llm_falls_back_to_json_schema(monkeypatch):
    calls: list[str] = []

    class FakeModel:
        def __init__(self, method: str) -> None:
            self.method = method

        def invoke(self, _messages: list[dict[str, str]]) -> str | None:
            calls.append(self.method)
            if self.method == "function_calling":
                return None
            return '{"java_source": "package p; public class A {}", "refactoring_summary": "Fallback JSON"}'

    monkeypatch.setattr(workflow_module, "_structured_refactor_model", lambda _name, method: FakeModel(method))

    output = workflow_module._invoke_refactor_llm("test-model", [{"role": "user", "content": "x"}])

    assert output.java_source.strip().startswith("package p;")
    assert calls == ["function_calling", "json_schema"]


def test_execute_refactor_action_file_scope_move_method_is_not_blocked(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "src" / "main" / "java" / "com" / "example" / "A.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.example;\npublic class A { int x; }", encoding="utf-8")

    class FakeSmell:
        smell_id = "src/main/java/com/example/A.java:1:0"
        file_path = str(source)
        smell_type = "Feature Envy"
        severity = 1
        detection_reason = "test"
        line_number = 1

    def fake_invoke(_model: str, _messages: list[dict[str, str]]) -> workflow_module.JavaRefactorOutput:
        return workflow_module.JavaRefactorOutput(
            java_source="package com.example;\npublic class A { int y; }",
            refactoring_summary="refactored",
        )

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["sg", "-p"]:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")
        assert cmd[:2] == ["git", "diff"]
        return types.SimpleNamespace(returncode=0, stdout="diff", stderr="")

    monkeypatch.setattr(workflow_module, "_invoke_refactor_llm", fake_invoke)
    monkeypatch.setattr(workflow_module.subprocess, "run", fake_run)

    ok, modified_files, reason = workflow_module._execute_refactor_action(
        repo,
        0,
        FakeSmell(),
        "Move Method",
        "test-model",
        "file",
    )

    assert ok is True
    assert modified_files == [source]
    assert reason == "execution_ok"


def test_execute_refactor_action_project_scope_allows_multiple_files(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    source_a = repo / "src" / "main" / "java" / "com" / "example" / "A.java"
    source_b = repo / "src" / "main" / "java" / "com" / "example" / "B.java"
    source_a.parent.mkdir(parents=True)
    source_b.parent.mkdir(parents=True, exist_ok=True)
    source_a.write_text("package com.example;\npublic class A { int x; }", encoding="utf-8")
    source_b.write_text("package com.example;\npublic class B { int y; }", encoding="utf-8")

    class FakeSmell:
        smell_id = "src/main/java/com/example/A.java:1:0"
        file_path = str(source_a)
        smell_type = "Feature Envy"
        severity = 1
        detection_reason = "test"
        line_number = 1

    def fake_invoke(_model: str, _messages: list[dict[str, str]]) -> workflow_module.JavaRefactorOutput:
        return workflow_module.JavaRefactorOutput(
            files=[
                RefactorFilePatch(file_path="src/main/java/com/example/A.java", java_source="package com.example;\npublic class A { int a; }"),
                RefactorFilePatch(file_path="src/main/java/com/example/B.java", java_source="package com.example;\npublic class B { int b; }"),
            ],
            refactoring_summary="multi-file refactor",
        )

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["sg", "-p"]:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")
        assert cmd[:2] == ["git", "diff"]
        assert "src/main/java/com/example/A.java" in cmd
        assert "src/main/java/com/example/B.java" in cmd
        return types.SimpleNamespace(returncode=0, stdout="diff", stderr="")

    monkeypatch.setattr(workflow_module, "_invoke_refactor_llm", fake_invoke)
    monkeypatch.setattr(workflow_module.subprocess, "run", fake_run)

    ok, modified_files, reason = workflow_module._execute_refactor_action(
        repo,
        0,
        FakeSmell(),
        "Move Method",
        "test-model",
        "project",
    )

    assert ok is True
    assert set(modified_files) == {source_a, source_b}
    assert source_a.read_text(encoding="utf-8").startswith("package com.example;")
    assert source_b.read_text(encoding="utf-8").startswith("package com.example;")
    assert reason == "execution_ok"


def test_execute_refactor_action_labels_no_change(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "src" / "main" / "java" / "com" / "example" / "A.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.example;\npublic class A { int x; }", encoding="utf-8")

    class FakeSmell:
        smell_id = "src/main/java/com/example/A.java:1:0"
        file_path = str(source)
        smell_type = "LongMethod"
        severity = 1
        detection_reason = "test"
        line_number = 1

    monkeypatch.setattr(workflow_module, "_resolve_smell_file", lambda _repo, _smell: source)
    monkeypatch.setattr(workflow_module, "_preflight_refactor_target", lambda _repo, _target_file, _smell: True)

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["sg", "-p"]:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")
        assert cmd[:2] == ["git", "diff"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(workflow_module.subprocess, "run", fake_run)

    action_types: list[str] = []

    class FakeAction:
        def addSuccessFields(self, **_kwargs):
            return None

    @contextmanager
    def fake_start_action(action_type, **kwargs):
        action_types.append(action_type)
        yield FakeAction()

    monkeypatch.setattr(workflow_module, "start_action", fake_start_action)

    class FakeStructured:
        def invoke(self, _messages):
            return workflow_module.JavaRefactorOutput(java_source=source.read_text(), refactoring_summary="noop")

    class FakeModel:
        def with_structured_output(self, _schema, *, method="function_calling"):
            return FakeStructured()

    monkeypatch.setattr(workflow_module, "make_openrouter_chat_model", lambda _model: FakeModel())
    workflow_module._structured_refactor_model.cache_clear()

    ok, modified_files, reason = workflow_module._execute_refactor_action(
        repo,
        0,
        FakeSmell(),
        "extract",
        "test-model",
        "file",
    )

    assert ok is False
    assert modified_files == []
    assert reason == "llm_no_change"
