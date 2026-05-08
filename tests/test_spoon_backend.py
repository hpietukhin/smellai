"""Tests for Spoon-backed refactoring execution path.

These tests exercise the `run_spoon_refactor` tool contract without invoking a
real Gradle build by monkeypatching process execution.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import agents.tools.edit_tools as edit_tools
from agents.tools.edit_tools import run_spoon_refactor


@pytest.fixture
def java_fixture_project(tmp_path: Path) -> Path:
    """Small Java project layout used by Spoon tool tests."""
    root = tmp_path / "proj"
    root.mkdir()
    return root


@pytest.fixture
def fake_spoon_plugin_root(tmp_path: Path) -> Path:
    """Minimal fake Spoon plugin checkout with a gradle wrapper script."""
    root = tmp_path / "spoon-plugin"
    root.mkdir(parents=True)
    gradlew = root / "gradlew"
    gradlew.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"fake gradlew\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    os.chmod(gradlew, 0o755)
    return root


def test_run_spoon_refactor_errors_without_gradlew(java_fixture_project: Path, tmp_path: Path) -> None:
    plugin_root = tmp_path / "no-gradlew"
    plugin_root.mkdir()

    out = run_spoon_refactor.invoke(
        {
            "target_project_root": str(java_fixture_project),
            "spoon_plugin_root": str(plugin_root),
            "operation": "Rename Method",
            "processor_fqcn": "com.example.RenameMethodProcessor",
        }
    )

    assert out.startswith("ERROR: gradlew not found")


def test_run_spoon_refactor_warns_for_non_structural_operation(
    monkeypatch: pytest.MonkeyPatch,
    java_fixture_project: Path,
    fake_spoon_plugin_root: Path,
) -> None:
    called: dict[str, bool] = {"run": False}

    def fail_if_called(*args, **kwargs) -> SimpleNamespace:
        called["run"] = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(edit_tools.subprocess, "run", fail_if_called)

    out = run_spoon_refactor.invoke(
        {
            "target_project_root": str(java_fixture_project),
            "spoon_plugin_root": str(fake_spoon_plugin_root),
            "operation": "Replace with Logger",
            "processor_fqcn": "com.example.ReplaceWithLoggerProcessor",
        }
    )

    assert out.startswith("WARN: operation 'Replace with Logger' is usually 'ast-grep', not spoon")
    assert called["run"] is False


@pytest.mark.parametrize(
    "operation",
    [
        "Rename Method",
        "Move Method",
        "Extract Class",
        "Move Class",
        "Pull Up Method",
        "Push Down Attribute",
        "Replace Conditional with Polymorphism",
    ],
)
def test_run_spoon_refactor_invokes_gradle_for_structural_ops(
    monkeypatch: pytest.MonkeyPatch,
    java_fixture_project: Path,
    fake_spoon_plugin_root: Path,
    operation: str,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, cwd=None, **kwargs) -> SimpleNamespace:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = kwargs["env"]
        captured["capture_output"] = kwargs.get("capture_output")
        captured["text"] = kwargs.get("text")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(edit_tools.subprocess, "run", fake_run)

    out = run_spoon_refactor.invoke(
        {
            "target_project_root": str(java_fixture_project),
            "spoon_plugin_root": str(fake_spoon_plugin_root),
            "operation": operation,
            "processor_fqcn": "com.example.Processor",
        }
    )

    assert out.startswith("OK: Spoon gradle build executed")
    assert captured["cmd"] == [str(fake_spoon_plugin_root / "gradlew"), "build", "-q"]
    assert captured["cwd"] == fake_spoon_plugin_root
    assert captured["capture_output"] is True
    assert captured["text"] is True

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["TARGET_PROJECT_ROOT"] == str(java_fixture_project)
    assert env["SPOON_PROCESSOR_FQCN"] == "com.example.Processor"


@pytest.mark.parametrize(
    "stdout,stderr,exit_code,msg",
    [
        ("", "refactor failed", 1, "ERROR: spoon gradle run failed"),
        ("", "", 2, "ERROR: spoon gradle run failed"),
    ],
)
def test_run_spoon_refactor_propagates_gradle_failure(
    monkeypatch: pytest.MonkeyPatch,
    java_fixture_project: Path,
    fake_spoon_plugin_root: Path,
    stdout: str,
    stderr: str,
    exit_code: int,
    msg: str,
) -> None:
    def fake_run(*args, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(returncode=exit_code, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(edit_tools.subprocess, "run", fake_run)

    out = run_spoon_refactor.invoke(
        {
            "target_project_root": str(java_fixture_project),
            "spoon_plugin_root": str(fake_spoon_plugin_root),
            "operation": "Rename Method",
            "processor_fqcn": "com.example.RenameMethodProcessor",
        }
    )

    assert out.startswith(msg)
    assert stderr in out or stdout in out
