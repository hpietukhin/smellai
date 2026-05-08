"""Tests for structural edit tools (text, ast-grep, spoon advisory path)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from agents.tools.edit_tools import (
    structural_backend_advice,
    suggest_structural_backend,
    replace_in_file,
    replace_in_file_git_safe,
    run_ast_grep_rewrite,
    run_ast_grep_rewrite_git_safe,
    run_spoon_refactor,
)


def test_structural_backend_advice_mapping():
    assert structural_backend_advice("Rename Method") == "spoon"
    assert structural_backend_advice("Move Method") == "spoon"
    assert structural_backend_advice("Rename Local Variable") == "ast-grep"
    assert structural_backend_advice("Replace with Logger") == "ast-grep"
    assert structural_backend_advice("Unknown Refactor") == "text"


def test_suggest_structural_backend_tool():
    out = suggest_structural_backend.invoke({"operation": "Rename Method"})
    assert out == "spoon"


def test_replace_in_file_success(tmp_path: Path):
    f = tmp_path / "A.java"
    f.write_text("class A { int x = 1; }", encoding="utf-8")
    out = replace_in_file.invoke(
        {
            "path": str(f),
            "old_text": "int x = 1;",
            "new_text": "int y = 1;",
        }
    )
    assert out.startswith("OK:")
    assert "int y = 1;" in f.read_text(encoding="utf-8")


@pytest.fixture
def java_fixture_project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    src = root / "src"
    src.mkdir(parents=True)
    (src / "A.java").write_text(
        """
class A {
  void m() {
    System.out.println(\"hi\");
  }
}
""".strip(),
        encoding="utf-8",
    )
    return root


def test_run_ast_grep_rewrite_missing_binary(java_fixture_project: Path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    out = run_ast_grep_rewrite.invoke(
        {
            "project_root": str(java_fixture_project),
            "language": "java",
            "pattern": "System.out.println($X)",
            "rewrite": "log.info($X)",
            "target_path": "src",
        }
    )
    assert out.startswith("ERROR: ast-grep")


@pytest.mark.skipif(shutil.which("sg") is None, reason="ast-grep (sg) not installed")
def test_run_ast_grep_rewrite_real(java_fixture_project: Path):
    out = run_ast_grep_rewrite.invoke(
        {
            "project_root": str(java_fixture_project),
            "language": "java",
            "pattern": "System.out.println($X)",
            "rewrite": "log.info($X)",
            "target_path": "src",
        }
    )
    assert out.startswith("OK:")
    content = (java_fixture_project / "src" / "A.java").read_text(encoding="utf-8")
    assert "log.info(" in content


@pytest.fixture
def fake_spoon_plugin_root(tmp_path: Path) -> Path:
    root = tmp_path / "spoon-plugin"
    root.mkdir(parents=True)
    gradlew = root / "gradlew"
    gradlew.write_text(
        "#!/usr/bin/env bash\n"
        "test -n \"$TARGET_PROJECT_ROOT\" || exit 3\n"
        "test -n \"$SPOON_PROCESSOR_FQCN\" || exit 4\n"
        "exit 0\n",
        encoding="utf-8",
    )
    os.chmod(gradlew, 0o755)
    return root


def test_run_spoon_refactor_warns_for_non_structural(
    java_fixture_project: Path,
    fake_spoon_plugin_root: Path,
):
    out = run_spoon_refactor.invoke(
        {
            "target_project_root": str(java_fixture_project),
            "spoon_plugin_root": str(fake_spoon_plugin_root),
            "operation": "Rename Local Variable",
            "processor_fqcn": "com.example.RenameLocalVariableProcessor",
        }
    )
    assert out.startswith("WARN:")


def test_run_spoon_refactor_executes_gradle(
    java_fixture_project: Path,
    fake_spoon_plugin_root: Path,
):
    out = run_spoon_refactor.invoke(
        {
            "target_project_root": str(java_fixture_project),
            "spoon_plugin_root": str(fake_spoon_plugin_root),
            "operation": "Rename Method",
            "processor_fqcn": "com.example.RenameMethodProcessor",
        }
    )
    assert out.startswith("OK:")


@pytest.fixture
def git_java_project(java_fixture_project: Path) -> Path:
    root = java_fixture_project
    import subprocess

    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True, text=True)
    return root


def test_replace_in_file_git_safe_commit_and_rollback(git_java_project: Path):
    import subprocess

    file_path = git_java_project / "src" / "A.java"
    before = subprocess.run(["git", "rev-parse", "HEAD"], cwd=git_java_project, check=True, capture_output=True, text=True).stdout.strip()
    out = replace_in_file_git_safe.invoke(
        {
            "project_root": str(git_java_project),
            "path": str(file_path),
            "old_text": "System.out.println(\"hi\");",
            "new_text": "log.info(\"hi\");",
        }
    )
    assert out.startswith("OK:")
    assert "Rollback: git reset --hard" in out
    assert "log.info(" in file_path.read_text(encoding="utf-8")

    subprocess.run(["git", "reset", "--hard", before], cwd=git_java_project, check=True, capture_output=True, text=True)
    assert "System.out.println(\"hi\");" in file_path.read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("sg") is None, reason="ast-grep (sg) not installed")
def test_run_ast_grep_rewrite_git_safe_creates_commit(git_java_project: Path):
    import subprocess

    before = subprocess.run(["git", "rev-parse", "HEAD"], cwd=git_java_project, check=True, capture_output=True, text=True).stdout.strip()
    out = run_ast_grep_rewrite_git_safe.invoke(
        {
            "project_root": str(git_java_project),
            "language": "java",
            "pattern": "System.out.println($X)",
            "rewrite": "log.info($X)",
            "target_path": "src",
        }
    )
    assert out.startswith("OK:")
    after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=git_java_project, check=True, capture_output=True, text=True).stdout.strip()
    assert before != after
