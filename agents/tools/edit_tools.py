"""Minimal file edit tools for LangGraph agents.

Small, explicit tools inspired by simple code-assistant workflows.
Use these when an agent needs controlled read/write/replace actions.
"""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import uuid

from langchain_core.tools import tool


def _resolve(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    assert p.is_absolute(), "Resolved path must be absolute"
    return p


def _git(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *cmd], cwd=cwd, capture_output=True, text=True)


def _require_clean_git_repo(project_root: Path) -> tuple[bool, str]:
    r = _git(["rev-parse", "--is-inside-work-tree"], project_root)
    if r.returncode != 0:
        return False, "ERROR: target is not a git repository"
    s = _git(["status", "--porcelain"], project_root)
    if s.returncode != 0:
        return False, f"ERROR: git status failed: {s.stderr.strip()}"
    if s.stdout.strip():
        return False, "ERROR: git repo is dirty; commit/stash first for safe rollback"
    return True, ""


def _commit_edit(project_root: Path, paths: list[str], message: str) -> tuple[bool, str]:
    add = _git(["add", *paths], project_root)
    if add.returncode != 0:
        return False, f"ERROR: git add failed: {add.stderr.strip()}"
    commit = _git(["commit", "-m", message], project_root)
    if commit.returncode != 0:
        return False, f"ERROR: git commit failed: {commit.stderr.strip()}"
    sha = _git(["rev-parse", "HEAD"], project_root)
    if sha.returncode != 0:
        return False, "ERROR: cannot read HEAD"
    return True, sha.stdout.strip()


def structural_backend_advice(operation: str) -> str:
    """Return preferred backend for refactoring operation.

    Safe advisory only (no forcing): spoon | ast-grep | text
    """
    op = (operation or "").strip().lower()
    if op in {
        "rename method",
        "move method",
        "extract class",
        "move class",
        "pull up method",
        "push down method",
        "pull up field",
        "push down field",
        "pull up attribute",
        "push down attribute",
        "replace conditional with polymorphism",
    }:
        return "spoon"
    if op in {
        "rename local variable",
        "rename variable",
        "replace with logger",
        "extract method",
        "inline method",
    }:
        return "ast-grep"
    return "text"


@tool
def read_text_file(path: str) -> str:
    """Read UTF-8 text file content.

    Args:
        path: Absolute or relative file path.
    """
    p = _resolve(path)
    if not p.exists():
        return f"ERROR: file does not exist: {p}"
    if p.is_dir():
        return f"ERROR: path is a directory: {p}"
    return p.read_text(encoding="utf-8")


@tool
def write_text_file(path: str, content: str) -> str:
    """Create or overwrite a UTF-8 text file.

    Args:
        path: Absolute or relative file path.
        content: New file content.
    """
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    assert p.exists(), "File must exist after write"
    return f"OK: wrote {p} ({len(content)} chars)"


@tool
def replace_in_file(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in a UTF-8 file.

    Fails if old_text does not appear exactly once.

    Args:
        path: Absolute or relative file path.
        old_text: Exact existing text.
        new_text: Replacement text.
    """
    assert old_text, "old_text must be non-empty"
    p = _resolve(path)
    if not p.exists() or p.is_dir():
        return f"ERROR: invalid file path: {p}"

    content = p.read_text(encoding="utf-8")
    count = content.count(old_text)
    if count != 1:
        return f"ERROR: expected exactly 1 occurrence, found {count}"

    updated = content.replace(old_text, new_text)
    p.write_text(updated, encoding="utf-8")
    return f"OK: replaced text in {p}"


@tool
def run_ast_grep_rewrite(
    project_root: str,
    language: str,
    pattern: str,
    rewrite: str,
    target_path: str = "src",
) -> str:
    """Run ast-grep structural rewrite on project sources.

    Requires ast-grep (`sg`) installed.
    """
    root = _resolve(project_root)
    cmd = [
        "sg",
        "--lang",
        language,
        "--pattern",
        pattern,
        "--rewrite",
        rewrite,
        "--update-all",
        str(root / target_path),
    ]
    try:
        res = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    except FileNotFoundError:
        return "ERROR: ast-grep (`sg`) not found in PATH"
    if res.returncode != 0:
        return f"ERROR: ast-grep failed\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return f"OK: ast-grep rewrite applied\n{res.stdout.strip()}"


@tool
def run_ast_grep_rewrite_git_safe(
    project_root: str,
    language: str,
    pattern: str,
    rewrite: str,
    target_path: str = "src",
) -> str:
    """Run ast-grep rewrite and create rollback-able git commit."""
    root = _resolve(project_root)
    ok, msg = _require_clean_git_repo(root)
    if not ok:
        return msg
    before = _git(["rev-parse", "HEAD"], root).stdout.strip()
    out = run_ast_grep_rewrite.invoke(
        {
            "project_root": str(root),
            "language": language,
            "pattern": pattern,
            "rewrite": rewrite,
            "target_path": target_path,
        }
    )
    if not out.startswith("OK:"):
        return out
    success, sha_or_err = _commit_edit(
        root,
        [target_path],
        f"edit-tools(ast-grep): {pattern} -> {rewrite} [{uuid.uuid4().hex[:8]}]",
    )
    if not success:
        return sha_or_err
    return (
        f"OK: ast-grep rewrite committed at {sha_or_err}. "
        f"Rollback: git reset --hard {before}"
    )


@tool
def run_spoon_refactor(
    target_project_root: str,
    spoon_plugin_root: str,
    operation: str,
    processor_fqcn: str,
) -> str:
    """Advisory Spoon runner via Gradle plugin repository.

    This is a safe helper: it runs Gradle in spoon plugin repo and expects
    the target project to be configured to apply spoon processors.
    """
    target = _resolve(target_project_root)
    spoon_root = _resolve(spoon_plugin_root)
    gradlew = spoon_root / "gradlew"
    if not gradlew.exists():
        return f"ERROR: gradlew not found in {spoon_root}"

    advice = structural_backend_advice(operation)
    if advice != "spoon":
        return f"WARN: operation '{operation}' is usually '{advice}', not spoon"

    cmd = [str(gradlew), "build", "-q"]
    env = {
        **os.environ,
        "TARGET_PROJECT_ROOT": str(target),
        "SPOON_PROCESSOR_FQCN": processor_fqcn,
    }
    res = subprocess.run(cmd, cwd=spoon_root, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        return f"ERROR: spoon gradle run failed\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return "OK: Spoon gradle build executed. Ensure target project applies spoon plugin and configured processor."


@tool
def replace_in_file_git_safe(
    project_root: str,
    path: str,
    old_text: str,
    new_text: str,
) -> str:
    """Replace exact text and create rollback-able git commit."""
    root = _resolve(project_root)
    ok, msg = _require_clean_git_repo(root)
    if not ok:
        return msg
    before = _git(["rev-parse", "HEAD"], root).stdout.strip()
    out = replace_in_file.invoke(
        {"path": path, "old_text": old_text, "new_text": new_text}
    )
    if not out.startswith("OK:"):
        return out
    success, sha_or_err = _commit_edit(
        root,
        [str(_resolve(path))],
        f"edit-tools(text): replace in {Path(path).name} [{uuid.uuid4().hex[:8]}]",
    )
    if not success:
        return sha_or_err
    return f"OK: committed at {sha_or_err}. Rollback: git reset --hard {before}"


@tool
def suggest_structural_backend(operation: str) -> str:
    """Suggest preferred backend for a refactoring operation.

    Returns one of: spoon, ast-grep, text.
    """
    return structural_backend_advice(operation)
