"""Integration-style tests for Spoon-backed refactoring invocations across catalogue ops.

For each refactoring operation declared in ``domain.rules.REFACTORING_CATALOGUE``,
we run a Spoon-like subprocess (a lightweight test double) over a two-file Java
project and verify that:

1. the subprocess is executed,
2. both Java files are modified,
3. Java AST-signature snapshots (via ``javac -Xprint``) change,
4. the operation-specific AST markers exist in both files.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from domain.rules import REFACTORING_CATALOGUE


def _all_refactoring_operations() -> tuple[str, ...]:
    """Collect all distinct refactoring operations defined in the catalogue."""

    operations: set[str] = set()
    for refs in REFACTORING_CATALOGUE.values():
        for op, _ in refs:
            operations.add(op)
    return tuple(sorted(operations))


def _operation_token(operation: str) -> str:
    """Create a safe, stable AST marker token from a refactoring operation."""

    normalized = "".join(
        ch if ch.isalnum() else "_" for ch in operation.strip().lower()
    )
    normalized = normalized.strip("_")
    if not normalized:
        return "op"
    if normalized[0].isdigit():
        normalized = f"op_{normalized}"
    return normalized[:48]


def _write_catalogue_java_project(root: Path) -> tuple[Path, Path]:
    """Create a two-file Java project used by every operation test."""

    src_dir = root / "src" / "main" / "java" / "com" / "example"
    src_dir.mkdir(parents=True)

    source_file = src_dir / "Source.java"
    target_file = src_dir / "Target.java"

    source_file.write_text(
        """package com.example;

public class Source {
    private int value = 1;

    public int compute(int x) {
        return value + x;
    }
}
""",
        encoding="utf-8",
    )
    target_file.write_text(
        """package com.example;

public class Target {
    private final Source source = new Source();

    public int delegate(int x) {
        return source.compute(x);
    }
}
""",
        encoding="utf-8",
    )

    return source_file, target_file


def _make_spoon_plugin(tmp_path: Path) -> Path:
    """Create a fake Spoon plugin root with a gradlew entry point.

    The script applies operation-specific AST markers to both Java files so tests
    can assert AST-tree changes deterministically without depending on a real Spoon
    runtime in this environment.
    """

    plugin_root = tmp_path / "spoon-plugin"
    plugin_root.mkdir(parents=True)
    gradlew = plugin_root / "gradlew"

    gradlew.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${TARGET_PROJECT_ROOT:?TARGET_PROJECT_ROOT is required}
OP=${SPOON_PROCESSOR_FQCN:-op}

python3 - "$PROJECT_ROOT" "$OP" <<'PY'
import os
import pathlib
import re
import sys


def slugify(op: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in op.strip().lower())
    normalized = normalized.strip("_")
    if not normalized:
        normalized = "op"
    if normalized[0].isdigit():
        normalized = f"op_{normalized}"
    return normalized[:48]


def insert_before_last_brace(text: str, snippet: str) -> str:
    idx = text.rfind("}")
    if idx < 0:
        raise RuntimeError("java source has no closing brace")
    return f"{text[:idx]}{snippet}{text[idx:]}"


project_root = pathlib.Path(sys.argv[1])
operation = sys.argv[2]
token = slugify(operation)

source = project_root / "src" / "main" / "java" / "com" / "example" / "Source.java"
target = project_root / "src" / "main" / "java" / "com" / "example" / "Target.java"

for path in (source, target):
    if not path.exists():
        raise SystemExit(f"missing source file: {path}")

source_code = source.read_text(encoding="utf-8")
target_code = target.read_text(encoding="utf-8")

method_name = f"spoon_{token}"
field_name = f"field_{token}"

method_snippet = f"\n    public int {method_name}() {{ return {len(token)}; }}\n"
field_snippet = f"\n    private int {field_name} = {len(operation)};\n"

if method_name not in source_code:
    source_code = insert_before_last_brace(source_code, method_snippet)

if field_name not in target_code:
    target_code = insert_before_last_brace(target_code, field_snippet)

source.write_text(source_code, encoding="utf-8")
target.write_text(target_code, encoding="utf-8")
PY
""",
        encoding="utf-8",
    )
    gradlew.chmod(0o755)
    return plugin_root


def _java_ast_signature(project_root: Path) -> str:
    """Return the javac -Xprint output for the project's Java files."""

    java_files = sorted(
        str(path)
        for path in (project_root / "src" / "main" / "java").rglob("*.java")
        if path.is_file()
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        compile_result = subprocess.run(
            ["javac", "-Xprint", "-d", tmpdir, *java_files],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    if compile_result.returncode != 0:
        raise AssertionError(
            "javac -Xprint failed before/after AST check\n"
            f"stdout={compile_result.stdout}\nstderr={compile_result.stderr}"
        )
    return compile_result.stdout.strip()


def _write_marker_expectations(operation: str, source_file: Path, target_file: Path) -> tuple[str, str]:
    """Compute expected marker names that the fake Spoon script injects."""

    token = _operation_token(operation)
    return (f"spoon_{token}", f"field_{token}")


@pytest.mark.skipif(shutil.which("javac") is None, reason="JDK javac not available")
def test_spoon_catalogue_backed_operations_modify_two_files_via_subprocess(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    source_file, target_file = _write_catalogue_java_project(project_root)
    plugin_root = _make_spoon_plugin(tmp_path)
    gradlew = plugin_root / "gradlew"

    baseline_source = source_file.read_text(encoding="utf-8")
    baseline_target = target_file.read_text(encoding="utf-8")
    baseline_ast = _java_ast_signature(project_root)

    env = dict(os.environ)
    for operation in _all_refactoring_operations():
        # Fresh checkout per operation keeps operations independent.
        # We recreate files to avoid marker accumulation across operations.
        source_file.write_text(baseline_source, encoding="utf-8")
        target_file.write_text(baseline_target, encoding="utf-8")

        before_source = source_file.read_text(encoding="utf-8")
        before_target = target_file.read_text(encoding="utf-8")

        marker_method, marker_field = _write_marker_expectations(operation, source_file, target_file)

        env["TARGET_PROJECT_ROOT"] = str(project_root)
        env["SPOON_PROCESSOR_FQCN"] = operation

        run = subprocess.run(
            [str(gradlew), "build", "-q"],
            cwd=plugin_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        assert run.returncode == 0, run.stderr

        after_source = source_file.read_text(encoding="utf-8")
        after_target = target_file.read_text(encoding="utf-8")
        after_ast = _java_ast_signature(project_root)

        assert after_source != before_source, f"operation '{operation}' did not change Source.java"
        assert after_target != before_target, f"operation '{operation}' did not change Target.java"
        assert marker_method in after_source
        assert marker_field in after_target

        assert marker_method in after_ast
        assert marker_field in after_ast
        assert after_ast != baseline_ast

        # ensure both files are part of AST output and syntactically valid
        assert "class Source" in after_ast
        assert "class Target" in after_ast

        # keep baseline AST as safety net per-iteration (it is unchanged by resets)
        assert baseline_ast == _java_ast_signature(project_root)
        baseline_ast = after_ast
