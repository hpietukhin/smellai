"""Shared runtime helpers for SWE-Refactor evaluation agents.

This module centralizes workspace setup and verification primitives used by both
full SWE LangGraph agents and mini-swe ablations. Agent modules should depend on
these coarse-grained helpers instead of importing individual repo/build/test/JDK
utilities directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from agents.tools.java_test_tools import run_tests_if_present
from swe_refactor.dataset import RefactoringRecord
from swe_refactor.utils import (
    clone_repository,
    compile_project,
    force_checkout_commit,
    get_previous_commit,
    get_repo_url,
    replace_java_code,
    switch_java_version,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SetupResult:
    """Result of preparing a repository workspace for one SWE sample."""

    project_path: Path
    parent_commit: str | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class VerificationResult:
    """Compile/test verification result for a refactoring attempt."""

    compile_success: bool
    test_success: bool
    error: str | None = None


def setup_project_workspace(record: RefactoringRecord, workspace_path: Path) -> SetupResult:
    """Clone, checkout parent commit, and switch JDK for ``record``.

    Returns a structured result instead of raising, matching the existing agent
    behavior where setup failures become model outputs.
    """
    project_path = workspace_path / record.projectName

    try:
        repo_url = get_repo_url(record.projectName)
    except KeyError:
        error = f"Unknown project: {record.projectName}"
        LOGGER.error(error)
        return SetupResult(project_path=project_path, error=error)

    if not project_path.exists():
        success = clone_repository(repo_url, project_path)
        if not success:
            return SetupResult(
                project_path=project_path,
                error=f"Failed to clone {repo_url}",
            )

    parent_commit = get_previous_commit(project_path, record.commitId)
    if not parent_commit:
        return SetupResult(
            project_path=project_path,
            error=f"Failed to get parent of {record.commitId}",
        )

    success = force_checkout_commit(project_path, parent_commit)
    if not success:
        return SetupResult(
            project_path=project_path,
            parent_commit=parent_commit,
            error=f"Failed to checkout {parent_commit}",
        )

    success = switch_java_version(record.compileJDK, project_path)
    if not success:
        LOGGER.warning("Failed to switch JDK to %s", record.compileJDK)

    LOGGER.info(
        "SWE setup complete — %s @ %s in %s",
        record.projectName,
        parent_commit[:8],
        project_path,
    )
    return SetupResult(project_path=project_path, parent_commit=parent_commit)


def verify_refactoring(
    record: RefactoringRecord,
    project_path: Path,
    *,
    refactored_code: str | None = None,
    refactored_target_code: str | None = None,
) -> VerificationResult:
    """Optionally write generated code, then compile and run tests.

    ``refactored_code=None`` supports agent implementations (mini-swe-agent) that
    mutate the working tree themselves before verification.
    """
    if refactored_code is not None:
        source_file = project_path / record.filePathBefore
        success = replace_java_code(source_file, refactored_code)
        if not success:
            return VerificationResult(
                compile_success=False,
                test_success=False,
                error=f"Failed to write {source_file}",
            )

    if refactored_target_code:
        target_file = project_path / record.filePathAfter
        replace_java_code(target_file, refactored_target_code)

    compile_result = compile_project(project_path, record.compileCommand)
    if not compile_result.success:
        error_summary = "\n".join(
            compile_result.error_summary or ["Unknown compile error"]
        )
        LOGGER.warning("SWE verification compile failed:\n%s", error_summary)
        return VerificationResult(
            compile_success=False,
            test_success=False,
            error=error_summary,
        )

    LOGGER.info("SWE verification compilation succeeded")

    test_success = run_tests_if_present(project_path, record.hasTestC)
    if record.hasTestC:
        if test_success:
            LOGGER.info("SWE verification tests passed")
        else:
            LOGGER.warning("SWE verification tests failed")

    return VerificationResult(
        compile_success=True,
        test_success=test_success,
        error=None,
    )


__all__ = [
    "SetupResult",
    "VerificationResult",
    "setup_project_workspace",
    "verify_refactoring",
]
