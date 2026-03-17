"""Java test analysis functions for pipeline stages A, D, and J.

Stages (per conf.tex pipeline):
  A - load source, detect build system (Maven or Gradle)
  D - run full test suite, record pre-refactoring baseline
  J - run test suite after refactoring, compare before/after metrics

No LLM is needed: all work is deterministic shell execution.
"""

from __future__ import annotations

from agents.tools.java_test_tools import (
    TestRunSummary,
    detect_build_system,
    run_tests,
)


def run_java_test_analysis(
    project_path: str,
    *,
    clean: bool = True,
    timeout: int = 300,
) -> dict:
    """Detect build system and run Java tests (stages A + D/J).

    Args:
        project_path: Path to the Java project directory.
        clean: Whether to run clean before tests.
        timeout: Timeout in seconds for the test command.

    Returns:
        dict with keys: project_path, build_system, summary (TestRunSummary).
    """
    build_system = detect_build_system(project_path)
    if build_system is None:
        return {
            "project_path": project_path,
            "build_system": None,
            "summary": None,
            "error": f"No Java build system detected in {project_path}",
        }

    summary: TestRunSummary = run_tests(
        project_path, build_system, clean=clean, timeout=timeout
    )

    return {
        "project_path": project_path,
        "build_system": build_system,
        "summary": summary,
    }
