#!/usr/bin/env python3
"""Workflow for running Java test analysis (pipeline stages A + D).

Usage:
    uv run workflows/java_test_workflow.py --project /path/to/java/project
    uv run workflows/java_test_workflow.py --project /path/to/java/project --json
    uv run workflows/java_test_workflow.py --project /path/to/java/project --verbose
"""

from __future__ import annotations

import json as json_lib
import logging
import sys
from pathlib import Path

from agents.java_test.agent import run_java_test_analysis
from agents.tools.java_test_tools import test_summary_to_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _validate_project(project: str) -> Path | None:
    project_path = Path(project)
    if not project_path.exists() or not project_path.is_dir():
        logger.error("Project path is not a directory: %s", project_path)
        return None
    return project_path


def _print_failed_tests(summary) -> None:
    if summary.failed <= 0 and summary.errors <= 0:
        return
    print(f"\n{'-' * 40}")
    print("Failed tests:")
    for test in summary.tests:
        if test.status in ("FAIL", "ERROR"):
            print(f"  [{test.status}] {test.name}")
            if test.error_message:
                print(f"         {test.error_message}")


def _print_text_summary(result: dict) -> None:
    summary = result["summary"]
    print(f"\n{'=' * 80}")
    print("Java Test Analysis Results")
    print(f"{'=' * 80}\n")
    print(f"Project:      {result['project_path']}")
    print(f"Build system: {result['build_system']}")
    print(f"Result:       {'PASS' if summary.success else 'FAIL'}")
    print(
        f"Tests:        {summary.passed}/{summary.total} passed"
        f"  ({summary.failed} failed, {summary.errors} errors,"
        f" {summary.skipped} skipped)"
    )
    print(f"Duration:     {summary.duration:.2f}s")
    _print_failed_tests(summary)
    print(f"\n{'=' * 80}\n")


def _print_result(result: dict, as_json: bool) -> None:
    summary = result["summary"]
    if as_json:
        output = {
            "project_path": result["project_path"],
            **test_summary_to_dict(summary),
        }
        print(json_lib.dumps(output, indent=2))
    else:
        _print_text_summary(result)


def main(
    project: str,
    no_clean: bool = False,
    timeout: int = 300,
    verbose: bool = False,
    json: bool = False,
) -> int:
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    project_path = _validate_project(project)
    if project_path is None:
        return 1

    logger.info("Analyzing Java tests in: %s", project_path)

    try:
        result = run_java_test_analysis(
            str(project_path),
            clean=not no_clean,
            timeout=timeout,
        )
        if "error" in result:
            logger.error(result["error"])
            return 1

        _print_result(result, json)
        logger.info("Analysis completed successfully")
        return 0

    except Exception as e:
        logger.error("Error during analysis: %s", e, exc_info=verbose)
        return 1


if __name__ == "__main__":
    import fire

    sys.exit(fire.Fire(main))
