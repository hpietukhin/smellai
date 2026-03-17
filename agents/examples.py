#!/usr/bin/env python3
"""Example: Using the Java test analysis functions (pipeline stages A, D, J)."""

from agents.java_test.agent import run_java_test_analysis


def basic_example():
    """Basic usage: detect build system and run tests."""
    result = run_java_test_analysis("/path/to/your/java/project")

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    summary = result["summary"]
    print(f"Build system: {result['build_system']}")
    print(f"Result: {'PASS' if summary.success else 'FAIL'}")
    print(f"Tests: {summary.passed}/{summary.total} passed")


def baseline_vs_after_example():
    """Pipeline stages D + J: capture baseline, then compare after refactoring."""
    project_path = "/path/to/your/java/project"

    # Stage D: pre-refactoring baseline
    before = run_java_test_analysis(project_path)

    # ... apply refactoring here ...

    # Stage J: post-refactoring verification
    after = run_java_test_analysis(project_path)

    if before["summary"] and after["summary"]:
        delta = after["summary"].passed - before["summary"].passed
        print(f"Tests passed delta: {delta:+d}")
        print(f"Behavior preserved: {after['summary'].success}")


if __name__ == "__main__":
    # basic_example()
    # baseline_vs_after_example()
    print("Replace '/path/to/your/java/project' with an actual path and uncomment an example.")
