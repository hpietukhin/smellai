"""Enhanced test execution utilities with wrapper and custom command support.

This module extends the basic test execution capabilities to support:
1. Maven/Gradle wrapper scripts (mvnw, gradlew)
2. Custom build commands from datasets
3. Fallback to system-installed build tools

Designed for integration with SWE-Refactor and similar datasets.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which
from typing import Literal

from agents.tools.java_test_tools import TestRunSummary, detect_build_system, _parse_maven_results, _parse_gradle_results


def has_wrapper(project_path: Path, build_system: Literal["maven", "gradle"]) -> bool:
    """Check if project has wrapper scripts.

    Args:
        project_path: Path to project directory
        build_system: Build system type

    Returns:
        True if wrapper exists
    """
    if build_system == "maven":
        return (project_path / "mvnw").exists()
    else:  # gradle
        return (project_path / "gradlew").exists()


def get_build_command(
    project_path: str | Path,
    *,
    custom_command: str | None = None,
    prefer_wrapper: bool = True,
    clean: bool = True,
) -> list[str] | None:
    """Determine the build command to use for testing.

    Priority order:
    1. Custom command (if provided)
    2. Wrapper scripts (if prefer_wrapper=True and they exist)
    3. System-installed tools (mvn/gradle)
    4. None (if nothing available)

    Args:
        project_path: Path to project directory
        custom_command: Custom command from dataset (e.g., "mvn clean test -DskipTests=false")
        prefer_wrapper: Whether to prefer wrapper over system installation
        clean: Whether to include clean step

    Returns:
        Command as list of strings, or None if no build tool available
    """
    project = Path(project_path)

    # Priority 1: Custom command
    if custom_command:
        # Parse custom command string into list
        # Handle common patterns in SWE-Refactor dataset
        return custom_command.split()

    # Detect build system
    build_system = detect_build_system(str(project))
    if not build_system:
        return None

    # Priority 2: Wrapper scripts
    if prefer_wrapper and has_wrapper(project, build_system):
        if build_system == "maven":
            cmd = ["./mvnw"]
        else:  # gradle
            cmd = ["./gradlew"]

        if clean:
            cmd.append("clean")
        cmd.append("test")
        return cmd

    # Priority 3: System installation
    if build_system == "maven" and which("mvn"):
        cmd = ["mvn"]
    elif build_system == "gradle" and which("gradle"):
        cmd = ["gradle"]
    else:
        return None

    if clean:
        cmd.append("clean")
    cmd.append("test")
    return cmd


def run_tests_enhanced(
    project_path: str | Path,
    *,
    custom_command: str | None = None,
    prefer_wrapper: bool = True,
    clean: bool = True,
    timeout: int = 300,
) -> TestRunSummary:
    """Run tests with enhanced command selection.

    This function provides more flexible test execution compared to the basic
    run_tests() function, supporting:
    - Custom build commands from datasets
    - Wrapper scripts
    - Fallback to system tools

    Args:
        project_path: Path to project directory
        custom_command: Custom build command (e.g., from SWE-Refactor dataset)
        prefer_wrapper: Whether to prefer wrapper over system installation
        clean: Whether to run clean before tests
        timeout: Timeout in seconds

    Returns:
        TestRunSummary with results

    Raises:
        RuntimeError: If no build tool is available
    """
    project = Path(project_path)
    build_system = detect_build_system(str(project))

    if not build_system:
        return TestRunSummary(
            build_system="maven",  # Default, but will have error
            exit_code=-1,
            stderr="No Java build system detected. Looking for pom.xml or build.gradle",
        )

    # Get command
    cmd = get_build_command(
        project,
        custom_command=custom_command,
        prefer_wrapper=prefer_wrapper,
        clean=clean,
    )

    if not cmd:
        error_msg = (
            "No Maven/Gradle found. "
            "Install via SDKMAN or add wrappers to project."
        )
        return TestRunSummary(
            build_system=build_system,
            exit_code=-1,
            stderr=error_msg,
        )

    # Execute command
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        summary = TestRunSummary(
            build_system=build_system,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

        # Parse test results
        if build_system == "maven":
            summary.tests = _parse_maven_results(project)
        else:
            summary.tests = _parse_gradle_results(project)

        # Calculate summary statistics
        summary.total = len(summary.tests)
        for test in summary.tests:
            if test.status == "PASS":
                summary.passed += 1
            elif test.status == "FAIL":
                summary.failed += 1
            elif test.status == "ERROR":
                summary.errors += 1
            elif test.status == "SKIPPED":
                summary.skipped += 1
            summary.duration += test.duration

        return summary

    except subprocess.TimeoutExpired:
        return TestRunSummary(
            build_system=build_system,
            exit_code=-1,
            stderr=f"Test execution timed out after {timeout} seconds",
        )
    except Exception as e:
        return TestRunSummary(
            build_system=build_system,
            exit_code=-1,
            stderr=f"Error running tests: {str(e)}",
        )
