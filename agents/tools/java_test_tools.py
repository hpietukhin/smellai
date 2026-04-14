"""Java test runner tools for LangGraph agent.

This module provides tools for detecting build systems, running tests,
and parsing test results for Java projects. Supports Maven and Gradle.
"""

from __future__ import annotations

import logging
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from langchain_core.tools import tool

LOGGER = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Individual test result."""

    name: str
    status: Literal["PASS", "FAIL", "ERROR", "SKIPPED"]
    duration: float = 0.0
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    failure_trace: Optional[str] = None


@dataclass
class TestCounts:
    """Aggregated counts from a test run."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration: float = 0.0


@dataclass
class TestRunSummary:
    """Summary of test run."""

    build_system: Literal["maven", "gradle"]
    exit_code: int = 0
    tests: list[TestResult] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    counts: TestCounts = field(default_factory=TestCounts)

    @property
    def success(self) -> bool:
        """Check if all tests passed."""
        return self.exit_code == 0 and self.counts.failed == 0 and self.counts.errors == 0


def detect_build_system(project_path: str) -> Optional[Literal["maven", "gradle"]]:
    """Detect Java build system in the project.

    Args:
        project_path: Path to the project directory

    Returns:
        "maven", "gradle", or None if no build system found
    """
    project = Path(project_path)

    if (project / "pom.xml").exists():
        return "maven"

    if (project / "build.gradle").exists() or (project / "build.gradle.kts").exists():
        return "gradle"

    return None


def run_cmd_and_parse(
    cmd: list[str],
    project: Path,
    build_system: Literal["maven", "gradle"],
    timeout: int = 300,
) -> TestRunSummary:
    """Execute a test command and parse results.

    Args:
        cmd: Command to execute
        project: Path to the project directory
        build_system: Build system type (for report parsing)
        timeout: Timeout in seconds

    Returns:
        TestRunSummary with results
    """
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
        summary.counts.total = len(summary.tests)
        for test in summary.tests:
            if test.status == "PASS":
                summary.counts.passed += 1
            elif test.status == "FAIL":
                summary.counts.failed += 1
            elif test.status == "ERROR":
                summary.counts.errors += 1
            elif test.status == "SKIPPED":
                summary.counts.skipped += 1
            summary.counts.duration += test.duration

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


def run_tests(
    project_path: str,
    build_system: Literal["maven", "gradle"],
    *,
    clean: bool = True,
    timeout: int = 300,
) -> TestRunSummary:
    """Run tests using the detected build system.

    Args:
        project_path: Path to the project directory
        build_system: Build system to use ("maven" or "gradle")
        clean: Whether to run clean before tests
        timeout: Timeout in seconds

    Returns:
        TestRunSummary with results
    """
    project = Path(project_path)

    if build_system == "maven":
        cmd = ["mvn"]
    else:  # gradle
        cmd = ["gradle"]

    if clean:
        cmd.append("clean")
    cmd.append("test")

    return run_cmd_and_parse(cmd, project, build_system, timeout)


def _parse_test_xml_reports(report_dir: Path) -> list[TestResult]:
    """Parse JUnit-style XML test reports (used by both Maven Surefire and Gradle).

    Args:
        report_dir: Path to the directory containing TEST-*.xml files

    Returns:
        List of TestResult objects
    """
    results = []

    if not report_dir.exists():
        return results

    for xml_file in report_dir.glob("TEST-*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for testcase in root.findall("testcase"):
                name = f"{testcase.get('classname', '')}.{testcase.get('name', '')}"
                duration = float(testcase.get("time", "0"))

                # Check for failure
                failure = testcase.find("failure")
                error = testcase.find("error")
                skipped = testcase.find("skipped")

                if failure is not None:
                    status = "FAIL"
                    error_message = failure.get("message", "")
                    error_type = failure.get("type", "")
                    failure_trace = failure.text
                elif error is not None:
                    status = "ERROR"
                    error_message = error.get("message", "")
                    error_type = error.get("type", "")
                    failure_trace = error.text
                elif skipped is not None:
                    status = "SKIPPED"
                    error_message = None
                    error_type = None
                    failure_trace = None
                else:
                    status = "PASS"
                    error_message = None
                    error_type = None
                    failure_trace = None

                results.append(
                    TestResult(
                        name=name,
                        status=status,
                        duration=duration,
                        error_message=error_message,
                        error_type=error_type,
                        failure_trace=failure_trace,
                    )
                )
        except ET.ParseError as e:
            LOGGER.warning("Skipping malformed XML file %s: %s", xml_file, e)
            continue

    return results


def _parse_maven_results(project_path: Path) -> list[TestResult]:
    """Parse Maven Surefire XML test reports."""
    return _parse_test_xml_reports(project_path / "target" / "surefire-reports")


def _parse_gradle_results(project_path: Path) -> list[TestResult]:
    """Parse Gradle test XML reports."""
    return _parse_test_xml_reports(project_path / "build" / "test-results" / "test")


# LangChain tools for agent


@tool
def detect_java_build_system(project_path: str) -> str:
    """Detect Java build system (Maven or Gradle) in a project.

    Args:
        project_path: Path to the Java project directory

    Returns:
        String describing the detected build system
    """
    build_system = detect_build_system(project_path)

    if build_system is None:
        return f"No Java build system detected in {project_path}. Looking for pom.xml (Maven) or build.gradle/build.gradle.kts (Gradle)."

    return f"Detected {build_system.upper()} build system in {project_path}"


@tool
def run_java_tests(project_path: str, clean: bool = True) -> dict:
    """Run Java tests and return results summary.

    Args:
        project_path: Path to the Java project directory
        clean: Whether to run clean before tests (default: True)

    Returns:
        Dictionary with test results including pass/fail counts and failed test details
    """
    build_system = detect_build_system(project_path)

    if build_system is None:
        return {
            "error": f"No Java build system detected in {project_path}",
            "success": False,
        }

    summary = run_tests(project_path, build_system, clean=clean)

    # Prepare response
    result = test_summary_to_dict(summary)

    # Add failed test details
    if summary.counts.failed > 0 or summary.counts.errors > 0:
        result["failed_tests"] = [
            {
                "name": test.name,
                "status": test.status,
                "error_type": test.error_type,
                "error_message": test.error_message,
                "failure_trace": (
                    test.failure_trace[:500] + "..."
                    if test.failure_trace and len(test.failure_trace) > 500
                    else test.failure_trace
                ),
            }
            for test in summary.tests
            if test.status in ("FAIL", "ERROR")
        ]

    return result


def run_tests_if_present(project_path, has_tests: bool) -> bool:
    """Run tests when ``has_tests`` is True and a build system is detected.

    Returns True if tests passed (or no tests were run), False otherwise.
    """
    if not has_tests:
        return True
    build_system = detect_build_system(str(project_path))
    if build_system:
        test_result = run_tests(str(project_path), build_system)
        return test_result.success
    return True


def test_summary_to_dict(summary: TestRunSummary) -> dict:
    """Convert a TestRunSummary to the standard result dict (9 core fields)."""
    return {
        "build_system": summary.build_system,
        "success": summary.success,
        "total": summary.counts.total,
        "passed": summary.counts.passed,
        "failed": summary.counts.failed,
        "errors": summary.counts.errors,
        "skipped": summary.counts.skipped,
        "duration": round(summary.counts.duration, 2),
        "exit_code": summary.exit_code,
    }


@tool
def get_test_output(project_path: str) -> str:
    """Get recent test output from Maven or Gradle.

    Args:
        project_path: Path to the Java project directory

    Returns:
        Recent test output (stdout and stderr)
    """
    build_system = detect_build_system(project_path)

    if build_system is None:
        return f"No Java build system detected in {project_path}"

    summary = run_tests(project_path, build_system, clean=False, timeout=60)

    output = "=== Test Output ===\n\n"

    if summary.stdout:
        output += "STDOUT:\n" + summary.stdout[-2000:] + "\n\n"

    if summary.stderr:
        output += "STDERR:\n" + summary.stderr[-2000:] + "\n"

    return output


# Helper function to get all tools
def get_java_test_tools() -> list:
    """Get all Java test tools for LangGraph agent.

    Returns:
        List of LangChain tools
    """
    return [
        detect_java_build_system,
        run_java_tests,
        get_test_output,
    ]
