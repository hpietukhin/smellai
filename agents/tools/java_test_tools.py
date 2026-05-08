"""Java test runner tools for LangGraph agent.

This module provides tools for detecting build systems, running tests,
and parsing test results for Java projects. Supports Maven and Gradle.
"""

from __future__ import annotations

import logging
import os
import select
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

LOGGER = logging.getLogger(__name__)
BuildSystem = Literal["maven", "gradle"]
TestStatus = Literal["PASS", "FAIL", "ERROR", "SKIPPED"]
REPORT_PATHS: dict[BuildSystem, str] = {
    "maven": "target/surefire-reports",
    "gradle": "build/test-results/test",
}
XML_TAG_TO_STATUS: dict[str, TestStatus] = {
    "failure": "FAIL",
    "error": "ERROR",
    "skipped": "SKIPPED",
}


@dataclass
class TestResult:
    """Individual test result."""

    __test__ = False

    name: str
    status: TestStatus
    duration: float = 0.0
    error_message: str | None = None
    error_type: str | None = None
    failure_trace: str | None = None


@dataclass
class TestCounts:
    """Aggregated counts from a test run."""

    __test__ = False

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration: float = 0.0


@dataclass
class TestRunSummary:
    """Summary of test run."""

    __test__ = False

    build_system: BuildSystem
    exit_code: int = 0
    tests: list[TestResult] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    counts: TestCounts = field(default_factory=TestCounts)

    @property
    def success(self) -> bool:
        """Check if all tests passed."""
        return self.exit_code == 0 and self.counts.failed == 0 and self.counts.errors == 0


def detect_build_system(project_path: str) -> BuildSystem | None:
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


def _run_streaming_command(cmd: list[str], project: Path, timeout: int) -> tuple[int, str, str]:
    LOGGER.info("Streaming Java test command: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(project),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    deadline = time.monotonic() + timeout
    fd = proc.stdout.fileno()
    while proc.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            proc.kill()
            raise subprocess.TimeoutExpired(cmd, timeout, output="\n".join(lines))
        ready, _, _ = select.select([fd], [], [], min(1.0, remaining))
        if ready:
            line = proc.stdout.readline()
            if line:
                print(f"[java-test] {line}", end="", flush=True)
                lines.append(line.rstrip())
    for line in proc.stdout:
        print(f"[java-test] {line}", end="", flush=True)
        lines.append(line.rstrip())
    return proc.wait(), "\n".join(lines), ""


def _run_captured_command(cmd: list[str], project: Path, timeout: int) -> tuple[int, str, str]:
    result = subprocess.run(
        cmd,
        cwd=str(project),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


def _compute_counts(tests: list[TestResult]) -> TestCounts:
    by_status = Counter(test.status for test in tests)
    return TestCounts(
        total=len(tests),
        passed=by_status["PASS"],
        failed=by_status["FAIL"],
        errors=by_status["ERROR"],
        skipped=by_status["SKIPPED"],
        duration=sum(test.duration for test in tests),
    )


def run_cmd_and_parse(
    cmd: list[str],
    project: Path,
    build_system: BuildSystem,
    timeout: int = 2,
) -> TestRunSummary:
    """Execute a test command and parse results."""
    runner = _run_streaming_command if os.environ.get("JAVA_TEST_STREAM_LOGS", "") else _run_captured_command
    try:
        exit_code, stdout, stderr = runner(cmd, project, timeout)
    except subprocess.TimeoutExpired as exc:
        output = exc.output if isinstance(exc.output, str) else ""
        return TestRunSummary(
            build_system=build_system,
            exit_code=-1,
            stdout=output,
            stderr=f"Test execution timed out after {timeout} seconds",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return TestRunSummary(
            build_system=build_system,
            exit_code=-1,
            stderr=f"Error running tests: {exc}",
        )

    tests = _collect_test_reports(project, build_system)
    return TestRunSummary(
        build_system=build_system,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        tests=tests,
        counts=_compute_counts(tests),
    )


def run_tests(
    project_path: str,
    build_system: BuildSystem,
    *,
    clean: bool = True,
    timeout: int = 2,
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


def _testcase_duration(testcase: ET.Element) -> float:
    try:
        return float(testcase.get("time", "0"))
    except ValueError:
        return 0.0


def _extract_test_result(testcase: ET.Element) -> TestResult:
    name = f"{testcase.get('classname', '')}.{testcase.get('name', '')}"
    duration = _testcase_duration(testcase)
    for tag, status in XML_TAG_TO_STATUS.items():
        element = testcase.find(tag)
        if element is not None:
            return TestResult(
                name=name,
                status=status,
                duration=duration,
                error_message=element.get("message") or None,
                error_type=element.get("type") or None,
                failure_trace=element.text,
            )
    return TestResult(name=name, status="PASS", duration=duration)


def _parse_test_xml_reports(report_dir: Path) -> list[TestResult]:
    """Parse JUnit-style XML test reports (used by both Maven Surefire and Gradle)."""
    if not report_dir.exists():
        return []

    results: list[TestResult] = []
    for xml_file in report_dir.glob("TEST-*.xml"):
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError as exc:
            LOGGER.warning("Skipping malformed XML file %s: %s", xml_file, exc)
            continue
        results.extend(_extract_test_result(testcase) for testcase in root.findall("testcase"))
    return results


def _parse_reports_from_dirs(report_dirs: list[Path]) -> list[TestResult]:
    seen: set[Path] = set()
    results: list[TestResult] = []
    for report_dir in report_dirs:
        resolved = report_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        results.extend(_parse_test_xml_reports(report_dir))
    return results


def _collect_test_reports(project_path: Path, build_system: BuildSystem) -> list[TestResult]:
    """Parse JUnit XML test reports, avoiding deep repository scans on common layouts."""
    relative_report_path = REPORT_PATHS[build_system]
    common_dirs: list[Path] = []
    direct = project_path / relative_report_path
    if direct.exists():
        common_dirs.append(direct)

    excluded_modules = {".git", ".gradle", ".mvn", "build", "target", "node_modules"}
    for child in sorted(project_path.iterdir()):
        if not child.is_dir() or child.name in excluded_modules:
            continue
        module_reports = child / relative_report_path
        if module_reports.exists():
            common_dirs.append(module_reports)

    results = _parse_reports_from_dirs(common_dirs)
    if results:
        return results

    deep_dirs = sorted(project_path.glob(f"**/{relative_report_path}"))
    return _parse_reports_from_dirs(deep_dirs)


def _parse_maven_results(project_path: Path) -> list[TestResult]:
    """Backward-compatible Maven report parser wrapper."""
    return _collect_test_reports(project_path, "maven")


def _parse_gradle_results(project_path: Path) -> list[TestResult]:
    """Backward-compatible Gradle report parser wrapper."""
    return _collect_test_reports(project_path, "gradle")


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
def run_java_tests(project_path: str, clean: bool = True) -> dict[str, object]:
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


def run_tests_if_present(project_path: str | Path, has_tests: bool) -> bool:
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


def test_summary_to_dict(summary: TestRunSummary) -> dict[str, object]:
    """Convert a TestRunSummary to the standard result dict (9 core fields)."""
    counts = asdict(summary.counts)
    counts["duration"] = round(summary.counts.duration, 2)
    return {
        "build_system": summary.build_system,
        "success": summary.success,
        "exit_code": summary.exit_code,
        **counts,
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

    summary = run_tests(project_path, build_system, clean=False, timeout=2)

    output = "=== Test Output ===\n\n"

    if summary.stdout:
        output += "STDOUT:\n" + summary.stdout[-2000:] + "\n\n"

    if summary.stderr:
        output += "STDERR:\n" + summary.stderr[-2000:] + "\n"

    return output


