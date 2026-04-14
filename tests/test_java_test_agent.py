"""Tests for Java test analysis agent and tools."""

from agents.tools.java_test_tools import (
    detect_build_system,
    TestCounts,
    TestResult,
    TestRunSummary,
)


def test_detect_maven_project(tmp_path):
    """Test Maven project detection."""
    # Create fake pom.xml
    (tmp_path / "pom.xml").write_text("<project></project>")

    result = detect_build_system(str(tmp_path))
    assert result == "maven"


def test_detect_gradle_project(tmp_path):
    """Test Gradle project detection."""
    # Create fake build.gradle
    (tmp_path / "build.gradle").write_text("plugins { id 'java' }")

    result = detect_build_system(str(tmp_path))
    assert result == "gradle"


def test_detect_gradle_kts_project(tmp_path):
    """Test Gradle Kotlin DSL project detection."""
    # Create fake build.gradle.kts
    (tmp_path / "build.gradle.kts").write_text("plugins { java }")

    result = detect_build_system(str(tmp_path))
    assert result == "gradle"


def test_detect_no_build_system(tmp_path):
    """Test no build system detected."""
    result = detect_build_system(str(tmp_path))
    assert result is None


def test_maven_priority_over_gradle(tmp_path):
    """Test Maven takes priority when both exist."""
    (tmp_path / "pom.xml").write_text("<project></project>")
    (tmp_path / "build.gradle").write_text("plugins { id 'java' }")

    result = detect_build_system(str(tmp_path))
    assert result == "maven"


def test_test_counts_defaults():
    """TestCounts base class should have zero defaults."""
    counts = TestCounts()
    assert counts.total == 0
    assert counts.passed == 0
    assert counts.failed == 0
    assert counts.errors == 0
    assert counts.skipped == 0
    assert counts.duration == 0.0


def test_test_run_summary_has_counts():
    """TestRunSummary should contain a TestCounts via composition."""
    summary = TestRunSummary(build_system="maven")
    assert isinstance(summary.counts, TestCounts)
    assert summary.counts.total == 0
    assert summary.counts.duration == 0.0


def test_test_result_dataclass():
    """Test TestResult dataclass."""
    result = TestResult(
        name="com.example.TestClass.testMethod",
        status="FAIL",
        duration=1.5,
        error_message="Expected 5 but got 3",
        error_type="AssertionError",
    )

    assert result.name == "com.example.TestClass.testMethod"
    assert result.status == "FAIL"
    assert result.duration == 1.5
    assert result.error_message == "Expected 5 but got 3"


def test_test_run_summary_success():
    """Test TestRunSummary success property."""
    summary = TestRunSummary(
        build_system="maven",
        exit_code=0,
        counts=TestCounts(total=10, passed=10, failed=0, errors=0),
    )

    assert summary.success is True


def test_test_run_summary_failure():
    """Test TestRunSummary failure detection."""
    summary = TestRunSummary(
        build_system="maven",
        exit_code=1,
        counts=TestCounts(total=10, passed=8, failed=2, errors=0),
    )

    assert summary.success is False


def test_ysoserial_detection(tmp_path):
    """Test detection with real ysoserial cloned repo."""
    from repo_utils import clone_repository

    repo_url = "https://github.com/frohoff/ysoserial.git"

    # Clone into tmp_path
    # clone_repository creates a subdirectory with the repo name in the target_dir
    repo_name = clone_repository(repo_url, target_dir=tmp_path)
    repo_path = tmp_path / repo_name

    # Verify detection
    result = detect_build_system(str(repo_path))
    assert result == "maven"


def test_test_run_summary_with_errors():
    """Test TestRunSummary with errors."""
    summary = TestRunSummary(
        build_system="gradle",
        exit_code=1,
        counts=TestCounts(total=5, passed=4, failed=0, errors=1),
    )

    assert summary.success is False
