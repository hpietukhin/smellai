"""Tests for Java test analysis agent and tools."""

from contextlib import contextmanager
from types import SimpleNamespace

from agents.java_test import agent as java_test_agent
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


def test_maven_setup_mapping_is_limited_to_full_ready_repos(tmp_path):
    """Only curated full-ready eval repos get maven_setup.md commands."""
    philip = tmp_path / "philib"
    philip.mkdir()
    assert java_test_agent._maven_setup_command_for(str(philip)).command == "mvn test"

    lyra = tmp_path / "lyra"
    lyra.mkdir()
    lyra_command = java_test_agent._maven_setup_command_for(str(lyra)).command
    assert "ChannelRecoveryTest" not in lyra_command
    assert "RetryPolicyTest" in lyra_command

    ircbot = tmp_path / "ircbot"
    ircbot.mkdir()
    assert java_test_agent._maven_setup_command_for(str(ircbot)) is None


def test_run_java_test_analysis_uses_curated_command_for_ready_repo(tmp_path, monkeypatch):
    project = tmp_path / "philib"
    project.mkdir()
    (project / "pom.xml").write_text("<project></project>")

    captured: dict[str, list[object]] = {"cmd": [], "timeout": []}

    def fake_run_cmd_and_parse(cmd, project_path, build_system, timeout=300):
        captured["cmd"].append(cmd)
        captured["project_path"] = project_path
        captured["build_system"] = build_system
        captured["timeout"].append(timeout)
        return TestRunSummary(build_system="maven", exit_code=0)

    monkeypatch.setattr(java_test_agent, "run_cmd_and_parse", fake_run_cmd_and_parse)

    result = java_test_agent.run_java_test_analysis(str(project), timeout=123)

    assert result["command_source"] == "evals/maven_setup.md"
    assert result["command"] == "mvn test"
    assert len(captured["cmd"]) >= 1
    test_cmd = captured["cmd"][-1]
    assert test_cmd[:2] == ["bash", "-lc"]
    assert "sdk use java 8.0.442-amzn" in test_cmd[2]
    assert "sdk use maven 3.6.3" in test_cmd[2]
    test_command_line = test_cmd[2].splitlines()[-1]
    assert test_command_line.startswith(("mvn ", "mvnd "))
    assert " test" in test_command_line
    assert captured["build_system"] == "maven"
    assert captured["timeout"][-1] == 123


def test_run_java_test_analysis_prefers_targeted_tests_for_changed_test_file(tmp_path, monkeypatch):
    project = tmp_path / "philib"
    project.mkdir()
    (project / "pom.xml").write_text("<project></project>")
    changed = str(project / "src" / "test" / "java" / "com" / "example" / "FooTest.java")

    captured: dict[str, list[object]] = {"cmd": [], "timeout": []}

    def fake_run_cmd_and_parse(cmd, project_path, build_system, timeout=300):
        captured["cmd"].append(cmd)
        captured["project_path"] = project_path
        captured["build_system"] = build_system
        captured["timeout"].append(timeout)
        return TestRunSummary(build_system="maven", exit_code=0)

    monkeypatch.setattr(java_test_agent, "run_cmd_and_parse", fake_run_cmd_and_parse)

    result = java_test_agent.run_java_test_analysis(str(project), timeout=123, target_files=[changed])

    assert result["command_source"] == "targeted-maven-tests"
    assert result["command"] == "mvn clean -Dtest=com.example.FooTest test"
    assert len(captured["cmd"]) >= 1
    test_cmd = captured["cmd"][-1]
    assert test_cmd[:2] == ["bash", "-lc"]
    assert "-Dtest=com.example.FooTest" in test_cmd[2]
    assert "clean" in test_cmd[2] and " test" in test_cmd[2]
    assert captured["build_system"] == "maven"
    assert captured["timeout"][-1] == 123


def test_run_java_test_analysis_falls_back_for_non_ready_maven_repo(tmp_path, monkeypatch):
    project = tmp_path / "ircbot"
    project.mkdir()
    (project / "pom.xml").write_text("<project></project>")

    called = {}

    def fake_run_tests(project_path, build_system, *, clean=True, timeout=300):
        called["project_path"] = project_path
        called["build_system"] = build_system
        called["clean"] = clean
        called["timeout"] = timeout
        return TestRunSummary(build_system="maven", exit_code=0)

    monkeypatch.setattr(java_test_agent, "run_tests", fake_run_tests)

    result = java_test_agent.run_java_test_analysis(str(project), clean=False, timeout=77)

    assert result["command_source"] == "default"
    assert result["command"] is None
    assert called == {
        "project_path": str(project),
        "build_system": "maven",
        "clean": False,
        "timeout": 77,
    }


def test_run_java_test_analysis_uses_code_agent_repair(tmp_path, monkeypatch):
    project = tmp_path / "ircbot"
    project.mkdir()
    (project / "pom.xml").write_text("<project></project>")

    calls = {"run_tests": 0}

    def fake_run_tests(project_path, build_system, *, clean=True, timeout=300):
        calls["run_tests"] += 1
        if calls["run_tests"] == 1:
            return TestRunSummary(build_system="maven", exit_code=1, stderr="baseline failed")
        return TestRunSummary(build_system="maven", exit_code=0)

    def fake_code_agent_repair(
        project_path,
        summary,
        *,
        verification_command,
        model_name=None,
        step_limit=0,
        cost_limit=0.0,
        timeout=0,
        allowed_write_paths=None,
        compile_mode=False,
    ):
        calls["code_agent_project_path"] = project_path
        calls["code_agent_verification_command"] = verification_command
        calls["code_agent_model_name"] = model_name
        calls["code_agent_step_limit"] = step_limit
        calls["code_agent_cost_limit"] = cost_limit
        calls["code_agent_timeout"] = timeout
        calls["code_agent_allowed_write_paths"] = allowed_write_paths
        calls["compile_mode"] = compile_mode
        return {"attempted": True, "applied": True, "changed_files": ["pom.xml"]}
    monkeypatch.setattr(java_test_agent, "run_tests", fake_run_tests)
    monkeypatch.setattr(java_test_agent, "_code_agent_repair_checkout", fake_code_agent_repair)
    monkeypatch.setattr(
        java_test_agent,
        "run_cmd_and_parse",
        lambda *args, **kwargs: TestRunSummary(build_system="maven", exit_code=0),
    )

    result = java_test_agent.run_java_test_analysis(str(project), timeout=77)

    assert result["summary"].success is True
    assert result["code_agent_repair"]["attempted"] is True
    assert calls["run_tests"] == 2
    assert calls["code_agent_project_path"] == str(project)
    assert "sdk use java 8.0.442-amzn" in calls["code_agent_verification_command"]
    assert " clean test" in calls["code_agent_verification_command"]
    assert calls["code_agent_step_limit"] == 2
    assert calls["code_agent_cost_limit"] == 0.0
    assert calls["code_agent_timeout"] == 77
    assert calls.get("compile_mode") is True


def test_detect_maven_compiler_source_uses_help_evaluate(tmp_path, monkeypatch):
    project = tmp_path / "legacy"
    project.mkdir()
    (project / "pom.xml").write_text("<project></project>", encoding="utf-8")
    java_test_agent._MAVEN_COMPILER_SOURCE_CACHE.clear()
    captured = {}

    def fake_run(cmd, cwd, capture_output, text, timeout, check):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["timeout"] = timeout
        return SimpleNamespace(returncode=0, stdout="1.6\n", stderr="")

    monkeypatch.setattr(java_test_agent.subprocess, "run", fake_run)

    detected = java_test_agent._detect_maven_compiler_source(str(project), timeout=9)

    assert detected == "1.6"
    assert captured["cmd"][:2] == ["bash", "-lc"]
    assert "mvn -q help:evaluate -Dexpression=maven.compiler.source -DforceStdout" in captured["cmd"][2]
    assert captured["cwd"] == project.resolve()
    assert captured["timeout"] == 9


def test_detect_maven_compiler_source_falls_back_to_plugin_configuration(tmp_path, monkeypatch):
    project = tmp_path / "lyra"
    project.mkdir()
    (project / "pom.xml").write_text(
        """<project xmlns=\"http://maven.apache.org/POM/4.0.0\">
  <build>
    <plugins>
      <plugin>
        <artifactId>maven-compiler-plugin</artifactId>
        <configuration>
          <source>1.6</source>
          <target>1.6</target>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>
""",
        encoding="utf-8",
    )
    java_test_agent._MAVEN_COMPILER_SOURCE_CACHE.clear()

    def fake_run(cmd, cwd, capture_output, text, timeout, check):
        return SimpleNamespace(returncode=0, stdout="null object or invalid expression\n", stderr="")

    monkeypatch.setattr(java_test_agent.subprocess, "run", fake_run)

    detected = java_test_agent._detect_maven_compiler_source(str(project), timeout=9)

    assert detected == "1.6"


def test_repair_prompt_context_includes_detected_java_source(tmp_path, monkeypatch):
    project = tmp_path / "tap4j"
    project.mkdir()
    (project / "pom.xml").write_text("<project></project>", encoding="utf-8")
    monkeypatch.setattr(java_test_agent, "_detect_maven_compiler_source", lambda project_path: "1.6")

    context = java_test_agent._repair_prompt_context(str(project))

    assert "Determine Java version for project:" in context
    assert "maven.compiler.source=1.6" in context
    assert "Do not use Java language features newer than source level 1.6" in context


def test_run_java_test_analysis_records_mlflow_actions_for_targeted_run(tmp_path, monkeypatch):
    project = tmp_path / "philib"
    project.mkdir()
    (project / "pom.xml").write_text("<project></project>")
    changed_test = project / "src" / "test" / "java" / "com" / "example" / "FooTest.java"

    def fake_run_cmd_and_parse(cmd, project_path, build_system, timeout=300):
        assert build_system == "maven"
        assert timeout in {12, 120}
        return TestRunSummary(build_system="maven", exit_code=0)

    monkeypatch.setattr(java_test_agent, "run_cmd_and_parse", fake_run_cmd_and_parse)

    action_types: list[str] = []

    @contextmanager
    def fake_start_action(action_type, **kwargs):
        action_types.append(action_type)
        yield object()

    monkeypatch.setattr(java_test_agent, "start_action", fake_start_action)
    result = java_test_agent.run_java_test_analysis(
        str(project),
        timeout=12,
        clean=True,
        target_files=[str(changed_test)],
        enable_code_agent_repair=False,
    )

    assert result["summary"].success is True
    assert result["command_source"] == "targeted-maven-tests"

    assert "run_java_test_analysis" in action_types
    assert "detect_build_system" in action_types
    assert "resolve_targeted_maven_tests" in action_types
    assert "select_java_test_path" in action_types
    assert "java_test_run_once" in action_types
