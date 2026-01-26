"""Build and compilation utilities for Java projects."""

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass
class CompileResult:
    """Result of compilation attempt."""

    success: bool
    command: str
    stdout: str
    stderr: str
    error_summary: list[str] | None = None


def run_command(
    command: str | list[str],
    cwd: str | Path,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Run shell command and capture output.

    Args:
        command: Command to run (string or list)
        cwd: Working directory
        timeout: Timeout in seconds

    Returns:
        CompletedProcess with result

    Raises:
        subprocess.TimeoutExpired: If command times out
        subprocess.CalledProcessError: If command fails (non-zero exit)
        FileNotFoundError: If command not found
    """
    cwd = Path(cwd)

    result = subprocess.run(
        command,
        shell=isinstance(command, str),
        text=True,
        capture_output=True,
        cwd=str(cwd),
        timeout=timeout,
        check=True,  # Raises CalledProcessError on non-zero exit
    )

    LOGGER.info("Command succeeded: %s", command)
    return result


def compile_project(
    project_path: str | Path,
    compile_command: str | None = None,
) -> CompileResult:
    """Compile Java project with Gradle fallback strategy.

    Attempts compilation using provided command or auto-detected build system.
    For Gradle, tries multiple fallback commands to skip common blockers
    (checkstyle, spotless, etc.).

    Args:
        project_path: Path to project root
        compile_command: Explicit compile command (e.g., "./gradlew clean build -x test")
                        If None, auto-detects Maven or Gradle

    Returns:
        CompileResult with success status and logs

    Raises:
        ValueError: If no build system detected
    """
    project_path = Path(project_path)

    # Determine compile command
    if compile_command is None:
        if (project_path / "pom.xml").exists():
            compile_command = "mvn clean compile -DskipTests"
        elif (project_path / "build.gradle").exists() or (project_path / "build.gradle.kts").exists():
            compile_command = "./gradlew clean build -x test"
        else:
            raise ValueError(f"No build system detected in {project_path} (no pom.xml or build.gradle)")

    # Try primary command
    try:
        result = run_command(compile_command, project_path)
        return CompileResult(
            success=True,
            command=compile_command,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.CalledProcessError as e:
        # For Gradle failures, try fallback commands (skip common blockers)
        if "gradlew" in compile_command:
            fallback_commands = [
                "./gradlew clean build -x test -x checkstyleMain",
                "./gradlew clean build -x test -x spotlessJavaCheck",
                "./gradlew clean build -x test -x enforceRules",
                "./gradlew clean build -x test -x spotlessJava",
            ]

            for fallback_cmd in fallback_commands:
                LOGGER.info("Primary command failed, trying fallback: %s", fallback_cmd)
                try:
                    fallback_result = run_command(fallback_cmd, project_path)
                    return CompileResult(
                        success=True,
                        command=fallback_cmd,
                        stdout=fallback_result.stdout,
                        stderr=fallback_result.stderr,
                    )
                except subprocess.CalledProcessError:
                    continue  # Try next fallback

        # All attempts failed - extract error summary
        LOGGER.error("Compilation failed with command: %s", compile_command)
        error_summary = _extract_error_summary(e.stdout + e.stderr)

        return CompileResult(
            success=False,
            command=compile_command,
            stdout=e.stdout,
            stderr=e.stderr,
            error_summary=error_summary,
        )


def _extract_error_summary(output: str) -> list[str]:
    """Extract [ERROR] lines from Maven/Gradle output.

    Args:
        output: Combined stdout/stderr from build

    Returns:
        List of error message lines
    """
    # Remove ANSI escape codes
    ansi_escape = re.compile(r'\x1B\[[0-9;]*[a-zA-Z]')
    clean_output = ansi_escape.sub('', output)

    # Extract [ERROR] lines
    error_lines = re.findall(r'\[ERROR\].*', clean_output)

    return error_lines[:20]  # Limit to first 20 errors
