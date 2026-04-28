"""Build and compilation utilities for Java projects."""

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

GRADLE_FALLBACK_COMMANDS = [
    "./gradlew clean build -x test -x checkstyleMain",
    "./gradlew clean build -x test -x spotlessJavaCheck",
    "./gradlew clean build -x test -x enforceRules",
    "./gradlew clean build -x test -x spotlessJava",
]


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
    """Compile Java project with auto-detected build system and Gradle fallbacks."""
    project_path = Path(project_path)
    _validate_project_path(project_path)
    compile_command = compile_command or _detect_compile_command(project_path)

    try:
        return _successful_compile(compile_command, project_path)
    except subprocess.CalledProcessError as error:
        if "gradlew" in compile_command:
            fallback_result = _try_gradle_fallbacks(project_path)
            if fallback_result is not None:
                return fallback_result

        LOGGER.error("Compilation failed with command: %s", compile_command)
        stdout = getattr(error, "stdout", "")
        stderr = getattr(error, "stderr", "")
        return CompileResult(
            success=False,
            command=compile_command,
            stdout=stdout,
            stderr=stderr,
            error_summary=_extract_error_summary(stdout + stderr),
        )


def _validate_project_path(project_path: Path) -> None:
    if not project_path.exists():
        raise ValueError(f"Project path does not exist: {project_path}")
    if not project_path.is_dir():
        raise ValueError(f"Project path is not a directory: {project_path}")


def _detect_compile_command(project_path: Path) -> str:
    if (project_path / "pom.xml").exists():
        return "mvn clean compile -DskipTests"
    if (project_path / "build.gradle").exists() or (project_path / "build.gradle.kts").exists():
        return "./gradlew clean build -x test"
    raise ValueError(
        f"No build system detected in {project_path} (no pom.xml or build.gradle)"
    )


def _successful_compile(command: str, project_path: Path) -> CompileResult:
    result = run_command(command, project_path)
    return CompileResult(
        success=True,
        command=command,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _try_gradle_fallbacks(project_path: Path) -> CompileResult | None:
    for fallback_command in GRADLE_FALLBACK_COMMANDS:
        LOGGER.info("Primary command failed, trying fallback: %s", fallback_command)
        try:
            return _successful_compile(fallback_command, project_path)
        except subprocess.CalledProcessError as fallback_error:
            LOGGER.warning(
                "Fallback command failed (exit code %d): %s",
                fallback_error.returncode,
                fallback_command,
            )
    return None


def _extract_error_summary(output: str) -> list[str]:
    """Extract [ERROR] lines from Maven/Gradle output."""
    # Remove ANSI escape codes
    ansi_escape = re.compile(r'\x1B\[[0-9;]*[a-zA-Z]')
    clean_output = ansi_escape.sub('', output)

    # Extract [ERROR] lines
    error_lines = re.findall(r'\[ERROR\].*', clean_output)

    return error_lines[:20]  # Limit to first 20 errors
