"""JDK version switching utilities using jenv."""

import logging
import re
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _run_jenv(args: list[str], project_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["jenv", *args],
        cwd=str(project_path),
        check=True,
        capture_output=True,
        text=True,
    )


def switch_java_version(version: int, project_path: str | Path) -> bool:
    """Switch Java version using jenv local command.

    Returns True on success and False when jenv is unavailable or the switch
    fails. Invalid project_path still raises ValueError.
    """
    project_path = Path(project_path)

    if not project_path.is_dir():
        raise ValueError(
            f"Project path does not exist or is not a directory: {project_path}"
        )

    try:
        _run_jenv(["local", str(version)], project_path)
        result = _run_jenv(["version"], project_path)
    except FileNotFoundError:
        LOGGER.error("jenv not found in PATH")
        return False
    except subprocess.CalledProcessError as e:
        LOGGER.error("Failed to set Java version: %s", e.stderr)
        return False

    if not re.search(rf'\b{version}\b', result.stdout):
        LOGGER.error(
            "Failed to verify Java %s switch. Current: %s",
            version,
            result.stdout.strip(),
        )
        return False

    LOGGER.info("Successfully switched to Java %s in %s", version, project_path)
    return True


def get_current_java_version(project_path: str | Path) -> str:
    """Get currently active Java version via jenv."""
    try:
        result = _run_jenv(["version"], Path(project_path))
        return result.stdout.strip()
    except FileNotFoundError as e:
        LOGGER.error("jenv not found in PATH")
        raise FileNotFoundError(
            "jenv not installed. Install jenv and ensure it's in PATH. "
            "See: https://github.com/jenv/jenv"
        ) from e
    except subprocess.CalledProcessError as e:
        LOGGER.error("jenv version command failed: %s", e.stderr)
        raise
