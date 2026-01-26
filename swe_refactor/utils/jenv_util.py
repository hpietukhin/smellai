"""JDK version switching utilities using jenv."""

import logging
import re
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def switch_java_version(version: int, project_path: str | Path) -> None:
    """Switch Java version using jenv local command.

    Args:
        version: Java version number (e.g., 11, 17)
        project_path: Path to project directory where .java-version will be set

    Raises:
        ValueError: If project_path does not exist or is not a directory
        FileNotFoundError: If jenv not installed or not in PATH
        subprocess.CalledProcessError: If jenv command failed
        RuntimeError: If version switch verification failed
    """
    project_path = Path(project_path)

    # Validate input
    if not project_path.is_dir():
        raise ValueError(f"Project path does not exist or is not a directory: {project_path}")

    # Set local Java version
    try:
        subprocess.run(
            ["jenv", "local", str(version)],
            cwd=str(project_path),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        LOGGER.error("jenv not found in PATH")
        raise FileNotFoundError(
            "jenv not installed. Install jenv and ensure it's in PATH. "
            "See: https://github.com/jenv/jenv"
        ) from e
    except subprocess.CalledProcessError as e:
        LOGGER.error("Failed to set Java version: %s", e.stderr)
        raise

    # Verify switch
    result = subprocess.run(
        ["jenv", "version"],
        cwd=str(project_path),
        capture_output=True,
        text=True,
        check=True,
    )

    # Use word boundary regex for robust verification
    if not re.search(rf'\b{version}\b', result.stdout):
        error_msg = f"Failed to verify Java {version} switch. Current: {result.stdout.strip()}"
        LOGGER.error(error_msg)
        raise RuntimeError(error_msg)

    LOGGER.info("Successfully switched to Java %s in %s", version, project_path)


def get_current_java_version(project_path: str | Path) -> str:
    """Get currently active Java version in project.

    Args:
        project_path: Path to project directory

    Returns:
        Java version string from jenv

    Raises:
        FileNotFoundError: If jenv not installed or not in PATH
        subprocess.CalledProcessError: If jenv command failed
    """
    try:
        result = subprocess.run(
            ["jenv", "version"],
            cwd=str(project_path),
            capture_output=True,
            text=True,
            check=True,
        )
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
