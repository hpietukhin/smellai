"""JDK version switching utilities using jenv."""

import logging
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def switch_java_version(version: int, project_path: str | Path) -> bool:
    """Switch Java version using jenv local command.

    Args:
        version: Java version number (e.g., 11, 17)
        project_path: Path to project directory where .java-version will be set

    Returns:
        True if switch succeeded, False otherwise
    """
    project_path = Path(project_path)

    try:
        # Set local Java version for this project
        subprocess.run(
            ["jenv", "local", str(version)],
            cwd=str(project_path),
            check=True,
            capture_output=True,
            text=True,
        )

        # Verify switch
        result = subprocess.run(
            ["jenv", "version"],
            cwd=str(project_path),
            capture_output=True,
            text=True,
            check=True,
        )

        if str(version) in result.stdout:
            LOGGER.info("Successfully switched to Java %s in %s", version, project_path)
            return True

        LOGGER.warning(
            "Failed to verify Java %s switch. Current: %s",
            version,
            result.stdout.strip(),
        )
        return False

    except subprocess.CalledProcessError as e:
        LOGGER.error("Failed to switch Java version: %s", e)
        return False
    except FileNotFoundError:
        LOGGER.error(
            "jenv not found. Install jenv and ensure it's in PATH. "
            "See: https://github.com/jenv/jenv"
        )
        return False


def get_current_java_version(project_path: str | Path) -> str | None:
    """Get currently active Java version in project.

    Args:
        project_path: Path to project directory

    Returns:
        Java version string or None if unable to determine
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
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
