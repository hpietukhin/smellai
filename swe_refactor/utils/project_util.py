"""Project manipulation utilities (git, file operations)."""

import logging
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def clone_repository(
    repo_url: str,
    target_dir: str | Path,
    *,
    shallow: bool = False,
) -> None:
    """Clone git repository to target directory.

    Args:
        repo_url: Git repository URL
        target_dir: Destination directory
        shallow: If True, performs shallow clone (--depth 1)

    Raises:
        subprocess.CalledProcessError: If git clone fails
        FileExistsError: If target_dir already exists (fail-fast, no silent skip)
    """
    target_dir = Path(target_dir)

    if target_dir.exists():
        raise FileExistsError(f"Target directory already exists: {target_dir}")

    cmd = ["git", "clone"]
    if shallow:
        cmd.extend(["--depth", "1"])
    cmd.extend([repo_url, str(target_dir)])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        LOGGER.info("Cloned %s to %s", repo_url, target_dir)
    except subprocess.CalledProcessError as e:
        LOGGER.error("Failed to clone repository: %s", e.stderr)
        raise


def force_checkout_commit(
    project_path: str | Path,
    commit_id: str,
) -> None:
    """Force checkout to specific commit, discarding local changes.

    Args:
        project_path: Path to git repository
        commit_id: Commit hash or ref to checkout

    Raises:
        ValueError: If project_path is not a git repository
        subprocess.CalledProcessError: If git commands fail
    """
    project_path = Path(project_path)

    if not (project_path / ".git").exists():
        raise ValueError(f"Not a git repository: {project_path}")

    try:
        # Reset to HEAD
        subprocess.run(
            ["git", "reset", "--hard", "HEAD"],
            cwd=str(project_path),
            check=True,
            capture_output=True,
            text=True,
        )

        # Checkout commit
        subprocess.run(
            ["git", "checkout", "-f", commit_id],
            cwd=str(project_path),
            check=True,
            capture_output=True,
            text=True,
        )

        LOGGER.info("Checked out commit %s in %s", commit_id, project_path)
    except subprocess.CalledProcessError as e:
        LOGGER.error("Failed to checkout commit %s: %s", commit_id, e.stderr)
        raise


def get_previous_commit(
    project_path: str | Path,
    commit_id: str,
) -> str:
    """Get parent commit hash.

    Args:
        project_path: Path to git repository
        commit_id: Commit hash

    Returns:
        Parent commit hash

    Raises:
        ValueError: If project_path is not a git repository
        subprocess.CalledProcessError: If git command fails (e.g., commit has no parent)
    """
    project_path = Path(project_path)

    if not (project_path / ".git").exists():
        raise ValueError(f"Not a git repository: {project_path}")

    try:
        result = subprocess.run(
            ["git", "rev-parse", f"{commit_id}~1"],
            cwd=str(project_path),
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        LOGGER.error("Failed to get previous commit for %s: %s", commit_id, e.stderr)
        raise


def replace_java_code(
    file_path: str | Path,
    new_code: str,
) -> None:
    """Replace content of Java file with new code.

    Args:
        file_path: Path to Java file
        new_code: New file content

    Raises:
        ValueError: If file_path does not have .java extension
        FileNotFoundError: If parent directory does not exist
        OSError: If file write fails
    """
    file_path = Path(file_path)

    # Validate Java file
    if file_path.suffix != ".java":
        raise ValueError(f"File is not a Java file: {file_path}")

    # Ensure parent directory exists
    if not file_path.parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {file_path.parent}")

    try:
        file_path.write_text(new_code, encoding="utf-8")
        LOGGER.info("Replaced code in %s", file_path)
    except Exception as e:
        LOGGER.error("Failed to write file %s: %s", file_path, e)
        raise
