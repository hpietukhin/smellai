"""Project manipulation utilities (git, file operations)."""

import logging
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _run_git(args: list[str], project_path: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(project_path) if project_path else None,
        check=True,
        capture_output=True,
        text=True,
    )


def clone_repository(
    repo_url: str,
    target_dir: str | Path,
    *,
    shallow: bool = False,
) -> bool:
    """Clone git repository to target directory.

    Returns True on success and False on operational failures so callers in the
    current agent workflow can branch without wrapping every filesystem/git
    operation in try/except. Invalid input still raises ValueError.
    """
    if not repo_url or not repo_url.strip():
        raise ValueError("Repository URL cannot be empty")

    target_dir = Path(target_dir)

    if target_dir.exists():
        LOGGER.info("Target directory already exists: %s", target_dir)
        return True

    clone_args = ["clone"]
    if shallow:
        clone_args.extend(["--depth", "1"])
    clone_args.extend([repo_url, str(target_dir)])

    try:
        _run_git(clone_args)
        LOGGER.info("Cloned %s to %s", repo_url, target_dir)
        return True
    except subprocess.CalledProcessError as e:
        LOGGER.error("Failed to clone repository: %s", e.stderr)
        return False


def force_checkout_commit(
    project_path: str | Path,
    commit_id: str,
) -> bool:
    """Force checkout to specific commit, discarding local changes.

    Returns True on success and False on git failures. Invalid input still
    raises ValueError.
    """
    if not commit_id or not commit_id.strip():
        raise ValueError("Commit ID cannot be empty")

    project_path = Path(project_path)

    if not (project_path / ".git").exists():
        raise ValueError(f"Not a git repository: {project_path}")

    try:
        _run_git(["reset", "--hard", "HEAD"], project_path)
        _run_git(["checkout", "-f", commit_id], project_path)

        LOGGER.info("Checked out commit %s in %s", commit_id, project_path)
        return True
    except subprocess.CalledProcessError as e:
        LOGGER.error("Failed to checkout commit %s: %s", commit_id, e.stderr)
        return False


def get_previous_commit(
    project_path: str | Path,
    commit_id: str,
) -> str | None:
    """Get parent commit hash, or None on operational failure."""
    if not commit_id or not commit_id.strip():
        raise ValueError("Commit ID cannot be empty")

    project_path = Path(project_path)

    if not (project_path / ".git").exists():
        raise ValueError(f"Not a git repository: {project_path}")

    try:
        return _run_git(["rev-parse", f"{commit_id}~1"], project_path).stdout.strip()
    except subprocess.CalledProcessError as e:
        LOGGER.error("Failed to get previous commit for %s: %s", commit_id, e.stderr)
        return None


def replace_java_code(
    file_path: str | Path,
    new_code: str,
) -> bool:
    """Replace content of a .java file. Returns False on I/O errors."""
    file_path = Path(file_path)

    if file_path.suffix != ".java":
        raise ValueError(f"File is not a Java file: {file_path}")

    if not file_path.parent.exists():
        LOGGER.error("Parent directory does not exist: %s", file_path.parent)
        return False

    if not file_path.exists():
        LOGGER.error("File does not exist: %s", file_path)
        return False

    try:
        file_path.write_text(new_code, encoding="utf-8")
        LOGGER.info("Replaced code in %s", file_path)
        return True
    except OSError as e:
        LOGGER.error("Failed to write file %s: %s", file_path, e)
        return False
