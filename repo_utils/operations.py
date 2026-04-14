"""Repository operation utilities for transparent source code handling.

This module provides utilities for cloning repositories, checking out commits,
and finding project roots. Designed for integration with evaluation datasets
like SWE-Refactor and RefactoringMiner.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import git

logger = logging.getLogger(__name__)


class RepositoryError(Exception):
    """Exception raised for repository operation errors."""
    pass


def clone_repository(
    repo_url: str,
    target_dir: str | Path,
    *,
    shallow: bool = False,
    branch: str | None = None,
    pull_if_exists: bool = False,
) -> git.Repo:
    """Clone a Git repository.

    Args:
        repo_url: Git repository URL (HTTPS or SSH)
        target_dir: Directory to clone into
        shallow: Whether to perform shallow clone (depth=1)
        branch: Specific branch to clone (None = default branch)
        pull_if_exists: If True, pull latest changes when repo already exists;
                        if False, return existing repo as-is

    Returns:
        git.Repo object

    Raises:
        RepositoryError: If cloning fails
    """
    target_path = Path(target_dir)

    if target_path.exists():
        logger.warning(f"Target directory already exists: {target_path}")
        try:
            repo = git.Repo(target_path)
        except git.InvalidGitRepositoryError:
            raise RepositoryError(
                f"Directory exists but is not a valid Git repository: {target_path}"
            )
        if pull_if_exists:
            logger.info(f"Pulling latest changes in {target_path}")
            repo.remotes.origin.pull()
        return repo

    try:
        logger.info(f"Cloning repository {repo_url} to {target_path}")

        kwargs = {}
        if shallow:
            kwargs["depth"] = 1
        if branch:
            kwargs["branch"] = branch

        repo = git.Repo.clone_from(repo_url, target_path, **kwargs)
        logger.info(f"Successfully cloned repository to {target_path}")
        return repo

    except git.GitCommandError as e:
        raise RepositoryError(f"Failed to clone repository: {e}")


def checkout_commit(
    repo_path: str | Path,
    commit_sha: str,
    *,
    force: bool = False,
) -> git.Repo:
    """Checkout a specific commit in a repository.

    Args:
        repo_path: Path to Git repository
        commit_sha: Commit SHA to checkout
        force: Whether to force checkout (discard local changes)

    Returns:
        git.Repo object

    Raises:
        RepositoryError: If checkout fails
    """
    repo_path = Path(repo_path)

    try:
        repo = git.Repo(repo_path)
    except git.InvalidGitRepositoryError:
        raise RepositoryError(f"Not a valid Git repository: {repo_path}")

    try:
        if force:
            # Reset to clean state first
            repo.git.reset("--hard")
            logger.info("Repository reset to clean state")

        # Checkout the specified commit
        repo.git.checkout(commit_sha, force=force)
        logger.info(f"Checked out commit: {commit_sha}")
        return repo

    except git.GitCommandError as e:
        raise RepositoryError(f"Failed to checkout commit {commit_sha}: {e}")


def get_previous_commit(
    repo_path: str | Path,
    commit_sha: str,
) -> str:
    """Get the parent commit SHA of a given commit.

    Args:
        repo_path: Path to Git repository
        commit_sha: Commit SHA

    Returns:
        Parent commit SHA

    Raises:
        RepositoryError: If commit has no parent or operation fails
    """
    repo_path = Path(repo_path)

    try:
        repo = git.Repo(repo_path)
        commit = repo.commit(commit_sha)

        if not commit.parents:
            raise RepositoryError(f"Commit {commit_sha} has no parents (initial commit)")

        # Return first parent (handles merge commits by taking first parent)
        return commit.parents[0].hexsha

    except git.GitCommandError as e:
        raise RepositoryError(f"Failed to get previous commit for {commit_sha}: {e}")


def find_project_root(
    repo_path: str | Path,
    build_system: Literal["maven", "gradle"] | None = None,
) -> Path | None:
    """Find the root directory of a Java project.

    Searches for pom.xml (Maven) or build.gradle/build.gradle.kts (Gradle)
    in the repository. If build_system is specified, only looks for that type.

    Args:
        repo_path: Path to repository
        build_system: Specific build system to look for (None = any)

    Returns:
        Path to project root, or None if not found
    """
    repo_path = Path(repo_path)

    if not repo_path.exists():
        logger.warning(f"Repository path does not exist: {repo_path}")
        return None

    # Define search patterns based on build system
    if build_system == "maven":
        patterns = ["pom.xml"]
    elif build_system == "gradle":
        patterns = ["build.gradle", "build.gradle.kts"]
    else:
        # Search for any build system
        patterns = ["pom.xml", "build.gradle", "build.gradle.kts"]

    # Search in repo directory and subdirectories
    for pattern in patterns:
        matches = list(repo_path.rglob(pattern))
        if matches:
            # Return the first match (could be enhanced to handle multiple)
            project_root = matches[0].parent
            logger.info(f"Found project root at {project_root}")
            return project_root

    logger.warning(f"No project root found in {repo_path}")
    return None


def clone_and_checkout(
    repo_url: str,
    commit_sha: str,
    target_dir: str | Path,
    *,
    shallow: bool = False,
    force_checkout: bool = True,
) -> tuple[git.Repo, Path | None]:
    """Clone repository and checkout specific commit in one operation.

    Convenience function that combines cloning and checkout. Also attempts
    to find the project root.

    Args:
        repo_url: Git repository URL
        commit_sha: Commit SHA to checkout
        target_dir: Directory to clone into
        shallow: Whether to perform shallow clone
        force_checkout: Whether to force checkout

    Returns:
        Tuple of (git.Repo object, project root path or None)

    Raises:
        RepositoryError: If operation fails
    """
    # Clone repository
    repo = clone_repository(repo_url, target_dir, shallow=shallow)

    # Checkout commit
    checkout_commit(target_dir, commit_sha, force=force_checkout)

    # Find project root
    project_root = find_project_root(target_dir)

    return repo, project_root
