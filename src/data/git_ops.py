"""
Git repository operations for code smell detection.

This module provides functions to interact with Git repositories, including:
- Deriving GitHub URLs from project names
- Finding commits before specific dates
- Sparse file checkout for efficient code retrieval
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from git import GitCommandError, Repo


def derive_repo_url(project_name: str) -> str:
    """
    Convert DACOS project name to GitHub repository URL.

    Project names in DACOS follow the format: "org_repo" where underscore
    separates the GitHub organization from the repository name.

    Args:
        project_name: Project name in format "org_repo" (e.g., "alibaba_arthas")

    Returns:
        GitHub repository URL (e.g., "https://github.com/alibaba/arthas")

    Raises:
        ValueError: If project_name doesn't contain an underscore

    Examples:
        >>> derive_repo_url("alibaba_arthas")
        'https://github.com/alibaba/arthas'
        >>> derive_repo_url("watabou_pixel-dungeon")
        'https://github.com/watabou/pixel-dungeon'
    """
    if "_" not in project_name:
        raise ValueError(
            f"Invalid project_name format: '{project_name}'. "
            "Expected format: 'org_repo' with underscore separator."
        )

    parts = project_name.split("_", 1)  # Split on first underscore only
    org = parts[0]
    repo = parts[1]

    return f"https://github.com/{org}/{repo}"


def get_commit_before_date(
    repo_url: str, before_date: str = "2023-01-24"
) -> Optional[str]:
    """
    Find the latest commit before a specific date.

    Clones the repository (shallow clone with depth 1000) and searches for
    the most recent commit before the given date. This is used to get the
    code version before the DACOS dataset cutoff date.

    Args:
        repo_url: GitHub repository URL
        before_date: Date string in format "YYYY-MM-DD" (default: "2023-01-24")

    Returns:
        Commit SHA string, or None if no commits found before date

    Raises:
        GitCommandError: If repository doesn't exist or git operations fail
        ValueError: If date format is invalid

    Examples:
        >>> get_commit_before_date("https://github.com/alibaba/arthas", "2023-01-24")
        'abc123def456...'  # Returns actual commit SHA
    """
    temp_dir = None

    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="dacos_git_")

        # Clone repository (shallow clone for efficiency)
        print(f"Cloning repository: {repo_url} (depth=1000)...")
        repo = Repo.clone_from(
            repo_url,
            temp_dir,
            depth=1000,  # Shallow clone for speed
            no_checkout=True,  # Don't checkout files, just get history
        )

        # Get commit before date using git log
        commits = list(repo.iter_commits(max_count=1, before=before_date))

        if not commits:
            print(f"⚠ No commits found before {before_date}")
            return None

        commit_sha = commits[0].hexsha
        commit_date = commits[0].committed_datetime
        print(
            f"✓ Found commit: {commit_sha[:8]} (date: {commit_date.strftime('%Y-%m-%d')})"
        )

        return commit_sha

    except GitCommandError as e:
        if "not found" in str(e).lower():
            raise GitCommandError(
                f"Repository not found: {repo_url}. Please check the URL.", 128
            ) from e
        raise GitCommandError(f"Git operation failed: {e}", 128) from e

    except Exception as e:
        raise ValueError(f"Failed to get commit before date: {e}") from e

    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def clone_and_read_file(repo_url: str, commit_sha: str, file_path: str) -> str:
    """
    Clone repository, checkout specific commit, and read a file.

    Uses sparse checkout to efficiently download only the required file.
    This is optimized for reading single files from large repositories.

    Args:
        repo_url: GitHub repository URL
        commit_sha: Git commit SHA to checkout
        file_path: Relative path to file in repository

    Returns:
        File content as string

    Raises:
        GitCommandError: If git operations fail
        FileNotFoundError: If file doesn't exist at the specified commit
        UnicodeDecodeError: If file is not valid text

    Examples:
        >>> content = clone_and_read_file(
        ...     "https://github.com/alibaba/arthas",
        ...     "abc123def456",
        ...     "src/main/java/com/example/Test.java"
        ... )
        >>> print(content[:50])
        'package com.example;\\n\\npublic class Test {...'
    """
    temp_dir = None

    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="dacos_sparse_")

        print(f"Sparse cloning: {repo_url}")
        print(f"  Commit: {commit_sha[:8]}")
        print(f"  File: {file_path}")

        # Initialize empty git repo
        repo = Repo.init(temp_dir)

        # Add remote
        origin = repo.create_remote("origin", repo_url)

        # Enable sparse checkout
        with repo.config_writer() as config:
            config.set_value("core", "sparseCheckout", True)

        # Specify which file to checkout
        sparse_checkout_file = Path(temp_dir) / ".git" / "info" / "sparse-checkout"
        sparse_checkout_file.parent.mkdir(parents=True, exist_ok=True)
        sparse_checkout_file.write_text(file_path + "\n")

        # Fetch the specific commit
        origin.fetch(commit_sha, depth=1)

        # Checkout the commit
        repo.git.checkout(commit_sha)

        # Read the file
        file_full_path = Path(temp_dir) / file_path

        if not file_full_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path} in commit {commit_sha[:8]}. "
                f"Please verify the file path is correct."
            )

        # Read file content
        try:
            content = file_full_path.read_text(encoding="utf-8")
            print(f"✓ File read successfully ({len(content)} characters)")
            return content

        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ["latin-1", "cp1252"]:
                try:
                    content = file_full_path.read_text(encoding=encoding)
                    print(
                        f"✓ File read successfully with {encoding} encoding ({len(content)} characters)"
                    )
                    return content
                except UnicodeDecodeError:
                    continue

            raise UnicodeDecodeError(
                "utf-8",
                b"",
                0,
                0,
                f"Failed to decode file: {file_path}. File may be binary.",
            )

    except GitCommandError as e:
        if "pathspec" in str(e).lower() or "did not match" in str(e).lower():
            raise FileNotFoundError(
                f"File or commit not found: {file_path} at {commit_sha[:8]}"
            ) from e
        raise GitCommandError(
            f"Git operation failed while cloning {repo_url}: {e}", 128
        ) from e

    except Exception as e:
        raise RuntimeError(f"Failed to clone and read file: {e}") from e

    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
