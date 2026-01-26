"""Utilities for SWE-Refactor integration."""

from .repos import get_repo_url, PROJECT_REPOS
from .jenv_util import switch_java_version, get_current_java_version
from .build_util import compile_project, run_command, CompileResult
from .project_util import (
    clone_repository,
    force_checkout_commit,
    get_previous_commit,
    replace_java_code,
)

__all__ = [
    "get_repo_url",
    "PROJECT_REPOS",
    "switch_java_version",
    "get_current_java_version",
    "compile_project",
    "run_command",
    "CompileResult",
    "clone_repository",
    "force_checkout_commit",
    "get_previous_commit",
    "replace_java_code",
]
