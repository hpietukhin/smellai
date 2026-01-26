# SWE-Refactor Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate SWE-Refactor benchmark (1,099 pure Java refactorings) into SmellAI multi-agent system with Phase 1 baseline replication.

**Architecture:** Reuse existing SmellAI components (java_test_tools, mlflow_utils, LangGraph patterns) + port minimal utilities from original SWE-Refactor code (jenv, git, build commands) + create LangGraph agent for A0 (setup) → A5 (LLM generate) → A6 (verify) workflow.

**Tech Stack:** LangGraph, MLflow, LiteLLM, jenv (JDK switching), Maven/Gradle (via existing java_test_tools), RefactoringMiner (verification)

---

## Phase 1: Foundation - Utility Layer

### Task 1: Port project repository mapping

**Files:**
- Create: `swe_refactor/utils/__init__.py`
- Create: `swe_refactor/utils/repos.py`
- Reference: `/tmp/SWE-Refactor/code/repos.txt`

**Step 1: Write repos.py with project-to-URL mapping**

```python
"""Project repository URL mappings for SWE-Refactor dataset."""

# Mapping from project name to GitHub repository URL
PROJECT_REPOS = {
    "checkstyle": "https://github.com/checkstyle/checkstyle.git",
    "commons-lang": "https://github.com/apache/commons-lang.git",
    "commons-io": "https://github.com/apache/commons-io.git",
    "hibernate-orm": "https://github.com/hibernate/hibernate-orm.git",
    "hibernate-search": "https://github.com/hibernate/hibernate-search.git",
    "javaparser": "https://github.com/javaparser/javaparser.git",
    "junit4": "https://github.com/junit-team/junit4.git",
    "junit5": "https://github.com/junit-team/junit5.git",
    "mockito": "https://github.com/mockito/mockito.git",
    "pmd": "https://github.com/pmd/pmd.git",
}


def get_repo_url(project_name: str) -> str:
    """Get repository URL for project name.

    Args:
        project_name: Name of the project (e.g., "checkstyle")

    Returns:
        GitHub clone URL

    Raises:
        KeyError: If project name not found
    """
    return PROJECT_REPOS[project_name]
```

**Step 2: Write __init__.py to export repos module**

```python
"""Utilities for SWE-Refactor integration."""

from .repos import get_repo_url, PROJECT_REPOS

__all__ = ["get_repo_url", "PROJECT_REPOS"]
```

**Step 3: Commit**

```bash
git add swe_refactor/utils/__init__.py swe_refactor/utils/repos.py
git commit -m "feat(swe-refactor): add project repository mappings"
```

---

### Task 2: Port JDK version switching utility

**Files:**
- Create: `swe_refactor/utils/jenv_util.py`
- Reference: `/tmp/SWE-Refactor/code/compile_experiment.py:333-358` (switch_java_version)

**Step 1: Write jenv_util.py with switch_java_version**

```python
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
```

**Step 2: Add to __init__.py**

```python
from .repos import get_repo_url, PROJECT_REPOS
from .jenv_util import switch_java_version, get_current_java_version

__all__ = [
    "get_repo_url",
    "PROJECT_REPOS",
    "switch_java_version",
    "get_current_java_version",
]
```

**Step 3: Commit**

```bash
git add swe_refactor/utils/jenv_util.py swe_refactor/utils/__init__.py
git commit -m "feat(swe-refactor): add jenv JDK switching utility"
```

---

### Task 3: Port build and compilation utilities

**Files:**
- Create: `swe_refactor/utils/build_util.py`
- Reference: `/tmp/SWE-Refactor/code/compile_experiment.py:14-71` (run_command, compile_project)

**Step 1: Write build_util.py with compile functions**

```python
"""Build and compilation utilities for Java projects."""

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
) -> tuple[bool, subprocess.CompletedProcess]:
    """Run shell command and capture output.

    Args:
        command: Command to run (string or list)
        cwd: Working directory
        timeout: Timeout in seconds

    Returns:
        Tuple of (success: bool, result: CompletedProcess)
    """
    cwd = Path(cwd)

    try:
        result = subprocess.run(
            command,
            shell=isinstance(command, str),
            text=True,
            capture_output=True,
            cwd=str(cwd),
            timeout=timeout,
        )
        success = result.returncode == 0

        if success:
            LOGGER.info("Command succeeded: %s", command)
        else:
            LOGGER.warning("Command failed (code %d): %s", result.returncode, command)

        return success, result

    except subprocess.TimeoutExpired as e:
        LOGGER.error("Command timed out after %ds: %s", timeout, command)
        return False, subprocess.CompletedProcess(
            args=command,
            returncode=-1,
            stdout="",
            stderr=f"Timeout after {timeout}s",
        )


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
    """
    project_path = Path(project_path)

    # Determine compile command
    if compile_command is None:
        if (project_path / "pom.xml").exists():
            compile_command = "mvn clean compile -DskipTests"
        elif (project_path / "build.gradle").exists() or (project_path / "build.gradle.kts").exists():
            compile_command = "./gradlew clean build -x test"
        else:
            return CompileResult(
                success=False,
                command="",
                stdout="",
                stderr="No build system detected (no pom.xml or build.gradle)",
            )

    # Try primary command
    success, result = run_command(compile_command, project_path)

    if success:
        return CompileResult(
            success=True,
            command=compile_command,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    # For Gradle failures, try fallback commands (skip common blockers)
    if "gradlew" in compile_command:
        fallback_commands = [
            "./gradlew clean build -x test -x checkstyleMain",
            "./gradlew clean build -x test -x spotlessJavaCheck",
            "./gradlew clean build -x test -x enforceRules",
            "./gradlew clean build -x test -x spotlessJava",
        ]

        for fallback_cmd in fallback_commands:
            LOGGER.info("Trying fallback: %s", fallback_cmd)
            success, fallback_result = run_command(fallback_cmd, project_path)

            if success:
                return CompileResult(
                    success=True,
                    command=fallback_cmd,
                    stdout=fallback_result.stdout,
                    stderr=fallback_result.stderr,
                )

    # All attempts failed - extract error summary
    error_summary = _extract_error_summary(result.stdout + result.stderr)

    return CompileResult(
        success=False,
        command=compile_command,
        stdout=result.stdout,
        stderr=result.stderr,
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
```

**Step 2: Add to __init__.py**

```python
from .repos import get_repo_url, PROJECT_REPOS
from .jenv_util import switch_java_version, get_current_java_version
from .build_util import compile_project, run_command, CompileResult

__all__ = [
    "get_repo_url",
    "PROJECT_REPOS",
    "switch_java_version",
    "get_current_java_version",
    "compile_project",
    "run_command",
    "CompileResult",
]
```

**Step 3: Commit**

```bash
git add swe_refactor/utils/build_util.py swe_refactor/utils/__init__.py
git commit -m "feat(swe-refactor): add build compilation utilities with Gradle fallbacks"
```

---

### Task 4: Port git and project manipulation utilities

**Files:**
- Create: `swe_refactor/utils/project_util.py`
- Reference: `/tmp/SWE-Refactor/code/compile_experiment.py:24-88` (checkout, replace_java_code, get_previous_commit)

**Step 1: Write project_util.py with git operations**

```python
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
) -> bool:
    """Clone git repository to target directory.

    Args:
        repo_url: Git repository URL
        target_dir: Destination directory
        shallow: If True, performs shallow clone (--depth 1)

    Returns:
        True if clone succeeded, False otherwise
    """
    target_dir = Path(target_dir)

    if target_dir.exists():
        LOGGER.warning("Target directory already exists: %s", target_dir)
        return True

    cmd = ["git", "clone"]
    if shallow:
        cmd.extend(["--depth", "1"])
    cmd.extend([repo_url, str(target_dir)])

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        LOGGER.info("Cloned %s to %s", repo_url, target_dir)
        return True
    except subprocess.CalledProcessError as e:
        LOGGER.error("Failed to clone repository: %s", e)
        return False


def force_checkout_commit(
    project_path: str | Path,
    commit_id: str,
) -> bool:
    """Force checkout to specific commit, discarding local changes.

    Args:
        project_path: Path to git repository
        commit_id: Commit hash or ref to checkout

    Returns:
        True if checkout succeeded, False otherwise
    """
    project_path = Path(project_path)

    try:
        # Reset to HEAD
        subprocess.run(
            ["git", "reset", "--hard", "HEAD"],
            cwd=str(project_path),
            check=True,
            capture_output=True,
        )

        # Checkout commit
        subprocess.run(
            ["git", "checkout", "-f", commit_id],
            cwd=str(project_path),
            check=True,
            capture_output=True,
        )

        LOGGER.info("Checked out commit %s in %s", commit_id, project_path)
        return True

    except subprocess.CalledProcessError as e:
        LOGGER.error("Failed to checkout commit %s: %s", commit_id, e)
        return False


def get_previous_commit(
    project_path: str | Path,
    commit_id: str,
) -> str | None:
    """Get parent commit hash.

    Args:
        project_path: Path to git repository
        commit_id: Commit hash

    Returns:
        Parent commit hash or None if failed
    """
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
        LOGGER.error("Failed to get previous commit for %s: %s", commit_id, e)
        return None


def replace_java_code(
    file_path: str | Path,
    new_code: str,
) -> bool:
    """Replace content of Java file with new code.

    Args:
        file_path: Path to Java file
        new_code: New file content

    Returns:
        True if replacement succeeded, False otherwise
    """
    file_path = Path(file_path)

    try:
        file_path.write_text(new_code, encoding="utf-8")
        LOGGER.info("Replaced code in %s", file_path)
        return True

    except Exception as e:
        LOGGER.error("Failed to replace code in %s: %s", file_path, e)
        return False
```

**Step 2: Add to __init__.py**

```python
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
```

**Step 3: Commit**

```bash
git add swe_refactor/utils/project_util.py swe_refactor/utils/__init__.py
git commit -m "feat(swe-refactor): add git and file manipulation utilities"
```

---

## Phase 2: Dataset Layer

### Task 5: Create Pydantic models for SWE-Refactor dataset

**Files:**
- Create: `swe_refactor/dataset.py`
- Reference: `/tmp/SWE-Refactor/pure_refactoring_data.json` (structure)

**Step 1: Write dataset.py with Pydantic models**

```python
"""Dataset models and loader for SWE-Refactor benchmark."""

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)


class RefactoringRecord(BaseModel):
    """Single refactoring record from SWE-Refactor dataset."""

    # Identifiers
    projectName: str = Field(description="Project name (e.g., 'checkstyle')")
    commitId: str = Field(description="Full commit hash")
    type: Literal[
        "Extract Method",
        "Move Method",
        "Inline Method",
        "Extract And Move Method",
        "Move And Rename Method",
        "Move And Inline Method",
    ] = Field(description="Refactoring type")

    # File paths
    filePathBefore: str = Field(description="Source file path before refactoring")
    filePathAfter: str = Field(description="Target file path after refactoring")

    # Source code
    sourceCodeBeforeForWhole: str = Field(description="Full source file before")
    sourceCodeAfterForWhole: str = Field(description="Full source file after")

    # Build configuration
    compileJDK: int = Field(description="JDK version for compilation (e.g., 11)")
    compileCommand: str = Field(description="Build command to use")
    hasTestC: bool = Field(description="Whether commit has test coverage")

    # Optional metadata (may not be in all records)
    description: str | None = None
    packageNameBefore: str | None = None
    classNameBefore: str | None = None
    methodNameBefore: str | None = None


class SWERefactorDataset:
    """Loader for SWE-Refactor dataset."""

    def __init__(self, dataset_path: str | Path):
        """Initialize dataset loader.

        Args:
            dataset_path: Path to pure_refactoring_data.json
        """
        self.dataset_path = Path(dataset_path)
        self.records: list[RefactoringRecord] = []

    def load(self) -> list[RefactoringRecord]:
        """Load dataset from JSON file.

        Returns:
            List of RefactoringRecord objects

        Raises:
            FileNotFoundError: If dataset file not found
            ValueError: If JSON parsing fails
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        with open(self.dataset_path) as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Expected JSON array of refactoring records")

        self.records = [RefactoringRecord(**record) for record in data]

        LOGGER.info("Loaded %d refactoring records from %s", len(self.records), self.dataset_path)

        return self.records

    def filter_by_type(
        self,
        refactoring_type: str,
    ) -> list[RefactoringRecord]:
        """Filter records by refactoring type.

        Args:
            refactoring_type: Type to filter (e.g., "Extract Method")

        Returns:
            Filtered list of records
        """
        return [r for r in self.records if r.type == refactoring_type]

    def filter_by_project(
        self,
        project_name: str,
    ) -> list[RefactoringRecord]:
        """Filter records by project name.

        Args:
            project_name: Project to filter (e.g., "checkstyle")

        Returns:
            Filtered list of records
        """
        return [r for r in self.records if r.projectName == project_name]

    def get_commit_records(
        self,
        commit_id: str,
    ) -> list[RefactoringRecord]:
        """Get all refactorings in a specific commit.

        Args:
            commit_id: Commit hash (can be short or full)

        Returns:
            List of records from this commit
        """
        return [r for r in self.records if r.commitId.startswith(commit_id)]


def load_swe_refactor_dataset(
    dataset_path: str | Path = "/tmp/SWE-Refactor/pure_refactoring_data.json",
) -> list[RefactoringRecord]:
    """Convenience function to load SWE-Refactor dataset.

    Args:
        dataset_path: Path to dataset JSON file

    Returns:
        List of RefactoringRecord objects
    """
    dataset = SWERefactorDataset(dataset_path)
    return dataset.load()
```

**Step 2: Commit**

```bash
git add swe_refactor/dataset.py
git commit -m "feat(swe-refactor): add Pydantic models and dataset loader"
```

---

## Phase 3: LangGraph Agent

### Task 6: Create agent configuration

**Files:**
- Create: `agents/swe_eval/__init__.py`
- Create: `agents/swe_eval/config.py`
- Reference: `agents/rminer_eval/config.py` (pattern)

**Step 1: Write config.py with agent settings**

```python
"""Configuration for SWE evaluation agent."""

from enum import Enum


class SWEEvalAgentConfig(str, Enum):
    """Configuration keys for SWE evaluation agent."""

    MODEL_NAME = "model_name"
    WORKSPACE_DIR = "workspace_dir"
    MAX_RETRIES = "max_retries"
    COMPILE_TIMEOUT = "compile_timeout"
    TEST_TIMEOUT = "test_timeout"


DEFAULT_CONFIG = {
    SWEEvalAgentConfig.MODEL_NAME: "claude-sonnet-4-5-20250929",
    SWEEvalAgentConfig.WORKSPACE_DIR: "/tmp/swe-eval-workspace",
    SWEEvalAgentConfig.MAX_RETRIES: 3,
    SWEEvalAgentConfig.COMPILE_TIMEOUT: 600,  # 10 minutes
    SWEEvalAgentConfig.TEST_TIMEOUT: 600,  # 10 minutes
}
```

**Step 2: Write __init__.py to export agent components**

```python
"""LangGraph agent for SWE-Refactor evaluation."""

from .config import SWEEvalAgentConfig, DEFAULT_CONFIG

__all__ = [
    "SWEEvalAgentConfig",
    "DEFAULT_CONFIG",
]
```

**Step 3: Commit**

```bash
git add agents/swe_eval/__init__.py agents/swe_eval/config.py
git commit -m "feat(swe-eval): add agent configuration"
```

---

### Task 7: Create refactoring prompt templates

**Files:**
- Create: `agents/swe_eval/prompts.py`

**Step 1: Write prompts.py with type-specific templates**

```python
"""Prompt templates for refactoring generation by type."""

from typing import Literal


SYSTEM_PROMPT = """You are an expert Java refactoring assistant.

Your task is to apply a specific refactoring to Java code while preserving behavior.

Key requirements:
- Output ONLY valid Java code (the complete refactored file)
- Preserve all functionality and behavior
- Follow Java naming conventions
- Maintain proper indentation and formatting
- Do NOT add comments explaining the refactoring
- Do NOT add extra features or "improvements"
"""


def get_refactoring_prompt(
    refactoring_type: Literal[
        "Extract Method",
        "Move Method",
        "Inline Method",
        "Extract And Move Method",
        "Move And Rename Method",
        "Move And Inline Method",
    ],
    source_code_before: str,
    target_code_before: str | None,
    file_path_before: str,
    file_path_after: str,
) -> str:
    """Generate refactoring prompt for specific type.

    Args:
        refactoring_type: Type of refactoring to perform
        source_code_before: Full source file content
        target_code_before: Full target file content (for Move refactorings)
        file_path_before: Source file path
        file_path_after: Target file path

    Returns:
        Formatted prompt string
    """
    if refactoring_type == "Extract Method":
        return _extract_method_prompt(source_code_before, file_path_before)

    elif refactoring_type == "Inline Method":
        return _inline_method_prompt(source_code_before, file_path_before)

    elif refactoring_type == "Move Method":
        return _move_method_prompt(
            source_code_before,
            target_code_before,
            file_path_before,
            file_path_after,
        )

    elif refactoring_type == "Extract And Move Method":
        return _extract_and_move_prompt(
            source_code_before,
            target_code_before,
            file_path_before,
            file_path_after,
        )

    elif refactoring_type in ("Move And Rename Method", "Move And Inline Method"):
        return _move_method_prompt(
            source_code_before,
            target_code_before,
            file_path_before,
            file_path_after,
        )

    raise ValueError(f"Unknown refactoring type: {refactoring_type}")


def _extract_method_prompt(source_code: str, file_path: str) -> str:
    """Prompt for Extract Method refactoring."""
    return f"""# Extract Method Refactoring

File: {file_path}

Analyze the code below and apply Extract Method refactoring to improve code organization.
Identify code that should be extracted into a separate method and perform the extraction.

Return the complete refactored file content.

## Source Code

```java
{source_code}
```

Output the refactored code:
"""


def _inline_method_prompt(source_code: str, file_path: str) -> str:
    """Prompt for Inline Method refactoring."""
    return f"""# Inline Method Refactoring

File: {file_path}

Analyze the code below and apply Inline Method refactoring.
Identify a method that should be inlined into its callers and perform the inlining.

Return the complete refactored file content.

## Source Code

```java
{source_code}
```

Output the refactored code:
"""


def _move_method_prompt(
    source_code: str,
    target_code: str | None,
    source_path: str,
    target_path: str,
) -> str:
    """Prompt for Move Method refactoring."""
    target_section = ""
    if target_code:
        target_section = f"""
## Target File (Before)

File: {target_path}

```java
{target_code}
```
"""

    return f"""# Move Method Refactoring

Move a method from the source class to the target class.

## Source File (Before)

File: {source_path}

```java
{source_code}
```
{target_section}

Identify which method should be moved from {source_path} to {target_path}.
Return TWO complete refactored files in this format:

// FILE: {source_path}
```java
<refactored source code>
```

// FILE: {target_path}
```java
<refactored target code>
```
"""


def _extract_and_move_prompt(
    source_code: str,
    target_code: str | None,
    source_path: str,
    target_path: str,
) -> str:
    """Prompt for Extract And Move Method refactoring."""
    target_section = ""
    if target_code:
        target_section = f"""
## Target File (Before)

File: {target_path}

```java
{target_code}
```
"""

    return f"""# Extract And Move Method Refactoring

Extract code into a new method AND move it to the target class (compound refactoring).

## Source File (Before)

File: {source_path}

```java
{source_code}
```
{target_section}

Identify code to extract into a method, then move that method to {target_path}.
Return TWO complete refactored files in this format:

// FILE: {source_path}
```java
<refactored source code>
```

// FILE: {target_path}
```java
<refactored target code>
```
"""
```

**Step 2: Add to agents/swe_eval/__init__.py**

```python
from .config import SWEEvalAgentConfig, DEFAULT_CONFIG
from .prompts import get_refactoring_prompt, SYSTEM_PROMPT

__all__ = [
    "SWEEvalAgentConfig",
    "DEFAULT_CONFIG",
    "get_refactoring_prompt",
    "SYSTEM_PROMPT",
]
```

**Step 3: Commit**

```bash
git add agents/swe_eval/prompts.py agents/swe_eval/__init__.py
git commit -m "feat(swe-eval): add refactoring prompt templates by type"
```

---

### Task 8: Create LangGraph agent with A0/A5/A6 workflow

**Files:**
- Create: `agents/swe_eval/agent.py`
- Reference: `agents/rminer_eval/agent.py` (LangGraph pattern)
- Reuse: `agents/tools/java_test_tools.py` (test execution)

**Step 1: Write agent.py state and nodes**

```python
"""LangGraph agent for SWE-Refactor evaluation workflow.

Workflow: A0 (setup) → A5 (generate) → A6 (verify)
"""

import logging
import re
from pathlib import Path
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from swe_refactor.dataset import RefactoringRecord
from swe_refactor.utils import (
    clone_repository,
    compile_project,
    force_checkout_commit,
    get_previous_commit,
    get_repo_url,
    replace_java_code,
    switch_java_version,
)
from agents.tools.java_test_tools import detect_build_system, run_tests
from agents.swe_eval.config import DEFAULT_CONFIG, SWEEvalAgentConfig
from agents.swe_eval.prompts import SYSTEM_PROMPT, get_refactoring_prompt

LOGGER = logging.getLogger(__name__)


class SWEEvalState(dict):
    """State for SWE evaluation agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    record: RefactoringRecord
    workspace_path: Path
    project_path: Path
    refactored_code: str | None
    refactored_target_code: str | None  # For Move operations
    compile_success: bool
    test_success: bool
    retry_count: int
    error_message: str | None


def create_swe_eval_agent(model_name: str | None = None) -> StateGraph:
    """Create LangGraph agent for SWE-Refactor evaluation.

    Args:
        model_name: LLM model to use. If None, uses default from config.

    Returns:
        Compiled LangGraph StateGraph
    """
    if model_name is None:
        model_name = DEFAULT_CONFIG[SWEEvalAgentConfig.MODEL_NAME]

    model = ChatLiteLLM(model=model_name)

    # Agent nodes

    def a0_setup(state: SWEEvalState) -> dict:
        """A0: Setup - Clone repo, checkout parent commit."""
        record = state["record"]
        workspace_path = state["workspace_path"]

        LOGGER.info("A0: Setting up workspace for %s @ %s", record.projectName, record.commitId[:8])

        # Clone repository
        repo_url = get_repo_url(record.projectName)
        project_path = workspace_path / record.projectName

        if not project_path.exists():
            success = clone_repository(repo_url, project_path)
            if not success:
                return {
                    "error_message": f"Failed to clone {repo_url}",
                    "project_path": project_path,
                }

        # Checkout parent commit
        parent_commit = get_previous_commit(project_path, record.commitId)
        if not parent_commit:
            return {
                "error_message": f"Failed to get parent of {record.commitId}",
                "project_path": project_path,
            }

        success = force_checkout_commit(project_path, parent_commit)
        if not success:
            return {
                "error_message": f"Failed to checkout {parent_commit}",
                "project_path": project_path,
            }

        # Switch to correct JDK
        switch_java_version(record.compileJDK, project_path)

        LOGGER.info("A0: Setup complete. Project at %s, commit %s", project_path, parent_commit[:8])

        return {
            "project_path": project_path,
            "error_message": None,
        }

    def a5_generate(state: SWEEvalState) -> dict:
        """A5: Generate - LLM generates refactored code."""
        record = state["record"]

        LOGGER.info("A5: Generating refactoring (%s)", record.type)

        # Prepare prompt
        target_code = None
        if record.filePathBefore != record.filePathAfter:
            # Move refactoring - need target file
            target_code = record.sourceCodeAfterForWhole  # Placeholder

        prompt = get_refactoring_prompt(
            record.type,
            record.sourceCodeBeforeForWhole,
            target_code,
            record.filePathBefore,
            record.filePathAfter,
        )

        # Add retry context if this is a retry
        if state.get("retry_count", 0) > 0:
            error_msg = state.get("error_message", "")
            prompt += f"\n\n## Previous Attempt Failed\n\nCompilation error:\n{error_msg}\n\nPlease fix and try again."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = model.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Extract code from response
        refactored_code, refactored_target = _extract_code_from_response(
            response_text,
            record.filePathBefore,
            record.filePathAfter,
        )

        if not refactored_code:
            return {
                "error_message": "Failed to extract code from LLM response",
                "refactored_code": None,
            }

        LOGGER.info("A5: Generated %d chars of refactored code", len(refactored_code))

        return {
            "refactored_code": refactored_code,
            "refactored_target_code": refactored_target,
            "messages": [response],
        }

    def a6_verify(state: SWEEvalState) -> dict:
        """A6: Verify - Compile and test refactored code."""
        record = state["record"]
        project_path = state["project_path"]
        refactored_code = state["refactored_code"]

        LOGGER.info("A6: Verifying refactored code")

        # Apply refactored code to file
        source_file = project_path / record.filePathBefore
        success = replace_java_code(source_file, refactored_code)

        if not success:
            return {
                "error_message": f"Failed to write {source_file}",
                "compile_success": False,
                "test_success": False,
            }

        # For Move operations, also update target file
        if state.get("refactored_target_code"):
            target_file = project_path / record.filePathAfter
            replace_java_code(target_file, state["refactored_target_code"])

        # Compile
        compile_result = compile_project(project_path, record.compileCommand)

        if not compile_result.success:
            error_summary = "\n".join(compile_result.error_summary or ["Unknown compile error"])
            LOGGER.warning("A6: Compilation failed:\n%s", error_summary)
            return {
                "compile_success": False,
                "test_success": False,
                "error_message": error_summary,
            }

        LOGGER.info("A6: Compilation succeeded")

        # Run tests (if commit has test coverage)
        test_success = True
        if record.hasTestC:
            build_system = detect_build_system(str(project_path))
            if build_system:
                test_result = run_tests(str(project_path), build_system)
                test_success = test_result.success

                if not test_success:
                    LOGGER.warning("A6: Tests failed (%d failures)", test_result.failed)
                else:
                    LOGGER.info("A6: Tests passed (%d tests)", test_result.total)

        return {
            "compile_success": True,
            "test_success": test_success,
            "error_message": None,
        }

    def should_retry(state: SWEEvalState) -> Literal["retry", "end"]:
        """Decide whether to retry generation after failure."""
        max_retries = DEFAULT_CONFIG[SWEEvalAgentConfig.MAX_RETRIES]

        if state.get("compile_success", False):
            return "end"

        if state.get("retry_count", 0) >= max_retries:
            LOGGER.warning("Max retries (%d) reached", max_retries)
            return "end"

        return "retry"

    # Build graph
    workflow = StateGraph(SWEEvalState)

    workflow.add_node("a0_setup", a0_setup)
    workflow.add_node("a5_generate", a5_generate)
    workflow.add_node("a6_verify", a6_verify)

    workflow.set_entry_point("a0_setup")
    workflow.add_edge("a0_setup", "a5_generate")
    workflow.add_edge("a5_generate", "a6_verify")

    workflow.add_conditional_edges(
        "a6_verify",
        should_retry,
        {
            "retry": "a5_generate",
            "end": END,
        },
    )

    return workflow.compile()


def _extract_code_from_response(
    response: str,
    source_file: str,
    target_file: str,
) -> tuple[str | None, str | None]:
    """Extract Java code from LLM response.

    Handles both single-file and multi-file responses.

    Args:
        response: LLM response text
        source_file: Expected source file path
        target_file: Expected target file path

    Returns:
        Tuple of (source_code, target_code). target_code is None for single-file refactorings.
    """
    # Check for multi-file format (Move operations)
    if f"// FILE: {source_file}" in response and f"// FILE: {target_file}" in response:
        return _extract_multi_file(response, source_file, target_file)

    # Single-file format
    pattern = r'```java\s*(.*?)\s*```'
    matches = re.findall(pattern, response, re.DOTALL)

    if not matches:
        LOGGER.warning("No Java code block found in response")
        return None, None

    # Return first code block
    return matches[0].strip(), None


def _extract_multi_file(
    response: str,
    source_file: str,
    target_file: str,
) -> tuple[str | None, str | None]:
    """Extract source and target code from multi-file response."""
    # Split by file markers
    source_pattern = rf'// FILE: {re.escape(source_file)}\s*```java\s*(.*?)\s*```'
    target_pattern = rf'// FILE: {re.escape(target_file)}\s*```java\s*(.*?)\s*```'

    source_match = re.search(source_pattern, response, re.DOTALL)
    target_match = re.search(target_pattern, response, re.DOTALL)

    source_code = source_match.group(1).strip() if source_match else None
    target_code = target_match.group(1).strip() if target_match else None

    return source_code, target_code


def invoke_agent(
    agent: StateGraph,
    record: RefactoringRecord,
    workspace_path: str | Path,
) -> dict:
    """Invoke agent for single refactoring record.

    Args:
        agent: Compiled LangGraph agent
        record: Refactoring record to process
        workspace_path: Base workspace directory

    Returns:
        Dictionary with evaluation results
    """
    workspace_path = Path(workspace_path)
    workspace_path.mkdir(parents=True, exist_ok=True)

    result = agent.invoke({
        "messages": [],
        "record": record,
        "workspace_path": workspace_path,
        "project_path": None,
        "refactored_code": None,
        "refactored_target_code": None,
        "compile_success": False,
        "test_success": False,
        "retry_count": 0,
        "error_message": None,
    })

    return {
        "project": record.projectName,
        "commit": record.commitId,
        "type": record.type,
        "compile_success": result.get("compile_success", False),
        "test_success": result.get("test_success", False),
        "error": result.get("error_message"),
    }
```

**Step 2: Update agents/swe_eval/__init__.py**

```python
from .config import SWEEvalAgentConfig, DEFAULT_CONFIG
from .prompts import get_refactoring_prompt, SYSTEM_PROMPT
from .agent import create_swe_eval_agent, invoke_agent

__all__ = [
    "SWEEvalAgentConfig",
    "DEFAULT_CONFIG",
    "get_refactoring_prompt",
    "SYSTEM_PROMPT",
    "create_swe_eval_agent",
    "invoke_agent",
]
```

**Step 3: Commit**

```bash
git add agents/swe_eval/agent.py agents/swe_eval/__init__.py
git commit -m "feat(swe-eval): add LangGraph agent with A0/A5/A6 workflow"
```

---

## Phase 4: MLflow Integration

### Task 9: Create MLflow dataset factory

**Files:**
- Create: `mlflow_utils/datasets/swe_factory.py`
- Reference: `mlflow_utils/datasets/rminer_factory.py` (pattern)

**Step 1: Write swe_factory.py to create MLflow dataset**

```python
"""MLflow dataset factory for SWE-Refactor benchmark."""

import logging
from pathlib import Path

from mlflow.genai.datasets import create_dataset

from swe_refactor.dataset import load_swe_refactor_dataset

LOGGER = logging.getLogger(__name__)


def create_swe_refactor_dataset(
    dataset_path: str | Path,
    name: str = "swe-refactor-dataset",
    *,
    limit: int | None = None,
    refactoring_type: str | None = None,
    project_name: str | None = None,
) -> str:
    """Create MLflow GenAI dataset from SWE-Refactor JSON.

    Args:
        dataset_path: Path to pure_refactoring_data.json
        name: Dataset name in MLflow
        limit: Limit number of records (for testing)
        refactoring_type: Filter by refactoring type (e.g., "Extract Method")
        project_name: Filter by project (e.g., "checkstyle")

    Returns:
        Dataset ID
    """
    records = load_swe_refactor_dataset(dataset_path)

    # Apply filters
    if refactoring_type:
        records = [r for r in records if r.type == refactoring_type]
        LOGGER.info("Filtered to %d records of type '%s'", len(records), refactoring_type)

    if project_name:
        records = [r for r in records if r.projectName == project_name]
        LOGGER.info("Filtered to %d records from project '%s'", len(records), project_name)

    if limit:
        records = records[:limit]
        LOGGER.info("Limited to %d records", limit)

    # Convert to MLflow GenAI format
    genai_records = [
        {
            "inputs": {
                "project_name": r.projectName,
                "commit_id": r.commitId,
                "type": r.type,
                "source_code_before": r.sourceCodeBeforeForWhole,
                "file_path_before": r.filePathBefore,
                "file_path_after": r.filePathAfter,
            },
            "outputs": {
                "source_code_after": r.sourceCodeAfterForWhole,
            },
            "metadata": {
                "compile_jdk": r.compileJDK,
                "compile_command": r.compileCommand,
                "has_test_coverage": r.hasTestC,
            },
        }
        for r in records
    ]

    dataset = create_dataset(
        name=name,
        description=f"SWE-Refactor benchmark: {len(genai_records)} pure Java refactorings",
        records=genai_records,
    )

    LOGGER.info("Created MLflow dataset '%s' (ID: %s) with %d records", name, dataset.dataset_id, len(genai_records))

    return dataset.dataset_id
```

**Step 2: Commit**

```bash
git add mlflow_utils/datasets/swe_factory.py
git commit -m "feat(mlflow): add SWE-Refactor dataset factory"
```

---

### Task 10: Create workflow script with MLflow evaluation

**Files:**
- Create: `workflows/swe_eval_workflow.py`
- Reference: `workflows/rminer_eval_workflow.py` (CLI pattern)
- Reuse: `mlflow_utils.setup_mlflow_tracking`

**Step 1: Write swe_eval_workflow.py**

```python
#!/usr/bin/env python3
"""MLflow GenAI evaluation workflow for SWE-Refactor agent.

Evaluates the agent's ability to generate correct refactorings.

Scorers:
- compile_success_rate: fraction of generated code that compiles
- test_pass_rate: fraction of compilable code that passes tests
- overall_success_rate: fraction that both compiles and passes tests

Usage:
    # Evaluate single commit
    uv run workflows/swe_eval_workflow.py --commit 65655da4 --project checkstyle

    # Evaluate using dataset
    uv run workflows/swe_eval_workflow.py --dataset /tmp/SWE-Refactor/pure_refactoring_data.json --limit 10

    # Use different model
    uv run workflows/swe_eval_workflow.py --dataset <path> --model gpt-4o

    # Draw agent graph
    uv run workflows/swe_eval_workflow.py --draw-graph
"""

import argparse
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from agents.swe_eval import create_swe_eval_agent, invoke_agent
from swe_refactor.dataset import load_swe_refactor_dataset
from mlflow_utils import setup_mlflow_tracking

load_dotenv()


def compile_success_scorer(outputs: dict, inputs: dict) -> float:
    """Score: 1.0 if compilation succeeded, 0.0 otherwise."""
    return 1.0 if outputs.get("compile_success", False) else 0.0


def test_pass_scorer(outputs: dict, inputs: dict) -> float:
    """Score: 1.0 if tests passed, 0.0 otherwise (NA if no compilation)."""
    if not outputs.get("compile_success", False):
        return 0.0
    return 1.0 if outputs.get("test_success", False) else 0.0


def overall_success_scorer(outputs: dict, inputs: dict) -> float:
    """Score: 1.0 if both compile and tests pass, 0.0 otherwise."""
    compile_ok = outputs.get("compile_success", False)
    test_ok = outputs.get("test_success", False)
    return 1.0 if (compile_ok and test_ok) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate SWE-Refactor agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        help="Path to pure_refactoring_data.json",
        default="/tmp/SWE-Refactor/pure_refactoring_data.json",
    )
    parser.add_argument("--commit", help="Specific commit to evaluate")
    parser.add_argument("--project", help="Project name (with --commit)")
    parser.add_argument("--experiment", default="swe-refactor-evaluation")
    parser.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking URI",
    )
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929")
    parser.add_argument("--limit", type=int, help="Limit number of records")
    parser.add_argument(
        "--workspace",
        default="/tmp/swe-eval-workspace",
        help="Workspace directory for cloned repos",
    )
    parser.add_argument(
        "--draw-graph",
        action="store_true",
        help="Draw agent graph to PNG",
    )
    args = parser.parse_args()

    if args.draw_graph:
        print("Generating agent graph...")
        agent = create_swe_eval_agent(model_name=args.model)
        try:
            png_bytes = agent.get_graph().draw_mermaid_png()
            output_path = "swe_eval_agent_graph.png"
            with open(output_path, "wb") as f:
                f.write(png_bytes)
            print(f"Graph saved to {output_path}")
        except Exception as e:
            print(f"Failed to draw graph: {e}")
        return 0

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    records = load_swe_refactor_dataset(dataset_path)

    # Filter by commit if specified
    if args.commit:
        if not args.project:
            print("--project required with --commit", file=sys.stderr)
            return 1
        records = [
            r
            for r in records
            if r.commitId.startswith(args.commit) and r.projectName == args.project
        ]
        print(f"Filtered to {len(records)} records from commit {args.commit}")

    if args.limit:
        records = records[: args.limit]

    if not records:
        print("No records to evaluate", file=sys.stderr)
        return 1

    # Setup MLflow
    setup_mlflow_tracking(
        tracking_uri=args.tracking_uri,
        backend_uri="sqlite:///mlflow.db",
        experiment_name=args.experiment,
        auto_start_server=True,
    )

    print(f"Model: {args.model}")
    print(f"Records: {len(records)}")
    print(f"Workspace: {args.workspace}")

    # Create agent
    print("Creating agent...")
    agent = create_swe_eval_agent(model_name=args.model)

    # Convert to MLflow GenAI format
    genai_records = [
        {
            "inputs": {
                "project_name": r.projectName,
                "commit_id": r.commitId,
                "type": r.type,
            },
            "outputs": {},
            "metadata": {"record": r.model_dump()},
        }
        for r in records
    ]

    def predict_fn(
        project_name: str,
        commit_id: str,
        type: str,
        **metadata,
    ) -> dict:
        """Prediction function for MLflow evaluation."""
        record_dict = metadata.get("record", {})
        from swe_refactor.dataset import RefactoringRecord

        record = RefactoringRecord(**record_dict)
        return invoke_agent(agent, record, args.workspace)

    print(f"Running evaluation on {len(genai_records)} records...")

    results = mlflow.genai.evaluate(
        data=genai_records,
        predict_fn=predict_fn,
        scorers=[
            compile_success_scorer,
            test_pass_scorer,
            overall_success_scorer,
        ],
    )

    run_id = results.run_id

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for metric_name, metric_value in results.metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}: {metric_value}")

    print("=" * 60)
    print(f"MLflow run ID: {run_id}")

    if run_id != "N/A" and args.tracking_uri.startswith("http://"):
        exp_id = getattr(results, "experiment_id", "?")
        print(f"View results: {args.tracking_uri}/#/experiments/{exp_id}/runs/{run_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Step 2: Make executable**

```bash
chmod +x workflows/swe_eval_workflow.py
```

**Step 3: Commit**

```bash
git add workflows/swe_eval_workflow.py
git commit -m "feat(swe-eval): add MLflow evaluation workflow with CLI"
```

---

## Phase 5: Testing & Validation

### Task 11: Create dataset creation script

**Files:**
- Create: `scripts/create_swe_dataset.py`
- Reference: `rminer/create_rminer_dataset.py` (pattern)

**Step 1: Write create_swe_dataset.py**

```python
#!/usr/bin/env python3
"""Create MLflow GenAI dataset from SWE-Refactor JSON.

Usage:
    uv run scripts/create_swe_dataset.py --name swe-refactor-full
    uv run scripts/create_swe_dataset.py --name checkstyle-only --project checkstyle
    uv run scripts/create_swe_dataset.py --name extract-method --type "Extract Method" --limit 100
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from mlflow_utils import setup_mlflow_tracking
from mlflow_utils.datasets.swe_factory import create_swe_refactor_dataset

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create SWE-Refactor MLflow dataset")
    parser.add_argument(
        "--dataset",
        default="/tmp/SWE-Refactor/pure_refactoring_data.json",
        help="Path to pure_refactoring_data.json",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Dataset name in MLflow",
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--type",
        help="Filter by refactoring type (e.g., 'Extract Method')",
    )
    parser.add_argument(
        "--project",
        help="Filter by project name (e.g., 'checkstyle')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of records",
    )

    args = parser.parse_args()

    # Check dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    # Setup MLflow
    setup_mlflow_tracking(
        tracking_uri=args.tracking_uri,
        backend_uri="sqlite:///mlflow.db",
        auto_start_server=True,
    )

    # Create dataset
    print(f"Creating dataset '{args.name}' from {dataset_path}...")

    dataset_id = create_swe_refactor_dataset(
        dataset_path=dataset_path,
        name=args.name,
        limit=args.limit,
        refactoring_type=args.type,
        project_name=args.project,
    )

    print(f"\nDataset created successfully!")
    print(f"  Name: {args.name}")
    print(f"  ID: {dataset_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Step 2: Make executable and commit**

```bash
chmod +x scripts/create_swe_dataset.py
git add scripts/create_swe_dataset.py
git commit -m "feat(swe-eval): add dataset creation script"
```

---

### Task 12: Manual validation with single commit

**Files:**
- No new files
- Test with: checkstyle/65655da4 (4 Move Method refactorings, JDK 11)

**Step 1: Ensure jenv is configured**

Run: `jenv versions`
Expected: List of installed JDK versions including 11

If JDK 11 not listed:
```bash
# Find JDK 11 path
/usr/libexec/java_home -V

# Add to jenv
jenv add /Library/Java/JavaVirtualMachines/jdk-11.jdk/Contents/Home
```

**Step 2: Run evaluation on single commit**

Run:
```bash
uv run workflows/swe_eval_workflow.py \
  --commit 65655da4 \
  --project checkstyle \
  --model claude-sonnet-4-5-20250929 \
  --workspace /tmp/swe-eval-test
```

Expected output:
```
Model: claude-sonnet-4-5-20250929
Records: 4
Workspace: /tmp/swe-eval-test
Creating agent...
Running evaluation on 4 records...
...
============================================================
EVALUATION RESULTS
============================================================
compile_success_scorer: <score>
test_pass_scorer: <score>
overall_success_scorer: <score>
============================================================
```

**Step 3: Verify workspace structure**

Run: `ls -la /tmp/swe-eval-test/checkstyle`

Expected: Cloned checkstyle repository

**Step 4: Check MLflow UI**

Run: `open http://localhost:5000`

Expected: See experiment "swe-refactor-evaluation" with 1 run

**Step 5: If all checks pass, commit validation notes**

```bash
git add -A
git commit -m "test(swe-eval): validate workflow with checkstyle/65655da4"
```

---

## Unresolved Questions

1. **RefactoringMiner Integration (Phase 1 only)**: Do we need RefactoringMiner verification in Phase 1, or defer to Phase 2? Original paper uses it for AST verification.

2. **CodeBLEU Metric**: Should we implement CodeBLEU scoring in Phase 1 or Phase 2? Requires installing CodeBLEU library.

3. **Test Execution Strategy**: Should we run full test suite or only tests related to modified files? Full suite is safer but slower.

4. **Retry Logic**: Current plan has max 3 retries on compile failure. Should we also retry on test failures?

5. **Multi-file Refactoring Edge Cases**: For Move operations affecting >2 files, do we need to handle additional dependencies?

6. **Workspace Cleanup**: Should agent clean up cloned repos after each evaluation, or reuse them?

7. **JDK Version Fallback**: If exact JDK version not available in jenv, should we try closest version or fail fast?

---

## Next Steps After Phase 1

**Phase 2 (smell detection) will add:**
- A1 agent node (SonarQube BEFORE scan)
- A1 agent node (SonarQube AFTER scan)
- Smell delta comparison metrics
- Integration with existing `agents/dependency_analysis`
- New scorers: smells_eliminated, smells_created, net_quality_improvement

**Implementation order:**
1. Complete Phase 1 Tasks 1-12
2. Validate with 10-20 refactorings across different types
3. Analyze failure modes and improve prompts
4. Run full evaluation on 1,099 refactorings
5. Begin Phase 2 design doc

