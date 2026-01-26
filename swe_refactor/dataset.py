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
