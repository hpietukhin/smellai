"""Dataset models and loader for SWE-Refactor benchmark."""

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

LOGGER = logging.getLogger(__name__)


class RefactoringRecord(BaseModel):
    """Single refactoring record from SWE-Refactor dataset."""

    # Identifiers
    projectName: str = Field(description="Project name (e.g., 'checkstyle')")
    commitId: str = Field(description="Full commit hash")
    type: str = Field(description="Refactoring type; may be compound e.g. 'Extract Method+Move Method'")

    # File paths
    filePathBefore: str = Field(description="Source file path before refactoring")
    filePathAfter: str = Field(description="Target file path after refactoring")

    # Source code
    sourceCodeBeforeForWhole: str = Field(description="Full source file before")
    sourceCodeAfterForWhole: str = Field(description="Full source file after")

    # Build configuration
    compileJDK: int = Field(description="JDK version for compilation (e.g., 8, 11, 17, 21)")
    compileCommand: str = Field(description="Build command to use")
    compileResultBefore: bool | None = Field(default=None, description="Whether code compiled successfully before refactoring")
    compileResultCurrent: bool | None = Field(default=None, description="Whether code compiles successfully after refactoring")
    hasTestC: bool | None = Field(default=None, description="Whether commit has test coverage (None if unknown)")

    # Optional metadata (may not be in all records)
    isPureRefactoring: bool | None = None
    description: str | None = None
    packageNameBefore: str | None = None
    classNameBefore: str | None = None
    methodNameBefore: str | None = None

    @field_validator("compileJDK", mode="before")
    @classmethod
    def convert_jdk_version(cls, v):
        """Convert JDK version from float (1.8) or int (11, 17, 21) to int."""
        if isinstance(v, float):
            return 8 if v == 1.8 else int(round(v))
        return int(v)


class SWERefactorDataset:
    """Loader for SWE-Refactor dataset."""

    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        self.records: list[RefactoringRecord] = []

    def load(self) -> list[RefactoringRecord]:
        """Load dataset from JSON file."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        data = json.loads(self.dataset_path.read_text())

        if not isinstance(data, list):
            raise ValueError("Expected JSON array of refactoring records")

        self.records = [RefactoringRecord(**record) for record in data]

        LOGGER.info("Loaded %d refactoring records from %s", len(self.records), self.dataset_path)

        return self.records

    def filter_by_type(
        self,
        refactoring_type: str,
    ) -> list[RefactoringRecord]:
        return [r for r in self.records if r.type == refactoring_type]

    def filter_by_project(
        self,
        project_name: str,
    ) -> list[RefactoringRecord]:
        return [r for r in self.records if r.projectName == project_name]

    def get_commit_records(
        self,
        commit_id: str,
    ) -> list[RefactoringRecord]:
        """Get all refactorings in a specific commit (supports short hashes)."""
        return [r for r in self.records if r.commitId.startswith(commit_id)]


def load_swe_refactor_dataset(
    dataset_path: str | Path = "/tmp/SWE-Refactor/pure_refactoring_data.json",
) -> list[RefactoringRecord]:
    """Convenience function to load SWE-Refactor dataset."""
    dataset = SWERefactorDataset(dataset_path)
    return dataset.load()
