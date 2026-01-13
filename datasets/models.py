"""Pydantic models for dataset records."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ============================================================================
# Common Dataset Models (Target format)
# ============================================================================

class RecordInputs(BaseModel):
    """Input data for a refactoring task (what goes into the LLM)."""

    pair_id: str = Field(..., description="Unique identifier for this refactoring instance")
    code_before: str = Field(..., description="Code before refactoring")
    refactoring_type: str = Field(..., description="Type of refactoring to perform")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (varies by dataset)"
    )


class RecordExpectations(BaseModel):
    """Expected output / ground truth for evaluation."""

    code_after: str = Field(..., description="Ground truth refactored code")
    diff_hunks: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Diff hunks showing changes"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Dataset-specific metadata"
    )


class RecordTags(BaseModel):
    """Metadata tags for a record."""

    repository: str = Field(default="", description="Repository URL or name")
    commit_sha: str = Field(default="", description="Git commit SHA")
    dataset_source: str = Field(..., description="Dataset source (rminer, swe-refactor, etc.)")


class DatasetRecord(BaseModel):
    """Standard record format for all datasets (MLflow-compatible)."""

    inputs: RecordInputs
    expectations: RecordExpectations
    tags: RecordTags

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


# ============================================================================
# RMiner-Specific Models
# ============================================================================

class RMinerContext(BaseModel):
    """Context data specific to RMiner dataset."""

    sonar_issues: list[dict[str, Any]] = Field(
        default_factory=list,
        description="SonarQube issues detected in the code"
    )


class DiffHunk(BaseModel):
    """A hunk from git diff."""

    old_start: int = Field(..., description="Starting line in old file")
    old_count: int = Field(..., description="Number of lines in old file")
    new_start: int = Field(..., description="Starting line in new file")
    new_count: int = Field(..., description="Number of lines in new file")
    removed_lines: list[str] = Field(default_factory=list, description="Lines removed")
    added_lines: list[str] = Field(default_factory=list, description="Lines added")
    context_lines: list[str] = Field(default_factory=list, description="Context lines")


class RMinerExpectations(BaseModel):
    """Ground truth data specific to RMiner format."""

    num_refactorings: int = Field(..., description="Number of refactorings applied")
    num_hunks: int = Field(..., description="Number of diff hunks")
    diff_hunks: list[DiffHunk] = Field(..., description="Detailed diff hunks")
    refactoring_types: list[str] = Field(..., description="Types of refactorings")
    refactoring_descriptions: list[str] = Field(..., description="Descriptions of refactorings")
    file_path: str = Field(..., description="Path to the file being refactored")


class RMinerTags(BaseModel):
    """Tags specific to RMiner records."""

    repository: str = Field(default="", description="Repository URL")
    commit_sha: str = Field(default="", description="Commit SHA")
    status: str = Field(default="modified", description="File status (modified, added, etc.)")


class RMinerRecord(BaseModel):
    """RMiner-specific record format (before adaptation)."""

    inputs: dict[str, Any]  # Contains pair_id and sonar_issues
    expectations: RMinerExpectations
    tags: RMinerTags

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


# ============================================================================
# SWE-Refactor-Specific Models
# ============================================================================

class ClassHierarchy(BaseModel):
    """Class hierarchy information."""

    superclass: str | None = Field(None, description="Parent class name")
    subclasses: list[str] = Field(default_factory=list, description="Child class names")
    interfaces: list[str] = Field(default_factory=list, description="Implemented interfaces")


class MethodSignature(BaseModel):
    """Method signature information."""

    name: str = Field(..., description="Method name")
    parameters: list[str] = Field(default_factory=list, description="Parameter types")
    return_type: str = Field(default="", description="Return type")


class BuildConfiguration(BaseModel):
    """Build configuration for the project."""

    commit_id: str = Field(..., description="Git commit ID")
    jdk_version: str | int = Field(..., description="JDK version (e.g., 17)")
    build_command: str = Field(..., description="Build command (e.g., 'mvn clean package')")


class TestCoverage(BaseModel):
    """Test coverage information."""

    branch_coverage: float = Field(default=0.0, description="Branch coverage percentage")
    instruction_coverage: float = Field(default=0.0, description="Instruction coverage percentage")
    line_coverage: float = Field(default=0.0, description="Line coverage percentage")
    complexity_coverage: float = Field(default=0.0, description="Complexity coverage percentage")
    method_coverage: float = Field(default=0.0, description="Method coverage percentage")


class SWERefactorContext(BaseModel):
    """Context data specific to SWE-Refactor dataset (much richer than RMiner)."""

    class_content: str = Field(default="", description="Full class source code")
    class_hierarchy: ClassHierarchy = Field(
        default_factory=ClassHierarchy,
        description="Class hierarchy information"
    )
    callers: list[MethodSignature] = Field(
        default_factory=list,
        description="Methods that call the target method"
    )
    callees: list[MethodSignature] = Field(
        default_factory=list,
        description="Methods called by the target method"
    )
    project_structure: dict[str, Any] = Field(
        default_factory=dict,
        description="Project file structure"
    )
    build_config: BuildConfiguration | None = Field(
        None,
        description="Build configuration"
    )


class SWERefactorMetadata(BaseModel):
    """Metadata specific to SWE-Refactor."""

    refactoring_type: str = Field(..., description="Type of refactoring")
    is_compound: bool = Field(
        False,
        description="Whether this is a compound refactoring (multiple transformations)"
    )


class SWERefactorTags(BaseModel):
    """Tags specific to SWE-Refactor records."""

    repository: str = Field(default="", description="Repository name")
    commit_id: str = Field(default="", description="Commit ID")
    test_coverage: TestCoverage = Field(
        default_factory=TestCoverage,
        description="Test coverage metrics"
    )
    dataset_source: str = Field(default="swe-refactor", description="Dataset source")


class SWERefactorRecord(BaseModel):
    """SWE-Refactor-specific record format (before adaptation)."""

    inputs: dict[str, Any]  # Contains pair_id, target_method, refactoring_type, context
    expectations: dict[str, Any]  # Contains developer_written_code, refactoring_metadata
    tags: SWERefactorTags

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


# ============================================================================
# Dataset Metadata Models
# ============================================================================

class DatasetMetadata(BaseModel):
    """Metadata for a dataset."""

    name: str = Field(..., description="Dataset name")
    source: str = Field(..., description="Dataset source")
    total_records: int = Field(..., description="Total number of records")
    tags: dict[str, str] = Field(default_factory=dict, description="Additional tags")
