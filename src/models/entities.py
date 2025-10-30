"""
Pydantic models for code smell detection entities.

This module contains all data models used throughout the LLM-based code smell
detection pipeline. These models are designed for use with LangChain's structured
output features, enabling type-safe LLM responses.

**Structured Output with Pydantic Models**:
These models can be used with:
- LangChain's `with_structured_output()` for direct model binding
- `PydanticOutputParser` for parsing LLM text responses
- LiteLLM's `response_format` parameter for JSON schema validation

Models included:
- Ground truth annotations from DACOS database (SmellAnnotation)
- LLM-detected smells (SmellDetection)
- Evaluation results and scores (EvaluationResult, SmellEvaluation, EvaluationScore)
- DACOS sample records (DACOSSample)
"""

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SmellAnnotation(BaseModel):
    """
    Ground truth code smell annotation from DACOS database.

    Represents a known code smell annotation used as ground truth for evaluation.
    """

    smell_type: str = Field(
        description="Type of code smell (e.g., 'Complex Method', 'Long Method')"
    )
    is_present: bool = Field(
        description="Whether this smell is present in the code (from annotation flags)"
    )
    package_name: Optional[str] = Field(
        default=None, description="Package name where smell is located"
    )
    type_name: Optional[str] = Field(
        default=None, description="Class name where smell is located"
    )
    method_name: Optional[str] = Field(
        default=None, description="Method name where smell is located"
    )
    loc: Optional[int] = Field(
        default=None, description="Lines of code (LOC) metric", ge=0
    )
    cc: Optional[int] = Field(
        default=None, description="Cyclomatic complexity (CC) metric", ge=0
    )
    pc: Optional[int] = Field(
        default=None, description="Parameter count (PC) metric", ge=0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "smell_type": "Complex Method",
                "is_present": True,
                "package_name": "com.example.service",
                "type_name": "UserService",
                "method_name": "processUserData",
                "loc": 45,
                "cc": 12,
                "pc": 5,
            }
        }


class SmellDetection(BaseModel):
    """
    Code smell detected by the LLM.

    Represents a smell identified by the LLM detector agent during analysis.
    """

    smell_type: str = Field(description="Type of code smell detected")
    location: str = Field(
        description="Description of where the smell was found (class, method, line range)"
    )
    description: str = Field(description="Detailed description of the detected smell")
    severity: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description="Severity level of the detected smell"
    )
    refactoring_suggestion: str = Field(
        description="Suggested refactoring to fix the smell"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score of the detection (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "smell_type": "Long Method",
                "location": "UserService.processUserData() (lines 23-78)",
                "description": "Method has 45 lines and performs multiple unrelated tasks",
                "severity": "HIGH",
                "refactoring_suggestion": "Extract validation logic into separate methods",
                "confidence": 0.92,
            }
        }


class EvaluationScore(str, Enum):
    """Evaluation score levels for LLM-as-judge assessment."""

    EXCELLENT = "EXCELLENT"  # 5 points
    GOOD = "GOOD"  # 4 points
    ACCEPTABLE = "ACCEPTABLE"  # 3 points
    POOR = "POOR"  # 2 points
    INCORRECT = "INCORRECT"  # 1 point


class SmellEvaluation(BaseModel):
    """
    Evaluation of a single detected smell against ground truth.

    Represents the LLM-as-judge assessment of one detected smell.
    """

    detected_smell: str = Field(description="The smell type that was detected")
    location: str = Field(description="Location where the smell was detected")
    ground_truth_match: Optional[str] = Field(
        default=None, description="Matched ground truth smell type (if any)"
    )
    score: EvaluationScore = Field(description="Quality score for this detection")
    justification: str = Field(description="Explanation for the assigned score")

    class Config:
        json_schema_extra = {
            "example": {
                "detected_smell": "Complex Method",
                "location": "UserService.processUserData()",
                "ground_truth_match": "Complex Method",
                "score": "EXCELLENT",
                "justification": "Correctly identified Complex Method with accurate location",
            }
        }


class EvaluationResult(BaseModel):
    """
    Complete evaluation result for a code sample.

    Contains all detection evaluations and aggregate metrics.
    """

    sample_id: int = Field(description="DACOS sample ID")
    file_path: str = Field(description="Path to the evaluated file")
    overall_score: float = Field(
        description="Overall quality score (0.0 to 5.0)", ge=0.0, le=5.0
    )
    precision: float = Field(
        description="Precision metric (0.0 to 1.0)", ge=0.0, le=1.0
    )
    recall: float = Field(description="Recall metric (0.0 to 1.0)", ge=0.0, le=1.0)
    f1_score: float = Field(description="F1 score (0.0 to 1.0)", ge=0.0, le=1.0)
    evaluations: List[SmellEvaluation] = Field(
        description="Individual smell evaluations"
    )
    summary: str = Field(description="Overall summary of the evaluation")
    timestamp: str = Field(description="Evaluation timestamp (ISO format)")
    git_sha: str = Field(description="Git commit SHA of the evaluated code")

    class Config:
        json_schema_extra = {
            "example": {
                "sample_id": 12345,
                "file_path": "src/main/java/com/example/UserService.java",
                "overall_score": 4.2,
                "precision": 0.85,
                "recall": 0.90,
                "f1_score": 0.87,
                "evaluations": [],
                "summary": "Good detection quality with 2/2 smells correctly identified",
                "timestamp": "2024-10-19T10:30:00Z",
                "git_sha": "abc123def456",
            }
        }


class DACOSSample(BaseModel):
    """
    Complete DACOS database record with annotations.

    Represents a sample from the DACOS dataset including metadata,
    annotations, and derived information.
    """

    # Sample table fields
    id: int = Field(description="Sample ID")
    designite_id: Optional[int] = Field(default=None, description="Designite tool ID")
    has_smell: bool = Field(description="Whether sample contains smells")
    is_class: bool = Field(description="Whether sample is a class-level smell")
    path_to_file: str = Field(description="Relative path to file in repository")
    project_name: str = Field(description="Project name (format: org_repo)")
    sample_constraints: Optional[str] = Field(
        default=None, description="Sample constraints"
    )
    smells: Optional[int] = Field(default=None, description="Smell type ID")

    # Annotation fields (boolean flags for smell types)
    iscm: bool = Field(default=False, description="Is Complex Method")
    isim: bool = Field(default=False, description="Is Insufficient Modularization")
    islp: bool = Field(default=False, description="Is Long Parameter List")
    isma: bool = Field(default=False, description="Is Multifaceted Abstraction")

    # Smell information
    smell_name: Optional[str] = Field(
        default=None, description="Human-readable smell name"
    )
    smell_description: Optional[str] = Field(
        default=None, description="Smell description"
    )

    # Derived fields
    repo_url: Optional[str] = Field(default=None, description="GitHub repository URL")
    commit_sha: Optional[str] = Field(default=None, description="Git commit SHA")

    def ground_truth_smells(self) -> List[str]:
        """
        Get list of active ground truth smells based on annotation flags.

        Returns:
            List of smell type names that are marked as present
        """
        smells = []
        if self.iscm:
            smells.append("Complex Method")
        if self.isim:
            smells.append("Insufficient Modularization")
        if self.islp:
            smells.append("Long Parameter List")
        if self.isma:
            smells.append("Multifaceted Abstraction")
        return smells

    def to_annotations(self) -> List[SmellAnnotation]:
        """
        Convert annotation flags to SmellAnnotation objects.

        Returns:
            List of SmellAnnotation objects for all smell types
        """
        annotations = []

        smell_flags = [
            ("Complex Method", self.iscm),
            ("Insufficient Modularization", self.isim),
            ("Long Parameter List", self.islp),
            ("Multifaceted Abstraction", self.isma),
        ]

        for smell_type, is_present in smell_flags:
            annotations.append(
                SmellAnnotation(
                    smell_type=smell_type,
                    is_present=is_present,
                    package_name=None,  # Not available in DACOS schema
                    type_name=None,  # Would need to parse from path
                    method_name=None,  # Would need to parse from code
                )
            )

        return annotations

    class Config:
        json_schema_extra = {
            "example": {
                "id": 12345,
                "designite_id": 5678,
                "has_smell": True,
                "is_class": False,
                "path_to_file": "src/main/java/com/example/UserService.java",
                "project_name": "alibaba_arthas",
                "sample_constraints": None,
                "smells": 1,
                "iscm": True,
                "isim": False,
                "islp": False,
                "isma": False,
                "smell_name": "Complex Method",
                "smell_description": "Method with high cyclomatic complexity",
                "repo_url": "https://github.com/alibaba/arthas",
                "commit_sha": "abc123def456",
            }
        }
