"""Structured output schemas for the LangGraph ReAct agent.

These models mirror the structured output used in the reference pipeline
(`pipeline_reference/pipeline.py`) and allow the agent to return rich,
validated responses describing detected code smells and refactoring guidance.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class CodeSmellSeverity(str, Enum):
    """Severity levels for detected code smells."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class CodeSmellDetection(BaseModel):
    """Single code smell detection produced by the agent."""

    smell_type: str = Field(
        description=(
            "The DACOS smell category that best matches the finding. Examples: "
            "'Multifaceted Abstraction', 'Complex Method', 'Long Parameter List'."
        )
    )
    location: str = Field(
        description=(
            "Precise location of the smell (file path, class, method, or line span)."
        )
    )
    severity: CodeSmellSeverity = Field(
        description="Relative severity of the issue based on impact and scope."
    )
    description: str = Field(
        description="Short justification that explains why this is a smell."
    )
    refactoring_suggestion: str = Field(description="Actionable refactoring guidance.")
    refactoring_reference: Optional[str] = Field(
        default=None,
        description=(
            "Optional citation or reference identifier from DACOS/Designite "
            "materials that inspired the suggestion."
        ),
    )
    code_example: Optional[str] = Field(
        default=None,
        description="Illustrative snippet that demonstrates the recommended fix.",
    )


class CodeAnalysisResult(BaseModel):
    """Top level structured response returned by the agent."""

    analysis_summary: str = Field(
        description="High level summary describing the analysed snippet."
    )
    smells_detected: List[CodeSmellDetection] = Field(
        description="List of detected smells ordered from most to least severe."
    )
    evidence: Optional[str] = Field(
        default=None,
        description=(
            "Optional supporting context retrieved from DACOS (e.g., DataFrame "
            "rows, annotations, or smell descriptions)."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "analysis_summary": (
                    "The method `Foo.bar()` exhibits multiple indicators of a "
                    "Complex Method smell and the constructor parameters form a "
                    "Long Parameter List."
                ),
                "smells_detected": [
                    {
                        "smell_type": "Complex Method",
                        "location": "Foo.bar()",
                        "severity": "HIGH",
                        "description": "Cyclomatic complexity above 15 with nested branches.",
                        "refactoring_suggestion": "Extract guard clauses and helper methods to reduce branching.",
                        "refactoring_reference": "DACOS::ComplexMethod::Refactor",
                    }
                ],
                "evidence": "Sample 42 from DACOS tagged as Complex Method with similarity score 0.82.",
            }
        }
