"""
Pydantic models for RefactoringMiner data structures.

This module contains data models for parsing and analyzing RefactoringMiner
benchmark data. These models support both the simplified oracle data (data.json)
and full RefactoringMiner output with location information.

Models included:
- Refactoring operations and their locations (Refactoring, RefactoringLocation)
- Commit information from RefactoringMiner (RMinerCommit)
- Clustering and analysis results (CommitCluster, RefactoringStats)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RefactoringLocation(BaseModel):
    """
    Location information for code elements involved in refactoring.

    This model represents the precise location of code elements (methods, classes,
    variables) before or after a refactoring operation. Location data is available
    in full RefactoringMiner output but not in the oracle benchmark data.json.
    """

    file_path: str = Field(description="Path to the file relative to repository root")
    start_line: int = Field(description="Starting line number", ge=1)
    end_line: int = Field(description="Ending line number", ge=1)
    start_column: Optional[int] = Field(
        default=None, description="Starting column (optional)", ge=0
    )
    end_column: Optional[int] = Field(
        default=None, description="Ending column (optional)", ge=0
    )
    code_element: Optional[str] = Field(
        default=None, description="Code element name (method, class, variable)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "src/main/java/com/example/UserService.java",
                "start_line": 42,
                "end_line": 58,
                "start_column": 4,
                "end_column": 5,
                "code_element": "processUserData",
            }
        }


class Refactoring(BaseModel):
    """
    Individual refactoring operation detected by RefactoringMiner.

    Represents a single refactoring with type, description, validation status,
    and optional location information for before/after states.
    """

    type: str = Field(
        description="Refactoring type (e.g., 'Extract Method', 'Rename Class')"
    )
    description: str = Field(
        description="Human-readable description of the refactoring"
    )
    validation: Optional[str] = Field(
        default=None,
        description="Validation status: 'TP' (true positive), 'FP' (false positive), or null",
    )
    comment: Optional[str] = Field(
        default=None, description="Validator comment explaining validation decision"
    )
    detection_tools: Optional[str] = Field(
        default=None,
        alias="detectionTools",
        description="Comma-separated list of tools that detected this refactoring",
    )
    validators: Optional[str] = Field(
        default=None, description="Names of human validators (if any)"
    )
    left_side_locations: List[RefactoringLocation] = Field(
        default_factory=list,
        alias="leftSideLocations",
        description="Locations of code elements before refactoring (from full JSON)",
    )
    right_side_locations: List[RefactoringLocation] = Field(
        default_factory=list,
        alias="rightSideLocations",
        description="Locations of code elements after refactoring (from full JSON)",
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "type": "Extract Method",
                "description": "Extract Method private validateUser(user User) : boolean extracted from public processRequest(request Request) : Response in class com.example.RequestHandler",
                "validation": "TP",
                "comment": None,
                "detectionTools": "RefactoringMiner, RefDiff",
                "validators": None,
                "leftSideLocations": [],
                "rightSideLocations": [],
            }
        }


class RMinerCommit(BaseModel):
    """
    Git commit with refactoring information from RefactoringMiner.

    Represents a commit that has been analyzed by RefactoringMiner,
    containing metadata and a list of detected refactorings.
    """

    id: int = Field(description="Numeric commit ID from benchmark database")
    repository: str = Field(description="Git repository URL")
    sha1: str = Field(description="Git commit SHA-1 hash")
    url: str = Field(description="Web URL to view the commit")
    author: str = Field(description="Commit author name")
    time: str = Field(description="Commit timestamp")
    refactorings: List[Refactoring] = Field(
        default_factory=list, description="List of refactorings detected in this commit"
    )
    ref_diff_execution_time: Optional[int] = Field(
        default=None,
        alias="refDiffExecutionTime",
        description="RefDiff execution time in milliseconds",
        ge=0,
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": 1100435,
                "repository": "https://github.com/realm/realm-java.git",
                "sha1": "6cf596df183b3c3a38ed5dd9bb3b0100c6548ebb",
                "url": "https://github.com/realm/realm-java/commit/6cf596df183b3c3a38ed5dd9bb3b0100c6548ebb",
                "author": "Christian Melchior",
                "time": "6/8/15 7:26 AM",
                "refactorings": [],
                "refDiffExecutionTime": 1250,
            }
        }


class CommitCluster(BaseModel):
    """
    Group of consecutive commits from the same repository.

    Represents a cluster of commits that are potentially related based on
    ID proximity and semantic similarity (refactoring types, file paths).
    """

    repository: str = Field(description="Git repository URL")
    commit_ids: List[int] = Field(
        description="Ordered list of commit IDs in this cluster"
    )
    commits: List[RMinerCommit] = Field(
        default_factory=list, description="Full commit data for analysis"
    )
    cluster_score: float = Field(
        default=0.0,
        description="Cluster quality score (0.0 to 1.0) based on proximity and semantics",
        ge=0.0,
        le=1.0,
    )

    def get_id_gaps(self) -> List[int]:
        """
        Calculate gaps between consecutive commit IDs.

        Returns:
            List of gap sizes between adjacent commits in cluster
        """
        if len(self.commit_ids) < 2:
            return []
        return [
            self.commit_ids[i + 1] - self.commit_ids[i]
            for i in range(len(self.commit_ids) - 1)
        ]

    def max_id_gap(self) -> int:
        """
        Return maximum gap between commit IDs.

        Returns:
            Maximum gap size, or 0 if cluster has fewer than 2 commits
        """
        gaps = self.get_id_gaps()
        return max(gaps) if gaps else 0

    def avg_id_gap(self) -> float:
        """
        Return average gap between commit IDs.

        Returns:
            Average gap size, or 0.0 if cluster has fewer than 2 commits
        """
        gaps = self.get_id_gaps()
        return sum(gaps) / len(gaps) if gaps else 0.0

    def total_refactorings(self) -> int:
        """
        Count total refactorings across all commits in cluster.

        Returns:
            Total number of refactorings
        """
        return sum(len(commit.refactorings) for commit in self.commits)

    class Config:
        json_schema_extra = {
            "example": {
                "repository": "https://github.com/JetBrains/intellij-community.git",
                "commit_ids": [1100842, 1100856, 1100868],
                "commits": [],
                "cluster_score": 0.87,
            }
        }


class RefactoringStats(BaseModel):
    """
    Statistical summary of refactoring analysis.

    Aggregates key metrics and distributions from analyzed commits,
    including refactoring types, validation status, and repository breakdown.
    """

    total_commits: int = Field(description="Total number of commits analyzed", ge=0)
    total_repositories: int = Field(
        description="Total number of unique repositories", ge=0
    )
    total_refactorings: int = Field(
        description="Total number of refactorings detected", ge=0
    )
    refactoring_type_counts: Dict[str, int] = Field(
        description="Count of each refactoring type"
    )
    validation_counts: Dict[str, int] = Field(
        description="Count of TP, FP, and null validations"
    )
    top_repositories: List[Dict[str, Any]] = Field(
        description="Top repositories by commit count"
    )
    clusters_found: int = Field(
        description="Number of consecutive commit clusters found", ge=0
    )
    clusters_detail: List[Dict[str, Any]] = Field(
        description="Details of each cluster (repo, size, score)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "total_commits": 100,
                "total_repositories": 10,
                "total_refactorings": 2847,
                "refactoring_type_counts": {
                    "Rename Method": 342,
                    "Move Class": 289,
                    "Extract Method": 267,
                },
                "validation_counts": {"TP": 2456, "FP": 298, "Other/null": 93},
                "top_repositories": [
                    {
                        "repository": "https://github.com/JetBrains/intellij-community.git",
                        "commit_count": 37,
                    }
                ],
                "clusters_found": 15,
                "clusters_detail": [],
            }
        }
