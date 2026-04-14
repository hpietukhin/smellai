"""DiffHunk model — internal to RMiner utilities."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DiffHunk(BaseModel):
    """A hunk from git diff."""

    old_start: int = Field(..., description="Starting line in old file")
    old_count: int = Field(..., description="Number of lines in old file")
    new_start: int = Field(..., description="Starting line in new file")
    new_count: int = Field(..., description="Number of lines in new file")
    removed_lines: list[str] = Field(default_factory=list, description="Lines removed")
    added_lines: list[str] = Field(default_factory=list, description="Lines added")
    context_lines: list[str] = Field(default_factory=list, description="Context lines")


__all__ = ["DiffHunk"]
