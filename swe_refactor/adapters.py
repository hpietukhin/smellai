"""Adapters between generic EvalSample records and SWE-Refactor models."""

from __future__ import annotations

from smellai_datasets.schema import EvalSample
from swe_refactor.dataset import RefactoringRecord


def sample_to_refactoring_record(sample: EvalSample) -> RefactoringRecord:
    """Build a ``RefactoringRecord`` from a SWE EvalSample.

    Notes
    -----
    - ``sourceCodeAfterForWhole`` is set to "" because agents generate their
      own refactored code; ground truth remains in ``sample.expectations``.
    - ``compileResultBefore`` / ``compileResultCurrent`` default to True because
      SWE-Refactor guarantees pre- and post-refactoring compilability.
    """
    if sample.source != "swe":
        raise ValueError(f"SWE adapter expects source='swe', got {sample.source!r}")

    inputs = sample.inputs
    return RefactoringRecord(
        projectName=inputs["project_name"],
        commitId=inputs["commit_id"],
        type=inputs["refactoring_type"],
        filePathBefore=inputs["file_path_before"],
        filePathAfter=inputs["file_path_after"],
        sourceCodeBeforeForWhole=inputs["class_before"],
        sourceCodeAfterForWhole="",
        compileJDK=inputs["jdk_version"],
        compileCommand=inputs["compile_command"],
        compileResultBefore=True,
        compileResultCurrent=True,
        hasTestC=sample.tags.get("has_tests", False),
        isPureRefactoring=sample.tags.get("is_pure", True),
    )


__all__ = ["sample_to_refactoring_record"]
