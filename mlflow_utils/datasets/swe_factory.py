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

    if refactoring_type:
        records = [r for r in records if r.type == refactoring_type]
        LOGGER.info(
            "Filtered to %d records of type '%s'", len(records), refactoring_type
        )

    if project_name:
        records = [r for r in records if r.projectName == project_name]
        LOGGER.info(
            "Filtered to %d records from project '%s'", len(records), project_name
        )

    if limit:
        records = records[:limit]
        LOGGER.info("Limited to %d records", limit)

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

    LOGGER.info(
        "Created MLflow dataset '%s' (ID: %s) with %d records",
        name,
        dataset.dataset_id,
        len(genai_records),
    )

    return dataset.dataset_id
