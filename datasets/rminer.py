"""RMiner dataset adaptee and adapter."""

from __future__ import annotations

from pathlib import Path

from .base import Dataset


class RMinerDataset:
    """
    The Adaptee contains useful behavior for RMiner dataset,
    but its interface is incompatible with the existing client code.
    The Adaptee needs some adaptation before the client code can use it.
    """

    def __init__(self, manifest_path: str, limit: int | None = None):
        """
        Initialize RMiner dataset.

        Args:
            manifest_path: Path to manifest.json file
            limit: Optional limit on number of records
        """
        self.manifest_path = Path(manifest_path)
        self.limit = limit
        self._cached_records = None

    def build_rminer_records(self) -> list[dict]:
        """
        Build records in RMiner-specific format.

        Uses the existing rminer/create_rminer_dataset.py:build_genai_records()
        function to create records.

        Returns:
            list[dict]: Records in RMiner format with structure:
                {
                    "inputs": {"pair_id": str, "sonar_issues": list},
                    "expectations": {
                        "num_refactorings": int,
                        "num_hunks": int,
                        "diff_hunks": list,
                        "refactoring_types": list,
                        "refactoring_descriptions": list,
                        "file_path": str,
                    },
                    "tags": {
                        "repository": str,
                        "commit_sha": str,
                        "status": str,
                    }
                }
        """
        if self._cached_records is not None:
            return self._cached_records

        # Import here to avoid circular dependencies
        from rminer.create_rminer_dataset import build_genai_records

        self._cached_records = build_genai_records(
            self.manifest_path,
            limit=self.limit
        )
        return self._cached_records

    def get_rminer_metadata(self) -> dict[str, str]:
        """
        Get RMiner-specific metadata.

        Returns:
            dict: Metadata including total_pairs count
        """
        records = self.build_rminer_records()
        return {
            "total_pairs": str(len(records)),
        }


class RMinerAdapter(Dataset, RMinerDataset):
    """
    The Adapter makes the RMinerDataset's interface compatible with the
    Dataset's interface via multiple inheritance.

    Example:
        >>> adapter = RMinerAdapter(
        ...     manifest_path="rminer_data/manifest.json",
        ...     limit=10
        ... )
        >>> records = adapter.request()
        >>> print(f"Dataset: {adapter.get_dataset_name()}")
        >>> print(f"Records: {len(records)}")
    """

    def request(self) -> list[dict]:
        """
        Adapt RMiner-specific format to common Dataset format.

        Translates build_rminer_records() to MLflow-compatible format.
        The RMiner format is already MLflow-compatible, so we return it directly.

        Returns:
            list[dict]: MLflow-compatible records
        """
        return self.build_rminer_records()

    def get_dataset_name(self) -> str:
        """
        Get dataset name.

        Returns:
            str: Dataset name in format "rminer-{limit}" or "rminer-all"
        """
        return f"rminer-{self.limit or 'all'}"

    def get_tags(self) -> dict[str, str]:
        """
        Get dataset-level tags/metadata.

        Returns:
            dict: Tags including source, type, and metadata
        """
        metadata = self.get_rminer_metadata()
        return {
            "source": "RefactoringMiner",
            "type": "atomic_refactorings",
            **metadata,
        }
