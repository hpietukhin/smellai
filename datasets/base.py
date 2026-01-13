"""Base dataset interface (Target in adapter pattern)."""

from abc import ABC, abstractmethod


class Dataset(ABC):
    """
    The Target defines the domain-specific interface used by client code.
    This is the common interface that all dataset adapters must implement.

    All datasets return MLflow-compatible evaluation records with:
    - inputs: dict (what goes into the LLM)
    - expectations: dict (ground truth for evaluation)
    - tags: dict (metadata)
    """

    @abstractmethod
    def request(self) -> list[dict]:
        """
        Return MLflow-compatible evaluation records.

        Returns:
            list[dict]: Records with structure:
                {
                    "inputs": {
                        "pair_id": str,
                        "code_before": str,
                        "refactoring_type": str,
                        "context": dict,
                    },
                    "expectations": {
                        "code_after": str,
                        "diff_hunks": list,
                        "metadata": dict,
                    },
                    "tags": {
                        "repository": str,
                        "commit_sha": str,
                        "dataset_source": str,
                    }
                }
        """
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Get the dataset name."""
        pass

    @abstractmethod
    def get_tags(self) -> dict[str, str]:
        """Get dataset-level tags/metadata."""
        pass
