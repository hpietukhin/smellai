"""CRUD operations for MLflow GenAI datasets."""

from __future__ import annotations

from typing import Any

import mlflow


class DatasetManager:
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)

    def list_datasets(self, experiment_name: str | None = None) -> list[Any]:
        """List datasets using mlflow.genai.datasets.search_datasets."""
        try:
            from mlflow.genai.datasets import search_datasets
            return search_datasets(experiment_name=experiment_name)
        except ImportError:
            print("Warning: mlflow.genai.datasets not available.")
            return []
        except Exception as e:
            print(f"Error listing datasets: {e}")
            return []

    def get_dataset(self, name: str) -> Any:
        try:
            from mlflow.genai.datasets import get_dataset
            return get_dataset(name)
        except ImportError:
            print("Warning: mlflow.genai.datasets not available.")
            return None
        except Exception as e:
            print(f"Error getting dataset: {e}")
            return None

    def create_dataset_from_records(
        self,
        records: list[dict],
        name: str,
        experiment: str,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """Create an MLflow GenAI dataset from pre-built records.

        Args:
            records: List of MLflow GenAI records (inputs/expectations/tags dicts)
            name: Dataset name
            experiment: MLflow experiment name
            tags: Optional dataset-level tags

        Returns:
            Dataset ID, or None on failure
        """
        try:
            from mlflow.genai.datasets import create_dataset as create_genai_dataset

            existing_exp = mlflow.get_experiment_by_name(experiment)
            experiment_id = (
                existing_exp.experiment_id if existing_exp
                else mlflow.create_experiment(experiment)
            )

            dataset = create_genai_dataset(
                name=name,
                experiment_id=[experiment_id],
                tags=tags or {},
            )
            dataset.merge_records(records)
            print(f"Registered MLflow dataset: {dataset.dataset_id}")
            return dataset.dataset_id
        except ImportError:
            print("Warning: mlflow.genai.datasets not available.")
            return None
        except Exception as e:
            print(f"Warning: Failed to register dataset: {e}")
            return None
