import mlflow
from typing import Optional, List, Any
from .factory import DatasetFactory


class DatasetManager:
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)

    def list_datasets(self, experiment_name: Optional[str] = None) -> List[Any]:
        """List datasets using mlflow.genai.datasets.search_datasets if available."""
        try:
            # Attempt to import from mlflow.genai.datasets
            # Note: This API might be experimental or specific to the user's mlflow version/extensions
            from mlflow.genai.datasets import search_datasets

            return search_datasets(experiment_name=experiment_name)
        except ImportError:
            print(
                "Warning: mlflow.genai.datasets not found or search_datasets not available."
            )
            return []
        except Exception as e:
            print(f"Error listing datasets: {e}")
            return []

    def get_dataset(self, name: str) -> Any:
        try:
            from mlflow.genai.datasets import get_dataset

            return get_dataset(name)
        except ImportError:
            print("Warning: mlflow.genai.datasets not found.")
            return None
        except Exception as e:
            print(f"Error getting dataset: {e}")
            return None

    def create_dataset(
        self, factory: DatasetFactory, experiment: str, tracking_uri: str
    ):
        """Creates a dataset using the provided factory."""
        print(f"Creating dataset using {factory.__class__.__name__}...")

        records = factory.create_records()

        # 1. Register Dataset in MLflow
        try:
            from mlflow.genai.datasets import create_dataset as create_genai_dataset

            # Ensure experiment exists
            existing_exp = mlflow.get_experiment_by_name(experiment)
            if existing_exp:
                experiment_id = existing_exp.experiment_id
            else:
                experiment_id = mlflow.create_experiment(experiment)

            dataset = create_genai_dataset(
                name=factory.get_dataset_name(),
                experiment_id=[experiment_id],
                tags=factory.get_tags(),
            )
            dataset.merge_records(records)
            print(f"Registered MLflow dataset: {dataset.dataset_id}")
        except ImportError:
            print(
                "Warning: mlflow.genai.datasets not found, skipping dataset registration."
            )
        except Exception as e:
            print(f"Warning: Failed to register dataset: {e}")
