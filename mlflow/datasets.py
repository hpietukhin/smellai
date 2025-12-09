import mlflow
import subprocess
import sys
from typing import Optional, List, Any


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
        self, manifest_path: str, limit: int, experiment: str, tracking_uri: str
    ):
        """Creates a dataset by running the rminer/create_rminer_dataset.py script."""
        print(f"Creating dataset from {manifest_path} (limit: {limit})...")
        cmd = [
            "uv",
            "run",
            "rminer/create_rminer_dataset.py",
            "--manifest",
            manifest_path,
            "--limit",
            str(limit),
            "--experiment",
            experiment,
            "--tracking-uri",
            tracking_uri,
        ]
        try:
            subprocess.run(cmd, check=True)
            print("Dataset creation script completed.")
        except subprocess.CalledProcessError as e:
            print(f"Error creating dataset: {e}")
            raise

    def delete_dataset(self, name: str):
        try:
            from mlflow.genai.datasets import delete_dataset

            delete_dataset(name)
            print(f"Dataset {name} deleted.")
        except ImportError:
            print("Warning: mlflow.genai.datasets not found.")
        except Exception as e:
            print(f"Error deleting dataset: {e}")
