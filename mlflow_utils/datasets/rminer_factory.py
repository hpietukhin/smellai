from pathlib import Path
from typing import List, Any, Dict
from .factory import DatasetFactory


class RMinerDatasetFactory(DatasetFactory):
    """Factory for creating RefactoringMiner datasets."""

    def __init__(self, manifest_path: str, limit: int):
        self.manifest_path = manifest_path
        self.limit = limit
        self._records = None

    def create_records(self) -> List[Any]:
        if self._records is None:
            from rminer.create_rminer_dataset import build_genai_records

            self._records = build_genai_records(
                Path(self.manifest_path), limit=self.limit
            )
        return self._records

    def get_dataset_name(self) -> str:
        return f"rminer-dataset-{self.limit or 'all'}"

    def get_tags(self) -> Dict[str, str]:
        records = self.create_records()
        return {
            "source": "RefactoringMiner",
            "total_pairs": str(len(records)),
        }
