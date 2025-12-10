from abc import ABC, abstractmethod
from typing import List, Any, Dict


class DatasetFactory(ABC):
    """Abstract factory for creating datasets."""

    @abstractmethod
    def create_records(self) -> List[Any]:
        """Create the list of records for the dataset."""
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Get the name for the dataset."""
        pass

    @abstractmethod
    def get_tags(self) -> Dict[str, str]:
        """Get tags for the dataset."""
        pass
