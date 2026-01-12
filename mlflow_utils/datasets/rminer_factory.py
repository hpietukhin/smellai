from pathlib import Path
from typing import List, Any, Dict
from .factory import DatasetFactory


class RMinerDatasetFactory(DatasetFactory):
    """Factory for creating RefactoringMiner datasets.

    # TODO SPEC-008: Implement adapter for new dataset format (specification to be provided).
    # Current implementation hardcodes RefactoringMiner JSON parsing.
    # Need generic adapter interface for different input formats.
    # HIGH priority.
    # (See TECHNICAL_SPECIFICATION.md §4.3)

    # TODO SPEC-014: Implement adapter for new dataset format.
    # This is a duplicate of SPEC-008, same implementation task.
    # (See TECHNICAL_SPECIFICATION.md §5.5)

    # TODO SPEC-013: Verify how new dataset handles refactorings spanning multiple files.
    # Current pair_id format is commit_sha:file_path.
    # Need to verify handling of multi-file refactorings (Move Class, Pull Up Method).
    # MEDIUM priority.
    # (See TECHNICAL_SPECIFICATION.md §5.5)
    """

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
