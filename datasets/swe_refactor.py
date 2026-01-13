"""SWE-Refactor dataset adaptee and adapter."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .base import Dataset


class SWERefactorDataset:
    """
    The Adaptee for SWE-Refactor dataset with its specific interface.

    SWE-Refactor contains 1,099 pure real-world Java refactorings from 18 projects.
    Each sample has 6 components:
    1. Target Method: original code before refactoring
    2. Refactoring Type: specific operation to apply
    3. Repository and Code Structure: multi-level info
    4. Developer-Written Code: ground truth refactored code
    5. Build Configuration: commit ID, JDK version, build commands
    6. Test Coverage: coverage data from JaCoCo
    """

    def __init__(self, data_path: str, limit: int | None = None):
        """
        Initialize SWE-Refactor dataset.

        Args:
            data_path: Path to SWE-Refactor.zip or extracted directory
            limit: Optional limit on number of records
        """
        self.data_path = Path(data_path)
        self.limit = limit
        self._cached_records = None

    def build_swe_records(self) -> list[dict]:
        """
        Build records in SWE-Refactor format with comprehensive context.

        Returns:
            list[dict]: Records in SWE-Refactor format with structure:
                {
                    "inputs": {
                        "pair_id": str,
                        "target_method": str,
                        "refactoring_type": str,
                        "context": {
                            "class_content": str,
                            "class_hierarchy": dict,
                            "callers": list,
                            "callees": list,
                            "project_structure": dict,
                            "build_config": dict,
                        }
                    },
                    "expectations": {
                        "developer_written_code": str,
                        "refactoring_metadata": dict,
                    },
                    "tags": {
                        "repository": str,
                        "commit_id": str,
                        "test_coverage": dict,
                    }
                }
        """
        if self._cached_records is not None:
            return self._cached_records

        records = []

        # Check if data_path is a ZIP file or directory
        if self.data_path.suffix == '.zip':
            records = self._load_from_zip()
        else:
            records = self._load_from_directory()

        if self.limit:
            records = records[:self.limit]

        self._cached_records = records
        return self._cached_records

    def _load_from_zip(self) -> list[dict]:
        """
        Load records from SWE-Refactor.zip file.

        The ZIP file should contain JSON files with refactoring data.
        """
        records = []

        try:
            with zipfile.ZipFile(self.data_path, 'r') as zip_file:
                # Find all JSON files in the ZIP
                json_files = [
                    name for name in zip_file.namelist()
                    if name.endswith('.json')
                ]

                for json_file in json_files:
                    with zip_file.open(json_file) as f:
                        data = json.load(f)
                        # Convert SWE-Refactor format to our standard format
                        record = self._convert_swe_format(data)
                        if record:
                            records.append(record)

        except (zipfile.BadZipFile, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Failed to load SWE-Refactor data from ZIP: {e}")

        return records

    def _load_from_directory(self) -> list[dict]:
        """
        Load records from extracted SWE-Refactor directory.
        """
        records = []

        try:
            # Find all JSON files in the directory
            json_files = list(self.data_path.rglob('*.json'))

            for json_file in json_files:
                with open(json_file) as f:
                    data = json.load(f)
                    # Convert SWE-Refactor format to our standard format
                    record = self._convert_swe_format(data)
                    if record:
                        records.append(record)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Failed to load SWE-Refactor data from directory: {e}")

        return records

    def _convert_swe_format(self, data: dict) -> dict | None:
        """
        Convert SWE-Refactor format to standardized record format.

        Args:
            data: Raw data from SWE-Refactor JSON

        Returns:
            dict | None: Converted record, or None if invalid
        """
        try:
            # Extract the 6 components from SWE-Refactor format
            # Note: Field names differ from design doc - using actual dataset fields
            source_before = data.get('sourceCodeBeforeRefactoring', '')
            source_after = data.get('sourceCodeAfterRefactoring', '')
            refactoring_type = data.get('type', '')
            refactoring_desc = data.get('description', '')
            commit_id = data.get('commitId', '')
            project_name = data.get('projectName', '')
            compile_command = data.get('compileCommand', '')
            compile_jdk = data.get('compileJDK', 11)
            has_test = data.get('hasTestC', False)
            coverage_info = data.get('coverageInfo', {})

            # Extract file paths
            file_path_before = data.get('filePathBefore', '')
            file_path_after = data.get('filePathAfter', '')

            # Extract full file context
            source_before_whole = data.get('sourceCodeBeforeForWhole', '')
            source_after_whole = data.get('sourceCodeAfterForWhole', '')

            # Create unique ID
            unique_id = data.get('uniqueId', commit_id)

            # Build configuration
            build_config = {
                'commit_id': commit_id,
                'jdk_version': compile_jdk,
                'build_command': compile_command,
                'has_tests': has_test,
            }

            # Create standardized record
            record = {
                'inputs': {
                    'pair_id': unique_id,
                    'target_method': source_before,
                    'refactoring_type': refactoring_type,
                    'context': {
                        'class_content': source_before_whole,
                        'class_hierarchy': {},  # Not provided in actual dataset
                        'callers': [],  # Available in callInfo but needs parsing
                        'callees': [],  # Available in invokedMethodSet
                        'project_structure': {},  # Not directly available
                        'build_config': build_config,
                        'file_path_before': file_path_before,
                        'file_path_after': file_path_after,
                    }
                },
                'expectations': {
                    'developer_written_code': source_after,
                    'code_after_whole': source_after_whole,
                    'refactoring_metadata': {
                        'type': refactoring_type,
                        'description': refactoring_desc,
                        'is_compound': '+' in refactoring_type,
                        'is_pure': data.get('isPureRefactoring', False),
                    },
                },
                'tags': {
                    'repository': project_name,
                    'commit_id': commit_id,
                    'test_coverage': coverage_info,
                    'dataset_source': 'swe-refactor',
                    'has_tests': has_test,
                }
            }

            return record

        except (KeyError, TypeError) as e:
            print(f"Warning: Failed to convert SWE-Refactor record: {e}")
            print(f"Available keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
            return None

    def get_swe_metadata(self) -> dict[str, str]:
        """
        Get SWE-Refactor-specific metadata.

        Returns:
            dict: Metadata including total refactorings count and types
        """
        records = self.build_swe_records()

        # Count refactoring types
        atomic_count = 0
        compound_count = 0
        for record in records:
            refactoring_type = record.get('inputs', {}).get('refactoring_type', '')
            if '+' in refactoring_type:
                compound_count += 1
            else:
                atomic_count += 1

        return {
            'total_refactorings': str(len(records)),
            'atomic_refactorings': str(atomic_count),
            'compound_refactorings': str(compound_count),
        }


class SWERefactorAdapter(Dataset, SWERefactorDataset):
    """
    The Adapter makes the SWERefactorDataset's interface compatible with
    the Dataset's interface via multiple inheritance.

    Example:
        >>> adapter = SWERefactorAdapter(
        ...     data_path="swe_refactor/SWE-Refactor.zip",
        ...     limit=10
        ... )
        >>> records = adapter.request()
        >>> print(f"Dataset: {adapter.get_dataset_name()}")
        >>> print(f"Records: {len(records)}")
    """

    def request(self) -> list[dict]:
        """
        Adapt SWE-Refactor format to common Dataset format.

        Includes repository-level context, build config, and test coverage.

        Returns:
            list[dict]: MLflow-compatible records
        """
        return self.build_swe_records()

    def get_dataset_name(self) -> str:
        """
        Get dataset name.

        Returns:
            str: Dataset name in format "swe-refactor-{limit}" or "swe-refactor-all"
        """
        return f"swe-refactor-{self.limit or 'all'}"

    def get_tags(self) -> dict[str, str]:
        """
        Get dataset-level tags/metadata.

        Returns:
            dict: Tags including source, type, and metadata
        """
        metadata = self.get_swe_metadata()
        return {
            'source': 'SWE-Refactor',
            'type': 'atomic_and_compound_refactorings',
            **metadata,
        }
