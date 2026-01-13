"""Dataset adapter pattern for unified dataset access.

This module provides a common interface for different refactoring datasets
using the adapter pattern with multiple inheritance.
"""

from .base import Dataset
from .rminer import RMinerDataset, RMinerAdapter
from .swe_refactor import SWERefactorDataset, SWERefactorAdapter

__all__ = [
    "Dataset",
    "RMinerDataset",
    "RMinerAdapter",
    "SWERefactorDataset",
    "SWERefactorAdapter",
]
