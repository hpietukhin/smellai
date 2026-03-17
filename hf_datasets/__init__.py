"""HuggingFace Datasets integration for SmellAI research datasets."""

from .converter import rminer_to_hf, swe_refactor_to_hf, tdd_to_hf
from .chain_builder import build_commit_chains
from .preprocessor import deduplicate, split, filter_by, save, load
from .mlflow_bridge import hf_to_genai_records, load_for_evaluation

__all__ = [
    "rminer_to_hf",
    "swe_refactor_to_hf",
    "tdd_to_hf",
    "build_commit_chains",
    "deduplicate",
    "split",
    "filter_by",
    "save",
    "load",
    "hf_to_genai_records",
    "load_for_evaluation",
]
