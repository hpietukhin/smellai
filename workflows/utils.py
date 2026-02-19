"""Shared utilities for analysis workflows (manifest loading, matplotlib, logging)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logger with standard format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def load_manifest(manifest_path: Path) -> Any:
    """Load a manifest JSON file.

    Handles:
    - List format → returned as-is
    - Dict with "pairs" key → returns ``data["pairs"]``
    - Dict format → returned as-is (e.g. smell co-occurrence manifests)
    """
    if not manifest_path.exists():
        logger.error("Manifest file not found: %s", manifest_path)
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "pairs" in data:
        return data["pairs"]
    return data


def save_matplotlib_graph(output_file: str) -> None:
    """Save and close the current matplotlib figure."""
    import matplotlib.pyplot as plt  # local import to keep module lightweight

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file)
    logger.info("Graph saved to %s", output_file)
    plt.close()
