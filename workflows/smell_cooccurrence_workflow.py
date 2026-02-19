#!/usr/bin/env python3
"""
Smell Co-occurrence Analysis Workflow.

This workflow analyzes the smell co-occurrence manifest to visualize
relationships between detected code smells based on dependency rules.

Usage:
    uv run workflows/smell_cooccurrence_workflow.py --manifest tests/test_data/smell_cooccurrence/smells_manifest.json
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx

from agents.dependency_analysis.agent import DEPENDENCY_RULES
from workflows.utils import configure_logging, load_manifest, save_matplotlib_graph

configure_logging()
logger = logging.getLogger(__name__)


def visualize_file_dependencies(
    filename: str, smells: List[Dict[str, Any]], output_file: str
) -> None:
    """
    Visualize smell dependencies for a single file using NetworkX.
    """
    G = nx.DiGraph()

    # Extract smell types present in this file
    present_smell_types = {s["type"] for s in smells}

    # Add nodes for all present smells
    for smell_type in present_smell_types:
        G.add_node(smell_type, color="lightblue", style="filled")

    # Add edges based on rules
    for smell in present_smell_types:
        if smell in DEPENDENCY_RULES:
            rules = DEPENDENCY_RULES[smell]

            # Positive dependencies (Refactoring 'smell' might solve 'pos_smell')
            for pos_smell in rules.get("positive", []):
                if pos_smell in present_smell_types:
                    G.add_edge(smell, pos_smell, color="green", label="solves")

            # Negative dependencies (Refactoring 'smell' might create 'neg_smell')
            for neg_smell in rules.get("negative", []):
                # We add the edge even if the target smell isn't currently present,
                # to show the risk. But we need to add the node if it's missing.
                if neg_smell not in present_smell_types:
                    G.add_node(neg_smell, color="lightsalmon", style="dashed")

                G.add_edge(smell, neg_smell, color="red", label="risks creating")

    if G.number_of_nodes() == 0:
        logger.warning(f"No nodes to visualize for {filename}.")
        return

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=1.5, iterations=50)

    # Draw nodes
    node_colors = [G.nodes[n].get("color", "gray") for n in G.nodes()]
    # node_styles = [G.nodes[n].get("style", "solid") for n in G.nodes()] # NetworkX draw doesn't support style list directly easily

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2500, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

    # Draw edges
    edge_colors = [G.edges[e].get("color", "black") for e in G.edges()]
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, arrows=True, arrowsize=20, width=1.5
    )

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f"Smell Dependencies: {filename}")
    save_matplotlib_graph(output_file)


def analyze_cooccurrences(manifest: Dict[str, Any]) -> None:
    """
    Analyze and visualize smell co-occurrences for each file in the manifest.
    """
    files = manifest.get("files", [])
    logger.info(f"Analyzing {len(files)} files from manifest...")

    for i, file_entry in enumerate(files):
        filename = file_entry.get("filename", f"file_{i}")
        smells = file_entry.get("smells", [])

        logger.info(f"Processing {filename} with {len(smells)} smells...")

        # Print summary
        print(f"\nFile: {filename}")
        print(f"  Smells: {', '.join([s['type'] for s in smells])}")

        # Visualize
        output_filename = f"smell_deps_{filename.replace('.java', '')}.png"
        visualize_file_dependencies(filename, smells, output_filename)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze smell co-occurrences.")
    parser.add_argument(
        "--manifest",
        default="tests/test_data/smell_cooccurrence/smells_manifest.json",
        help="Path to smells_manifest.json",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)

    analyze_cooccurrences(manifest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
