#!/usr/bin/env python3
"""
Composite Refactoring Analysis Workflow.

This workflow analyzes the RefactoringMiner manifest data to detect and explore
composite refactorings (commits containing multiple refactoring operations).
It combines logic inspired by 'detect_composites.py' and 'explore_rminer.py'.

Usage:
    uv run workflows/composite_analysis_workflow.py --manifest rminer_data/manifest.json
"""

import argparse
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import matplotlib.pyplot as plt

from domain.rules import DEPENDENCY_RULES
from workflows.utils import configure_logging, load_manifest, save_matplotlib_graph

configure_logging()
logger = logging.getLogger(__name__)


# Mapping from Refactoring Type to the Smells they typically address
REFACTORING_TO_SMELL_MAP = {
    "Extract Method": ["Long Method", "Complex Method"],
    "Extract Variable": ["Complex Method"],
    "Decompose Conditional": ["Complex Method", "Conditional Complexity"],
    "Introduce Parameter Object": ["Long Parameter List"],
    "Extract Class": ["Large Class", "God Class"],
    "Extract Subclass": ["Large Class", "God Class"],
    "Extract Interface": ["Large Class", "God Class"],
    "Consolidate Conditional Expression": ["Duplicated Conditions"],
    "Consolidate Duplicate Conditional Fragments": ["Duplicated Conditions"],
    # Add more mappings as needed
}


def smells_for_refactoring(refactoring_type: str) -> list[str]:
    """Return smell types targeted by a refactoring type label."""
    return [
        smell
        for label, smells in REFACTORING_TO_SMELL_MAP.items()
        if label in refactoring_type
        for smell in smells
    ]


def get_dependencies_for_refactorings(refactoring_types: List[str]) -> Dict[str, Any]:
    """Analyze positive/negative smell dependencies for refactoring types."""
    analysis = {"positive": Counter(), "negative": Counter(), "targeted_smells": set()}

    for refactoring_type in refactoring_types:
        matched_smells = smells_for_refactoring(refactoring_type)
        analysis["targeted_smells"].update(matched_smells)

        for smell in matched_smells:
            rules = DEPENDENCY_RULES.get(smell, {})
            analysis["positive"].update(rules.get("positive", []))
            analysis["negative"].update(rules.get("negative", []))

    return analysis


def visualize_dependencies(
    refactoring_types: List[str], output_file: str = "refactoring_dependencies.png"
) -> None:
    """
    Visualize refactoring dependencies using NetworkX.
    """
    G = nx.DiGraph()

    # Collect nodes and edges
    refactorings = set(refactoring_types)
    smells = set()

    # Add Refactoring -> Targeted Smell edges
    for ref_type in refactorings:
        G.add_node(ref_type, type="refactoring", color="lightblue")
        for smell in smells_for_refactoring(ref_type):
            smells.add(smell)
            G.add_node(smell, type="smell", color="lightgreen")
            G.add_edge(ref_type, smell, label="targets", color="black")

    # Add Smell -> Smell dependencies
    for smell in smells:
        if smell in DEPENDENCY_RULES:
            rules = DEPENDENCY_RULES[smell]

            # Positive dependencies (Smells that might be removed)
            for pos_smell in rules.get("positive", []):
                if (
                    pos_smell in smells
                ):  # Only draw if both nodes exist to keep graph clean
                    G.add_edge(smell, pos_smell, label="positive", color="green")

            # Negative dependencies (Smells that might be created)
            for neg_smell in rules.get("negative", []):
                if neg_smell in smells:
                    G.add_edge(smell, neg_smell, label="negative", color="red")

    if G.number_of_nodes() == 0:
        logger.warning("No nodes to visualize.")
        return

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Draw nodes
    node_colors = [G.nodes[n].get("color", "gray") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Draw edges
    edge_colors = [G.edges[e].get("color", "black") for e in G.edges()]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowsize=20)

    # Draw edge labels (optional, can be cluttered)
    # edge_labels = nx.get_edge_attributes(G, "label")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Refactoring Dependencies Graph")
    save_matplotlib_graph(output_file)



def analyze_composites(pairs: List[Dict[str, Any]], draw_graph: bool = False) -> None:
    """
    Analyze pairs to detect and report composite refactorings.

    A composite refactoring is defined here as:
    1. A single pair entry containing multiple refactoring types (separated by '|').
    2. Multiple pairs belonging to the same commit.
    """
    logger.info(f"Analyzing {len(pairs)} pairs for composite refactorings...")

    # 1. Single-entry composites (multiple types in one entry)
    single_entry_composites = []
    all_refactoring_types = set()

    for pair in pairs:
        ref_type = pair.get("refactoring_type", "")
        if ref_type:
            # Split by | and add to set
            types = ref_type.split("|")
            all_refactoring_types.update(types)

        if "|" in ref_type:
            single_entry_composites.append(pair)

    logger.info(
        f"Found {len(single_entry_composites)} pairs with multiple refactoring types (Single-entry composites)."
    )

    # 2. Group by commit to find Multi-entry composites
    commits = defaultdict(list)
    for pair in pairs:
        commit_sha = pair.get("commit_sha")
        if commit_sha:
            commits[commit_sha].append(pair)

    multi_entry_composites = {sha: p for sha, p in commits.items() if len(p) > 1}
    logger.info(
        f"Found {len(multi_entry_composites)} commits with multiple pair entries (Multi-entry composites)."
    )

    if draw_graph:
        logger.info("Generating dependency graph for all found refactoring types...")
        visualize_dependencies(list(all_refactoring_types))

    # Detailed Report
    print("\n" + "=" * 60)
    print("COMPOSITE REFACTORING ANALYSIS")
    print("=" * 60)

    print(f"\nTotal Pairs Processed: {len(pairs)}")
    print(f"Single-entry Composites: {len(single_entry_composites)}")
    print(f"Multi-entry Composites (Commits): {len(multi_entry_composites)}")

    # Intersection (Commits that are both)
    composite_commits = set(multi_entry_composites.keys())
    for pair in single_entry_composites:
        if pair.get("commit_sha"):
            composite_commits.add(pair.get("commit_sha"))

    print(f"Total Unique Composite Commits: {len(composite_commits)}")

    if single_entry_composites:
        print("\nTop 5 Single-entry Composites (by refactoring count):")
        # Sort by number of refactorings
        sorted_single = sorted(
            single_entry_composites,
            key=lambda p: len(p.get("refactoring_type", "").split("|")),
            reverse=True,
        )
        for i, pair in enumerate(sorted_single[:5]):
            types = pair.get("refactoring_type", "").split("|")
            print(
                f"{i + 1}. Commit: {pair.get('commit_sha')[:8]} - {len(types)} refactorings"
            )
            print(f"   Types: {', '.join(types[:3])}{'...' if len(types) > 3 else ''}")
            print(f"   Repo: {pair.get('repository')}")

            # Dependency Analysis
            deps = get_dependencies_for_refactorings(types)
            if deps["targeted_smells"]:
                print(f"   Targeted Smells: {', '.join(deps['targeted_smells'])}")
                if deps["positive"]:
                    print(
                        f"   Potential Positive Effects (Smells Removed): {', '.join([f'{k}({v})' for k, v in deps['positive'].most_common(3)])}"
                    )
                if deps["negative"]:
                    print(
                        f"   Potential Negative Effects (Smells Created): {', '.join([f'{k}({v})' for k, v in deps['negative'].most_common(3)])}"
                    )

    if multi_entry_composites:
        print("\nTop 5 Multi-entry Composites (by pair count):")
        sorted_multi = sorted(
            multi_entry_composites.items(), key=lambda item: len(item[1]), reverse=True
        )
        for i, (sha, p_list) in enumerate(sorted_multi[:5]):
            print(f"{i + 1}. Commit: {sha[:8]} - {len(p_list)} pairs")
            repo = p_list[0].get("repository")
            print(f"   Repo: {repo}")

            # Collect all refactoring types in this commit
            all_types = []
            for p in p_list:
                all_types.extend(p.get("refactoring_type", "").split("|"))

            # Dependency Analysis
            deps = get_dependencies_for_refactorings(all_types)
            if deps["targeted_smells"]:
                print(
                    f"   Targeted Smells: {', '.join(list(deps['targeted_smells'])[:5])}..."
                )
                if deps["positive"]:
                    print(
                        f"   Potential Positive Effects: {', '.join([f'{k}({v})' for k, v in deps['positive'].most_common(3)])}"
                    )
                if deps["negative"]:
                    print(
                        f"   Potential Negative Effects: {', '.join([f'{k}({v})' for k, v in deps['negative'].most_common(3)])}"
                    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze composite refactorings in manifest."
    )
    parser.add_argument(
        "--manifest", default=os.environ.get("RMINER_MANIFEST_PATH", "rminer_data/manifest.json"), help="Path to manifest.json"
    )
    parser.add_argument(
        "--draw-graph", action="store_true", help="Draw dependency graph"
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    pairs = load_manifest(manifest_path)

    analyze_composites(pairs, draw_graph=args.draw_graph)

    return 0


if __name__ == "__main__":
    sys.exit(main())
