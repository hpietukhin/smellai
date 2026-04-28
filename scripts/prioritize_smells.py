#!/usr/bin/env python3
"""
Script to prioritize and visualize code smell refactoring sequences.

Based on the methodology:
1. Identify smells and their locations.
2. Determine dependencies (Positive Impact PZ) between smells.
   - If refactoring Smell A helps resolve Smell B, A has a positive impact on B.
3. Calculate Importance (PZ) for each smell.
4. Order refactorings by max(PZ).
5. Visualize the sequence and dependencies.

Usage:
    uv run scripts/prioritize_smells.py --input tests/test_data/smell_cooccurrence/smells_manifest.json --visualize
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List

import networkx as nx

from agents.dependency_analysis.scorer import STANDARD_SCORE, ScoringContext
from store.rules import DEPENDENCY_RULES
from swe_refactor.persistence.models import SmellEvent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Model: SmellEvent (imported from swe_refactor.persistence.models)
# -----------------------------------------------------------------------------
# SmellEvent is used directly — no separate SmellInstance class.
# In-memory use: SmellEvent(smell_id=..., smell_type=..., file_path=..., severity=...)
# DB use: same model with session_id/iteration/action populated before saving.


def smell_json_to_instances(data: Any) -> "List[SmellEvent]":
    """Parse SmellEvent list from JSON data (smells_manifest or flat list format)."""
    smells: List[SmellEvent] = []
    if isinstance(data, dict) and "files" in data:
        for file_entry in data["files"]:
            filename = file_entry.get("filename", "UnknownFile")
            for i, smell in enumerate(file_entry.get("smells", [])):
                smell_id = f"{filename}_{i}_{smell.get('type', '').replace(' ', '')}"
                smells.append(
                    SmellEvent(
                        smell_id=smell_id,
                        smell_type=smell.get("type", "Unknown"),
                        file_path=filename,
                        line_number=int(smell.get("location", 0) or 0),
                        severity=smell.get("severity", "LOW"),
                    )
                )
    elif isinstance(data, list):
        for item in data:
            smells.append(
                SmellEvent(
                    smell_id=str(item.get("id", len(smells) + 1)),
                    smell_type=item.get("smell_type", "Unknown"),
                    file_path=item.get("location", "Unknown"),
                    line_number=0,
                    severity=item.get("severity", "LOW"),
                )
            )
    return smells


# -----------------------------------------------------------------------------
# Knowledge Base: Impact Rules
# -----------------------------------------------------------------------------

# We use the centralized DEPENDENCY_RULES from store.rules.
# This ensures consistency across the project.
# DEPENDENCY_RULES structure:
# {
#     "SmellType": {
#         "positive": ["SmellA", "SmellB"], # Refactoring SmellType helps solve these
#         "negative": ["SmellC"]            # Refactoring SmellType might create these
#     }
# }

# -----------------------------------------------------------------------------
# Prioritizer Logic
# -----------------------------------------------------------------------------


def _count_edges_by_type(graph: nx.DiGraph, node: str, edge_type: str) -> int:
    return sum(
        1 for _, _, data in graph.out_edges(node, data=True)
        if data.get("type") == edge_type
    )


def _severity_color(severity_score: int) -> str:
    if severity_score >= 3:
        return "#ff9999"  # High
    if severity_score == 2:
        return "#ffcc99"  # Medium
    return "#99ff99"  # Low


class SmellPrioritizer:
    def __init__(self, smells: List[SmellEvent]):
        self.smells = smells
        self.graph = nx.DiGraph()
        self._build_dependency_graph()

    def _build_dependency_graph(self):
        """
        Builds a graph where nodes are smells and edges represent dependencies.
        - Green edges: Positive Impact (PZ) - Refactoring A helps B.
        - Red edges: Negative Impact (NZ) - Refactoring A might create B.
        """
        # Add all nodes
        for smell in self.smells:
            self.graph.add_node(smell.smell_id, data=smell)

        # Add edges based on rules and location
        for i, smell_a in enumerate(self.smells):
            for j, smell_b in enumerate(self.smells):
                if i == j:
                    continue

                # Check if they are in the same context (Class/File)
                loc_a = smell_a.location.split(":")[0]
                loc_b = smell_b.location.split(":")[0]

                in_same_context = loc_a == loc_b

                if in_same_context:
                    rules = DEPENDENCY_RULES.get(smell_a.smell_type, {})

                    # Positive Impact (Green)
                    positive_impacts = rules.get("positive", [])
                    if smell_b.smell_type in positive_impacts:
                        self.graph.add_edge(
                            smell_a.smell_id, smell_b.smell_id, type="positive", color="green"
                        )

                    # Negative Impact (Red)
                    negative_impacts = rules.get("negative", [])
                    if smell_b.smell_type in negative_impacts:
                        self.graph.add_edge(
                            smell_a.smell_id, smell_b.smell_id, type="negative", color="red"
                        )

    def calculate_priorities(
        self, score_fn: Callable = STANDARD_SCORE
    ) -> List[Dict[str, Any]]:
        """Greedy planner: at each step picks the highest-scoring available smell.

        Implements Algorithm 1 from the paper using the spec formula (Eq. 2):
          P_i^conc = f_i · w_sev · sev(s_i) + Σpos_out^conc − w_neg · Σneg_out^abs

        score_fn can be swapped for experiments via scorer() from scorer.py.

        # TODO SPEC-010: Implement cycle detection mechanism and max-step limit.
        # If refactoring A creates smell B, and refactoring B creates smell A,
        # mark as outlier and prevent infinite loops.
        # HIGH priority.
        # (See TECHNICAL_SPECIFICATION.md §4.4)
        """
        freq_map = Counter(s.smell_type for s in self.smells)
        working_graph = self.graph.copy()
        sequence = []

        while working_graph.number_of_nodes() > 0:
            scores = {}
            for node in working_graph.nodes():
                smell = working_graph.nodes[node]["data"]
                pos_out = _count_edges_by_type(working_graph, node, "positive")
                neg_out = _count_edges_by_type(working_graph, node, "negative")
                ctx = ScoringContext(
                    freq=freq_map[smell.smell_type],
                    pos_out=pos_out,
                    neg_out=neg_out,
                )
                scores[node] = score_fn(smell, ctx)

            if not scores:
                break

            best_node = max(scores, key=scores.get)
            best_smell = working_graph.nodes[best_node]["data"]
            pos_impacts = _count_edges_by_type(working_graph, best_node, "positive")
            neg_impacts = _count_edges_by_type(working_graph, best_node, "negative")

            sequence.append({
                "order": len(sequence) + 1,
                "smell_id": best_node,
                "smell_type": best_smell.smell_type,
                "location": best_smell.location,
                "pz_score": scores[best_node],
                "positive_impacts": pos_impacts,
                "negative_impacts": neg_impacts,
            })

            working_graph.remove_node(best_node)

        return sequence

    def visualize(self, output_path: Path):
        """Generates a diagram of the smell dependencies and importance."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
        except ImportError:
            LOGGER.error("matplotlib is required for visualization.")
            return

        if self.graph.number_of_nodes() == 0:
            LOGGER.warning("Graph is empty, nothing to visualize.")
            return

        plt.figure(figsize=(14, 10))

        # Layout
        pos = nx.spring_layout(self.graph, k=2.0, seed=42, iterations=100)

        # Define shape mapping for different smell types
        SHAPE_MAP = {
            "God Class": "o",           # circle
            "Long Method": "s",         # square
            "Feature Envy": "^",        # triangle up
            "Duplicated Code": "D",     # diamond
            "Complex Method": "v",      # triangle down
            "Long Parameter List": "p", # pentagon
            "Data Clumps": "h",         # hexagon
            "Lazy Class": "*",          # star
            "Speculative Generality": "P",  # plus (filled)
            "Message Chains": "X",      # x (filled)
            "Middle Man": "<",          # triangle left
            "Inappropriate Intimacy": ">",  # triangle right
            "Shotgun Surgery": "8",     # octagon
            "Divergent Change": "d",    # thin diamond
        }

        DEFAULT_SHAPE = "o"  # circle for unknown smell types

        # Group nodes by smell type and severity for batch drawing
        nodes_by_shape_and_color = {}
        labels = {}

        for node in self.graph.nodes():
            smell = self.graph.nodes[node]["data"]
            out_degree = self.graph.out_degree(node)
            size = (smell.severity_score + out_degree) * 600

            # Get shape for this smell type
            shape = SHAPE_MAP.get(smell.smell_type, DEFAULT_SHAPE)

            color = _severity_color(smell.severity_score)

            # Group by (shape, color) for batch drawing
            key = (shape, color)
            if key not in nodes_by_shape_and_color:
                nodes_by_shape_and_color[key] = {"nodes": [], "sizes": []}

            nodes_by_shape_and_color[key]["nodes"].append(node)
            nodes_by_shape_and_color[key]["sizes"].append(size)

            # Label: Type + Location (shortened)
            short_loc = smell.location.split(":")[0]
            if len(short_loc) > 20:
                short_loc = "..." + short_loc[-17:]
            labels[node] = f"{smell.smell_type}\n{short_loc}"

        # Draw nodes grouped by shape and color
        for (shape, color), data in nodes_by_shape_and_color.items():
            node_positions = [pos[node] for node in data["nodes"]]
            x_coords = [p[0] for p in node_positions]
            y_coords = [p[1] for p in node_positions]

            plt.scatter(
                x_coords,
                y_coords,
                s=data["sizes"],
                c=[color] * len(data["nodes"]),
                marker=shape,
                alpha=0.9,
                edgecolors="black",
                linewidths=1.5,
                zorder=3
            )

        # Draw edges with colors
        edge_colors = [self.graph[u][v]["color"] for u, v in self.graph.edges()]

        nx.draw_networkx_edges(
            self.graph,
            pos,
            width=1.5,
            alpha=0.6,
            edge_color=edge_colors,
            arrowsize=20,
            connectionstyle="arc3,rad=0.1"
        )
        nx.draw_networkx_labels(
            self.graph, pos, labels=labels, font_size=8, font_weight="bold"
        )

        # Build legend with all unique smell types present in the graph
        unique_smells = set(self.graph.nodes[node]["data"].smell_type for node in self.graph.nodes())

        legend_elements = []

        # Add smell type shapes to legend
        for smell_type in sorted(unique_smells):
            shape = SHAPE_MAP.get(smell_type, DEFAULT_SHAPE)
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=shape,
                    color="w",
                    label=smell_type,
                    markerfacecolor="gray",
                    markersize=10,
                    markeredgecolor="black",
                )
            )

        # Add separator
        if legend_elements:
            legend_elements.append(Line2D([0], [0], color="none", label=""))

        # Add severity colors
        legend_elements.extend([
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="High Severity (3)",
                markerfacecolor="#ff9999",
                markersize=10,
                markeredgecolor="black",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Medium Severity (2)",
                markerfacecolor="#ffcc99",
                markersize=10,
                markeredgecolor="black",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Low Severity (1)",
                markerfacecolor="#99ff99",
                markersize=10,
                markeredgecolor="black",
            ),
            Line2D([0], [0], color="none", label=""),
            Line2D([0], [0], color="green", lw=2, label="Positive Impact (Solves)"),
            Line2D([0], [0], color="red", lw=2, label="Negative Impact (Risks)"),
        ])

        plt.legend(handles=legend_elements, loc="upper right", title="Legend", fontsize=8)

        plt.title(
            "Code Smell Dependency & Importance Graph\n(Larger nodes = Higher Priority/PZ = Severity + Positive Impact)",
            fontsize=14,
        )
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        LOGGER.info(f"Visualization saved to {output_path}")
        plt.close()


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------


def generate_sample_data() -> List[SmellEvent]:
    """Generates sample data for demonstration."""
    samples = [
        ("1", "God Class",           "com.app.UserManager",                 "HIGH"),
        ("2", "Long Method",         "com.app.UserManager.createUser",      "HIGH"),
        ("3", "Feature Envy",        "com.app.UserManager.validateAddress", "MEDIUM"),
        ("4", "Duplicated Code",     "com.app.UserManager.updateUser",      "LOW"),
        ("5", "Complex Method",      "com.app.UserManager.createUser",      "MEDIUM"),
        ("6", "Long Parameter List", "com.app.UserManager.createUser",      "LOW"),
        ("7", "Data Clumps",         "com.app.OrderService",                "MEDIUM"),
    ]
    return [
        SmellEvent(smell_id=s[0], smell_type=s[1], file_path=s[2], severity=s[3])
        for s in samples
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Prioritize code smells for refactoring."
    )
    parser.add_argument("--input", type=Path, help="Input JSON file with smells")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prioritization.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization diagram"
    )
    parser.add_argument(
        "--viz-output",
        type=Path,
        default=Path("smell_priority_graph.png"),
        help="Visualization output path",
    )

    args = parser.parse_args()

    smells = []
    if args.input and args.input.exists():
        with open(args.input, "r") as f:
            data = json.load(f)
        smells = smell_json_to_instances(data)
    else:
        LOGGER.info("No input file provided or found. Using sample data.")
        smells = generate_sample_data()

    LOGGER.info(f"Processing {len(smells)} smells...")

    prioritizer = SmellPrioritizer(smells)

    # 1. Calculate Sequence
    sequence = prioritizer.calculate_priorities()

    # 2. Output Results
    LOGGER.info("\nRecommended Refactoring Sequence (Max PZ Strategy):")
    print(f"{'Order':<6} | {'PZ':<4} | {'Smell Type':<25} | {'Location'}")
    print("-" * 80)
    for item in sequence:
        print(
            f"{item['order']:<6} | {item['pz_score']:<4} | {item['smell_type']:<25} | {item['location']}"
        )

    with open(args.output, "w") as f:
        json.dump(sequence, f, indent=2)
    LOGGER.info(f"\nDetailed sequence saved to {args.output}")

    # 3. Visualize
    if args.visualize:
        prioritizer.visualize(args.viz_output)


if __name__ == "__main__":
    main()
