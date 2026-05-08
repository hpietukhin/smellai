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
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

from domain.dependency_graph import DependencyGraph
from domain.refactoring_tree import RefactoringTree, State
from domain.models import SmellEvent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Model: SmellEvent (canonical domain model)
# -----------------------------------------------------------------------------
# SmellEvent is used directly — no separate SmellInstance class.
# In-memory use: SmellEvent(smell_id=..., smell_type=..., file_path=..., severity=...).
# DB/session metadata belongs to swe_refactor.persistence.models.SmellEventRecord.


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
# Prioritizer CLI adapter
# -----------------------------------------------------------------------------


def _to_visualization_graph(dep_graph: DependencyGraph) -> nx.DiGraph:
    """Convert DependencyGraph to a visualization-friendly nx.DiGraph."""
    graph = nx.DiGraph()
    for smell_id in dep_graph.all_smell_ids():
        smell = SmellEvent(
            smell_id=smell_id,
            smell_type=dep_graph.smell_type_of(smell_id),
            file_path=dep_graph.node_data(smell_id).get("file_path", "Unknown"),
            line_number=int(dep_graph.node_data(smell_id).get("line_number", 0) or 0),
            severity=dep_graph.node_data(smell_id).get("severity", "LOW"),
        )
        graph.add_node(smell_id, data=smell)

    for source, target, data in dep_graph.graph.edges(data=True):
        relation = data.get("relation", "")
        graph.add_edge(
            source,
            target,
            type=relation,
            color="green" if relation == "positive" else "red",
        )
    return graph


def _severity_color(severity_score: int) -> str:
    if severity_score >= 3:
        return "#ff9999"  # High
    if severity_score == 2:
        return "#ffcc99"  # Medium
    return "#99ff99"  # Low


class SmellPrioritizer:
    """CLI wrapper around DependencyGraph + RefactoringTree.

    Keeps the visualization and CLI API stable.
    """

    def __init__(self, smells: List[SmellEvent]):
        self.smells = smells
        self.dep_graph = DependencyGraph.from_events(smells)
        self.graph = _to_visualization_graph(self.dep_graph)

    def calculate_priorities(self) -> List[Dict[str, Any]]:
        """Run greedy planner and return legacy priority list format."""
        initial = State(frozenset(e.smell_id for e in self.smells))
        tree = RefactoringTree(initial, self.dep_graph)
        plan = tree.greedy()

        sequence = []
        for i, action in enumerate(plan.actions):
            smell_type = self.dep_graph.smell_type_of(action.smell_id)
            data = self.dep_graph.node_data(action.smell_id)
            file_path = data.get("file_path", "")
            line_number = data.get("line_number", 0)
            sequence.append({
                "order": i + 1,
                "smell_id": action.smell_id,
                "smell_type": smell_type,
                "file_path": file_path,
                "line_number": line_number,
                "location": f"{file_path}:{line_number}",
                "pz_score": self.dep_graph.score(action.smell_id),
                "positive_impacts": len(self.dep_graph.positive_neighbors(action.smell_id)),
                "negative_impacts": len(self.dep_graph.negative_neighbors(action.smell_id)),
                "suggested_refactorings": [action.ref_type],
            })
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
        unique_smells = {
            self.graph.nodes[node]["data"].smell_type for node in self.graph.nodes()
        }

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
    create_user_path = "com.app.UserManager.createUser"
    samples = [
        ("1", "God Class",           "com.app.UserManager",                 "HIGH"),
        ("2", "Long Method",         create_user_path,                       "HIGH"),
        ("3", "Feature Envy",        "com.app.UserManager.validateAddress", "MEDIUM"),
        ("4", "Duplicated Code",     "com.app.UserManager.updateUser",      "LOW"),
        ("5", "Complex Method",      create_user_path,                       "MEDIUM"),
        ("6", "Long Parameter List", create_user_path,                       "LOW"),
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
