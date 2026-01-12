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
from typing import Dict, List, Any
from dataclasses import dataclass

import networkx as nx

from agents.dependency_analysis.agent import DEPENDENCY_RULES

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------


@dataclass
class SmellInstance:
    id: str
    smell_type: str
    location: str  # e.g., "com.example.MyClass" or "com.example.MyClass.myMethod"
    severity: str  # HIGH, MEDIUM, LOW
    description: str = ""

    @property
    def severity_score(self) -> int:
        # Map severity strings to numeric scores
        # Handle "critical", "major", "minor" from manifest as well
        s = self.severity.upper()
        if s in ["BLOCKER", "CRITICAL", "HIGH"]:
            return 3
        elif s in ["MAJOR", "MEDIUM"]:
            return 2
        else:
            return 1


# -----------------------------------------------------------------------------
# Knowledge Base: Impact Rules
# -----------------------------------------------------------------------------

# We use the centralized DEPENDENCY_RULES from the agent module.
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


class SmellPrioritizer:
    def __init__(self, smells: List[SmellInstance]):
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
            self.graph.add_node(smell.id, data=smell)

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
                            smell_a.id, smell_b.id, type="positive", color="green"
                        )

                    # Negative Impact (Red)
                    negative_impacts = rules.get("negative", [])
                    if smell_b.smell_type in negative_impacts:
                        self.graph.add_edge(
                            smell_a.id, smell_b.id, type="negative", color="red"
                        )

    def calculate_priorities(self) -> List[Dict[str, Any]]:
        """
        Calculates the priority sequence based on PZ (Positive Impact).
        PZ = Intrinsic Severity + Impact on other smells (Out-degree weights).

        # TODO SPEC-010: Implement cycle detection mechanism and max-step limit.
        # If refactoring A creates smell B, and refactoring B creates smell A,
        # mark as outlier and prevent infinite loops.
        # HIGH priority.
        # (See TECHNICAL_SPECIFICATION.md §4.4)

        # TODO SPEC-011: Investigate Airflow capabilities for handling problematic cyclic dependencies.
        # Research whether Airflow can help manage cyclic dependency situations.
        # LOW priority.
        # (See TECHNICAL_SPECIFICATION.md §4.4)
        """
        # We will simulate the "remove max PZ" process

        working_graph = self.graph.copy()
        sequence = []

        while working_graph.number_of_nodes() > 0:
            scores = {}
            for node in working_graph.nodes():
                smell = working_graph.nodes[node]["data"]

                # Base Score (Intrinsic)
                pz = smell.severity_score

                # Impact Score (Dependencies)
                # Count outgoing POSITIVE edges in the CURRENT graph
                impact_count = 0
                for _, _, data in working_graph.out_edges(node, data=True):
                    if data.get("type") == "positive":
                        impact_count += 1

                # We can weight the impact. Let's say helping another smell is worth 2 points.
                pz += impact_count * 2

                scores[node] = pz

            # Find max PZ
            if not scores:
                break

            best_node = max(scores, key=scores.get)
            best_score = scores[best_node]
            best_smell = working_graph.nodes[best_node]["data"]

            # Count impacts for reporting
            positive_impacts = 0
            negative_impacts = 0
            for _, _, data in working_graph.out_edges(best_node, data=True):
                if data.get("type") == "positive":
                    positive_impacts += 1
                elif data.get("type") == "negative":
                    negative_impacts += 1

            sequence.append(
                {
                    "order": len(sequence) + 1,
                    "smell_id": best_node,
                    "smell_type": best_smell.smell_type,
                    "location": best_smell.location,
                    "pz_score": best_score,
                    "positive_impacts": positive_impacts,
                    "negative_impacts": negative_impacts,
                }
            )

            # Remove from graph
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

            # Color by severity
            score = smell.severity_score
            if score >= 3:
                color = "#ff9999"  # Red-ish (High)
            elif score == 2:
                color = "#ffcc99"  # Orange-ish (Medium)
            else:
                color = "#99ff99"  # Green-ish (Low)

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


def generate_sample_data() -> List[SmellInstance]:
    """Generates sample data for demonstration."""
    return [
        SmellInstance("1", "God Class", "com.app.UserManager", "HIGH"),
        SmellInstance("2", "Long Method", "com.app.UserManager.createUser", "HIGH"),
        SmellInstance(
            "3", "Feature Envy", "com.app.UserManager.validateAddress", "MEDIUM"
        ),
        SmellInstance("4", "Duplicated Code", "com.app.UserManager.updateUser", "LOW"),
        SmellInstance(
            "5", "Complex Method", "com.app.UserManager.createUser", "MEDIUM"
        ),
        SmellInstance(
            "6", "Long Parameter List", "com.app.UserManager.createUser", "LOW"
        ),
        SmellInstance(
            "7", "Data Clumps", "com.app.OrderService", "MEDIUM"
        ),  # Different class
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

            # Handle smells_manifest.json format (dict with "files" list)
            if isinstance(data, dict) and "files" in data:
                for file_entry in data["files"]:
                    filename = file_entry.get("filename", "UnknownFile")
                    for i, smell in enumerate(file_entry.get("smells", [])):
                        # Construct a unique ID
                        smell_id = (
                            f"{filename}_{i}_{smell.get('type').replace(' ', '')}"
                        )
                        # Construct location
                        loc = f"{filename}:{smell.get('location', '')}"

                        smells.append(
                            SmellInstance(
                                id=smell_id,
                                smell_type=smell.get("type", "Unknown"),
                                location=loc,
                                severity=smell.get("severity", "LOW"),
                                description=smell.get("description", ""),
                            )
                        )
            # Handle flat list format (if used directly)
            elif isinstance(data, list):
                for item in data:
                    smells.append(
                        SmellInstance(
                            id=str(item.get("id", len(smells) + 1)),
                            smell_type=item.get("smell_type", "Unknown"),
                            location=item.get("location", "Unknown"),
                            severity=item.get("severity", "LOW"),
                            description=item.get("description", ""),
                        )
                    )
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
