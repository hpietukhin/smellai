#!/usr/bin/env python3
"""
Interactive Smell Prioritization Visualizer using NiceGUI and ECharts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import networkx as nx
from nicegui import ui, app

# Import logic from existing script
# We need to make sure the python path allows this import or we adjust PYTHONPATH
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.prioritize_smells import (
    SmellPrioritizer,
    SmellInstance,
    generate_sample_data,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrioritizationVisualizer:
    def __init__(self):
        self.smells: List[SmellInstance] = []
        self.prioritizer: SmellPrioritizer | None = None
        self.sequence: List[Dict[str, Any]] = []
        self.chart = None
        self.current_step = 0
        self.layout_pos = {}

        # UI Elements
        self.step_slider = None
        self.details_label = None
        self.sequence_table = None

    def load_data(self, content: str | list | dict = None):
        """Load smells from content or generate sample."""
        self.smells = []
        data = content

        if data is None:
            self.smells = generate_sample_data()
        else:
            if isinstance(data, str):
                data = json.loads(data)

            # Logic adapted from prioritize_smells.py main()
            if isinstance(data, dict) and "files" in data:
                for file_entry in data["files"]:
                    filename = file_entry.get("filename", "UnknownFile")
                    for i, smell in enumerate(file_entry.get("smells", [])):
                        smell_id = (
                            f"{filename}_{i}_{smell.get('type').replace(' ', '')}"
                        )
                        loc = f"{filename}:{smell.get('location', '')}"
                        self.smells.append(
                            SmellInstance(
                                id=smell_id,
                                smell_type=smell.get("type", "Unknown"),
                                location=loc,
                                severity=smell.get("severity", "LOW"),
                                description=smell.get("description", ""),
                            )
                        )
            elif isinstance(data, list):
                for item in data:
                    self.smells.append(
                        SmellInstance(
                            id=str(item.get("id", len(self.smells) + 1)),
                            smell_type=item.get("smell_type", "Unknown"),
                            location=item.get("location", "Unknown"),
                            severity=item.get("severity", "LOW"),
                            description=item.get("description", ""),
                        )
                    )

        self.prioritizer = SmellPrioritizer(self.smells)
        self.sequence = self.prioritizer.calculate_priorities()
        self.current_step = 0

        # Pre-calculate layout
        if self.prioritizer.graph.number_of_nodes() > 0:
            self.layout_pos = nx.spring_layout(
                self.prioritizer.graph, k=2.0, seed=42, iterations=100
            )
        else:
            self.layout_pos = {}

        self.update_chart()

    def get_echart_options(self) -> Dict[str, Any]:
        if not self.prioritizer or not self.layout_pos:
            logger.warning("No prioritizer or layout_pos - returning empty options")
            return {}

        graph = self.prioritizer.graph

        logger.info(
            f"Generating chart with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
        )

        # Determine resolved nodes based on current step
        # The sequence is the order of resolution.
        # If current_step = 0, nothing resolved.
        # If current_step = 1, first item in sequence is resolved.
        resolved_ids = set()
        for i in range(self.current_step):
            if i < len(self.sequence):
                resolved_ids.add(self.sequence[i]["smell_id"])

        # Create lookup for PZ scores and priority order
        pz_lookup = {item["smell_id"]: item for item in self.sequence}

        nodes = []
        for node_id in graph.nodes():
            smell = graph.nodes[node_id]["data"]
            x, y = self.layout_pos[node_id]

            # Get PZ score and priority order for this node
            pz_data = pz_lookup.get(node_id, {})
            pz_score = pz_data.get("pz_score", smell.severity_score)
            priority_order = pz_data.get("order", "?")
            positive_impacts = pz_data.get("positive_impacts", 0)
            negative_impacts = pz_data.get("negative_impacts", 0)

            # Visual attributes
            is_resolved = node_id in resolved_ids

            # Size based on PZ score (higher PZ = larger node)
            size = pz_score * 10 + 20

            color = "#999"  # Default gray
            border_color = "black"
            border_width = 1

            if not is_resolved:
                if smell.severity_score >= 3:
                    color = "#ff9999"  # Red
                elif smell.severity_score == 2:
                    color = "#ffcc99"  # Orange
                else:
                    color = "#99ff99"  # Green

                # Highlight current smell (highest priority remaining)
                if (
                    self.current_step < len(self.sequence)
                    and node_id == self.sequence[self.current_step]["smell_id"]
                ):
                    border_color = "#2196F3"  # Blue border for next smell
                    border_width = 3
            else:
                color = "#e0e0e0"  # Light gray for resolved

            # Label shows type + priority order + PZ score
            label_text = f"#{priority_order} {smell.smell_type}\nPZ={pz_score}"

            nodes.append(
                {
                    "id": node_id,
                    "name": label_text,
                    "value": pz_score,
                    "x": x * 500,
                    "y": y * 500,
                    "symbolSize": size,
                    "itemStyle": {
                        "color": color,
                        "borderColor": border_color,
                        "borderWidth": border_width,
                    },
                    "label": {"show": True, "formatter": "{b}", "fontSize": 10},
                    # Custom data for click handler and tooltip
                    "custom_data": {
                        "location": smell.location,
                        "description": smell.description,
                        "severity": smell.severity,
                        "pz_score": pz_score,
                        "priority_order": priority_order,
                        "positive_impacts": positive_impacts,
                        "negative_impacts": negative_impacts,
                        "severity_score": smell.severity_score,
                    },
                }
            )

        edges = []
        for u, v, data in graph.edges(data=True):
            edge_type = data.get("type", "")

            # Positive dependencies = green, Negative = red
            if edge_type == "positive":
                color = "#4CAF50"  # Green
                line_type = "solid"
            elif edge_type == "negative":
                color = "#F44336"  # Red
                line_type = "dashed"
            else:
                color = "#999"
                line_type = "solid"

            # Fade edges if source or target is resolved
            if u in resolved_ids or v in resolved_ids:
                color = "#eee"

            # Show edge labels for positive/negative
            show_label = edge_type in ["positive", "negative"]
            label_text = (
                "+"
                if edge_type == "positive"
                else "−"
                if edge_type == "negative"
                else ""
            )

            edges.append(
                {
                    "source": u,
                    "target": v,
                    "lineStyle": {
                        "color": color,
                        "width": 2,
                        "curveness": 0.1,
                        "type": line_type,
                    },
                    "label": {
                        "show": show_label
                        and (u not in resolved_ids and v not in resolved_ids),
                        "formatter": label_text,
                        "fontSize": 14,
                        "fontWeight": "bold",
                    },
                }
            )

        # Build title with current step info
        title_text = "Smell Dependency Graph - PZ Prioritization"
        if self.current_step > 0:
            title_text += f" (Step {self.current_step}/{len(self.sequence)})"

        option = {
            "title": {
                "text": title_text,
                "subtext": "PZ = Severity + (Positive Impacts × 2)\nGreen edges = positive deps, Red dashed = negative deps",
                "left": "center",
            },
            "tooltip": {"show": True, "formatter": "{b}<br/>PZ Score: {c}"},
            "legend": {
                "data": [
                    "Positive Impact (+)",
                    "Negative Impact (−)",
                    "High Severity",
                    "Medium Severity",
                    "Low Severity",
                ],
                "bottom": 10,
            },
            "series": [
                {
                    "type": "graph",
                    "layout": "none",
                    "data": nodes,
                    "links": edges,
                    "roam": True,
                    "label": {"position": "right"},
                    "emphasis": {"focus": "adjacency", "label": {"show": True}},
                }
            ],
        }
        return option

    def update_chart(self):
        if self.chart:
            # Update options in-place because .options has no setter
            new_options = self.get_echart_options()
            self.chart.options.clear()
            self.chart.options.update(new_options)
            self.chart.update()

        if self.step_slider:
            self.step_slider.value = self.current_step
            self.step_slider.max = len(self.sequence)

        # Update sequence table
        if self.sequence_table and self.sequence:
            table_md = "| # | Smell | PZ | +/− |\n|---|---|---|---|\n"
            for item in self.sequence[:10]:  # Show top 10
                order = item["order"]
                smell_type = item["smell_type"]
                pz_score = item["pz_score"]
                pos = item["positive_impacts"]
                neg = item["negative_impacts"]

                # Highlight current step
                if order == self.current_step + 1:
                    table_md += f"| **{order}** | **{smell_type}** | **{pz_score}** | **+{pos}/−{neg}** |\n"
                else:
                    table_md += (
                        f"| {order} | {smell_type} | {pz_score} | +{pos}/−{neg} |\n"
                    )

            if len(self.sequence) > 10:
                table_md += f"\n_... and {len(self.sequence) - 10} more_"

            self.sequence_table.content = table_md

    def on_step_change(self, e):
        self.current_step = e.value
        self.update_chart()

    def handle_click(self, e):
        if e.args.get("componentType") == "series" and e.args.get("dataType") == "node":
            data_index = e.args.get("dataIndex")
            # Retrieve node data from options (assuming order matches)
            node_data = self.chart.options["series"][0]["data"][data_index]
            custom = node_data.get("custom_data", {})

            # Build PZ calculation explanation
            severity_score = custom.get("severity_score", 0)
            positive_impacts = custom.get("positive_impacts", 0)
            negative_impacts = custom.get("negative_impacts", 0)
            pz_score = custom.get("pz_score", 0)

            pz_formula = f"PZ = {severity_score} (severity) + {positive_impacts} × 2 (positive impacts) = {pz_score}"

            self.details_label.content = (
                f"### Priority #{custom.get('priority_order')}\n\n"
                f"**Type:** {node_data.get('id', '').split('_')[2] if '_' in node_data.get('id', '') else 'Unknown'}\n\n"
                f"**Location:** `{custom.get('location')}`\n\n"
                f"**Severity:** {custom.get('severity')} (score={severity_score})\n\n"
                f"---\n\n"
                f"**PZ Score:** {pz_score}\n\n"
                f"**Formula:** `{pz_formula}`\n\n"
                f"**Positive Dependencies:** {positive_impacts} (helps resolve other smells)\n\n"
                f"**Negative Dependencies:** {negative_impacts} (may create new smells)\n\n"
                f"---\n\n"
                f"**Description:** {custom.get('description') or 'N/A'}"
            )
        else:
            self.details_label.content = (
                "Click on a smell node to see prioritization details."
            )


visualizer = PrioritizationVisualizer()


@ui.page("/")
def main_page():
    # Header
    with ui.header(elevated=True).classes("items-center justify-between"):
        ui.label("Smell Prioritization Visualizer").classes("text-h6")

    # Left Drawer for Controls
    with ui.left_drawer(top_corner=True, bottom_corner=True).classes("p-4"):
        ui.markdown("### Controls")
        ui.button(
            "Load Sample Data", on_click=lambda: (visualizer.load_data(None))
        ).classes("w-full")

        ui.separator().classes("my-4")

        ui.markdown("### Playback")
        visualizer.step_slider = ui.slider(
            min=0, max=10, value=0, on_change=visualizer.on_step_change
        ).props("label-always")

        with ui.row().classes("w-full justify-between"):
            ui.button(
                "Prev",
                on_click=lambda: (
                    setattr(
                        visualizer,
                        "current_step",
                        max(0, visualizer.current_step - 1),
                    ),
                    visualizer.update_chart(),
                ),
            )
            ui.button(
                "Next",
                on_click=lambda: (
                    setattr(
                        visualizer,
                        "current_step",
                        min(len(visualizer.sequence), visualizer.current_step + 1),
                    ),
                    visualizer.update_chart(),
                ),
            )

        ui.separator().classes("my-4")

        ui.markdown("### Priority Sequence")
        with ui.scroll_area().classes("h-[200px] w-full"):
            visualizer.sequence_table = ui.markdown(
                "_Load data to see sequence_"
            ).classes("text-xs")

        ui.separator().classes("my-4")
        ui.markdown("### Smell Details")
        with ui.scroll_area().classes("w-full"):
            visualizer.details_label = ui.markdown(
                "Click on a smell node to see prioritization details."
            ).classes("text-sm")

        ui.separator().classes("my-4")
        ui.upload(
            label="Upload JSON Manifest",
            on_upload=lambda e: (
                visualizer.load_data(e.content.read().decode("utf-8"))
            ),
        ).classes("w-full")

    # Main content area with chart
    # Pre-load sample data to get initial options
    visualizer.load_data(None)
    initial_options = visualizer.get_echart_options()

    # Create a container that takes full viewport height minus header
    with ui.element("div").style(
        "width: 100%; height: calc(100vh - 50px); padding: 16px; box-sizing: border-box;"
    ):
        visualizer.chart = ui.echart(
            options=initial_options, on_point_click=visualizer.handle_click
        ).style("width: 100%; height: 100%; min-height: 600px;")

    # Update UI elements after chart is created
    visualizer.update_chart()


ui.run(title="Smell Visualizer", port=8080)
