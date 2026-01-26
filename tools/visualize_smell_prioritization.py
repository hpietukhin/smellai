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

from scripts.prioritize_smells import SmellPrioritizer, SmellInstance, generate_sample_data

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
                        smell_id = f"{filename}_{i}_{smell.get('type').replace(' ', '')}"
                        loc = f"{filename}:{smell.get('location', '')}"
                        self.smells.append(SmellInstance(
                            id=smell_id,
                            smell_type=smell.get("type", "Unknown"),
                            location=loc,
                            severity=smell.get("severity", "LOW"),
                            description=smell.get("description", "")
                        ))
            elif isinstance(data, list):
                for item in data:
                    self.smells.append(SmellInstance(
                        id=str(item.get("id", len(self.smells) + 1)),
                        smell_type=item.get("smell_type", "Unknown"),
                        location=item.get("location", "Unknown"),
                        severity=item.get("severity", "LOW"),
                        description=item.get("description", "")
                    ))
        
        self.prioritizer = SmellPrioritizer(self.smells)
        self.sequence = self.prioritizer.calculate_priorities()
        self.current_step = 0
        
        # Pre-calculate layout
        if self.prioritizer.graph.number_of_nodes() > 0:
            self.layout_pos = nx.spring_layout(self.prioritizer.graph, k=2.0, seed=42, iterations=100)
        else:
            self.layout_pos = {}

        self.update_chart()

    def get_echart_options(self) -> Dict[str, Any]:
        if not self.prioritizer or not self.layout_pos:
            return {}

        graph = self.prioritizer.graph
        
        # Determine resolved nodes based on current step
        # The sequence is the order of resolution.
        # If current_step = 0, nothing resolved.
        # If current_step = 1, first item in sequence is resolved.
        resolved_ids = set()
        for i in range(self.current_step):
            if i < len(self.sequence):
                resolved_ids.add(self.sequence[i]["smell_id"])

        nodes = []
        for node_id in graph.nodes():
            smell = graph.nodes[node_id]["data"]
            x, y = self.layout_pos[node_id]
            
            # Visual attributes
            is_resolved = node_id in resolved_ids
            
            # Size based on severity (1, 2, 3) -> (20, 40, 60)
            size = (smell.severity_score) * 15 + 10
            
            color = "#999" # Default gray
            if not is_resolved:
                if smell.severity_score >= 3:
                     color = "#ff9999"  # Red
                elif smell.severity_score == 2:
                    color = "#ffcc99"  # Orange
                else:
                    color = "#99ff99"  # Green
            else:
                 color = "#e0e0e0" # Light gray for resolved

            nodes.append({
                "id": node_id,
                "name": smell.smell_type, # Label shown
                "value": smell.severity_score,
                "x": x * 500, # Scale up
                "y": y * 500,
                "symbolSize": size,
                "itemStyle": {"color": color},
                "label": {"show": True, "formatter": "{b}"},
                # Custom data for click handler
                "custom_data": {
                    "location": smell.location,
                    "description": smell.description,
                    "severity": smell.severity
                }
            })

        edges = []
        for u, v, data in graph.edges(data=True):
            color = data.get("color", "black")
            # Fade edges if source or target is resolved
            if u in resolved_ids or v in resolved_ids:
                color = "#eee"
                
            edges.append({
                "source": u,
                "target": v,
                "lineStyle": {"color": color, "width": 2, "curveness": 0.1},
                 "label": {"show": False}
            })

        option = {
            "title": {"text": "Smell Dependency Graph"},
            "tooltip": {"show": True},
            "series": [
                {
                    "type": "graph",
                    "layout": "none",
                    "data": nodes,
                    "links": edges,
                    "roam": True,
                    "label": {"position": "right"},
                    "emphasis": {
                        "focus": "adjacency",
                        "label": {"show": True}
                    }
                }
            ]
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

    def on_step_change(self, e):
        self.current_step = e.value
        self.update_chart()

    def handle_click(self, e):
        if e.args.get("componentType") == "series" and e.args.get("dataType") == "node":
             data_index = e.args.get("dataIndex")
             # Retrieve node data from options (assuming order matches)
             node_data = self.chart.options["series"][0]["data"][data_index]
             custom = node_data.get("custom_data", {})
             
             self.details_label.content = (
                 f"**Type:** {node_data['name']}\n\n"
                 f"**Location:** `{custom.get('location')}`\n\n"
                 f"**Severity:** {custom.get('severity')}\n\n"
                 f"**Description:** {custom.get('description')}"
             )
        else:
            self.details_label.content = "Select a node to see details."

visualizer = PrioritizationVisualizer()

@ui.page('/')
def main_page():
    
    # Header
    with ui.header().classes(replace='row items-center') as header:
        ui.label('Smell Prioritization Visualizer').classes('text-lg font-bold')

    # Main Layout
    with ui.row().classes('w-full h-full'):
        # Sidebar / Controls
        with ui.column().classes('w-1/4 p-4 border-r'):
            ui.markdown("### Controls")
            ui.button("Load Sample Data", on_click=lambda: (visualizer.load_data(None))).classes('w-full')
            
            ui.separator().classes('my-4')
            
            ui.markdown("### Playback")
            visualizer.step_slider = ui.slider(min=0, max=10, value=0, on_change=visualizer.on_step_change).props('label-always')
            
            with ui.row().classes('w-full justify-between'):
                ui.button("Prev", on_click=lambda: (
                    setattr(visualizer, 'current_step', max(0, visualizer.current_step - 1)),
                    visualizer.update_chart()
                ))
                ui.button("Next", on_click=lambda: (
                    setattr(visualizer, 'current_step', min(len(visualizer.sequence), visualizer.current_step + 1)),
                    visualizer.update_chart()
                ))

            ui.separator().classes('my-4')
            ui.markdown("### Details")
            visualizer.details_label = ui.markdown("Select a node to see details.").classes('text-sm')
            
            ui.upload(label="Upload JSON Manifest", on_upload=lambda e: (
                visualizer.load_data(e.content.read().decode('utf-8'))
            )).classes('w-full mt-auto')

        # Chart Area
        with ui.column().classes('w-3/4 h-[80vh] p-4'):
             visualizer.chart = ui.echart(options={}, on_point_click=visualizer.handle_click).classes('w-full h-full')

    # Initialize with sample data
    visualizer.load_data(None)

ui.run(title="Smell Visualizer", port=8080)
