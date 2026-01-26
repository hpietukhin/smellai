#!/usr/bin/env python3
"""
Interactive Smell Prioritization Visualizer using NiceGUI and ECharts.

Loads agent execution data from analytics database and displays:
- Agent execution timeline with node invocations
- Smell dependency graph with current agent position
- Decision rationale (PZ scores, dependencies)
- Iteration details (before/after smells, outcomes)
- Tool call logs for debugging
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import glob as globlib

import networkx as nx
from nicegui import ui, app

# Import logic from existing script
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.prioritize_smells import (
    SmellPrioritizer,
    SmellInstance,
    generate_sample_data,
)

from swe_refactor.persistence.database import AnalyticsDB
from swe_refactor.persistence.models import (
    SmellEvent,
    RefactoringAttempt,
    ToolCall,
    TokenUsage,
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

        # Analytics database
        self.analytics_db: Optional[AnalyticsDB] = None
        self.current_session: Optional[str] = None
        self.sessions: List[str] = []

        # Agent execution data
        self.refactoring_attempts: List[RefactoringAttempt] = []
        self.tool_calls: List[ToolCall] = []
        self.smell_events_by_iteration: Dict[int, List[SmellEvent]] = {}

        # Current iteration for timeline playback
        self.current_iteration = 0
        self.max_iterations = 0

        # UI Elements
        self.step_slider = None
        self.details_label = None
        self.sequence_table = None
        self.iteration_details = None
        self.timeline_chart = None
        self.tool_logs_area = None
        self.diff_viewer = None
        self.manifest_info_label = None

        # Iteration controls (enabled only in database mode)
        self.iteration_slider = None
        self.iteration_prev_btn = None
        self.iteration_next_btn = None

    def load_database(self, db_path: str):
        """Load agent execution data from analytics database."""
        self.analytics_db = AnalyticsDB(db_path)

        # Get all sessions
        from sqlmodel import Session, select

        with Session(self.analytics_db.engine) as session:
            stmt = select(SmellEvent.session_id).distinct()
            self.sessions = list(session.exec(stmt).all())

        logger.info(f"Loaded database with {len(self.sessions)} sessions")

        if self.sessions:
            self.load_session(self.sessions[0])

    def load_session(self, session_id: str):
        """Load data for a specific session."""
        if not self.analytics_db:
            logger.error("No database loaded")
            return

        self.current_session = session_id
        logger.info(f"Loading session {session_id[:8]}")

        # Load refactoring attempts
        from sqlmodel import Session, select

        with Session(self.analytics_db.engine) as session:
            stmt = (
                select(RefactoringAttempt)
                .where(RefactoringAttempt.session_id == session_id)
                .order_by(RefactoringAttempt.iteration)
            )
            self.refactoring_attempts = list(session.exec(stmt).all())

            # Load tool calls
            stmt = (
                select(ToolCall)
                .where(ToolCall.session_id == session_id)
                .order_by(ToolCall.timestamp)
            )
            self.tool_calls = list(session.exec(stmt).all())

            # Load smell events by iteration
            self.smell_events_by_iteration = {}
            stmt = (
                select(SmellEvent)
                .where(SmellEvent.session_id == session_id)
                .order_by(SmellEvent.iteration, SmellEvent.smell_id)
            )
            all_events = list(session.exec(stmt).all())

            for event in all_events:
                if event.iteration not in self.smell_events_by_iteration:
                    self.smell_events_by_iteration[event.iteration] = []
                self.smell_events_by_iteration[event.iteration].append(event)

        self.max_iterations = len(self.refactoring_attempts)
        self.current_iteration = 0

        # Load smells from first iteration for prioritization visualization
        self.load_iteration_smells(0)

        # Update timeline chart
        if self.timeline_chart:
            timeline_options = self.get_timeline_options()
            if timeline_options:
                self.timeline_chart.options.clear()
                self.timeline_chart.options.update(timeline_options)
                self.timeline_chart.update()

        # Update iteration slider if callback registered
        if hasattr(self, "update_iteration_slider"):
            self.update_iteration_slider()

        # Enable iteration controls since we're in database mode
        self._enable_iteration_controls(True)

        logger.info(
            f"Loaded {self.max_iterations} iterations, {len(self.tool_calls)} tool calls"
        )

    def load_iteration_smells(self, iteration: int):
        """Load smells for a specific iteration and rebuild prioritization."""
        events = self.smell_events_by_iteration.get(iteration, [])

        # Filter only "detected" events for this iteration
        detected_events = [e for e in events if e.action.value == "detected"]

        # Convert to SmellInstance format
        self.smells = []
        for event in detected_events:
            self.smells.append(
                SmellInstance(
                    id=event.smell_id,
                    smell_type=event.smell_type,
                    location=f"{event.file_path}:{event.line_number}",
                    severity=event.severity,
                    description=f"{event.smell_type} at {event.file_path}:{event.line_number}",
                )
            )

        # Rebuild prioritization graph
        if self.smells:
            self.prioritizer = SmellPrioritizer(self.smells)
            self.sequence = self.prioritizer.calculate_priorities()
        else:
            self.prioritizer = None
            self.sequence = []

        self.current_step = 0

        # Pre-calculate layout
        if self.prioritizer and self.prioritizer.graph.number_of_nodes() > 0:
            self.layout_pos = nx.spring_layout(
                self.prioritizer.graph, k=2.0, seed=42, iterations=100
            )
        else:
            self.layout_pos = {}

        self.update_chart()

    def _enable_iteration_controls(self, enabled: bool):
        """Enable or disable iteration timeline controls."""
        if self.iteration_slider:
            self.iteration_slider.enabled = enabled
        if self.iteration_prev_btn:
            self.iteration_prev_btn.enabled = enabled
        if self.iteration_next_btn:
            self.iteration_next_btn.enabled = enabled

    def load_example_manifest(self, manifest_path: str):
        """Load smell data from example manifest JSON."""
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            logger.info(f"Loading manifest: {manifest.get('name', 'Unknown')}")
            self.load_data(manifest)

            # Disable iteration controls (not available for examples)
            self._enable_iteration_controls(False)

            # Update UI if available (set title/description)
            if hasattr(self, "manifest_info_label"):
                info = f"**{manifest.get('name', 'Example')}**\n\n"
                info += f"{manifest.get('description', '')}\n\n"
                info += f"Project: {manifest.get('project', 'N/A')} | Commit: {manifest.get('commit', 'N/A')[:8]}\n\n"

                if (
                    "refactoring_sequence" in manifest
                    and manifest["refactoring_sequence"]
                ):
                    info += "**Refactoring Sequence:**\n"
                    for i, step in enumerate(manifest["refactoring_sequence"], 1):
                        info += f"{i}. {step}\n"

                self.manifest_info_label.content = info

            ui.notify(f"Loaded: {manifest.get('name', 'Example')}", type="positive")

        except Exception as e:
            ui.notify(f"Error loading manifest: {e}", type="negative")
            logger.error(f"Manifest load error: {e}")

    def load_data(self, content: str | list | dict | None = None):
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
                    "label": {"show": True, "formatter": "{b}", "fontSize": 7},
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
                        "fontSize": 10,
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
                "textStyle": {"fontSize": 14},
                "subtextStyle": {"fontSize": 10},
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
                "textStyle": {"fontSize": 10},
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
                f"**Priority #{custom.get('priority_order')}**\n\n"
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

    def get_timeline_options(self) -> Dict[str, Any]:
        """Generate ECharts timeline showing agent node execution or refactoring attempts."""
        # Try to show tool calls if available
        if self.tool_calls:
            return self._get_tool_calls_timeline()

        # Fallback: show refactoring attempts timeline
        if self.refactoring_attempts:
            return self._get_refactoring_timeline()

        return {}

    def _get_tool_calls_timeline(self) -> Dict[str, Any]:
        """Generate timeline from tool calls."""
        # Group tool calls by node
        node_groups = {}
        for call in self.tool_calls:
            if call.node_name not in node_groups:
                node_groups[call.node_name] = []
            node_groups[call.node_name].append(call)

        # Create Gantt-style chart data
        categories = list(node_groups.keys())
        data = []

        # Get min timestamp for relative positioning
        min_time = min(call.timestamp for call in self.tool_calls)

        for i, (node_name, calls) in enumerate(node_groups.items()):
            for call in calls:
                start_offset = (call.timestamp - min_time).total_seconds()
                end_offset = start_offset + (call.duration_ms / 1000)

                data.append(
                    {
                        "name": f"{node_name} (iter {call.iteration})",
                        "value": [i, start_offset, end_offset, call.duration_ms],
                        "itemStyle": {
                            "color": self._get_node_color(node_name),
                        },
                    }
                )

        return {
            "title": {
                "text": "Agent Execution Timeline",
                "left": "center",
                "textStyle": {"fontSize": 12},
            },
            "tooltip": {"formatter": "{b}<br/>Duration: {c}ms"},
            "xAxis": {
                "type": "value",
                "name": "Time (seconds)",
                "nameTextStyle": {"fontSize": 10},
                "axisLabel": {"fontSize": 9},
            },
            "yAxis": {
                "type": "category",
                "data": categories,
                "name": "Agent Node",
                "nameTextStyle": {"fontSize": 10},
                "axisLabel": {"fontSize": 9},
            },
            "series": [
                {
                    "type": "custom",
                    "renderItem": """function (params, api) {
                        var categoryIndex = api.value(0);
                        var start = api.coord([api.value(1), categoryIndex]);
                        var end = api.coord([api.value(2), categoryIndex]);
                        var height = api.size([0, 1])[1] * 0.6;
                        
                        return {
                            type: 'rect',
                            shape: {
                                x: start[0],
                                y: start[1] - height / 2,
                                width: end[0] - start[0],
                                height: height
                            },
                            style: api.style()
                        };
                    }""",
                    "encode": {"x": [1, 2], "y": 0},
                    "data": data,
                }
            ],
        }

    def _get_refactoring_timeline(self) -> Dict[str, Any]:
        """Generate timeline from refactoring attempts."""
        data = []
        categories = ["Refactoring"]

        for i, attempt in enumerate(self.refactoring_attempts):
            color = "#4CAF50" if attempt.outcome == "success" else "#F44336"

            data.append(
                {
                    "name": f"Iter {attempt.iteration}: {attempt.refactoring_type}",
                    "value": [0, i, i + 0.8, attempt.iteration],
                    "itemStyle": {"color": color},
                }
            )

        return {
            "title": {
                "text": "Refactoring Attempts Timeline",
                "left": "center",
                "textStyle": {"fontSize": 12},
            },
            "tooltip": {"formatter": "{b}"},
            "xAxis": {
                "type": "value",
                "name": "Iteration",
                "nameTextStyle": {"fontSize": 10},
                "axisLabel": {"fontSize": 9},
            },
            "yAxis": {
                "type": "category",
                "data": categories,
                "nameTextStyle": {"fontSize": 10},
                "axisLabel": {"fontSize": 9},
            },
            "series": [
                {
                    "type": "custom",
                    "renderItem": """function (params, api) {
                        var categoryIndex = api.value(0);
                        var start = api.coord([api.value(1), categoryIndex]);
                        var end = api.coord([api.value(2), categoryIndex]);
                        var height = api.size([0, 1])[1] * 0.6;
                        
                        return {
                            type: 'rect',
                            shape: {
                                x: start[0],
                                y: start[1] - height / 2,
                                width: end[0] - start[0],
                                height: height
                            },
                            style: api.style()
                        };
                    }""",
                    "encode": {"x": [1, 2], "y": 0},
                    "data": data,
                }
            ],
        }

    def _get_node_color(self, node_name: str) -> str:
        """Get color for agent node."""
        colors = {
            "A0_setup": "#607D8B",
            "A1_detect_smells": "#2196F3",
            "A2_prioritize_smells": "#4CAF50",
            "A3_select_next_smell": "#FF9800",
            "A4_map_smell_to_refactoring": "#9C27B0",
            "A5_generate": "#F44336",
            "A6_verify": "#00BCD4",
        }
        return colors.get(node_name, "#999")

    def get_iteration_details_markdown(self) -> str:
        """Generate markdown showing details of current iteration."""
        if not self.refactoring_attempts or self.current_iteration >= len(
            self.refactoring_attempts
        ):
            return "_No iteration data available_"

        attempt = self.refactoring_attempts[self.current_iteration]
        events = self.smell_events_by_iteration.get(attempt.iteration, [])

        # Count events by action
        detected = len([e for e in events if e.action.value == "detected"])
        resolved = attempt.smells_resolved
        created = attempt.smells_created

        md = f"**Iteration {attempt.iteration}**\n\n"
        md += f"**Target Smell:** `{attempt.smell_id}`\n\n"
        md += f"**Refactoring Type:** {attempt.refactoring_type}\n\n"
        md += f"**Outcome:** {'✅ Success' if attempt.outcome == 'success' else '❌ ' + attempt.outcome}\n\n"
        md += f"**Retries:** {attempt.retries}\n\n"
        md += "---\n\n"
        md += f"**Smells Detected:** {detected}\n\n"
        md += f"**Smells Resolved:** {resolved} 🎯\n\n"
        md += f"**Smells Created:** {created} {'⚠️' if created > 0 else ''}\n\n"
        md += f"**Net Impact:** {resolved - created:+d}\n\n"

        # Add button to view diff if available
        if attempt.code_diff:
            md += "\n---\n\n"
            md += f"**Code Changes:** {len(attempt.code_diff.splitlines())} lines\n\n"
            md += "_See diff below_\n"

        return md

    def get_code_diff(self) -> str:
        """Get code diff for current iteration."""
        if not self.refactoring_attempts or self.current_iteration >= len(
            self.refactoring_attempts
        ):
            return ""

        attempt = self.refactoring_attempts[self.current_iteration]
        return attempt.code_diff or "_No diff available for this iteration_"

    def get_tool_logs_markdown(self) -> str:
        """Generate markdown showing tool calls for current iteration."""
        if not self.refactoring_attempts or self.current_iteration >= len(
            self.refactoring_attempts
        ):
            return "_No tool call data available_"

        iteration = self.refactoring_attempts[self.current_iteration].iteration
        calls = [c for c in self.tool_calls if c.iteration == iteration]

        if not calls:
            return "_No tool calls in this iteration_"

        md = f"**Tool Calls (Iteration {iteration})**\n\n"
        md += "| Node | Tool | Duration |\n|---|---|---|\n"

        for call in calls:
            md += (
                f"| {call.node_name} | {call.tool_name} | {call.duration_ms:.1f}ms |\n"
            )

        md += f"\n**Total:** {len(calls)} calls, {sum(c.duration_ms for c in calls):.1f}ms\n"

        return md

    def on_iteration_change(self, iteration: int):
        """Handle iteration change in timeline playback."""
        self.current_iteration = iteration
        self.load_iteration_smells(iteration)

        # Update iteration details
        if self.iteration_details:
            self.iteration_details.content = self.get_iteration_details_markdown()

        # Update tool logs
        if self.tool_logs_area:
            self.tool_logs_area.content = self.get_tool_logs_markdown()

        # Update diff viewer
        if self.diff_viewer:
            self.diff_viewer.content = self.get_code_diff()

    def update_chart(self):
        if self.chart:
            # Update options in-place because .options has no setter
            new_options = self.get_echart_options()
            if new_options:
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

        # Update iteration details if in database mode
        if self.analytics_db and self.iteration_details:
            self.iteration_details.content = self.get_iteration_details_markdown()

        if self.analytics_db and self.tool_logs_area:
            self.tool_logs_area.content = self.get_tool_logs_markdown()


visualizer = PrioritizationVisualizer()


@ui.page("/")
def main_page():
    # Header
    with ui.header(elevated=True).classes("items-center justify-between"):
        ui.label("AI Agent Execution Visualizer").classes("text-sm font-bold")

        # Session selector (shown when database is loaded)
        session_selector = (
            ui.select(
                options=[],
                label="Session",
                on_change=lambda e: visualizer.load_session(e.value)
                if e.value
                else None,
            )
            .classes("w-64 text-xs")
            .props("outlined dense")
        )
        session_selector.visible = False

    # Left Drawer for Controls
    with ui.left_drawer(top_corner=True, bottom_corner=True).classes("p-4"):
        ui.markdown("### Data Source").classes("text-xs")

        db_input = ui.input(
            label="Analytics DB Path",
            placeholder="test_analytics.db",
            value="test_analytics.db",
        ).classes("w-full text-xs")

        def load_db():
            try:
                visualizer.load_database(db_input.value)
                session_selector.options = visualizer.sessions
                session_selector.value = (
                    visualizer.sessions[0] if visualizer.sessions else None
                )
                session_selector.visible = True
                ui.notify(
                    f"Loaded {len(visualizer.sessions)} sessions", type="positive"
                )
            except Exception as e:
                ui.notify(f"Error loading database: {e}", type="negative")
                logger.error(f"Database load error: {e}")

        ui.button("Load Database", on_click=load_db, icon="database").classes(
            "w-full text-xs"
        )

        ui.button(
            "Load Sample Data",
            on_click=lambda: (
                visualizer.load_data(None),
                visualizer._enable_iteration_controls(False),
            ),
            icon="science",
        ).classes("w-full mt-2 text-xs")

        ui.separator().classes("my-4")

        # Example manifest loading
        ui.markdown("### Examples").classes("text-xs")

        # Get example manifest files
        manifest_dir = Path(__file__).parent / "example_manifests"
        example_files = (
            sorted(list(manifest_dir.glob("*.json"))) if manifest_dir.exists() else []
        )

        if example_files:
            example_options = {
                f.stem.replace("_", " ").title(): str(f) for f in example_files
            }

            ui.select(
                options=list(example_options.keys()),
                label="Load Example",
                on_change=lambda e: visualizer.load_example_manifest(
                    example_options[e.value]
                )
                if e.value
                else None,
            ).classes("w-full text-xs")

            # Info panel for loaded manifest
            with ui.scroll_area().classes("h-[120px] w-full mt-2"):
                visualizer.manifest_info_label = ui.markdown(
                    "_Select example above_"
                ).classes("text-[10px]")
        else:
            ui.label("No examples found").classes("text-[10px] text-gray-500")

        ui.separator().classes("my-4")

        # Iteration playback (for database mode)
        ui.markdown("### Iteration Timeline").classes("text-xs")

        visualizer.iteration_slider = (
            ui.slider(
                min=0,
                max=1,
                value=0,
                on_change=lambda e: visualizer.on_iteration_change(int(e.value)),
            )
            .props("label-always label='Iteration'")
            .classes("text-xs")
        )
        visualizer.iteration_slider.enabled = False  # Start disabled

        def update_iteration_slider():
            visualizer.iteration_slider.max = max(1, visualizer.max_iterations - 1)

        # Hook to update slider when database loads
        visualizer.update_iteration_slider = update_iteration_slider

        with ui.row().classes("w-full justify-between"):
            visualizer.iteration_prev_btn = ui.button(
                "Prev",
                on_click=lambda: (
                    visualizer.on_iteration_change(
                        max(0, visualizer.current_iteration - 1)
                    ),
                    setattr(
                        visualizer.iteration_slider,
                        "value",
                        visualizer.current_iteration,
                    ),
                ),
                icon="skip_previous",
            ).classes("text-xs")
            visualizer.iteration_prev_btn.enabled = False  # Start disabled

            visualizer.iteration_next_btn = ui.button(
                "Next",
                on_click=lambda: (
                    visualizer.on_iteration_change(
                        min(
                            visualizer.max_iterations - 1,
                            visualizer.current_iteration + 1,
                        )
                    ),
                    setattr(
                        visualizer.iteration_slider,
                        "value",
                        visualizer.current_iteration,
                    ),
                ),
                icon="skip_next",
            ).classes("text-xs")
            visualizer.iteration_next_btn.enabled = False  # Start disabled

        ui.separator().classes("my-4")

        # Smell prioritization playback
        ui.markdown("### Smell Priority").classes("text-xs")
        visualizer.step_slider = (
            ui.slider(min=0, max=10, value=0, on_change=visualizer.on_step_change)
            .props("label-always label='Step'")
            .classes("text-xs")
        )

        with ui.row().classes("w-full justify-between"):
            ui.button(
                "←",
                on_click=lambda: (
                    setattr(
                        visualizer,
                        "current_step",
                        max(0, visualizer.current_step - 1),
                    ),
                    visualizer.update_chart(),
                ),
            ).classes("text-xs")
            ui.button(
                "→",
                on_click=lambda: (
                    setattr(
                        visualizer,
                        "current_step",
                        min(len(visualizer.sequence), visualizer.current_step + 1),
                    ),
                    visualizer.update_chart(),
                ),
            ).classes("text-xs")

        ui.separator().classes("my-4")

        ui.markdown("### Priority Sequence").classes("text-xs")
        with ui.scroll_area().classes("h-[150px] w-full"):
            visualizer.sequence_table = ui.markdown(
                "_Load data to see sequence_"
            ).classes("text-[10px]")

        ui.separator().classes("my-4")
        ui.markdown("### Smell Details").classes("text-xs")
        with ui.scroll_area().classes("h-[200px] w-full"):
            visualizer.details_label = ui.markdown(
                "Click on a smell node to see prioritization details."
            ).classes("text-[10px]")

    # Right Drawer for Agent Execution Context
    with (
        ui.right_drawer(top_corner=True, bottom_corner=True, value=True)
        .classes("p-4 w-[500px]")
        .style("width: 500px;")
    ):
        ui.markdown("### Iteration Details").classes("text-xs")
        with ui.scroll_area().classes("h-[200px] w-full"):
            visualizer.iteration_details = ui.markdown(
                "_Load a database to see iteration details_"
            ).classes("text-[10px]")

        ui.separator().classes("my-4")

        ui.markdown("### Tool Call Logs").classes("text-xs")
        with ui.scroll_area().classes("h-[150px] w-full"):
            visualizer.tool_logs_area = ui.markdown(
                "_Load a database to see tool calls_"
            ).classes("text-[10px]")

        ui.separator().classes("my-4")

        ui.markdown("### Code Diff").classes("text-xs")
        with ui.scroll_area().classes("h-[300px] w-full"):
            visualizer.diff_viewer = ui.code("", language="diff").classes("text-[10px]")

    # Main content area - split between graph and timeline
    with ui.element("div").style(
        "width: 100%; height: calc(100vh - 50px); display: flex; flex-direction: column; gap: 16px; padding: 16px; box-sizing: border-box; background-color: #f5f5f5;"
    ):
        # Smell dependency graph (top, 60%)
        with ui.element("div").style(
            "flex: 3; min-height: 400px; background-color: white; border-radius: 8px;"
        ):
            visualizer.load_data(None)
            initial_options = visualizer.get_echart_options()

            visualizer.chart = ui.echart(
                options=initial_options or {}, on_point_click=visualizer.handle_click
            ).style("width: 100%; height: 100%;")

        # Timeline (bottom, 40%)
        with ui.element("div").style(
            "flex: 2; min-height: 250px; background-color: white; border-radius: 8px;"
        ):
            visualizer.timeline_chart = ui.echart(options={}).style(
                "width: 100%; height: 100%;"
            )

    # Update UI elements after chart is created
    visualizer.update_chart()


ui.run(title="AI Agent Execution Visualizer", port=8080)
