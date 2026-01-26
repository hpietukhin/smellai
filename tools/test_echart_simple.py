#!/usr/bin/env python3
"""Simple echart test to verify rendering"""

from nicegui import ui


@ui.page("/")
def main_page():
    ui.label("Simple EChart Test").classes("text-h4")

    options = {
        "xAxis": {"type": "category", "data": ["A", "B", "C"]},
        "yAxis": {"type": "value"},
        "series": [{"data": [10, 20, 30], "type": "bar"}],
    }

    with ui.element("div").style(
        "width: 100%; height: 400px; background-color: #f0f0f0; padding: 20px;"
    ):
        ui.echart(options).style("width: 100%; height: 100%; background-color: white;")


ui.run(title="Test EChart", port=8081)
