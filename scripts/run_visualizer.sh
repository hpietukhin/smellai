#!/usr/bin/env bash
# Launch NiceGUI-based agent execution visualizer
# Port: 8080 (hardcoded in visualizer)
# Access at: http://localhost:8080
#
# To change port, edit line 1074 in tools/visualize_smell_prioritization.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Starting NiceGUI visualizer on port 8080..."
echo "Access at: http://localhost:8080"
echo "Press Ctrl+C to stop"
echo ""

uv run python tools/visualize_smell_prioritization.py
