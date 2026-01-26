#!/usr/bin/env python3
"""Launch NiceGUI-based agent execution visualizer.

The visualizer runs on port 8080 (hardcoded in tools/visualize_smell_prioritization.py).
To change the port, edit line 1074 in that file.

Usage:
    uv run python scripts/run_visualizer.py
    ./scripts/run_visualizer.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Navigate to project root
    project_root = Path(__file__).parent.parent
    visualizer_path = project_root / "tools" / "visualize_smell_prioritization.py"

    if not visualizer_path.exists():
        print(f"Error: Visualizer not found at {visualizer_path}", file=sys.stderr)
        return 1

    print("Starting NiceGUI visualizer on port 8080...")
    print("Access at: http://localhost:8080")
    print("Press Ctrl+C to stop\n")

    try:
        subprocess.run(["python", str(visualizer_path)], cwd=project_root, check=True)
    except KeyboardInterrupt:
        print("\n\nShutting down visualizer...")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nError running visualizer: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
