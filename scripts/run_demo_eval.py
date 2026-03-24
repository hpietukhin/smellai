#!/usr/bin/env python3
"""Run composite evaluation demo with automatic SonarQube readiness check.

Usage:
    uv run python scripts/run_demo_eval.py
    uv run python scripts/run_demo_eval.py --records 2 --iterations 5
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Installing requests...")
    subprocess.run(["uv", "pip", "install", "requests"], check=True)
    import requests


def check_sonarqube_ready(max_attempts=12, delay=5):
    """Check if SonarQube is ready."""
    print("Checking SonarQube status...")
    for i in range(max_attempts):
        try:
            response = requests.get(
                "http://localhost:9000/api/system/status", timeout=5
            )
            if response.ok:
                status = response.json().get("status")
                if status == "UP":
                    print("✅ SonarQube is ready")
                    return True
                else:
                    print(f"⏳ SonarQube status: {status} (attempt {i+1}/{max_attempts})")
        except requests.exceptions.RequestException:
            print(f"⏳ Waiting for SonarQube... (attempt {i+1}/{max_attempts})")

        if i < max_attempts - 1:
            time.sleep(delay)

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Run composite evaluation demo with visualization"
    )
    parser.add_argument(
        "--records",
        type=int,
        default=1,
        help="Number of records to evaluate (default: 1 for quick demo)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Max refactoring iterations per record (default: 3)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="compound_demo.json",
        help="Dataset to use (default: compound_demo.json)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="demo_composite.db",
        help="Analytics database path (default: demo_composite.db)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="LLM model to use",
    )
    parser.add_argument(
        "--skip-sonar-check",
        action="store_true",
        help="Skip SonarQube readiness check",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    print("=" * 60)
    print("COMPOSITE EVALUATION DEMO")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Records to evaluate: {args.records}")
    print(f"Max iterations: {args.iterations}")
    print(f"Model: {args.model}")
    print(f"Analytics DB: {args.db}")
    print("=" * 60)
    print()

    # Check SonarQube
    if not args.skip_sonar_check:
        if not check_sonarqube_ready():
            print()
            print("❌ SonarQube not ready after 60 seconds")
            print()
            print("Troubleshooting:")
            print("  1. Check if SonarQube is running:")
            print("     docker ps | grep sonarqube")
            print()
            print("  2. Start SonarQube:")
            print("     docker compose -f sonarqube/docker-compose.yml up -d")
            print()
            print("  3. Check logs:")
            print("     docker logs smellai-sonarqube")
            print()
            print("  4. Access UI:")
            print("     open http://localhost:9000")
            print()
            return 1
    else:
        print("⚠️  Skipping SonarQube check")

    print()
    print("=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)
    print("This may take 5-15 minutes per record (model + compile time)")
    print(f"Estimated time: {args.records * 10}-{args.records * 15} minutes")
    print()

    # Run evaluation
    cmd = [
        "uv",
        "run",
        "workflows/swe_eval_workflow.py",
        "--dataset",
        args.dataset,
        "--enable-composite",
        "--max-refactorings",
        str(args.iterations),
        "--analytics-db",
        args.db,
        "--model",
        args.model,
        "--limit",
        str(args.records),
    ]

    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Evaluation failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        return 130

    print()
    print("=" * 60)
    print("✅ EVALUATION COMPLETE")
    print("=" * 60)
    print()
    print("Results saved to:")
    print(f"  📊 Analytics DB: {args.db}")
    print("  📈 MLflow DB: sqlite:///mlflow.db")
    print()
    print("Next steps:")
    print()
    print("  1. Visualize agent execution:")
    print("     ./scripts/run_visualizer.sh")
    print(f"     Then load: {args.db}")
    print()
    print("  2. View MLflow metrics:")
    print("     mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("     Open: http://localhost:5000")
    print()
    print("  3. Analyze results:")
    print("     - Smell dependency graph (visualizer)")
    print("     - Agent timeline (visualizer)")
    print("     - Compilation/test rates (MLflow)")
    print()
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
