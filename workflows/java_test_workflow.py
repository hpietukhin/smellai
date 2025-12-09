#!/usr/bin/env python3
"""Workflow for analyzing Java tests with LangGraph agent.

This script runs the Java test analysis agent on a project directory,
detecting the build system, running tests, and reporting results.

Usage:
    # Basic usage with defaults
    uv run workflows/java_test_workflow.py --project /path/to/java/project

    # Use specific model
    uv run workflows/java_test_workflow.py --project /path/to/java/project --model gpt-4

    # Verbose output
    uv run workflows/java_test_workflow.py --project /path/to/java/project --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from agents.java_test.agent import analyze_java_tests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main workflow entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze Java tests using LangGraph agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Path to Java project directory",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate project path
    project_path = Path(args.project)
    if not project_path.exists():
        logger.error(f"Project path does not exist: {project_path}")
        sys.exit(1)

    if not project_path.is_dir():
        logger.error(f"Project path is not a directory: {project_path}")
        sys.exit(1)

    # Run analysis
    logger.info(f"Analyzing Java tests in: {project_path}")
    logger.info(f"Using model: {args.model}")

    try:
        result = analyze_java_tests(
            str(project_path),
            model_name=args.model,
        )

        if args.json:
            # Output as JSON
            output = {
                "project_path": str(project_path),
                "model": args.model,
                "response": result["response"],
            }
            print(json.dumps(output, indent=2))
        else:
            # Human-readable output
            print(f"\n{'=' * 80}")
            print("Java Test Analysis Results")
            print(f"{'=' * 80}\n")
            print(f"Project: {project_path}")
            print(f"Model: {args.model}")
            print(f"\n{'-' * 80}\n")
            print(result["response"])
            print(f"\n{'=' * 80}\n")

        logger.info("Analysis completed successfully")

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
