#!/usr/bin/env python3
"""Thin compatibility wrapper for SWE evaluation.

Delegates to workflows/eval_workflow.py with --source swe.

Usage:
    uv run workflows/swe_eval_workflow.py --dataset /path/to/data.json
"""

from __future__ import annotations

import sys
from workflows.eval_workflow import main as unified_eval_main


def main(
    dataset: str = "/tmp/SWE-Refactor/pure_refactoring_data.json",
    experiment: str = "swe-refactor-evaluation",
    tracking_uri: str = "http://localhost:5000",
    model: str = "claude-sonnet-4-5-20250929",
    limit: int | None = None,
    workspace: str = "/tmp/swe-eval-workspace",
    draw_graph: bool = False,
    enable_composite: bool = False,
    max_refactorings: int = 5,
    analytics_db: str = "analytics.db",
    sonar_url: str = "http://localhost:9000",
    sonar_cache_dir: str = "./sonar_cache",
    with_sonar: bool = False,
) -> int:
    """SWE-Refactor evaluation workflow."""
    return unified_eval_main(
        source="swe",
        raw_path=dataset,
        experiment=experiment,
        tracking_uri=tracking_uri,
        model=model,
        limit=limit,
        workspace=workspace,
        draw_graph=draw_graph,
        enable_composite=enable_composite,
        max_refactorings=max_refactorings,
        analytics_db=analytics_db,
        sonar_url=sonar_url,
        sonar_cache_dir=sonar_cache_dir,
        with_sonar=with_sonar,
    )


if __name__ == "__main__":
    import fire

    sys.exit(fire.Fire(main))
