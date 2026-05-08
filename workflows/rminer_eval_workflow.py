#!/usr/bin/env python3
"""DEPRECATED thin compatibility wrapper for RMiner evaluation.

Redirect to Composite Refactorings 2020 flow instead:
- dataset.neo4j_graph.DatasetGraph.composite_refactoring(...)
- workflows/planner_eval_workflow.py

Delegates to workflows/eval_workflow.py with --source rminer.

Usage:
    uv run workflows/rminer_eval_workflow.py --manifest path/to/manifest.json
"""

from __future__ import annotations

import sys
import warnings
from workflows.eval_workflow import main as unified_eval_main


def main(
    manifest: str,
    experiment: str = "rminer-evaluation",
    tracking_uri: str = "http://localhost:5000",
    model: str = "gpt-4o-mini",
    limit: int | None = None,
    draw_graph: bool = False,
) -> int:
    """RMiner evaluation workflow.

    DEPRECATED: kept only for compatibility; prefer Composite Refactorings 2020 planner workflow.
    """
    warnings.warn(
        "workflows.rminer_eval_workflow is deprecated; use workflows/planner_eval_workflow.py with Composite Refactorings 2020 dataset.",
        DeprecationWarning,
        stacklevel=2,
    )
    return unified_eval_main(
        source="rminer",
        manifest=manifest,
        experiment=experiment,
        tracking_uri=tracking_uri,
        model=model,
        limit=limit,
        draw_graph=draw_graph,
    )


if __name__ == "__main__":
    import fire

    sys.exit(fire.Fire(main))
