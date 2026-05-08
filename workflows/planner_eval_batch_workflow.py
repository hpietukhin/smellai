#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from workflows.planner_eval_workflow import _build_stratified_summary_rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch planner-eval over a runnable batch list")
    p.add_argument("--batch-list", required=True, help="JSON: {cases:[{project,elements:[...],start_commit_hash,...}, ...]}")
    p.add_argument("--max-episodes", type=int, default=20)
    p.add_argument("--heuristic", choices=["element-based", "commit-based", "range-based"], default="range-based")
    p.add_argument("--planner", choices=["greedy", "befs", "developer"], default="befs")
    p.add_argument("--locality", choices=["none", "class", "file"], default="none")
    p.add_argument("--tracking-uri", default="http://localhost:5000")
    p.add_argument("--experiment", default="planner-eval-batch")
    p.add_argument("--run-name", default="planner-eval-batch")
    return p.parse_args()


def _load_batch_list(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text())
    cases = data.get("cases", [])
    assert isinstance(cases, list), "cases must be a list"
    for case in cases:
        assert "project" in case and "elements" in case, "case must include project,elements"
    return cases


def _aggregate_rows(rows: list[dict[str, float]]) -> dict:
    assert rows, "rows must be non-empty"
    return _build_stratified_summary_rows(rows)


def main() -> int:
    raise RuntimeError(
        "DONT RUN THIS DONT RUN THIS DEPRECATED: workflows/planner_eval_batch_workflow.py is disabled. "
        "Use workflows/composite_workflow_full.py batch with a generated batch list instead."
    )


if __name__ == "__main__":
    raise SystemExit(main())
