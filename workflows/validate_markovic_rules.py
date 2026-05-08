#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging

import mlflow

from dataset.neo4j_graph import DatasetGraph
from domain.dependency_rules_validator import DependencyRulesValidator

LOGGER = logging.getLogger(__name__)


def _parse_args():
    p = argparse.ArgumentParser(description="Validate Markovic dependency rules with MLflow logging")
    p.add_argument("--project", required=True)
    p.add_argument("--elements", required=True, help="CSV element FQNs")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--tracking-uri", default="http://localhost:5000")
    p.add_argument("--experiment", default="markovic-rules-validation")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)

    ds = DatasetGraph()
    if not ds.is_available():
        LOGGER.error("Neo4j is not available")
        return 1

    elements = {e.strip() for e in args.elements.split(",") if e.strip()}
    steps = ds.composite_refactoring(elements=elements, project=args.project, max_steps=args.max_steps)
    if len(steps) < 2:
        LOGGER.error("Need at least 2 steps for validation")
        return 1

    result = DependencyRulesValidator().validate_steps(steps)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    run_name = args.run_name or f"rules/{args.project.replace(' ', '_')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "project": args.project,
                "elements": ",".join(sorted(elements)),
                "max_steps": args.max_steps,
            }
        )
        mlflow.log_metrics(result.to_mlflow_metrics())

    LOGGER.info("Done: pos_f1=%.3f neg_f1=%.3f", result.positive_f1, result.negative_f1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
