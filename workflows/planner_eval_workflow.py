#!/usr/bin/env python3
"""Planner evaluation workflow against the Composite Refactorings 2020 dataset.

For each composite episode (element × project), runs both planners (BeFS and
Greedy) and the developer's simulated sequence, then logs per-episode metrics
to MLflow.

Usage::

    uv run workflows/planner_eval_workflow.py \\
        --project "JUnit4" \\
        --elements "org.junit.runners.BlockJUnit4ClassRunner,org.junit.runners.ParentRunner" \\
        --max-episodes 20

    uv run workflows/planner_eval_workflow.py \\
        --project "OkHttp" \\
        --elements "com.squareup.okhttp.internal.http.ResponseCacheAdapterTest.put_httpGet,com.squareup.okhttp.internal.http.ResponseCacheAdapterTest.put_responseChangesForbidden" \\
        --max-episodes 50 \\
        --tracking-uri http://localhost:5000 \\
        --experiment planner-eval-safe-maven

Metrics logged per episode (all as MLflow metrics):
    smells_initial, befs_reduces, greedy_reduces, dev_reduces,
    befs_eta, greedy_eta, befs_rho, greedy_rho,
    h_initial, h_befs_final, h_greedy_final, h_dev_final, dev_steps

Params logged per episode:
    project, initial_commit, initial_commit_order
"""
from __future__ import annotations

import sys
import logging
import argparse
import json
from statistics import median

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate planners (BeFS + Greedy) against Composite Refactorings 2020.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--project",
        default="JUnit4",
        help="Project name in the Neo4j dataset (e.g. 'JUnit4', 'OkHttp')",
    )
    p.add_argument(
        "--elements",
        default="org.junit.runners.BlockJUnit4ClassRunner,org.junit.runners.ParentRunner",
        help="Comma-separated element FQNs to trace",
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        default=20,
        help="Maximum number of composite steps to retrieve per element set",
    )
    p.add_argument(
        "--heuristic",
        choices=["element-based", "commit-based", "range-based"],
        default="range-based",
        help="Composite synthesis perspective for methodology-aligned logging",
    )
    p.add_argument(
        "--planner",
        choices=["greedy", "befs", "developer"],
        default="befs",
        help="Primary planner label for experiment spec",
    )
    p.add_argument(
        "--locality",
        choices=["none", "class", "file"],
        default="none",
        help="Dependency graph locality mode",
    )
    p.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking server URI",
    )
    p.add_argument(
        "--experiment",
        default="planner-eval",
        help="MLflow experiment name",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="MLflow run name (auto-generated from project + element if omitted)",
    )
    p.add_argument(
        "--validate-rules",
        action="store_true",
        help="Run Markovic dependency-rules validation on the same episode and log metrics",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return p.parse_args(argv)


def _run_name_for(project: str, elements: set[str]) -> str:
    """Generate a short run name from project + first element."""
    first_el = sorted(elements)[0].split(".")[-1] if elements else "unknown"
    safe_project = project.replace(" ", "_").lower()
    return f"{safe_project}/{first_el}"


def _namespace_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    assert prefix, "prefix is required"
    return {f"{prefix}.{k}": float(v) for k, v in metrics.items()}


def _relative_h_reduction(h_initial: float, h_final: float) -> float:
    if h_initial <= 0.0:
        return 0.0
    return (h_initial - h_final) / h_initial


def _with_derived_methodology_metrics(row: dict[str, float]) -> dict[str, float]:
    """Add methodology-consistent derived metrics used for stratified reporting."""
    enriched = dict(row)
    h_initial = enriched.get("h_initial")
    if h_initial is not None:
        if "h_befs_final" in enriched:
            enriched["befs_relative_h_reduction"] = _relative_h_reduction(h_initial, enriched["h_befs_final"])
        if "h_greedy_final" in enriched:
            enriched["greedy_relative_h_reduction"] = _relative_h_reduction(h_initial, enriched["h_greedy_final"])
        if "h_dev_final" in enriched:
            enriched["dev_relative_h_reduction"] = _relative_h_reduction(h_initial, enriched["h_dev_final"])
        if "h_rule_aware_final" in enriched:
            enriched["rule_aware_relative_h_reduction"] = _relative_h_reduction(h_initial, enriched["h_rule_aware_final"])
        if "h_no_rules_final" in enriched:
            enriched["no_rules_relative_h_reduction"] = _relative_h_reduction(h_initial, enriched["h_no_rules_final"])

    if "rule_aware_relative_h_reduction" in enriched and "no_rules_relative_h_reduction" in enriched:
        enriched["rule_gain_h"] = enriched["rule_aware_relative_h_reduction"] - enriched["no_rules_relative_h_reduction"]
    if "rule_aware_resolved" in enriched and "no_rules_resolved" in enriched:
        enriched["rule_gain_resolved"] = enriched["rule_aware_resolved"] - enriched["no_rules_resolved"]
    if "rule_aware_introduced" in enriched and "no_rules_introduced" in enriched:
        enriched["rule_gain_introduced"] = enriched["no_rules_introduced"] - enriched["rule_aware_introduced"]

    return enriched


def _summary_for_values(values: list[float]) -> dict[str, float]:
    assert values, "values must be non-empty"
    return {
        "n": float(len(values)),
        "mean": float(sum(values) / len(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _summarize_methodology_group(rows: list[dict[str, float]]) -> dict[str, object]:
    assert rows, "rows must be non-empty"
    preferred_metrics = [
        # Primary / rule-effectiveness outcomes from docs/composite_methodology.md.
        "success",
        "success_rate",
        "relative_h_reduction",
        "rule_aware_relative_h_reduction",
        "no_rules_relative_h_reduction",
        "befs_relative_h_reduction",
        "greedy_relative_h_reduction",
        "dev_relative_h_reduction",
        "rule_gain_h",
        "resolved",
        "rule_aware_resolved",
        "no_rules_resolved",
        "rule_gain_resolved",
        "introduced_smells",
        "created_smells",
        "rule_aware_introduced",
        "no_rules_introduced",
        "rule_gain_introduced",
        "rho",
        "befs_rho",
        "greedy_rho",
        "eta",
        "befs_eta",
        "greedy_eta",
        "steps_executed",
        "repair_attempts",
        # Paper-style diagnostic bridge to Sousa et al.
        "smells_initial",
        "smells_after_empirical",
        "smells_before",
        "smells_after",
        # Operational diagnostics.
        "llm_time_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_cost",
    ]
    metrics: dict[str, dict[str, float]] = {}
    for metric in preferred_metrics:
        values = [r[metric] for r in rows if metric in r]
        if values:
            metrics[metric] = _summary_for_values(values)
    return {"count": float(len(rows)), "metrics": metrics}


def _build_stratified_summary_rows(rows: list[dict[str, float]]) -> dict[str, object]:
    """Summarize methodology metrics by composite length and scope size.

    Composite length is `composite_size` / `ref_count`: small <=4, medium 5-10,
    large >10.  The metric set follows docs/composite_methodology.md: primary
    executable success / h reduction, rule-ablation gains, safety (`rho` /
    introduced smells), efficiency (`eta` / steps), paper-style smell incidence,
    and deterministic operational cost diagnostics.
    """
    from dataset.stratify import bucket_composite_size, bucket_scope_size

    assert rows, "rows must be non-empty"
    enriched_rows = [_with_derived_methodology_metrics(r) for r in rows]

    scope_groups: dict[str, list[dict[str, float]]] = {}
    comp_groups: dict[str, list[dict[str, float]]] = {}
    for row in enriched_rows:
        scope_bucket = bucket_scope_size(int(row["scope_size"]))
        composite_bucket = bucket_composite_size(int(row["composite_size"]))
        scope_groups.setdefault(scope_bucket, []).append(row)
        comp_groups.setdefault(composite_bucket, []).append(row)

    return {
        "bucket_definitions": {
            "composite_size": {
                "description": "Composite length measured as number of refactorings (`ref_count` / `composite_size`).",
                "small": "1-4 refactorings",
                "medium": "5-10 refactorings",
                "large": ">10 refactorings",
            },
            "scope_size": {
                "description": "Number of affected elements in the selected-scope episode.",
                "2": "<=2 elements",
                "3-5": "3-5 elements",
                "6-10": "6-10 elements",
                ">10": ">10 elements",
            },
        },
        "overall": _summarize_methodology_group(enriched_rows),
        "by_scope": {k: _summarize_methodology_group(v) for k, v in sorted(scope_groups.items())},
        "by_composite": {k: _summarize_methodology_group(v) for k, v in sorted(comp_groups.items())},
    }


def _build_validity_metadata(
    *,
    project: str,
    elements: set[str],
    heuristic: str,
    locality: str,
) -> dict:
    return {
        "construct_validity": "Smell-based outcomes are proxies; compile/test needed for stronger semantic claims.",
        "internal_validity": "Developer trajectory is an observed reference, not an optimal baseline.",
        "external_validity": "Results are bounded to Composite Refactorings 2020 and the generated runnable batch lists.",
        "filters": {
            "project": project,
            "elements": sorted(elements),
            "heuristic": heuristic,
            "locality": locality,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        import mlflow
    except ImportError:
        logger.error("mlflow is not installed. Run: uv add mlflow")
        return 1

    try:
        from dataset.neo4j_graph import DatasetGraph
    except ImportError as exc:
        logger.error("Could not import DatasetGraph: %s", exc)
        return 1

    from dataset.planner_eval import evaluate_composite
    from domain.dependency_rules_validator import DependencyRulesValidator
    from domain.experiment_axes import ExperimentSpec

    # --- Connect to Neo4j ---
    ds = DatasetGraph()
    if not ds.is_available():
        logger.error(
            "Neo4j is not reachable. Start the Composite Refactorings 2020 database first."
        )
        return 1
    logger.info("Neo4j connection OK")

    elements: set[str] = {e.strip() for e in args.elements.split(",") if e.strip()}
    project: str = args.project

    logger.info(
        "Project: %s | Elements: %s | max_episodes: %d",
        project,
        ", ".join(sorted(elements)),
        args.max_episodes,
    )

    # --- Fetch composite steps ---
    steps = ds.composite_refactoring(
        elements=elements,
        project=project,
        max_steps=args.max_episodes,
    )

    if not steps:
        logger.warning("No composite steps found for project=%r elements=%r", project, elements)
        return 0

    steps_with_smells = [s for s in steps if s.smells]
    logger.info(
        "Retrieved %d steps (%d with smells, %d with refactorings)",
        len(steps),
        len(steps_with_smells),
        sum(1 for s in steps if s.refactorings),
    )

    # --- Evaluate ---
    spec = ExperimentSpec(
        heuristic=args.heuristic,
        planner=args.planner,
        locality=args.locality,
        track_completion_risk=True,
    )
    result = evaluate_composite(steps, project=project, spec=spec)
    if result is None:
        logger.warning("No initial smells found — nothing to evaluate.")
        return 0

    # --- Log to MLflow ---
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    run_name = args.run_name or _run_name_for(project, elements)

    with mlflow.start_run(run_name=run_name):
        try:
            import pandas as pd  # type: ignore[import-untyped]
            from mlflow.data import from_pandas  # type: ignore[attr-defined]

            dataset_df = pd.DataFrame(
                [
                    {
                        "project": project,
                        "elements": ",".join(sorted(elements)),
                        "max_episodes": args.max_episodes,
                        "heuristic": args.heuristic,
                        "planner": args.planner,
                        "locality": args.locality,
                        "steps_retrieved": len(steps),
                        "steps_with_smells": len(steps_with_smells),
                    }
                ]
            )
            ds = from_pandas(
                dataset_df,
                source=f"composite_refactorings_2020:{project}",
                name="planner_eval_input",
            )
            mlflow.log_input(ds, context="evaluation")
        except Exception as exc:
            logger.warning("mlflow.log_input skipped: %s", exc)

        mlflow.log_params(result.to_mlflow_params())
        mlflow.log_params({"elements": ", ".join(sorted(elements))})

        if result.completion_risk_count < 0:
            logger.warning("Unexpected completion_risk_count=%s (diagnostic-only metric)", result.completion_risk_count)

        algo_metrics = _namespace_metrics("algo", result.to_mlflow_metrics())
        operational_metrics = _namespace_metrics("ops", {})
        mlflow.log_metrics(algo_metrics)
        mlflow.log_metrics(operational_metrics)

        if args.validate_rules:
            try:
                rv = DependencyRulesValidator().validate_steps(steps)
                mlflow.log_metrics(_namespace_metrics("rules", rv.to_mlflow_metrics()))
            except Exception as exc:
                logger.warning("Rule validation skipped: %s", exc)

        # Log full h-traces as a JSON artifact for downstream comparison.
        import tempfile
        import os

        h_traces = {
            "befs_h_trace": list(result.befs_h_trace),
            "greedy_h_trace": list(result.greedy_h_trace),
            "dev_h_trace": list(result.dev_h_trace),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="h_traces_", delete=False,
        ) as f:
            json.dump(h_traces, f, indent=2)
            tmp_path = f.name
        try:
            mlflow.log_artifact(tmp_path, artifact_path="h_traces")
        finally:
            os.unlink(tmp_path)

        touched: set[str] = set()
        n_refs = 0
        for st in steps:
            n_refs += len(st.refactorings)
            for ref in st.refactorings:
                touched.update(ref.changed_elements or [])
                touched.update(ref.produced_elements or [])
        episode_row = {
            "scope_size": float(max(len(touched), len(elements))),
            "composite_size": float(n_refs),
            "smells_initial": float(result.smells_initial),
            "smells_after_empirical": float(result.smells_after_empirical),
            "befs_eta": float(result.befs_eta),
            "greedy_eta": float(result.greedy_eta),
            "befs_rho": float(result.befs_rho),
            "greedy_rho": float(result.greedy_rho),
            "h_initial": float(result.h_initial),
            "h_befs_final": float(result.h_befs_final),
            "h_greedy_final": float(result.h_greedy_final),
            "h_dev_final": float(result.h_dev_final),
        }
        stratified = _build_stratified_summary_rows([episode_row])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="stratified_", delete=False,
        ) as f:
            json.dump(stratified, f, indent=2)
            s_path = f.name
        try:
            mlflow.log_artifact(s_path, artifact_path="stratified")
        finally:
            os.unlink(s_path)

        validity = _build_validity_metadata(
            project=project,
            elements=elements,
            heuristic=args.heuristic,
            locality=args.locality,
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="validity_", delete=False,
        ) as f:
            json.dump(validity, f, indent=2)
            v_path = f.name
        try:
            mlflow.log_artifact(v_path, artifact_path="validity")
        finally:
            os.unlink(v_path)

        logger.info(
            "Run logged: smells_initial=%d  befs_reduces=%s  greedy_reduces=%s  dev_reduces=%s",
            result.smells_initial,
            result.befs_reduces,
            result.greedy_reduces,
            result.dev_reduces,
        )
        logger.info(
            "  h: initial=%.2f  befs=%.2f  greedy=%.2f  dev=%.2f",
            result.h_initial,
            result.h_befs_final,
            result.h_greedy_final,
            result.h_dev_final,
        )
        logger.info(
            "  η: befs=%.2f  greedy=%.2f | ρ: befs=%.2f  greedy=%.2f | dev_steps=%d",
            result.befs_eta,
            result.greedy_eta,
            result.befs_rho,
            result.greedy_rho,
            result.dev_steps,
        )

    logger.info("MLflow run complete. Tracking URI: %s", args.tracking_uri)
    return 0


if __name__ == "__main__":
    sys.exit(main())
