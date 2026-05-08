from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections import Counter
from dataclasses import dataclass
import os

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.tracking.client import Run


@dataclass(frozen=True)
class SpanTiming:
    """Single span timing extracted from MLflow traces."""

    action: str
    span_name: str
    duration_ms: float
    step: str | None = None
    file: str | None = None


@dataclass(frozen=True)
class RunBottleneckSummary:
    """Condensed per-run timing summary."""

    run_id: str
    run_name: str
    status: str
    duration_seconds: float
    model: str
    steps_executed: float | None
    stop_reason: str | None
    success: float | None
    top_action: str | None
    top_action_ms: float
    top_spans: list[SpanTiming]
    top_actions: dict[str, float]


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Summarize MLflow experiment bottlenecks from traces and run metadata."
    )
    parser.add_argument(
        "--experiment-name",
        default="composite_workflow_full",
        help="MLflow experiment name (default: composite_workflow_full)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of latest runs to inspect (default: 20)",
    )
    parser.add_argument(
        "--top-spans",
        type=int,
        default=5,
        help="Number of hottest spans to show per run (default: 5)",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or http://localhost:5000)",
    )
    parser.add_argument(
        "--trace-max-spans",
        type=int,
        default=1000,
        help="Max number of spans to fetch for a run (default: 1000)",
    )
    return parser.parse_args()


def _resolve_tracking_uri(cli_tracking_uri: str | None) -> str:
    if cli_tracking_uri:
        return cli_tracking_uri
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _run_duration_seconds(run: Run) -> float | None:
    start_ms = run.info.start_time
    end_ms = run.info.end_time
    if start_ms is None or end_ms is None:
        return None
    return (end_ms - start_ms) / 1000.0


def _safe_run_param(run: Run, key: str) -> str | None:
    value = run.data.params.get(key)
    if value is None:
        return None
    return str(value)


def _safe_run_metric(run: Run, key: str) -> float | None:
    value = run.data.metrics.get(key)
    if value is None:
        return None
    return float(value)


def _span_attribute(span_attrs: object, key: str) -> str | None:
    if span_attrs is None:
        return None
    getter = getattr(span_attrs, "get", None)
    if not callable(getter):
        return None
    value = getter(key)
    if value is None:
        return None
    return str(value)


def _span_duration_ms(span: object) -> float | None:
    data = getattr(span, "_span", None)
    if data is None:
        return None
    start_ns = getattr(data, "_start_time", None)
    end_ns = getattr(data, "_end_time", None)
    if start_ns is None or end_ns is None:
        return None
    return (float(end_ns) - float(start_ns)) / 1_000_000.0


def _extract_span_timings(run_id: str, experiment_id: str, trace_max_spans: int) -> tuple[list[SpanTiming], dict[str, float]]:
    """Fetch and aggregate span timings for one run."""
    all_spans: list[SpanTiming] = []
    action_ms: dict[str, float] = {}

    try:
        traces = mlflow.search_traces(
            run_id=run_id,
            include_spans=True,
            return_type="list",
            max_results=trace_max_spans,
            locations=[experiment_id],
        )
    except Exception:
        # Fallback for legacy setups where trace location filtering fails.
        try:
            traces = mlflow.search_traces(
                run_id=run_id,
                include_spans=True,
                return_type="list",
                max_results=trace_max_spans,
            )
        except Exception:
            return all_spans, action_ms

    for trace in traces:
        if getattr(trace, "data", None) is None:
            continue
        for span in getattr(trace.data, "spans", []):
            duration_ms = _span_duration_ms(span)
            if duration_ms is None:
                continue

            attrs = getattr(span, "_attributes", None)
            action = _span_attribute(attrs, "action_type")
            if not action:
                continue

            file_path = _span_attribute(attrs, "file")
            step = _span_attribute(attrs, "step")
            span_name = getattr(span, "_span", None)
            span_name_value = str(getattr(span_name, "_name", "<unknown>")) if span_name is not None else "<unknown>"

            all_spans.append(
                SpanTiming(
                    action=action,
                    span_name=span_name_value,
                    duration_ms=duration_ms,
                    step=step,
                    file=file_path,
                )
            )
            action_ms[action] = action_ms.get(action, 0.0) + duration_ms

    all_spans.sort(key=lambda item: item.duration_ms, reverse=True)
    return all_spans, action_ms


def _collect_run_summary(
    run: Run,
    experiment_id: str,
    top_spans: int,
    trace_max_spans: int,
) -> RunBottleneckSummary | None:
    duration_seconds = _run_duration_seconds(run)
    if duration_seconds is None:
        return None

    model = _safe_run_param(run, "model") or "<unknown>"
    steps_executed = _safe_run_metric(run, "steps_executed")
    stop_reason = _safe_run_param(run, "stop_reason")
    success = _safe_run_metric(run, "success")
    run_name = run.info.run_name or "<unnamed>"

    span_timings, action_ms = _extract_span_timings(run.info.run_id, experiment_id, trace_max_spans)

    top_action = None
    top_action_ms = 0.0
    if action_ms:
        top_action, top_action_ms = max(action_ms.items(), key=lambda item: item[1])

    return RunBottleneckSummary(
        run_id=run.info.run_id,
        run_name=run_name,
        status=run.info.status,
        duration_seconds=duration_seconds,
        model=model,
        steps_executed=steps_executed,
        stop_reason=stop_reason,
        success=success,
        top_action=top_action,
        top_action_ms=top_action_ms,
        top_spans=span_timings[:top_spans],
        top_actions=action_ms,
    )


def _percent(value: float, total: float) -> float:
    if total <= 0.0:
        return 0.0
    return (value / total) * 100.0


def _print_summary(summaries: list[RunBottleneckSummary], top_spans: int) -> None:
    print("\nTop runs by runtime")
    print("-" * 120)
    header = (
        f"{'#':>2} {'Run ID':12} {'Duration(s)':>11} {'Status':10} {'Model':48} "
        f"{'Top action':22} {'Top action s':>12} {'Stop':15} {'Steps':>7}"
    )
    print(header)
    print("-" * len(header))

    for idx, summary in enumerate(sorted(summaries, key=lambda s: s.duration_seconds, reverse=True), 1):
        top_action = summary.top_action or "<none>"
        top_action_s = f"{summary.top_action_ms/1000:.2f}"
        steps = f"{summary.steps_executed:.0f}" if summary.steps_executed is not None else "-"
        stop_reason = summary.stop_reason or "-"
        print(
            f"{idx:>2} {summary.run_id[:12]:12} {summary.duration_seconds:11.2f} "
            f"{summary.status:10} {summary.model:48} {top_action:22} {top_action_s:12} "
            f"{stop_reason:15} {steps:>7}"
        )
        if summary.top_spans:
            print(f"    Top {top_spans} spans:")
            for span in summary.top_spans:
                step = f" step={span.step}" if span.step is not None else ""
                file = f" file={span.file}" if span.file is not None else ""
                print(
                    f"      - {span.action:24} {span.span_name:28} "
                    f"{span.duration_ms/1000:8.2f}s{step}{file}"
                )
        else:
            print("    No trace spans available")

    print("\nAggregate bottleneck distribution")
    print("-" * 120)
    counter: Counter[str] = Counter()
    for summary in summaries:
        if summary.top_action:
            counter[summary.top_action] += 1

    for action, count in counter.most_common():
        pct = _percent(count, len(summaries))
        print(f"  {action:24} {count:>3} runs ({pct:5.1f}%)")

    durations = [s.duration_seconds for s in summaries]
    if durations:
        avg = sum(durations) / len(durations)
        print(f"\nAverage duration: {avg:.2f}s across {len(summaries)} runs")


def _main() -> None:
    args = _parse_args()
    tracking_uri = _resolve_tracking_uri(args.tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        raise SystemExit(f"Experiment '{args.experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=args.runs,
    )

    summaries: list[RunBottleneckSummary] = []
    for run in runs:
        summary = _collect_run_summary(
            run=run,
            experiment_id=experiment.experiment_id,
            top_spans=args.top_spans,
            trace_max_spans=args.trace_max_spans,
        )
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise SystemExit("No completed runs found in the requested experiment.")

    _print_summary(summaries, top_spans=args.top_spans)


if __name__ == "__main__":
    _main()
