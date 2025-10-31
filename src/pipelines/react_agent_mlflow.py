"""MLflow evaluation pipeline for the LangGraph ReAct agent.

This module mirrors the patterns described in the MLflow GenAI documentation
(section *Evaluating Agents* / *Example: Evaluating a Tool-Calling Agent*) and
adapts them for the DACOS-backed refactoring workflow. The pipeline performs a
simple two-stage evaluation:

1. run the LangGraph ReAct agent to suggest a refactoring for a DACOS sample;
2. score the response with an LLM-as-judge scorer registered via ``make_judge``.

The evaluation dataset is generated straight from the DACOS MySQL catalogue via
``mysql_connector``; no git checkout is required. Each evaluation example
contains the minimal context the agent needs (sample metadata and smell
annotation) together with ground-truth expectations for the judge.

Environment variables:
- ``MLFLOW_TRACKING_URI`` (default: ``./mlruns``)
- ``MLFLOW_EXPERIMENT_NAME`` (default: ``react-agent-mlflow``)
- ``MODEL`` (optional LangChain chat model identifier for the agent)
- ``MLFLOW_JUDGE_MODEL`` (default: ``openai:/gpt-4o-mini``)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import mlflow
import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage
from mlflow.entities import AssessmentSource, AssessmentSourceType, SpanType
from mlflow.exceptions import MlflowException
from mlflow.genai import evaluate as mlflow_genai_evaluate
from mlflow.genai import scorer
from mlflow.genai.judges import make_judge

from src.agents.react_agent import graph
from src.agents.react_agent.context import Context
from src.agents.react_agent.utils import get_message_text
from src.data.mysql_connector import (
    fetch_balanced_smell_samples,
    fetch_sample_by_id,
    fetch_samples,
    fetch_samples_dataframe,
    get_connection_pool,
)
from src.models.entities import DACOSSample

# Lazily constructed global context so that environment overrides are respected
_AGENT_CONTEXT: Optional[Context] = None


logger = logging.getLogger(__name__)


LOGGED_METRICS_MESSAGE = "Logged metrics: %s"


def _get_output_text(row: pd.Series) -> Optional[str]:
    """Return the response text for a result row, normalizing non-str values."""

    output = row.get("output")
    if output is None:
        output = row.get("outputs")

    if output is None:
        return None

    if isinstance(output, str):
        return output

    try:
        return json.dumps(output)
    except (TypeError, ValueError):  # pragma: no cover - defensive path
        return str(output)


def _compute_positive_flags(
    output: str, expectations: Dict[str, Any], row: pd.Series
) -> Optional[Tuple[bool, float]]:
    """Return (mentions_smell, f1_score) when the example qualifies as positive."""

    mentions = row.get("mentions_smell")
    if mentions is None:
        mentions = mentions_smell(output, expectations)

    f1_score = row.get("smell_detection_f1")
    if f1_score is None:
        f1_score = smell_detection_f1(output, expectations)

    if mentions and (f1_score or 0.0) >= 0.5:
        return bool(mentions), float(f1_score or 0.0)

    return None


def _log_positive_trace_feedback(
    trace_id: Optional[str],
    *,
    similarity: float,
    sample_id: Any,
    smell_name: str,
    reference_sample_id: Any,
    mentions_smell_flag: bool,
    f1_score: float,
) -> None:
    """Attach similarity feedback and tags to the trace when possible."""

    if not trace_id:
        return

    try:
        mlflow.log_feedback(
            trace_id=trace_id,
            name="refactoring_similarity",
            value=similarity,
            rationale="Positive refactoring example with smell coverage",
            source=AssessmentSource(
                source_type=AssessmentSourceType.CODE,
                source_id="react_agent_mlflow_pipeline",
            ),
            metadata={
                "sample_id": sample_id,
                "smell_name": smell_name,
                "reference_sample_id": reference_sample_id,
                "mentions_smell": mentions_smell_flag,
                "smell_detection_f1": f1_score,
            },
        )
        mlflow.set_trace_tag(
            trace_id=trace_id,
            key="refactoring_success",
            value="true",
        )
    except MlflowException as exc:  # pragma: no cover - backend variability
        logger.debug("Unable to log positive feedback for trace %s: %s", trace_id, exc)


def _safe_update_current_trace(**kwargs: Any) -> None:
    """Update the active MLflow trace when both an active span and API support exist."""

    if not kwargs:
        return

    update_fn = getattr(mlflow, "update_current_trace", None)
    if update_fn is None:
        update_fn = getattr(mlflow, "update_trace", None)
        if update_fn:
            logger.debug("Falling back to legacy mlflow.update_trace API")

    if update_fn is None:
        logger.debug(
            "MLflow does not expose a trace update API; skipping update for %s",
            list(kwargs.keys()),
        )
        return

    get_span = getattr(mlflow, "get_current_active_span", None)
    if callable(get_span):
        try:
            span = get_span()
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive, depends on MLflow internals
            logger.debug("Failed to fetch current MLflow span: %s", exc)
            span = None
        if span is None or getattr(span, "trace_id", None) is None:
            logger.debug(
                "No active MLflow span available; skipping trace update for %s",
                list(kwargs.keys()),
            )
            return

    try:
        update_fn(**kwargs)
    except (
        MlflowException
    ) as exc:  # pragma: no cover - defensive, depends on backend capabilities
        logger.debug("MLflow refused trace update (%s); skipping", exc)


def _log_expectation_with_fallback(
    *,
    trace_id: str,
    name: str,
    value: Any,
    source: AssessmentSource,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an expectation, downgrading gracefully when the server lacks tracing APIs."""

    try:
        mlflow.log_expectation(
            trace_id=trace_id,
            name=name,
            value=value,
            source=source,
            metadata=metadata,
        )
        return
    except MlflowException as exc:
        if "error code 404" not in str(exc):
            raise

        if not getattr(_log_expectation_with_fallback, "_warned", False):
            logger.warning(
                "MLflow server does not expose the /api/3.0 tracing assessments endpoint; "
                "falling back to logging expectations as run artifacts instead."
            )
            _log_expectation_with_fallback._warned = True

        payload = {
            "trace_id": trace_id,
            "name": name,
            "value": value,
            "source_type": getattr(source.source_type, "name", str(source.source_type)),
            "source_id": source.source_id,
            "metadata": metadata or {},
        }
        artifact_path = f"expectations/{trace_id}-{name}.json"
        mlflow.log_dict(payload, artifact_path)


_log_expectation_with_fallback._warned = False  # type: ignore[attr-defined]


def _log_positive_refactoring_examples(df: pd.DataFrame) -> None:
    """Annotate MLflow traces with high-quality refactoring similarities."""

    if df.empty:
        return

    positives: List[Dict[str, Any]] = []
    reference_text_by_smell: Dict[str, str] = {}
    reference_sample_by_smell: Dict[str, Any] = {}
    similarities: List[float] = []

    for idx, row in df.iterrows():
        expectations = row.get("expectations") or {}
        smell_name = (expectations.get("smell_name") or "unknown").strip().lower()
        sample_id = expectations.get("sample_id")
        output = _get_output_text(row)
        if output is None:
            logger.debug(
                "No output available for row %s; skipping positive logging", idx
            )
            continue

        positive_flags = _compute_positive_flags(output, expectations, row)
        if positive_flags is None:
            continue

        mentions_flag, f1_score = positive_flags

        reference_output = reference_text_by_smell.get(smell_name)
        if reference_output is None:
            similarity = 1.0
            reference_text_by_smell[smell_name] = output
            reference_sample_by_smell[smell_name] = sample_id
        else:
            similarity = SequenceMatcher(None, output, reference_output).ratio()

        reference_sample_id = reference_sample_by_smell.get(smell_name)
        similarities.append(similarity)

        output_preview = output[:500]
        positive_entry: Dict[str, Any] = {
            "sample_id": sample_id,
            "smell_name": smell_name,
            "output_preview": output_preview,
            "output_length": len(output),
            "similarity": similarity,
            "reference_sample_id": reference_sample_id,
            "mentions_smell": mentions_flag,
            "smell_detection_f1": f1_score,
        }

        trace_id = row.get("trace_id")
        if trace_id:
            positive_entry["trace_id"] = trace_id
            _log_positive_trace_feedback(
                trace_id,
                similarity=similarity,
                sample_id=sample_id,
                smell_name=smell_name,
                reference_sample_id=reference_sample_id,
                mentions_smell_flag=mentions_flag,
                f1_score=f1_score,
            )

        positives.append(positive_entry)

    if not positives:
        return

    similarity_mean = sum(similarities) / len(similarities)
    mlflow.log_metric("refactoring_similarity/mean", similarity_mean)
    mlflow.log_metric("refactoring_similarity/count", len(similarities))

    positive_df = pd.DataFrame(positives)
    try:
        mlflow.log_table(positive_df, artifact_file="positive_refactorings.json")
    except AttributeError:
        mlflow.log_dict(
            positive_df.to_dict(orient="records"),
            "positive_refactorings.json",
        )


def _extract_outputs_from_tables(tables: Any) -> List[Any]:
    """Return the list of outputs from the evaluation result tables."""

    if not isinstance(tables, dict):
        return []

    eval_table = tables.get("eval_results_table")
    if eval_table is None:
        return []

    candidate_columns = ["outputs", "output", "response", "predictions"]
    for column in candidate_columns:
        if column in eval_table.columns:
            return eval_table[column].tolist()

    for column in eval_table.columns:
        if "output" in column:
            return eval_table[column].tolist()

    return []


def _run_parallel_evaluation(
    data: List[Dict[str, Any]],
    judges: Sequence[Any],
    *,
    max_concurrent: int,
    trace_registry: Sequence[Dict[str, Any]],
):
    """Evaluate samples in parallel, log metrics, and return results."""

    context = _get_agent_context()
    parallel_data = [
        {**row, "trace_id": trace_info.get("trace_id")}
        for row, trace_info in zip(data, trace_registry)
    ]

    evaluation_results = asyncio.run(
        _evaluate_samples_parallel(
            parallel_data, context, max_concurrent=max_concurrent
        )
    )

    outputs_df = pd.DataFrame(
        [
            {
                "inputs": row["inputs"]["inputs"],
                "output": row["output"],
                "expectations": row.get("expectations", {}),
                "trace_id": row.get("trace_id"),
            }
            for row in evaluation_results
        ]
    )

    scored_results = []
    metrics: Dict[str, Any] = {}

    for idx, row in outputs_df.iterrows():
        scores: Dict[str, Any] = {}
        scores["mentions_smell"] = mentions_smell(row["output"], row["expectations"])
        scores["smell_detection_f1"] = smell_detection_f1(
            row["output"], row["expectations"]
        )
        outputs_df.loc[idx, "mentions_smell"] = scores["mentions_smell"]
        outputs_df.loc[idx, "smell_detection_f1"] = scores["smell_detection_f1"]

        for judge in judges:
            judge_name = judge.name if hasattr(judge, "name") else str(judge)
            try:
                judge_result = judge(
                    inputs=row["inputs"],
                    outputs=row["output"],
                    expectations=row["expectations"],
                )
                scores[judge_name] = judge_result
                outputs_df.loc[idx, judge_name] = judge_result
            except Exception as exc:
                logger.warning("Judge %s failed for row %d: %s", judge_name, idx, exc)
                scores[judge_name] = None

        scored_results.append(scores)

    all_score_keys = {key for score in scored_results for key in score.keys()}
    for key in all_score_keys:
        values = [score[key] for score in scored_results if score.get(key) is not None]
        if not values:
            continue
        first = values[0]
        if isinstance(first, (bool, int, float)):
            metrics[f"{key}/mean"] = sum(values) / len(values)

    mlflow.log_metrics(metrics)
    logger.debug(LOGGED_METRICS_MESSAGE, metrics)

    _log_positive_refactoring_examples(outputs_df)

    class EvaluationResult:
        def __init__(self, metrics: Dict[str, Any], results: pd.DataFrame):
            self.metrics = metrics
            self.results = results

    return EvaluationResult(metrics=metrics, results=outputs_df)


def _log_sequential_positive_examples(
    result: Any,
    data: Sequence[Dict[str, Any]],
    trace_registry: Sequence[Dict[str, Any]],
) -> None:
    """Extract sequential outputs, derive heuristics, and log positives."""

    trace_ids = [entry.get("trace_id") for entry in trace_registry]
    expectations_payload = [row.get("expectations", {}) for row in data]
    inputs_payload = [row.get("inputs", {}).get("inputs") for row in data]

    outputs_list = _extract_outputs_from_tables(getattr(result, "tables", {}))

    if not outputs_list:
        logger.debug(
            "Evaluation results table did not expose outputs; falling back to empty list"
        )
        outputs_list = [None] * len(data)

    sequential_df = pd.DataFrame(
        {
            "inputs": inputs_payload,
            "output": outputs_list,
            "expectations": expectations_payload,
            "trace_id": trace_ids,
        }
    )

    sequential_df["mentions_smell"] = sequential_df.apply(
        lambda row: mentions_smell(row.get("output"), row.get("expectations")),
        axis=1,
    )
    sequential_df["smell_detection_f1"] = sequential_df.apply(
        lambda row: smell_detection_f1(row.get("output"), row.get("expectations")),
        axis=1,
    )

    _log_positive_refactoring_examples(sequential_df)


def _get_agent_context() -> Context:
    """Return the shared agent context, instantiating it on first use."""

    global _AGENT_CONTEXT
    if _AGENT_CONTEXT is None:
        _AGENT_CONTEXT = Context()
    return _AGENT_CONTEXT


async def _ainvoke_agent(
    messages: Sequence[BaseMessage], *, context: Context
) -> Dict[str, Any]:
    """Run the LangGraph agent asynchronously and return the final state."""

    return await graph.ainvoke({"messages": list(messages)}, context=context)


def _invoke_agent(
    messages: Sequence[BaseMessage], *, context: Context
) -> Dict[str, Any]:
    """Synchronous helper that wraps ``asyncio.run`` for the agent call."""

    return asyncio.run(_ainvoke_agent(messages, context=context))


def _extract_agent_response(state: Dict[str, Any]) -> str:
    """Return the final LLM output as plain text."""

    messages = state.get("messages", [])
    if not messages:
        raise ValueError("Agent returned no messages")
    return get_message_text(messages[-1])


def _resolve_smell_name(sample: DACOSSample) -> str:
    """Return a canonical smell name derived from DACOS annotations."""

    smell_name = sample.smell_name
    if smell_name:
        return smell_name

    smells = sample.ground_truth_smells()
    if smells:
        return smells[0]

    return "Unknown smell"


def _sample_to_prompt(
    sample: DACOSSample, *, smell_override: Optional[str] = None
) -> str:
    """Craft the user prompt that primes the agent for refactoring advice."""

    smell_name = smell_override or _resolve_smell_name(sample)
    smell_description = sample.smell_description or ""
    constraints = sample.sample_constraints or ""

    prompt_lines = [
        f"We are reviewing DACOS sample {sample.id}.",
        f"Repository slug: {sample.project_name}.",
        f"File: {sample.path_to_file}.",
        f"Annotated smell: {smell_name}.",
    ]

    if smell_description:
        prompt_lines.append(f"Smell description: {smell_description}.")
    if constraints:
        prompt_lines.append(f"Additional notes: {constraints}.")

    prompt_lines.extend(
        [
            "Use the available DACOS tools to ground your reasoning before answering.",
            "Provide a concise refactoring plan that directly addresses the annotated smell.",
        ]
    )

    return " \n".join(prompt_lines)


def _build_expectations(
    sample: DACOSSample, *, smell_override: Optional[str] = None
) -> Dict[str, Any]:
    """Assemble judge expectations for a given sample."""

    return {
        "sample_id": sample.id,
        "smell_name": smell_override or _resolve_smell_name(sample),
        "smell_description": sample.smell_description or "",
    }


def _build_inputs(
    sample: DACOSSample, *, smell_override: Optional[str] = None
) -> Dict[str, str]:
    """Prepare the ``inputs`` block passed to ``predict_fn``."""

    return {"inputs": _sample_to_prompt(sample, smell_override=smell_override)}


def _load_samples_from_ids(sample_ids: Iterable[int]) -> List[DACOSSample]:
    """Load specific samples, filtering out those without smell annotations."""

    samples: List[DACOSSample] = []
    for sample_id in sample_ids:
        sample = fetch_sample_by_id(sample_id)
        if sample is None:
            logger.warning("Sample %s could not be loaded; skipping", sample_id)
            continue

        if not sample.has_smell and not sample.ground_truth_smells():
            logger.warning(
                "Sample %s has no ground-truth smell annotation; skipping", sample_id
            )
            continue

        samples.append(sample)

    return samples


def _load_random_smelly_samples(limit: int) -> List[DACOSSample]:
    """Return a batch of samples that are already flagged as smelly."""

    samples = fetch_samples(has_smell=True, limit=limit)
    if samples:
        return samples

    df = fetch_samples_dataframe(limit=limit)
    fallback_ids = df["id"].tolist() if (not df.empty and "id" in df.columns) else []
    logger.warning(
        "No smelly samples returned by fetch_samples; fallback dataframe IDs: %s",
        fallback_ids,
    )
    return []


TEST_5_PRESET = "test_5"


def _load_eval_examples(
    sample_ids: Optional[Iterable[int]] = None,
    *,
    limit: int = 5,
    preset: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch DACOS samples with ground-truth smells for MLflow evaluation."""

    get_connection_pool()  # ensure credentials are valid before fetching

    if sample_ids and preset:
        raise ValueError("Cannot combine explicit sample IDs with a preset selector.")

    sample_entries: List[Tuple[DACOSSample, Optional[str]]]

    if preset == TEST_5_PRESET:
        balanced = fetch_balanced_smell_samples(per_smell=5)
        sample_entries = [(sample, smell_label) for sample, smell_label in balanced]
    elif sample_ids:
        requested_ids = [int(sid) for sid in sample_ids]
        logger.debug("Using explicit sample IDs: %s", requested_ids)
        samples = _load_samples_from_ids(requested_ids)
        sample_entries = [(sample, None) for sample in samples]
    else:
        samples = _load_random_smelly_samples(limit)
        sample_entries = [(sample, None) for sample in samples]

    if not sample_entries:
        raise ValueError(
            "No DACOS samples with ground-truth smells were loaded for evaluation."
        )

    examples: List[Dict[str, Any]] = []
    for sample, smell_override in sample_entries:
        examples.append(
            {
                "inputs": _build_inputs(sample, smell_override=smell_override),
                "expectations": _build_expectations(
                    sample, smell_override=smell_override
                ),
                "tags": {"smell_name": smell_override or _resolve_smell_name(sample)},
            }
        )

    logger.info("Prepared %d evaluation examples", len(examples))
    return examples


def _configure_mlflow() -> None:
    """Set tracking URI, experiment, and enable LangChain tracing."""

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000"
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "react-agent-mlflow")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    mlflow.langchain.autolog()


def _resolve_judge_models(
    judge_model: Optional[str] = None,
    judge_models: Optional[Sequence[str]] = None,
) -> List[str]:
    """Resolve the list of judge models to use for evaluation."""
    if judge_models:
        return list(judge_models)

    if judge_model:
        return [judge_model]

    # Default judge model from environment or fallback
    default_model = os.getenv("MLFLOW_JUDGE_MODEL", "openai:/gpt-4.1-mini")
    return [default_model]


def _format_judge_label(model: str) -> str:
    """Return a human-readable judge label including provider annotations when possible."""

    if ":/" not in model:
        return model

    provider, remainder = model.split(":/", 1)
    vendor_part, _, model_part = remainder.partition("/")

    if provider == "litellm" and vendor_part and model_part:
        return f"litellm annotation of {vendor_part} ({model_part})"

    if vendor_part and model_part:
        return f"{provider}::{vendor_part}/{model_part}"

    return f"{provider}::{remainder}"


def _quality_judge(model: str):
    """Create the LLM-as-judge scorer for refactoring quality."""

    instructions = (
        "Evaluate the refactoring guidance in {{ outputs }} for the DACOS sample "
        "described in {{ inputs }}. Consider the annotated smell from {{ expectations }} "
        "and score the answer using one of: excellent, good, acceptable, poor. "
        "The response should cite the smell explicitly, propose actionable steps, and remain grounded in the dataset."
    )

    judge_label = _format_judge_label(model)

    return make_judge(
        name=f"refactoring_quality::{judge_label}",
        instructions=instructions,
        model=model,
    )


@scorer
def mentions_smell(outputs: Any, expectations: Dict[str, Any]) -> bool:
    """Simple heuristic scorer that checks if the annotated smell is referenced."""

    smell = (expectations.get("smell_name") or "").strip()
    if not smell:
        return False

    if isinstance(outputs, str):
        payload = outputs
    else:
        payload = json.dumps(outputs)

    return smell.lower() in payload.lower()


@scorer
def smell_detection_f1(outputs: Any, expectations: Dict[str, Any]) -> float:
    """Calculate F1 score for smell detection based on ground truth.

    This scorer evaluates whether the agent correctly identifies the smell type.
    It calculates precision, recall, and F1 score based on:
    - True Positive: Agent mentions the correct smell that is present
    - False Positive: Agent mentions a smell that is not present
    - False Negative: Agent fails to mention a smell that is present
    """

    if isinstance(outputs, str):
        payload = outputs.lower()
    else:
        payload = json.dumps(outputs).lower()

    # Get expected smell
    expected_smell = (expectations.get("smell_name") or "").strip().lower()

    # Common smell types to check
    smell_types = [
        "complex method",
        "long method",
        "insufficient modularization",
        "long parameter list",
        "multifaceted abstraction",
    ]

    # Determine detected smells
    detected_smells = {smell for smell in smell_types if smell in payload}

    # Ground truth: the expected smell should be present
    true_smells = {expected_smell} if expected_smell else set()

    # Calculate metrics
    true_positives = len(detected_smells & true_smells)
    false_positives = len(detected_smells - true_smells)
    false_negatives = len(true_smells - detected_smells)

    # Calculate precision and recall
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    # Calculate F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return f1


@mlflow.trace(span_type=SpanType.CHAIN, attributes={"component": "react_agent"})
def predict_refactoring(inputs: str) -> str:
    """Prediction function invoked by ``mlflow.genai.evaluate`` with tracing."""

    context = _get_agent_context()
    _safe_update_current_trace(request_preview=inputs[:200])

    try:
        state = _invoke_agent([HumanMessage(content=inputs)], context=context)
        response = _extract_agent_response(state)
        _safe_update_current_trace(response_preview=response[:200])
        return response
    except Exception as exc:  # pragma: no cover - defensive path for runtime issues
        payload = json.dumps({"error": str(exc)})
        _safe_update_current_trace(
            tags={"react_agent.error": type(exc).__name__},
            response_preview=payload[:200],
        )
        return payload


async def _evaluate_single_sample(
    row: Dict[str, Any],
    context: Context,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Evaluate a single sample with rate limiting via semaphore."""

    async with semaphore:
        inputs = row["inputs"]["inputs"]
        message = HumanMessage(content=inputs)

        sample_id = row.get("expectations", {}).get("sample_id")

        try:
            state = await graph.ainvoke({"messages": [message]}, context=context)
            response = _extract_agent_response(state)

            return {
                **row,
                "output": response,
            }
        except Exception as exc:
            logger.exception(
                "Failed to evaluate sample %s: %s",
                sample_id,
                exc,
            )
            raise


async def _evaluate_samples_parallel(
    data: List[Dict[str, Any]],
    context: Context,
    *,
    max_concurrent: int = 5,
) -> List[Dict[str, Any]]:
    """Evaluate multiple samples in parallel using async with concurrency control.

    Args:
        data: List of evaluation examples with inputs, expectations, and tags
        context: Agent context with model and configuration
        max_concurrent: Maximum number of concurrent evaluations (default: 5)

    Returns:
        List of evaluation results with outputs and trace IDs
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    logger.info(
        "Starting parallel evaluation of %d samples with max_concurrent=%d",
        len(data),
        max_concurrent,
    )

    tasks = [_evaluate_single_sample(row, context, semaphore) for row in data]
    results = await asyncio.gather(*tasks)

    logger.info("Completed parallel evaluation of %d samples", len(results))
    return results


def evaluate_react_agent(
    sample_ids: Optional[Iterable[int]] = None,
    *,
    limit: int = 5,
    sample_preset: Optional[str] = None,
    judge_model: Optional[str] = None,
    judge_models: Optional[Sequence[str]] = None,
    max_concurrent: int = 5,
    use_parallel: bool = True,
):
    """Run the MLflow GenAI evaluation for the ReAct agent.

    Args:
        sample_ids: Optional iterable of DACOS sample identifiers to evaluate.
        limit: Maximum number of random samples (via dataframe query) when ``sample_ids`` is ``None``. Ignored when ``sample_preset`` is provided.
        sample_preset: Optional preset name for deterministic dataset selection (e.g. ``"test_5"``).
        judge_model: Optional override for the LLM-as-judge model identifier.
        judge_models: Optional list of LLM-as-judge model identifiers.
        max_concurrent: Maximum number of concurrent evaluations when using parallel mode (default: 5).
        use_parallel: Whether to use parallel evaluation (default: True). Set to False to use sequential MLflow evaluation.

    Returns:
        ``EvaluationResult`` from ``mlflow.genai.evaluate`` or custom parallel evaluation.
    """

    logger.info("Starting React agent evaluation (parallel=%s)", use_parallel)
    _configure_mlflow()

    if sample_preset:
        logger.info("Using sample preset: %s", sample_preset)

    effective_limit = limit
    if not sample_ids and not sample_preset:
        effective_limit = max(limit, 5)
        if effective_limit != limit:
            logger.info(
                "Adjusted limit from %d to %d to meet minimum dataset size",
                limit,
                effective_limit,
            )

    data = _load_eval_examples(sample_ids, limit=effective_limit, preset=sample_preset)

    if sample_preset:
        effective_limit = len(data)

    models = _resolve_judge_models(judge_model, judge_models)
    judges = [_quality_judge(model_id) for model_id in models]
    logger.info("Resolved judge models: %s", models)

    # Explicit MLflow run to log configuration & expectations as assessments.
    with mlflow.start_run(run_name="react-agent-eval"):
        logger.debug("Logging run parameters and expectations")
        mlflow.log_params(
            {
                "limit": effective_limit,
                "resolved_dataset_size": len(data),
                "provided_sample_ids": ",".join(
                    str(row.get("expectations", {}).get("sample_id"))
                    for row in data
                    if row.get("expectations", {}).get("sample_id") is not None
                ),
                "judge_models": ",".join(models),
                "sample_preset": sample_preset or "",
                "use_parallel": use_parallel,
                "max_concurrent": max_concurrent if use_parallel else 1,
            }
        )

        # Log expectations (ground truth smell info) to the trace layer so they appear
        # in the Traces UI; create a lightweight trace per sample for richer inspection.
        trace_registry: List[Dict[str, Any]] = []

        for row in data:
            exp = row.get("expectations", {})
            # Create a synthetic trace capturing input prompt & placeholder output
            trace_id = mlflow.log_trace(
                name="sample_refactoring",
                request=row.get("inputs", {}).get("inputs"),
                response=None,
                attributes={
                    "smell_name": exp.get("smell_name", "unknown"),
                    "sample_id": exp.get("sample_id"),
                },
                tags=row.get("tags", {}),
            )
            # Human annotated expectation source metadata
            source = AssessmentSource(
                source_type=AssessmentSourceType.HUMAN,
                source_id="dacos-catalogue",
            )
            _log_expectation_with_fallback(
                trace_id=trace_id,
                name="expected_smell_name",
                value=exp.get("smell_name"),
                source=source,
                metadata={"sample_id": exp.get("sample_id")},
            )
            _log_expectation_with_fallback(
                trace_id=trace_id,
                name="expected_smell_description",
                value=exp.get("smell_description"),
                source=source,
                metadata={"sample_id": exp.get("sample_id")},
            )

            trace_registry.append(
                {
                    "trace_id": trace_id,
                    "sample_id": exp.get("sample_id"),
                    "smell_name": exp.get("smell_name"),
                }
            )

        if use_parallel:
            logger.info(
                "Running parallel evaluation with max_concurrent=%d", max_concurrent
            )
            evaluation_result = _run_parallel_evaluation(
                data=data,
                judges=judges,
                max_concurrent=max_concurrent,
                trace_registry=trace_registry,
            )
            logger.info("Parallel evaluation completed")
            return evaluation_result

        result = mlflow_genai_evaluate(
            data=data,
            predict_fn=predict_refactoring,
            scorers=[*judges, mentions_smell, smell_detection_f1],
        )
        logger.info("mlflow.genai.evaluate completed")

        mlflow.log_metrics(
            {k: v for k, v in result.metrics.items() if isinstance(v, (int, float))}
        )
        logger.debug(LOGGED_METRICS_MESSAGE, result.metrics)

        _log_sequential_positive_examples(result, data, trace_registry)
        return result


def _parse_cli_args(argv: Optional[Sequence[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate the LangGraph ReAct agent with MLflow GenAI."
    )
    parser.add_argument(
        "--sample-ids",
        type=int,
        nargs="*",
        help="Explicit DACOS sample IDs to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Fallback sample count when --sample-ids is omitted",
    )
    parser.add_argument(
        "--sample-preset",
        choices=[TEST_5_PRESET],
        default=None,
        help="Shortcut dataset selection; 'test_5' loads up to five samples per smell type.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override MLFLOW_JUDGE_MODEL for the LLM-as-judge scorer",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("PIPELINE_LOG_LEVEL", "INFO"),
        help="Log level (e.g. DEBUG, INFO, WARNING). Defaults to PIPELINE_LOG_LEVEL env or INFO.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of concurrent evaluations when using parallel mode (default: 5)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel evaluation and use sequential MLflow evaluation instead",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_cli_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        force=True,
    )
    logger.debug("CLI arguments: %s", args)
    result = evaluate_react_agent(
        sample_ids=args.sample_ids,
        limit=args.limit,
        sample_preset=args.sample_preset,
        judge_model=args.judge_model,
        max_concurrent=args.max_concurrent,
        use_parallel=not args.no_parallel,
    )

    print("Evaluation finished. Metrics:")
    for metric, value in result.metrics.items():
        print(f"- {metric}: {value}")


if __name__ == "__main__":
    main()
