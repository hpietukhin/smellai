#!/usr/bin/env python3
"""Online-only composite workflow.

Loop per step:
  detect smells -> plan action (greedy/befs) -> execute LLM refactoring
  -> run java build/tests -> re-detect smells -> replan

This workflow is intentionally runtime-first (actual detector state is source of truth).
"""
from __future__ import annotations

import csv
import concurrent.futures
import contextlib
import contextvars
import functools
import importlib.util
import faulthandler
import io
import logging
import os
import re
import subprocess
import sys
import shutil
import tempfile
import time
import traceback
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import httpx
import javalang  # type: ignore[import-untyped]
from langchain_core.runnables import Runnable
from litellm.exceptions import APIConnectionError as LiteLLMAPIConnectionError
from litellm.exceptions import APIError as LiteLLMAPIError
from litellm.exceptions import BadRequestError as LiteLLMBadRequestError
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import typer
from typer_config.decorators import use_json_config
import orjson
from mlflow import langchain, litellm

from agents.java_test.agent import run_java_test_analysis
from agents.java_test.java_version import java_version_prompt_context
from agents.litellm_config import current_datetime_context, load_openrouter_env, make_openrouter_chat_model
from agents.observability import start_action
from agents.tools.java_inspection_tools import (
    JavaInspectorUnavailableError,
    resolve_smell_location_value,
    set_java_inspector_url,
)
from dataset.organic_detector import OrganicDetector
from domain.dependency_graph import DependencyGraph
from domain.detector import SmellDetectionError, SmellDetector, StaticDetector
from domain.refactoring_tree import RefactoringTree, State
from sonarqube.detector import SonarQubeDetector

LOGGER = logging.getLogger(__name__)

# Repair policy for java_test_analysis (Python-side configuration only).
JAVA_TEST_CODE_AGENT_ENABLED = True
JAVA_TEST_CODE_AGENT_MODEL = "openrouter/openai/gpt-oss-120b:free"
JAVA_TEST_CODE_AGENT_MAX_STEPS = 4
JAVA_TEST_CODE_AGENT_MAX_ATTEMPTS = 2
JAVA_TEST_CODE_AGENT_TIMEOUT = 180

_CURRENT_TRACKER: contextvars.ContextVar[Any | None] = contextvars.ContextVar("composite_workflow_tracker", default=None)
_CURRENT_STEP: contextvars.ContextVar[int | None] = contextvars.ContextVar("composite_workflow_step", default=None)

DEFAULT_REPOS_ROOT = Path("/Users/havriil.pietukhin/uni/masterThesis/code/repos")
DEFAULT_REFRACTOR_MODEL = os.environ.get("COMPOSITE_REFACTOR_MODEL", "openrouter/minimax/minimax-m2.7")
KNOWN_REPO_URLS = {
    "Apache Tomcat": "https://github.com/apache/tomcat.git",
    "JUnit4": "https://github.com/junit-team/junit4.git",
    "Lyra": "https://github.com/jhalterman/lyra.git",
    "OkHttp": "https://github.com/square/okhttp.git",
    "PhiCode Philib": "https://github.com/PhiCode/philib.git",
    "Tap4j": "https://github.com/tupilabs/tap4j.git",
}


def _configure_workflow_logging(verbose: bool = False) -> None:
    """Configure plain Python logging; MLflow stores per-run log files as artifacts."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )




@functools.lru_cache(maxsize=1)
def _load_workflow_env(env_file: str | Path = ".env") -> None:
    """Load workflow credentials from .env only once per process."""
    load_openrouter_env(str(env_file))
    os.environ.setdefault("OR_APP_NAME", "smellai-composite-workflow")


def _get_refactor_model(model_override: str | None = None) -> str:
    return model_override or os.environ.get("COMPOSITE_REFACTOR_MODEL", DEFAULT_REFRACTOR_MODEL)


@functools.lru_cache(maxsize=1)
def _enable_litellm_autologging(mlflow_module: Any) -> None:
    try:
        litellm.autolog()
    except (AttributeError, RuntimeError, ImportError) as exc:
        raise RuntimeError("MLflow LiteLLM autologging must be enabled for this workflow") from exc
    LOGGER.info("PHASE mlflow_litellm_autolog enabled")


@functools.lru_cache(maxsize=1)
def _enable_langchain_autologging(mlflow_module: Any) -> None:
    try:
        langchain.autolog(run_tracer_inline=True)
    except (AttributeError, RuntimeError, ImportError) as exc:
        raise RuntimeError("MLflow LangChain/LangGraph autologging must be enabled for this workflow") from exc
    LOGGER.info("PHASE mlflow_langchain_autolog enabled")


def _attach_mlflow_log_file(run_name: str) -> tuple[logging.FileHandler, Path]:
    log_dir = Path(tempfile.mkdtemp(prefix="composite_workflow_logs_"))
    log_path = log_dir / f"{_safe_name(run_name)}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler, log_path


def _detach_mlflow_log_file(handler: logging.FileHandler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.flush()
    handler.close()


_load_workflow_env()


@dataclass
class StepLog:
    step: int
    smell_count_before: int
    smell_count_after: int
    h_before: float
    h_after: float
    action_smell_id: str
    action_ref_type: str
    compile_passed: bool
    tests_passed: bool
    execution_ok: bool
    stop_reason: str | None = None


class RefactorFilePatch(BaseModel):
    """Single Java file replacement emitted by the refactor LLM."""

    file_path: str = Field(description="Path to Java file relative to repository root.")
    java_source: str = Field(
        description=(
            "Complete Java compilation unit. Must start with 'package', 'import', "
            "or a comment. No markdown fences and no prose."
        )
    )

    @field_validator("java_source")
    @classmethod
    def _validate_source_shape(cls, value: str) -> str:
        source = value.strip()
        if not source:
            raise ValueError("java_source must not be empty")
        if "```" in source:
            raise ValueError("java_source must not contain markdown fences")
        if not source.startswith(("package ", "import ", "/**", "/*", "//")):
            raise ValueError("java_source must start with package, import, or a Java comment")
        return source + ("\n" if not source.endswith("\n") else "")


class JavaRefactorOutput(BaseModel):
    """Structured response for Java refactoring output."""

    java_source: str | None = Field(
        default=None,
        description=(
            "Complete Java compilation unit for the primary target file. "
            "Use this for file-scope edits."
        ),
    )
    files: list[RefactorFilePatch] | None = Field(
        default=None,
        description=(
            "Complete Java source for one or more files used by multi-file refactorings. "
            "Use repository-relative paths."
        ),
    )
    refactoring_summary: str = Field(description="One-sentence description of the refactoring applied.")

    @field_validator("java_source")
    @staticmethod
    def _validate_source_shape(value: str | None) -> str | None:
        if value is None:
            return value
        source = value.strip()
        if not source:
            raise ValueError("java_source must not be empty")
        if "```" in source:
            raise ValueError("java_source must not contain markdown fences")
        if not source.startswith(("package ", "import ", "/**", "/*", "//")):
            raise ValueError("java_source must start with package, import, or a Java comment")
        return source + ("\n" if not source.endswith("\n") else "")

    @field_validator("files")
    @staticmethod
    def _validate_files(value: list[RefactorFilePatch] | None) -> list[RefactorFilePatch] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("files must not be empty when provided")
        return value

    @model_validator(mode="after")
    @staticmethod
    def _require_payload(value: "JavaRefactorOutput") -> "JavaRefactorOutput":
        if not value.java_source and not value.files:
            raise ValueError("at least one of 'java_source' or 'files' must be provided")
        return value


def _complete_h_trace(pre_step_h_trace: list[float], step_logs: list[StepLog]) -> list[float]:
    """Return a state-level h trace: initial h plus one terminal h per executed step.

    During online execution we observe ``h_before`` at step start and ``h_after``
    after verification/redetection. Evaluation metrics must use the terminal
    state, so the complete trace is ``[h0, step0.h_after, step1.h_after, ...]``.
    """
    if not pre_step_h_trace:
        return []
    return [float(pre_step_h_trace[0]), *[float(row.h_after) for row in step_logs]]


RefactorScope = Literal["file", "project", "auto"]

RefactorFailureReason = Literal[
    "llm_structured_none",
    "llm_no_change",
    "llm_invalid_java",
    "compile_fail",
    "tests_fail",
    "execution_error",
    "execution_ok",
]


class _RefactorLLMError(RuntimeError):
    """Error raised for explicit refactoring execution failure reasons."""

    def __init__(self, reason: RefactorFailureReason, message: str):
        super().__init__(message)
        self.reason = reason



def _extract_fallback_json_text(raw: str) -> str | None:
    """Extract a likely JSON payload from a model response string."""
    text = raw.strip()
    if not text:
        return None

    if "```" in text:
        start = text.find("```")
        if start >= 0:
            payload = text[start + 3 :].lstrip()
            if payload.startswith("json"):
                payload = payload[4:].lstrip()
            end = payload.find("```")
            if end >= 0:
                candidate = payload[:end].strip()
                if candidate:
                    return candidate

    start_json = text.find("{")
    if start_json < 0:
        return None
    end_json = text.rfind("}")
    if end_json <= start_json:
        return None
    return text[start_json : end_json + 1].strip()


def _coerce_refactor_output(raw: Any, *, context: str) -> JavaRefactorOutput:
    """Validate/transform candidate structured-refactor payload into model object."""
    if isinstance(raw, JavaRefactorOutput):
        return raw

    if isinstance(raw, str):
        try:
            candidate_json = _extract_fallback_json_text(raw)
        except ValueError:
            raise _RefactorLLMError(
                "llm_structured_none",
                f"{context}: response text could not be parsed as JSON",
            ) from None
        if candidate_json is None:
            raise _RefactorLLMError(
                "llm_structured_none",
                f"{context}: response text did not contain JSON payload",
            )
        try:
            parsed = orjson.loads(candidate_json)
        except orjson.JSONDecodeError as exc:
            raise _RefactorLLMError(
                "llm_structured_none",
                f"{context}: failed to parse JSON from text response ({exc})",
            ) from exc
        return _coerce_refactor_output(parsed, context=context)

    if isinstance(raw, dict):
        try:
            return JavaRefactorOutput.model_validate(raw)
        except ValidationError as exc:
            raise _RefactorLLMError(
                "llm_structured_none",
                f"{context}: payload did not match JavaRefactorOutput ({exc})",
            ) from exc

    content = None
    if hasattr(raw, "content"):
        raw_content = getattr(raw, "content")
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            parts = [
                item
                for item in raw_content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            if parts:
                content = parts[0]["text"].strip()
    if content is not None:
        return _coerce_refactor_output(content, context=context)

    raise _RefactorLLMError(
        "llm_structured_none",
        f"{context}: unsupported structured output type {type(raw)!r}",
    )


class WorkflowRunArgs(BaseSettings):
    """Validated external workflow configuration from CLI/JSON/batch records."""

    model_config = SettingsConfigDict(
        env_prefix="COMPOSITE_",
        env_file=".env",
        extra="ignore",
        validate_assignment=True,
    )

    planner: Literal["greedy", "befs"] = "befs"
    repos_root: str = str(DEFAULT_REPOS_ROOT)
    repo_url: str = ""
    start_commit_hash: str = Field(default="", min_length=1)
    worktree_suffix: str = ""
    detector_backend: Literal["organic", "sonar", "dummy", "static"] = "organic"
    locality: Literal["none", "class", "file"] = "none"
    max_steps: int = Field(default=5, ge=1)
    max_no_progress: int = Field(default=2, ge=0)
    retry_budget: int = Field(default=1, ge=0)
    timeout: int = Field(default=300, ge=1)
    organic_dir: str | None = None
    sonar_url: str = "http://localhost:9000"
    experiment: str = "composite_workflow_full"
    mlflow_healthcheck_experiment: str = "planner-eval"
    skip_mlflow_healthcheck: bool = False
    skip_java_inspector_healthcheck: bool = False
    run_name: str | None = None
    model: str | None = None
    refactor_scope: RefactorScope = "project"
    project: str = ""
    elements: str = ""
    h_reduction_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    eval_patch_script: str = ""
    targeted_testing: bool = True
    verbose: bool = False

    @field_validator("eval_patch_script")
    @classmethod
    def _ignore_eval_patch(cls, value: str) -> str:
        return ""


def _build_run_args(**kwargs: Any) -> WorkflowRunArgs:
    payload = dict(kwargs)
    # Deprecated compatibility option; intentionally ignored.
    payload["eval_patch_script"] = ""
    try:
        return WorkflowRunArgs.model_validate(payload)
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _safe_name(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return safe.strip("_") or "project"


def _profile_phase(phase: str, span_type: str = "TOOL"):
    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            step = _CURRENT_STEP.get()
            tracker = _CURRENT_TRACKER.get()
            started = time.monotonic()
            action_ok = False
            with start_action(action_type=phase, span_type=span_type, step=step) as action:
                try:
                    result = func(*args, **kwargs)
                    action_ok = True
                    return result
                finally:
                    elapsed_ms = (time.monotonic() - started) * 1000.0
                    if tracker is not None:
                        with suppress(AttributeError, RuntimeError, TypeError):
                            tracker.log_timing(phase, elapsed_ms, step=step)
                    if action_ok:
                        action.addSuccessFields(duration_ms=elapsed_ms, step=step)

        return _wrapper

    return _decorator


def _first_present(sources: tuple[dict[str, Any], ...], keys: tuple[str, ...]) -> Any | None:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value is not None:
                return value
    return None


def _case_args_from_batch_case(
    cfg: dict[str, Any],
    case_record: dict[str, Any],
    run_name: str,
) -> WorkflowRunArgs:
    refactor_scope = cfg.get("refactor_scope", "project")

    raw_meta = case_record.get("meta")
    nested_meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    sources: tuple[dict[str, Any], ...] = (case_record, nested_meta)

    project = case_record.get("project") or cfg.get("project")
    if not project:
        raise ValueError("batch case is missing project")

    repo_url = case_record.get("repo_url") or _resolve_repo_url(str(project), "")
    if not repo_url:
        raise ValueError(f"batch case {case_record.get('case_id')!r} is missing repo_url and project {project!r} is unknown")

    start_commit_hash = _first_present(sources, ("start_commit_hash", "start_commit", "commit_hash"))
    if not isinstance(start_commit_hash, str) or not start_commit_hash.strip():
        raise ValueError(f"batch case {case_record.get('case_id')!r} has invalid start_commit_hash")

    verification = case_record.get("baseline_verification")
    if verification is not None and verification.get("status") != "passed":
        raise ValueError(f"batch case {case_record.get('case_id')!r} is not baseline-verified")

    elements = _normalize_case_elements(_first_present(sources, ("elements",)) or case_record.get("elements") or [])
    case_cfg: dict[str, Any] = dict(cfg)
    case_cfg.update(
        {
            "project": project,
            "elements": elements,
            "repo_url": repo_url,
            "start_commit_hash": start_commit_hash,
            "run_name": run_name,
            "worktree_suffix": run_name,
            "refactor_scope": refactor_scope,
        }
    )
    return _build_args_from_config(case_cfg)


def _case_log_path(log_dir: Path, offset: int, case_id: str, case_args: WorkflowRunArgs) -> Path:
    run_token = _safe_name(case_args.run_name or case_id or f"case-{offset}")
    return log_dir / f"{offset:03d}-{run_token}.log"


def _derive_run_name_prefix(cfg: dict[str, Any], config_path: Path, now: datetime | None = None) -> str:
    """Create a searchable batch run prefix from stable eval parameters."""
    dt = now or datetime.now(UTC)
    date = dt.strftime("%Y%m%d")
    model = _safe_name(str(cfg.get("model") or _get_refactor_model()))[:60]
    batch = _safe_name(Path(str(cfg.get("batch_list") or cfg.get("manifest") or config_path.stem)).stem)[:40]
    return "-".join(
        [
            "full",
            date,
            f"batch-{batch}",
            f"planner-{_safe_name(str(cfg.get('planner', 'befs')))}",
            f"det-{_safe_name(str(cfg.get('detector_backend', 'organic')))}",
            f"loc-{_safe_name(str(cfg.get('locality', 'none')))}",
            f"model-{model}",
            f"steps-{int(cfg.get('max_steps', 5))}",
        ]
    )


def _workflow_cli_args(case_args: WorkflowRunArgs) -> list[str]:
    cli_args: list[str] = []
    for field_name in WorkflowRunArgs.model_fields:
        value = getattr(case_args, field_name)
        option = f"--{field_name.replace('_', '-') }"
        if isinstance(value, bool):
            if value:
                cli_args.append(option)
            elif field_name == "targeted_testing":
                cli_args.append(f"--no-{option[2:]}")
            continue
        if value is None:
            continue
        cli_args.extend([option, str(value)])
    return cli_args


class GitError(RuntimeError):
    """Raised when a git command fails."""


def _run_git(args: list[str], cwd: Path | None = None, *, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {result.stderr[-500:]}")
    return result


@contextlib.contextmanager
def _path_lock(lock_dir: Path, timeout: int = 300):
    started = time.monotonic()
    while True:
        try:
            lock_dir.mkdir(parents=True)
            break
        except FileExistsError:
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"Timed out waiting for lock: {lock_dir}")
            time.sleep(0.25)
    try:
        yield
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


def _resolve_repo_url(project: str, override: str) -> str:
    if override:
        return override
    url = KNOWN_REPO_URLS.get(project)
    if not url:
        raise ValueError(f"Unknown project={project!r}. Pass --repo-url explicitly.")
    return url


@_profile_phase("prepare_repo_checkout")
def _prepare_repo_checkout(
    project: str, repo_url: str, repos_root: Path, commit_hash: str, worktree_suffix: str = ""
) -> Path:
    if not commit_hash:
        raise ValueError("--start-commit-hash is required")
    slug = _safe_name(project)
    bare_dir = repos_root / "_bare" / f"{slug}.git"
    worktree_leaf = commit_hash[:12]
    if worktree_suffix:
        worktree_leaf = f"{worktree_leaf}-{_safe_name(worktree_suffix)[:80]}"
    worktree_dir = repos_root / "worktrees" / slug / worktree_leaf
    repos_root.mkdir(parents=True, exist_ok=True)
    bare_dir.parent.mkdir(parents=True, exist_ok=True)
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)

    # Git's bare repository and worktree metadata are shared per project; keep
    # checkout preparation serialized while allowing the expensive per-case
    # detect/refactor/test loop to run in parallel after checkout.
    with _path_lock(repos_root / "_locks" / f"{slug}.lock"):
        if not bare_dir.exists():
            _run_git(["clone", "--bare", "--filter=blob:none", repo_url, str(bare_dir)])
        else:
            _run_git(["--git-dir", str(bare_dir), "remote", "set-url", "origin", repo_url])
            _run_git(["--git-dir", str(bare_dir), "fetch", "--prune", "origin"])

        fetched = _run_git([
            "--git-dir",
            str(bare_dir),
            "fetch",
            "--depth=1",
            "--filter=blob:none",
            "origin",
            commit_hash,
        ], check=False)
        if fetched.returncode != 0:
            # Fallback for servers/commits where direct SHA fetch is restricted.
            _run_git(
                [
                    "--git-dir",
                    str(bare_dir),
                    "fetch",
                    "--filter=blob:none",
                    "--prune",
                    "origin",
                    "+refs/heads/*:refs/remotes/origin/*",
                    "+refs/tags/*:refs/tags/*",
                ]
            )

        if worktree_dir.exists():
            _run_git(["--git-dir", str(bare_dir), "worktree", "remove", "--force", str(worktree_dir)], check=False)
            if worktree_dir.exists():
                shutil.rmtree(worktree_dir, ignore_errors=True)

        _run_git([
            "--git-dir",
            str(bare_dir),
            "worktree",
            "add",
            "--detach",
            str(worktree_dir),
            commit_hash,
        ])
    return worktree_dir


def _select_detector(args: WorkflowRunArgs) -> SmellDetector:
    if args.detector_backend == "organic":
        organic_dir = Path(args.organic_dir) if args.organic_dir else None
        return OrganicDetector(organic_dir=organic_dir, timeout=args.timeout)
    if args.detector_backend == "sonar":
        return SonarQubeDetector(sonar_url=args.sonar_url)
    # "dummy" (preferred name) and legacy "static" are equivalent.
    return StaticDetector([])


def _filter_smells_to_elements(smells, elements_csv: str):
    elements = _parse_elements_arg(elements_csv)
    if not elements:
        return smells

    def matches(smell) -> bool:
        class_name = smell.class_name or ""
        method_sig = smell.method_signature or ""
        file_path = smell.file_path or ""
        for elem in elements:
            if class_name == elem:
                return True
            if class_name and elem.startswith(class_name + "."):
                return True
            if class_name.startswith(elem + "."):
                return True
            if method_sig and (elem.endswith("." + method_sig) or elem.endswith(method_sig)):
                return True
            if file_path and file_path.replace("/", ".").removesuffix(".java") == elem:
                return True
        return False

    return [s for s in smells if matches(s)]


@_profile_phase("detect_smells")
def _detect(detector: SmellDetector, repo_path: Path, elements_csv: str = ""):
    started = time.monotonic()
    LOGGER.info("PHASE detect_smells start repo=%s detector=%s", repo_path, detector.__class__.__name__)
    smells_all = detector.detect(repo_path)
    elapsed = time.monotonic() - started
    if not isinstance(smells_all, list):
        raise TypeError("Detector must return list[SmellEvent]")
    for s in smells_all:
        if not s.smell_id:
            raise ValueError("SmellEvent.smell_id must be non-empty")
    smells = _filter_smells_to_elements(smells_all, elements_csv)
    LOGGER.info(
        "PHASE detect_smells done smell_count=%d raw_smell_count=%d filtered=%s elapsed=%.2fs",
        len(smells),
        len(smells_all),
        bool(elements_csv),
        elapsed,
    )
    return smells


@_profile_phase("pick_next_action")
def _pick_next_action(smells, planner: str, locality: str):
    started = time.monotonic()
    LOGGER.info("PHASE planner start planner=%s locality=%s smell_count=%d", planner, locality, len(smells))
    LOGGER.info("PHASE dependency_graph start smell_count=%d", len(smells))
    dep_graph = DependencyGraph.from_events(smells, locality=locality)
    LOGGER.info("PHASE dependency_graph done nodes=%d edges=%d elapsed=%.2fs", len(dep_graph), dep_graph.graph.number_of_edges(), time.monotonic() - started)
    initial = State(frozenset(s.smell_id for s in smells))
    tree = RefactoringTree(initial, dep_graph)

    plan_started = time.monotonic()
    if planner == "greedy":
        plan = tree.greedy()
    else:
        plan = tree.befs()
    LOGGER.info(
        "PHASE planner done planner=%s actions=%d h_trace_len=%d elapsed=%.2fs total_elapsed=%.2fs",
        planner,
        len(plan.actions),
        len(plan.h_trace),
        time.monotonic() - plan_started,
        time.monotonic() - started,
    )

    if not plan.actions:
        return None, dep_graph, initial
    return plan.actions[0], dep_graph, initial


@functools.lru_cache(maxsize=16)
def _structured_refactor_model(model_name: str, method: str) -> Runnable[Any, object]:
    """Create a structured-output model for one method.

    Structured-output support varies by model/provider response path. This helper is
    intentionally small so the caller can attempt multiple methods (function
    calling, then JSON schema) as needed.
    """
    model = make_openrouter_chat_model(model_name)
    try:
        structured = model.with_structured_output(JavaRefactorOutput, method=method)
    except (NotImplementedError, AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Model {model_name!r} does not support structured output method {method!r}") from exc
    return cast(Runnable[Any, object], structured)


def _invoke_refactor_llm(model_name: str, messages: Sequence[dict[str, str]]) -> JavaRefactorOutput:
    """Call the refactoring model with fallback structured-output methods."""
    methods: tuple[str, ...] = ("function_calling", "json_schema")
    last_error: Exception | None = None

    for method in methods:
        LOGGER.info("PHASE llm_refactor_call method=%s model=%s", method, model_name)
        try:
            structured_model = _structured_refactor_model(model_name, method)
            result = structured_model.invoke(list(messages))
            LOGGER.debug("Structured output method=%s type=%s", method, type(result).__name__)
            if result is None:
                LOGGER.warning("Structured output returned None with method=%s for model=%s", method, model_name)
                last_error = _RefactorLLMError(
                    "llm_structured_none",
                    f"Structured output returned None with method {method}",
                )
                continue
            if isinstance(result, JavaRefactorOutput):
                return result
            try:
                return _coerce_refactor_output(result, context=f"method={method}")
            except _RefactorLLMError as exc:
                LOGGER.warning("Structured output parse/validation failed for method=%s: %s", method, exc)
                last_error = exc
                # Keep trying other structured methods if validation/parsing fails.
                if method == methods[-1]:
                    raise
        except (RuntimeError, _RefactorLLMError, ValueError) as exc:
            last_error = exc
            LOGGER.warning("Structured output call failed for method=%s: %s", method, exc)
            if method != methods[-1]:
                continue
            if isinstance(exc, _RefactorLLMError):
                raise
            raise _RefactorLLMError(
                "llm_structured_none",
                f"Unable to invoke structured output with method {method}: {exc}",
            ) from exc

    if last_error is not None:
        if isinstance(last_error, _RefactorLLMError):
            raise last_error
        raise _RefactorLLMError(
            "llm_structured_none",
            f"All structured-output methods exhausted; last error: {last_error}",
        ) from last_error
    raise _RefactorLLMError(
        "llm_structured_none",
        "All structured-output methods returned empty responses",
    )


def _validate_java_compilation_unit(source: str) -> tuple[bool, str | None]:
    try:
        javalang.parse.parse(source)
        return True, None
    except javalang.parser.JavaSyntaxError as exc:
        return False, f"java syntax: {exc.description} at {exc.at}"
    except javalang.tokenizer.LexerError as exc:
        return False, f"java lexer: {exc}"


def _java_path_suffixes(raw_path: str) -> list[str]:
    """Return plausible Java file suffixes, including outer class for inner-class paths."""
    normalized = raw_path.strip().replace("\\", "/")
    if not normalized.endswith(".java"):
        return []
    suffixes = [normalized]
    parts = normalized.split("/")
    if len(parts) <= 1:
        return suffixes

    # Some detectors report inner/nested classes as Outer/Inner.java even though
    # the physical source file is Outer.java. Add progressively shorter outer
    # source candidates while preserving the package/path prefix.
    stem = parts[-1].removesuffix(".java")
    prefix = parts[:-1]
    for index in range(len(prefix) - 1, -1, -1):
        outer = prefix[index]
        if outer and outer[0].isupper():
            suffixes.append("/".join([*prefix[: index + 1], outer + ".java"]))
            break
    if stem and stem[0].isupper() and len(prefix) >= 1:
        parent = prefix[-1]
        suffixes.append("/".join([*prefix[:-1], parent + ".java"]))

    return list(dict.fromkeys(suffixes))


_JAVA_WALK_EXCLUDED_DIRS = {".git", ".gradle", ".mvn", "build", "target", "node_modules"}


@functools.lru_cache(maxsize=16)
def _java_file_index(repo_path: str) -> tuple[str, ...]:
    """Return repo-relative Java files with build/cache directories pruned."""
    root = Path(repo_path).resolve()
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [dirname for dirname in dirnames if dirname not in _JAVA_WALK_EXCLUDED_DIRS]
        base = Path(dirpath)
        for filename in filenames:
            if not filename.endswith(".java"):
                continue
            path = base / filename
            try:
                files.append(path.relative_to(root).as_posix())
            except ValueError:
                continue
    return tuple(sorted(files))


def _find_indexed_java_file_by_suffix(repo_path: Path, suffixes: list[str]) -> Path | None:
    normalized_suffixes = [suffix.strip().replace("\\", "/").lstrip("/") for suffix in dict.fromkeys(suffixes) if suffix]
    for relative_path in _java_file_index(str(repo_path.resolve())):
        if any(relative_path.endswith(suffix) for suffix in normalized_suffixes):
            candidate = repo_path / relative_path
            if candidate.is_file():
                return candidate.resolve()
    return None


def _resolve_smell_file_with_inspector(repo_path: Path, reported_paths: list[str], class_name: str | None) -> Path | None:
    for reported_path in dict.fromkeys(path for path in reported_paths if path):
        try:
            result = resolve_smell_location_value(reported_path, class_name)
        except (JavaInspectorUnavailableError, httpx.HTTPError, OSError, ValueError) as exc:
            LOGGER.info("Java inspector smell resolution unavailable; using local index fallback: %s", exc)
            return None
        if not result.file:
            continue
        candidate = Path(result.file)
        if not candidate.is_absolute():
            candidate = repo_path / candidate
        if candidate.is_file():
            return candidate.resolve()
    return None


def _resolve_smell_file(repo_path: Path, smell: object) -> Path | None:
    candidates: list[Path] = []
    suffixes: list[str] = []
    reported_paths: list[str] = []
    file_path = str(getattr(smell, "file_path", "") or "")
    smell_id = str(getattr(smell, "smell_id", "") or "")
    raw_class_name = getattr(smell, "class_name", None)
    class_name = str(raw_class_name) if raw_class_name else None
    if file_path:
        reported_paths.append(file_path)
        candidates.append(repo_path / file_path)
        suffixes.extend(_java_path_suffixes(file_path))
    match = re.search(r":(?P<path>[^:\n]+\.java):\d+", smell_id)
    if match:
        smell_id_path = match.group("path")
        reported_paths.append(smell_id_path)
        candidates.append(repo_path / smell_id_path)
        suffixes.extend(_java_path_suffixes(smell_id_path))
    if class_name:
        class_path = class_name.replace(".", "/") + ".java"
        reported_paths.append(class_path)
        candidates.append(repo_path / class_path)
        suffixes.extend(_java_path_suffixes(class_path))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    inspector_match = _resolve_smell_file_with_inspector(repo_path, reported_paths, class_name)
    if inspector_match is not None:
        return inspector_match

    indexed_match = _find_indexed_java_file_by_suffix(repo_path, suffixes)
    if indexed_match is not None:
        return indexed_match

    basename = Path(file_path).name
    if basename:
        return _find_indexed_java_file_by_suffix(repo_path, [basename])
    return None


def _ast_grep_java_matches(repo_path: Path, target_file: Path, pattern: str) -> list[dict[str, Any]]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            ["sg", "-p", pattern, "--lang", "java", "--json=compact", str(target_file)],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError("ast-grep (`sg`) is required for refactor preflight but could not be executed") from exc
    except subprocess.SubprocessError as exc:
        raise RuntimeError("ast-grep (`sg`) preflight failed to complete") from exc
    elapsed_ms = (time.monotonic() - started) * 1000.0
    LOGGER.info("PHASE ast_grep_preflight command_ms=%.1f pattern=%s file=%s", elapsed_ms, pattern[:40], target_file.name)
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"ast-grep (`sg`) failed for {target_file}: {result.stderr[-500:]}")
    if not result.stdout.strip():
        return []
    try:
        parsed = orjson.loads(result.stdout)
    except orjson.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _range_contains_line(match: dict[str, Any], line: int) -> bool:
    span = match.get("range", {})
    if not isinstance(span, dict):
        return False
    start = span.get("start", {})
    end = span.get("end", {})
    if not isinstance(start, dict) or not isinstance(end, dict):
        return False
    start_line = int(start.get("line", 0)) + 1
    end_line = int(end.get("line", 0)) + 1
    return start_line <= line <= end_line


def _preflight_refactor_target(repo_path: Path, target_file: Path, smell) -> bool:
    """Cheap ast-grep guard before calling the LLM refactorer."""
    started = time.monotonic()
    try:
        rel = target_file.relative_to(repo_path)
    except ValueError:
        LOGGER.error("Refactor target escapes repo: %s", target_file)
        return False
    if "target" in rel.parts or target_file.suffix != ".java" or not target_file.exists():
        LOGGER.error("Invalid Java refactor target: %s", rel)
        return False

    line = int(getattr(smell, "line_number", 0) or 0)
    try:
        line_count = len(target_file.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError as exc:
        LOGGER.error("Cannot read preflight target %s: %s", rel, exc)
        return False
    if line > 0 and line > line_count:
        LOGGER.error("Smell line %d exceeds %s line count %d", line, rel, line_count)
        return False

    structural_ok = True
    if line > 0:
        patterns = ["$RET $METHOD($$$ARGS) { $$$BODY }", "class $C { $$$BODY }", "interface $C { $$$BODY }", "enum $C { $$$BODY }"]
        structural_ok = any(
            _range_contains_line(match, line)
            for pattern in patterns
            for match in _ast_grep_java_matches(repo_path, target_file, pattern)
        )
    elapsed_ms = (time.monotonic() - started) * 1000.0
    LOGGER.info(
        "PHASE refactor_preflight done file=%s line=%d structural_ok=%s elapsed_ms=%.1f",
        rel,
        line,
        structural_ok,
        elapsed_ms,
    )
    if not structural_ok:
        LOGGER.warning("No ast-grep structural context found for %s:%d; continuing with file-level guard", rel, line)
    return True


def _resolve_refactor_output_path(repo_path: Path, file_path: str, step_idx: int) -> Path:
    """Validate and resolve a repository Java file path emitted by the model."""
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = repo_path / candidate
    resolved = candidate.resolve()
    repo_root = repo_path.resolve()
    if not resolved.is_relative_to(repo_root):
        raise ValueError(f"refactor output path escapes repository: {file_path} (step={step_idx})")
    if resolved.suffix != ".java":
        raise ValueError(f"refactor output path must be a Java file: {resolved} (step={step_idx})")
    return resolved



@_profile_phase("llm_refactor")
def _execute_refactor_action(
    repo_path: Path,
    step_idx: int,
    smell,
    ref_type: str,
    model_name: str,
    refactor_scope: RefactorScope,
) -> tuple[bool, list[Path], RefactorFailureReason]:
    rel = None
    target_file: Path | None = None

    try:
        with start_action(action_type="resolve_refactor_target", step=step_idx):
            target_file = _resolve_smell_file(repo_path, smell)
        if target_file is None:
            LOGGER.error(
                "Could not resolve source file for smell_id=%s file_path=%s",
                smell.smell_id,
                getattr(smell, "file_path", None),
            )
            return False, [], "execution_error"

        rel = target_file.relative_to(repo_path) if target_file.is_relative_to(repo_path) else target_file

        with start_action(action_type="preflight_refactor_target", step=step_idx, file=str(rel)):
            if not _preflight_refactor_target(repo_path, target_file, smell):
                return False, [], "execution_error"

        with start_action(
            action_type="import_llm_client",
            step=step_idx,
            model_name=model_name,
            file=str(rel),
        ):
            if importlib.util.find_spec("langchain_litellm") is None:
                LOGGER.error("Cannot run mandatory LLM refactoring: langchain_litellm unavailable")
                return False, [], "execution_error"

        with start_action(action_type="read_refactor_source", step=step_idx, file=str(rel)):
            try:
                before = target_file.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                LOGGER.error("Cannot read target Java file as UTF-8: %s: %s", target_file, exc)
                return False, [], "execution_error"

        line = getattr(smell, "line_number", 0) or 0
        java_version_context = java_version_prompt_context(str(repo_path))
        is_file_scope = refactor_scope == "file"
        prompt = f"""Apply the requested Java refactoring.

Constraints:
- Return structured output only.
- If file scope: use `java_source` with the complete updated target file only.
- If project scope: you may use `files` to return complete updated Java source for multiple files.
- Do not include explanations, Markdown fences, or prose in returned Java source fields.
- Preserve behavior, package, imports, tests, and public API unless the refactoring requires a local change.
- Respect the Maven Java language level below; do not introduce newer Java syntax.
- For Java 1.6, avoid diamond operator, default interface methods, lambdas, method references, try-with-resources, multi-catch, streams, var, switch expressions, and records.
- Ensure any extracted/helper method signature exactly matches all call sites; if arguments mix strings and numbers, either convert call-site values explicitly or use a compatible signature.
- Make a small, compilable edit focused on the target smell.

Java language level:
{java_version_context}

Repository: {repo_path}
Step: {step_idx}
Scope: {"file" if is_file_scope else "project"}
File: {rel}
Smell id: {smell.smell_id}
Smell type: {smell.smell_type}
Severity: {smell.severity}
Location: line {line}
Detection reason: {getattr(smell, 'detection_reason', '') or ''}
Planned refactoring type: {ref_type}

Current file content:
```java
{before}
```"""

        LOGGER.info(
            "PHASE llm_refactor start step=%d file=%s smell_id=%s ref_type=%s model=%s scope=%s",
            step_idx,
            rel,
            smell.smell_id,
            ref_type,
            model_name,
            refactor_scope,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    f"{current_datetime_context()}\n"
                    "You are a careful Java refactoring agent. Produce structured output with complete compilable Java source only."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        with start_action(
            action_type="llm_refactor_call",
            step=step_idx,
            file=str(rel),
            smell_id=getattr(smell, "smell_id", ""),
            model_name=model_name,
            ref_type=ref_type,
        ):
            try:
                refactor_output = _invoke_refactor_llm(model_name, messages)
            except _RefactorLLMError as exc:
                LOGGER.exception("LLM refactoring failed (structured output): %s", exc)
                return False, [], exc.reason
            except (OSError, LiteLLMBadRequestError, LiteLLMAPIConnectionError, LiteLLMAPIError) as exc:
                LOGGER.exception("LLM API failed during refactor call: %s", exc)
                return False, [], "llm_structured_none"

        patch_entries: list[tuple[Path, str]] = []
        with start_action(action_type="validate_refactor_output", step=step_idx, file=str(rel)):
            if is_file_scope:
                if not refactor_output.java_source:
                    LOGGER.error("LLM output did not include java_source for file scope step=%d", step_idx)
                    return False, [], "llm_invalid_java"

                after = refactor_output.java_source
                if after.strip() == before.strip():
                    LOGGER.error("LLM refactoring produced no source change for %s", rel)
                    return False, [], "llm_no_change"
                if "class " not in after and "interface " not in after and "enum " not in after:
                    LOGGER.error("LLM refactoring output did not look like a Java compilation unit for %s", rel)
                    return False, [], "llm_invalid_java"
                is_valid_java, java_error = _validate_java_compilation_unit(after)
                if not is_valid_java:
                    LOGGER.error("LLM refactoring produced invalid Java for %s: %s", rel, java_error)
                    return False, [], "llm_invalid_java"
                patch_entries.append((target_file, after))
            else:
                if refactor_output.files:
                    seen: set[Path] = set()
                    for file_patch in refactor_output.files:
                        output_file = _resolve_refactor_output_path(repo_path, file_patch.file_path, step_idx)
                        if output_file in seen:
                            raise ValueError(f"Duplicate file path in model output: {output_file}")
                        seen.add(output_file)
                        source = file_patch.java_source
                        if "class " not in source and "interface " not in source and "enum " not in source:
                            LOGGER.error("LLM output did not look like a Java compilation unit for %s", output_file)
                            return False, [], "llm_invalid_java"
                        is_valid_java, java_error = _validate_java_compilation_unit(source)
                        if not is_valid_java:
                            LOGGER.error("LLM refactoring produced invalid Java for %s: %s", output_file, java_error)
                            return False, [], "llm_invalid_java"
                        patch_entries.append((output_file, source))
                elif refactor_output.java_source:
                    after = refactor_output.java_source
                    if after.strip() == before.strip():
                        LOGGER.error("LLM refactoring produced no source change for %s", rel)
                        return False, [], "llm_no_change"
                    if "class " not in after and "interface " not in after and "enum " not in after:
                        LOGGER.error("LLM refactoring output did not look like a Java compilation unit for %s", rel)
                        return False, [], "llm_invalid_java"
                    is_valid_java, java_error = _validate_java_compilation_unit(after)
                    if not is_valid_java:
                        LOGGER.error("LLM refactoring produced invalid Java for %s: %s", rel, java_error)
                        return False, [], "llm_invalid_java"
                    patch_entries.append((target_file, after))
                else:
                    LOGGER.error("LLM output did not include java_source/files for %s", rel)
                    return False, [], "llm_invalid_java"

        written_files: list[Path] = []
        with start_action(action_type="write_refactor_output", step=step_idx, file=str(rel)):
            for output_file, source in patch_entries:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                before_source = output_file.read_text(encoding="utf-8") if output_file.exists() else ""
                if source.strip() == before_source.strip():
                    continue
                output_file.write_text(source, encoding="utf-8")
                written_files.append(output_file)

        if not written_files:
            LOGGER.error("LLM refactoring produced no effective source change for %s", rel)
            return False, [], "llm_no_change"

        with start_action(action_type="verify_refactor_diff", step=step_idx, file=str(rel)):
            changed_paths = [str(path.relative_to(repo_path)) for path in written_files]
            diff = subprocess.run(["git", "diff", "--", *changed_paths], cwd=repo_path, capture_output=True, text=True)
            if diff.returncode != 0 or not diff.stdout.strip():
                LOGGER.error("No git diff after LLM refactoring for %s", rel)
                return False, [], "execution_error"
            LOGGER.info(
                "PHASE llm_refactor done step=%d file=%s changed_files=%d diff_chars=%d",
                step_idx,
                rel,
                len(written_files),
                len(diff.stdout),
            )

        return True, written_files, "execution_ok"

    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.exception("Unexpected error during refactor execution step=%d file=%s: %s", step_idx, rel, exc)
        return False, [], "execution_error"



def _rollback_repo(repo_path: Path) -> bool:
    r = subprocess.run(["bash", "-lc", "git reset --hard && git clean -fd"], cwd=repo_path, capture_output=True, text=True)
    ok = r.returncode == 0
    if not ok:
        LOGGER.error("Rollback failed: %s", r.stderr[-800:])
    return ok


@_profile_phase("mlflow_healthcheck")
def _run_mlflow_healthcheck(experiment_name: str, timeout: int) -> None:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "check_mlflow_health.sh"
    if not script_path.exists():
        raise FileNotFoundError(f"MLflow healthcheck script not found: {script_path}")
    env = {
        **os.environ,
        "EXPERIMENT_NAME": experiment_name,
    }
    LOGGER.info(
        "PHASE mlflow_healthcheck start experiment_name=%s script=%s",
        experiment_name,
        script_path,
    )
    result = subprocess.run(
        [str(script_path)],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.stdout:
        LOGGER.info("mlflow healthcheck stdout: %s", result.stdout[-2000:].strip())
    if result.stderr:
        LOGGER.warning("mlflow healthcheck stderr: %s", result.stderr[-2000:].strip())
    if result.returncode != 0:
        raise RuntimeError(f"MLflow healthcheck failed with {result.returncode}: {(result.stderr or result.stdout)[-1000:]}")
    LOGGER.info("PHASE mlflow_healthcheck done")


@_profile_phase("java_inspector_healthcheck")
def _run_java_inspector_healthcheck(repo_path: Path, timeout: int) -> str:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "check_java_inspector_health.sh"
    if not script_path.exists():
        raise FileNotFoundError(f"Java inspector healthcheck script not found: {script_path}")
    safe_repo_token = _safe_name(str(repo_path))
    env = {
        **os.environ,
        "JAVA_INSPECTOR_REPO_PATH": str(repo_path),
        "JAVA_INSPECTOR_LOG": f"/tmp/java_inspector_{safe_repo_token}.out",
        "JAVA_INSPECTOR_PID_FILE": f"/tmp/java_inspector_{safe_repo_token}.pid",
        "JAVA_INSPECTOR_URL_FILE": f"/tmp/java_inspector_{safe_repo_token}.url",
    }
    LOGGER.info("PHASE java_inspector_healthcheck start repo=%s script=%s", repo_path, script_path)
    result = subprocess.run(
        [str(script_path)],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.stdout:
        LOGGER.info("java inspector healthcheck stdout: %s", result.stdout[-2000:].strip())
    if result.stderr:
        LOGGER.warning("java inspector healthcheck stderr: %s", result.stderr[-2000:].strip())
    if result.returncode != 0:
        raise RuntimeError(
            f"Java inspector healthcheck failed with {result.returncode}: {(result.stderr or result.stdout)[-1000:]}"
        )
    for line in result.stdout.splitlines():
        if line.startswith("JAVA_INSPECTOR_URL="):
            inspector_url = line.split("=", 1)[1].strip()
            set_java_inspector_url(inspector_url)
            LOGGER.info("PHASE java_inspector_healthcheck done url=%s", inspector_url)
            return inspector_url
    raise RuntimeError("Java inspector healthcheck did not print JAVA_INSPECTOR_URL")


@_profile_phase("java_tests")
def _run_java_test_bool(
    repo_path: Path,
    timeout: int,
    llm_repair_model: str | None = None,
    target_files: list[str] | None = None,
) -> tuple[bool, bool]:
    started = time.monotonic()
    model_override = llm_repair_model or JAVA_TEST_CODE_AGENT_MODEL

    LOGGER.info(
        "PHASE java_tests start repo=%s timeout=%ss llm_repair_model=%s target_files=%s",
        repo_path,
        timeout,
        model_override,
        target_files,
    )
    result = run_java_test_analysis(
        str(repo_path),
        clean=False,
        timeout=timeout,
        llm_repair_model=model_override,
        target_files=target_files,
        enable_code_agent_repair=JAVA_TEST_CODE_AGENT_ENABLED,
        code_agent_model=model_override,
        code_agent_step_limit=JAVA_TEST_CODE_AGENT_MAX_STEPS,
        code_agent_cost_limit=0.0,
        code_agent_timeout=min(timeout, JAVA_TEST_CODE_AGENT_TIMEOUT),
        code_agent_max_attempts=JAVA_TEST_CODE_AGENT_MAX_ATTEMPTS,
    )
    if result.get("summary") is None:
        LOGGER.info("PHASE java_tests done build_system=%s summary=None elapsed=%.2fs", result.get("build_system"), time.monotonic() - started)
        return False, False
    repair = result.get("code_agent_repair") or {}
    if repair.get("attempted"):
        LOGGER.info(
            "PHASE java_tests code_agent_repair=%s pre_code_agent_exit_code=%s",
            repair,
            result.get("pre_code_agent_exit_code"),
        )
    summary = result["summary"]
    summary_command = result.get("command")
    if summary_command:
        LOGGER.info("PHASE java_tests command=%s", summary_command)
    compile_passed = bool(summary.exit_code == 0)
    tests_passed = bool(summary.success)
    LOGGER.info(
        "PHASE java_tests done build_system=%s exit_code=%s tests=%d failed=%d errors=%d elapsed=%.2fs",
        result.get("build_system"),
        summary.exit_code,
        summary.counts.total,
        summary.counts.failed,
        summary.counts.errors,
        time.monotonic() - started,
    )
    return compile_passed, tests_passed


def _parse_elements_arg(elements_arg: str) -> set[str]:
    raw = (elements_arg or "").strip()
    if not raw:
        return set()
    if raw.startswith("["):
        arr = orjson.loads(raw)
        if not isinstance(arr, list):
            raise ValueError("elements JSON must be a list")
        return {str(x).strip() for x in arr if str(x).strip()}
    return {e.strip() for e in raw.split(",") if e.strip()}



def _run_metadata(args: WorkflowRunArgs, repo_path: Path, repo_url: str) -> dict[str, Any]:
    metadata = args.model_dump(mode="json")
    metadata.update(
        {
            "repo_path": str(repo_path),
            "repo_url": repo_url,
            "model": _get_refactor_model(args.model),
            "java_test_repair_model": JAVA_TEST_CODE_AGENT_MODEL,
            "java_test_repair_steps": JAVA_TEST_CODE_AGENT_MAX_STEPS,
            "java_test_repair_attempts": JAVA_TEST_CODE_AGENT_MAX_ATTEMPTS,
        }
    )
    return metadata


def _maybe_reference_smells_after_empirical(project: str, elements_csv: str, max_steps: int) -> int | None:
    if not project or not elements_csv:
        return None
    try:
        from dataset.neo4j_graph import DatasetGraph

        ds = DatasetGraph()
        if not ds.is_available():
            return None
        elements = _parse_elements_arg(elements_csv)
        steps = ds.composite_refactoring(elements=elements, project=project, max_steps=max_steps)
        if not steps:
            return None
        return len(steps[-1].smells)
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        LOGGER.debug("Empirical smell lookup unavailable: %s", exc)
        return None


def main(args: WorkflowRunArgs) -> int:
    stack_heartbeat = int(os.environ.get("EVAL_STACK_HEARTBEAT_SECONDS", "0") or "0")
    if stack_heartbeat > 0:
        faulthandler.enable(file=sys.stderr)
        faulthandler.dump_traceback_later(stack_heartbeat, repeat=True, file=sys.stderr)

    _load_workflow_env()
    _configure_workflow_logging(verbose=args.verbose)

    LOGGER.info("PHASE workflow_start project=%s planner=%s detector=%s model=%s", args.project, args.planner, args.detector_backend, _get_refactor_model(args.model))
    if not args.project:
        raise ValueError("--project is required")
    if not args.skip_mlflow_healthcheck:
        _run_mlflow_healthcheck(
            experiment_name=args.mlflow_healthcheck_experiment,
            timeout=args.timeout,
        )
    repo_url = _resolve_repo_url(args.project, args.repo_url)
    LOGGER.info("PHASE repo_checkout start repo_url=%s commit=%s", repo_url, args.start_commit_hash)
    repo_path = _prepare_repo_checkout(
        project=args.project,
        repo_url=repo_url,
        repos_root=Path(args.repos_root).resolve(),
        commit_hash=args.start_commit_hash,
        worktree_suffix=args.worktree_suffix,
    )
    LOGGER.info("PHASE repo_checkout done repo_path=%s", repo_path)
    if not args.skip_java_inspector_healthcheck:
        _run_java_inspector_healthcheck(repo_path=repo_path, timeout=args.timeout)
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow is required for composite workflow logging/tracing") from exc

    _enable_litellm_autologging(mlflow)
    _enable_langchain_autologging(mlflow)

    detector = _select_detector(args)

    try:
        smells = _detect(detector, repo_path, args.elements)
    except SmellDetectionError as e:
        LOGGER.error("Initial smell detection failed: %s", e)
        return 1

    h_trace: list[float] = []
    step_logs: list[StepLog] = []
    retries_used = 0
    no_progress = 0
    stop_reason: str | None = "max_steps"

    initial_count = len(smells)
    if initial_count == 0:
        stop_reason = "smells_zero"

    mlflow.set_experiment(args.experiment)
    run_name = args.run_name or f"online/{repo_path.name}/{int(time.time())}"

    log_handler, log_path = _attach_mlflow_log_file(run_name)
    with mlflow.start_run(run_name=run_name):
        run_metadata = _run_metadata(args, repo_path, repo_url)
        mlflow.log_dict(run_metadata, "online/input.json")
        mlflow.log_params({key: value for key, value in run_metadata.items() if key not in {"project", "elements"}})

        # baseline build/test
        LOGGER.info("PHASE baseline_verification start")
        compile_passed_0, tests_passed_0 = _run_java_test_bool(
            repo_path,
            args.timeout,
            target_files=None,
        )
        LOGGER.info("PHASE baseline_verification done compile_passed=%s tests_passed=%s", compile_passed_0, tests_passed_0)
        mlflow.log_metrics(
            {
                "initial_smells": float(initial_count),
                "baseline_compile_passed": float(compile_passed_0),
                "baseline_tests_passed": float(tests_passed_0),
            }
        )

        for step_idx in range(args.max_steps):
            step_token = _CURRENT_STEP.set(step_idx)
            try:
                LOGGER.info("PHASE step_start step=%d current_smells=%d", step_idx, len(smells))
                if not smells:
                    stop_reason = "smells_zero"
                    break

                action, dep_graph, state = _pick_next_action(smells, args.planner, args.locality)
                h_before = state.h(dep_graph)
                h_trace.append(h_before)

                if action is None:
                    stop_reason = "no_action"
                    break
                if not action.smell_id:
                    raise ValueError("Action smell_id must be non-empty")

                target_smell = next((s for s in smells if s.smell_id == action.smell_id), None)
                if target_smell is None:
                    LOGGER.error("Selected action references missing smell_id=%s", action.smell_id)
                    stop_reason = "missing_action_smell"
                    break

                LOGGER.info("PHASE action_selected step=%d smell_id=%s ref_type=%s h_before=%.3f", step_idx, action.smell_id, action.ref_type, h_before)
                execution_ok, modified_files, failure_reason = _execute_refactor_action(
                    repo_path,
                    step_idx,
                    target_smell,
                    action.ref_type,
                    _get_refactor_model(args.model),
                    args.refactor_scope,
                )

                compile_passed = False
                tests_passed = False
                stop_reason = None
                target_files = [str(path) for path in (modified_files or [])] if args.targeted_testing else None

                if execution_ok:
                    compile_passed, tests_passed = _run_java_test_bool(
                        repo_path,
                        args.timeout,
                        target_files=target_files,
                    )
                    if not compile_passed:
                        stop_reason = "compile_fail"
                    elif not tests_passed:
                        stop_reason = "tests_fail"
                    else:
                        stop_reason = "execution_ok"
                else:
                    if failure_reason == "execution_ok":
                        failure_reason = "execution_error"
                    if failure_reason in {"llm_structured_none", "llm_no_change", "llm_invalid_java", "execution_error"}:
                        stop_reason = failure_reason
                    else:
                        stop_reason = "execution_error"

                if stop_reason != "execution_ok":
                    if retries_used < args.retry_budget:
                        retries_used += 1
                        if execution_ok and modified_files:
                            rolled_back = _rollback_repo(repo_path)
                            if not rolled_back:
                                raise RuntimeError("Rollback must succeed")
                        LOGGER.info("PHASE retry_redetect start step=%d", step_idx)
                        smells = _detect(detector, repo_path, args.elements)
                        step_logs.append(
                            StepLog(
                                step=step_idx,
                                smell_count_before=len(state.active),
                                smell_count_after=len(smells),
                                h_before=h_before,
                                h_after=h_before,
                                action_smell_id=action.smell_id,
                                action_ref_type=action.ref_type,
                                compile_passed=compile_passed,
                                tests_passed=tests_passed,
                                execution_ok=execution_ok,
                                stop_reason="retry",
                            )
                        )
                        continue
                    final_stop_reason = stop_reason
                    if execution_ok and modified_files:
                        rolled_back = _rollback_repo(repo_path)
                        LOGGER.info("PHASE failure_rollback step=%d rolled_back=%s", step_idx, rolled_back)
                    step_logs.append(
                        StepLog(
                            step=step_idx,
                            smell_count_before=len(state.active),
                            smell_count_after=len(state.active),
                            h_before=h_before,
                            h_after=h_before,
                            action_smell_id=action.smell_id,
                            action_ref_type=action.ref_type,
                            compile_passed=compile_passed,
                            tests_passed=tests_passed,
                            execution_ok=execution_ok,
                            stop_reason=final_stop_reason,
                        )
                    )
                    raise RuntimeError(
                        f"{final_stop_reason} reached: "
                        f"step={step_idx} smell_id={action.smell_id!r} "
                        f"ref_type={action.ref_type!r} execution_ok={execution_ok} "
                        f"compile_passed={compile_passed} tests_passed={tests_passed} "
                        f"retry_budget={args.retry_budget}"
                    )

                # Successful verify -> re-detect and replan next iteration
                LOGGER.info("PHASE post_action_detect start step=%d", step_idx)
                smells_after = _detect(detector, repo_path, args.elements)
                _, dep_graph_after, state_after = _pick_next_action(smells_after, args.planner, args.locality)
                h_after = state_after.h(dep_graph_after)

                if h_after >= h_before:
                    no_progress += 1
                else:
                    no_progress = 0

                step_logs.append(
                    StepLog(
                        step=step_idx,
                        smell_count_before=len(state.active),
                        smell_count_after=len(smells_after),
                        h_before=h_before,
                        h_after=h_after,
                        action_smell_id=action.smell_id,
                        action_ref_type=action.ref_type,
                        compile_passed=compile_passed,
                        tests_passed=tests_passed,
                        execution_ok=execution_ok,
                    )
                )

                mlflow.log_metrics(
                    {
                        "smells_before": float(len(state.active)),
                        "smells_after": float(len(smells_after)),
                        "h_before": float(h_before),
                        "h_after": float(h_after),
                        "compile_passed": float(compile_passed),
                        "tests_passed": float(tests_passed),
                    },
                    step=step_idx,
                )

                smells = smells_after
                if no_progress >= args.max_no_progress:
                    stop_reason = "no_progress"
                    break
            finally:
                _CURRENT_STEP.reset(step_token)

        if smells and stop_reason == "max_steps" and no_progress >= args.max_no_progress:
            stop_reason = "no_progress"

        h_trace = _complete_h_trace(h_trace, step_logs)
        final_count = len(smells)
        final_h = h_trace[-1] if h_trace else 0.0
        reached_goal = final_count == 0

        # Minimal success metric + must-have safety metric (rho)
        h0 = h_trace[0] if h_trace else 0.0
        relative_h_reduction = (h0 - final_h) / max(h0, 1e-9)
        compile_passed_final = bool(step_logs[-1].compile_passed) if step_logs else bool(compile_passed_0)
        tests_passed_final = bool(step_logs[-1].tests_passed) if step_logs else bool(tests_passed_0)
        success = bool(
            compile_passed_final
            and tests_passed_final
            and relative_h_reduction >= args.h_reduction_threshold
        )
        smells_created_total = 0
        for row in step_logs:
            smells_created_total += max(row.smell_count_after - row.smell_count_before, 0)
        rho = smells_created_total / max(len(step_logs), 1)

        empirical_after = _maybe_reference_smells_after_empirical(args.project, args.elements, args.max_steps)

        mlflow.log_metrics(
            {
                "final_smells": float(final_count),
                "reached_goal": float(reached_goal),
                "steps_executed": float(len(step_logs)),
                "final_h": float(final_h),
                "relative_h_reduction": float(relative_h_reduction),
                "compile_passed_final": float(compile_passed_final),
                "tests_passed_final": float(tests_passed_final),
                "success": float(success),
                "rho": float(rho),
            }
        )
        if empirical_after is not None:
            mlflow.log_metrics({"smells_after_empirical": float(empirical_after)})
        mlflow.log_params({"stop_reason": stop_reason})

        with tempfile.TemporaryDirectory(prefix="composite_workflow_full_") as td:
            step_path = Path(td) / "step_logs.jsonl"
            with step_path.open("w", encoding="utf-8") as f:
                for row in step_logs:
                    f.write(orjson.dumps(asdict(row)).decode("utf-8") + "\n")

            h_path = Path(td) / "h_trace.json"
            h_path.write_text(orjson.dumps({"h_trace": h_trace}, option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")

            mlflow.log_artifact(str(step_path), artifact_path="online")
            mlflow.log_artifact(str(h_path), artifact_path="online")
            _detach_mlflow_log_file(log_handler)
            mlflow.log_artifact(str(log_path), artifact_path="online/logs")

    summary = f"Done: initial={initial_count} final={len(smells)} steps={len(step_logs)} stop_reason={stop_reason}"
    LOGGER.info(summary)
    print(summary)
    return 0



def _read_config(config_path: Path) -> dict[str, Any]:
    path = config_path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    raw = orjson.loads(path.read_bytes())
    if not isinstance(raw, dict):
        raise ValueError("config JSON must be an object")
    cfg = raw.get("workflow", raw)
    if not isinstance(cfg, dict):
        raise ValueError("workflow config must be an object")
    return dict(cfg)



_WORKFLOW_ARG_FIELDS = set(WorkflowRunArgs.model_fields)


def _normalize_case_elements(elements: Any) -> str:
    if elements is None:
        return ""
    if isinstance(elements, (list, tuple, set)):
        return ",".join(str(x).strip() for x in elements if str(x).strip())
    return str(elements)


def _build_args_from_config(cfg: dict[str, Any]) -> WorkflowRunArgs:
    payload = {name: cfg[name] for name in _WORKFLOW_ARG_FIELDS if name in cfg}
    return _build_run_args(**payload)


def _case_args_from_config(cfg: dict[str, Any], ep: dict[str, Any], repo_url: str, run_name: str) -> WorkflowRunArgs:
    refactor_scope = cfg.get("refactor_scope", "project")
    meta = ep.get("meta") or {}
    start_commit_hash = cfg.get("start_commit_hash") or meta.get("start_commit_hash")
    if not start_commit_hash:
        raise ValueError("start_commit_hash missing in config or episode.meta")

    case_cfg: dict[str, Any] = dict(cfg)
    case_cfg.update(
        {
            "project": ep.get("project") or cfg.get("project"),
            "elements": _normalize_case_elements(ep.get("elements") or cfg.get("elements") or []),
            "repo_url": repo_url,
            "start_commit_hash": start_commit_hash,
            "run_name": run_name,
            "worktree_suffix": run_name,
            "refactor_scope": refactor_scope,
        }
    )
    return _build_args_from_config(case_cfg)


def _run_case_with_capture(case_args: WorkflowRunArgs) -> tuple[int, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        code = main(case_args)
    return code, buffer.getvalue()


app = typer.Typer(help="Typer entrypoints for configured full composite evaluation runs.")


@app.callback(invoke_without_command=True)
@use_json_config()
def run_case_direct(
    ctx: typer.Context,
    planner: str = typer.Option("befs", "--planner", help="Planner algorithm: greedy or befs."),
    repos_root: str = typer.Option(str(DEFAULT_REPOS_ROOT), "--repos-root", help="Root dir for managed repos/worktrees"),
    repo_url: str = typer.Option("", "--repo-url", help="Optional override git URL; auto-derived from --project when empty"),
    start_commit_hash: str = typer.Option("", "--start-commit-hash", help="Commit hash to checkout for the run"),
    worktree_suffix: str = typer.Option("", "--worktree-suffix", help="Optional unique suffix for case-level parallel worktree isolation"),
    detector_backend: str = typer.Option("organic", "--detector-backend", help="smell detection backend: organic, sonar, dummy, static"),
    locality: str = typer.Option("none", "--locality", help="Dependency locality: none/class/file"),
    max_steps: int = typer.Option(5, "--max-steps", help="Maximum number of refactoring steps."),
    max_no_progress: int = typer.Option(2, "--max-no-progress", help="Stop if no objective progress for this many steps."),
    retry_budget: int = typer.Option(1, "--retry-budget", help="How many failed steps to retry before stopping."),
    timeout: int = typer.Option(300, "--timeout", help="Per-step timeout for detection/refactor/test commands."),
    organic_dir: str | None = typer.Option(None, "--organic-dir", help="Optional Organic detector working directory."),
    sonar_url: str = typer.Option("http://localhost:9000", "--sonar-url", help="SonarQube URL for detector backend."),
    experiment: str = typer.Option("composite_workflow_full", "--experiment", help="MLflow experiment name."),
    mlflow_healthcheck_experiment: str = typer.Option(
        "planner-eval",
        "--mlflow-healthcheck-experiment",
        help="MLflow experiment name used by startup healthcheck script.",
    ),
    skip_mlflow_healthcheck: bool = typer.Option(False, "--skip-mlflow-healthcheck", help="Skip MLflow startup healthcheck."),
    skip_java_inspector_healthcheck: bool = typer.Option(False, "--skip-java-inspector-healthcheck", help="Skip Java inspector startup healthcheck."),
    run_name: str | None = typer.Option(None, "--run-name", help="Optional MLflow run name."),
    model: str | None = typer.Option(None, "--model", help="LLM model used for Java refactoring; defaults to COMPOSITE_REFACTOR_MODEL."),
    refactor_scope: RefactorScope = typer.Option("project", "--refactor-scope", help="Refactoring scope: file (single file) or project (multi-file allowed)."),
    project: str = typer.Option("", "--project", help="Dataset project name."),
    elements: str = typer.Option("", "--elements", help="CSV element FQNs to constrain the workflow."),
    h_reduction_threshold: float = typer.Option(0.7, "--h-reduction-threshold", help="Minimum relative h reduction to count as success."),
    eval_patch_script: str = typer.Option(
        "",
        "--eval-patch-script",
        help="Deprecated compatibility option; ignored. Build/test failures are handled by CodeAgent repair.",
    ),
    targeted_testing: bool = typer.Option(
        True,
        "--targeted-testing/--no-targeted-testing",
        help="Run post-refactor tests scoped to changed file when possible.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show phase-level progress logs."),
) -> None:
    """Run one workflow case. Parameters are loaded from --config or CLI."""
    if ctx.invoked_subcommand is not None:
        return

    args = _build_run_args(
        planner=planner,
        repos_root=repos_root,
        repo_url=repo_url,
        start_commit_hash=start_commit_hash,
        worktree_suffix=worktree_suffix,
        detector_backend=detector_backend,
        locality=locality,
        max_steps=max_steps,
        max_no_progress=max_no_progress,
        retry_budget=retry_budget,
        timeout=timeout,
        organic_dir=organic_dir,
        sonar_url=sonar_url,
        experiment=experiment,
        mlflow_healthcheck_experiment=mlflow_healthcheck_experiment,
        skip_mlflow_healthcheck=skip_mlflow_healthcheck,
        skip_java_inspector_healthcheck=skip_java_inspector_healthcheck,
        run_name=run_name,
        model=model,
        refactor_scope=refactor_scope,
        project=project,
        elements=elements,
        h_reduction_threshold=h_reduction_threshold,
        eval_patch_script=eval_patch_script,
        targeted_testing=targeted_testing,
        verbose=verbose,
    )
    _load_workflow_env(".env")
    code, output = _run_case_with_capture(args)
    if output:
        typer.echo(output)
    raise typer.Exit(code)


@app.command("batch")
def batch_from_config(
    config: Path = typer.Option(..., "--config", "-c", help="JSON config file, usually under evals/config."),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Maximum number of cases to run from the batch list."),
    start_index: int = typer.Option(1, min=1, help="1-based case index to start from."),
    concurrency: int | None = typer.Option(None, min=1, help="Case-level parallelism. Overrides config.concurrency."),
    list_cases: bool = typer.Option(False, "--list-cases", help="Print case ids selected by --config/--start-index/--limit and exit."),
) -> None:
    """Run the full workflow over a manifest using a JSON config."""
    cfg = _read_config(config)

    _load_workflow_env(".env")

    manifest_path = Path(cfg.get("batch_list") or cfg["manifest"])
    manifest = orjson.loads(manifest_path.read_bytes())
    episodes = list(manifest.get("episodes", []))
    if episodes:
        source = "episodes"
    else:
        episodes = list(manifest.get("cases", []))
        source = "cases"
    if not episodes:
        raise ValueError(f"No episodes/cases found in {manifest_path}")
    total_cases = len(episodes)
    selected = episodes[start_index - 1:]
    if limit is not None:
        selected = selected[:limit]

    typer.echo(
        f"Loaded batch_list={manifest_path} total_cases={total_cases} "
        f"selected_cases={len(selected)} start_index={start_index}"
    )
    if list_cases:
        for display_index, ep in enumerate(selected, start=start_index):
            typer.echo(f"{display_index}: {ep.get('case_id') or ep.get('project') or '<unknown-case>'}")
        return

    ready_repos_csv = Path(cfg.get("ready_repos_csv", "evals/helper/filer_ready_repos/maven_only_14_ready_commands.csv"))
    repo_urls = {r["project"]: r["repo_url"] for r in csv.DictReader(ready_repos_csv.open())}
    run_name_prefix = cfg.get("run_name_prefix") or _derive_run_name_prefix(cfg, config)

    case_jobs: list[tuple[int, str, WorkflowRunArgs]] = []
    for offset, ep in enumerate(selected, start=start_index):
        project = ep.get("project") or cfg.get("project")
        repo_url = cfg.get("repo_url") or repo_urls.get(project) or ""
        run_name = f"{run_name_prefix}-case-{offset}"
        if source == "cases":
            case_args = _case_args_from_batch_case(cfg, ep, run_name)
        else:
            case_args = _case_args_from_config(cfg, ep, repo_url, run_name)
        case_jobs.append((offset, ep.get("case_id", project), case_args))

    max_workers = int(concurrency or cfg.get("concurrency", 1))
    if max_workers < 1:
        raise ValueError("concurrency must be >= 1")
    show_case_output = bool(cfg.get("verbose", False))
    typer.echo(f"Running selected_cases={len(case_jobs)} with case-level concurrency={max_workers}")

    def run_case(job: tuple[int, str, WorkflowRunArgs]) -> tuple[int, str, int, str]:
        offset, case_id, case_args = job
        header = "\n" + "=" * 80 + f"\nCASE {offset} {case_id}\nRUN: {case_args.project}::{case_args.start_commit_hash[:12]}\n" + "=" * 80 + "\n"
        try:
            code, output = _run_case_with_capture(case_args)
        except (OSError, RuntimeError, ValueError, GitError, SmellDetectionError):
            output = traceback.format_exc()
            code = 1
        return offset, case_id, code, header + output + f"\nCASE {offset} EXIT {code}\n"

    failures: list[tuple[int, str, int]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {executor.submit(run_case, job): job for job in case_jobs}
        for future in concurrent.futures.as_completed(future_to_job):
            offset, case_id, code, output = future.result()
            status = "OK" if code == 0 else f"FAIL exit={code}"
            typer.echo(f"CASE {offset} {case_id}: {status}")
            if show_case_output or code != 0:
                typer.echo(output)
            if code != 0:
                failures.append((offset, case_id, code))

    failures.sort()
    if failures:
        typer.echo(f"FAILURES: {failures}", err=True)
        raise typer.Exit(1)
    typer.echo("ALL CASES PASSED")


@app.command("single", hidden=True)
def single_from_config(
    config: Path = typer.Option(..., "--config", "-c", help="JSON config file with one episode or project/elements."),
) -> None:
    """Run one full workflow case from JSON config."""
    cfg = _read_config(config)
    _load_workflow_env(".env")

    ep = cfg.get("episode")
    if ep is None:
        ep = {
            "project": cfg.get("project"),
            "elements": cfg.get("elements", []),
            "meta": cfg,
        }
    project = ep.get("project") or cfg.get("project")
    if not project:
        raise ValueError("project is required in config or episode.project")
    repo_url = cfg.get("repo_url") or _resolve_repo_url(project, "")
    run_name = cfg.get("run_name") or f"full-config-{config.stem}"
    case_args = _case_args_from_config(cfg, ep, repo_url, run_name)

    code, output = _run_case_with_capture(case_args)
    if output:
        typer.echo(output)
    raise typer.Exit(code)


if __name__ == "__main__":
    app()
