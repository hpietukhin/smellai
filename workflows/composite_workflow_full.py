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
import faulthandler
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from eliot import start_action, Logger
from eliot.stdlib import EliotHandler
import typer
from dotenv import load_dotenv
from typer_config.decorators import use_json_config
from mlflow import litellm

from agents.java_test.agent import run_java_test_analysis
from dataset.organic_detector import OrganicDetector
from domain.dependency_graph import DependencyGraph
from domain.detector import SmellDetectionError, SmellDetector, StaticDetector
from domain.refactoring_tree import RefactoringTree, State
from sonarqube.detector import SonarQubeDetector

LOGGER = logging.getLogger(__name__)

# CodeAgent repair policy for java_test_analysis (Python-side configuration only).
JAVA_TEST_CODE_AGENT_ENABLED = True
JAVA_TEST_CODE_AGENT_MODEL = "anthropic/claude-sonnet-4-20250514"
JAVA_TEST_CODE_AGENT_MAX_STEPS = 3
JAVA_TEST_CODE_AGENT_MAX_ATTEMPTS = 2
JAVA_TEST_CODE_AGENT_TIMEOUT = 60

_CURRENT_TRACKER: contextvars.ContextVar[Any | None] = contextvars.ContextVar("composite_workflow_tracker", default=None)
_CURRENT_STEP: contextvars.ContextVar[int | None] = contextvars.ContextVar("composite_workflow_step", default=None)

DEFAULT_REPOS_ROOT = Path("/Users/havriil.pietukhin/uni/masterThesis/code/repos")
DEFAULT_REFRACTOR_MODEL = os.environ.get("COMPOSITE_REFACTOR_MODEL", "claude-sonnet-4-5-20250929")
MLFLOW_TRACKING_URI = "http://localhost:5000"
KNOWN_REPO_URLS = {
    "Apache Tomcat": "https://github.com/apache/tomcat.git",
    "JUnit4": "https://github.com/junit-team/junit4.git",
    "Lyra": "https://github.com/jhalterman/lyra.git",
    "OkHttp": "https://github.com/square/okhttp.git",
    "PhiCode Philib": "https://github.com/PhiCode/philib.git",
    "Tap4j": "https://github.com/tupilabs/tap4j.git",
}

_LITELLM_AUTLOGGED = False
_ELIOT_LOG_DESTINATIONS: list[tuple[Any, Any]] = []


def _ensure_eliot_logging(verbose: bool = False, eliot_log_path: str | None = None) -> None:
    """Attach Eliot/stdlib logging handlers.

    The workflow uses Eliot for structured nested phase timing.
    Optionally write all Eliot messages to ``eliot_log_path`` for
    post-run analysis.
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), EliotHandler()],
        force=True,
    )

    if not eliot_log_path:
        return

    log_path = Path(eliot_log_path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output = log_path.open("ab")
    from eliot import FileDestination

    destination = FileDestination(output)
    Logger._destinations.add(destination)
    _ELIOT_LOG_DESTINATIONS.append((output, destination))


def _load_workflow_env(env_file: str | Path = ".env") -> None:
    """Load workflow credentials, especially OpenRouter for LiteLLM.

    LiteLLM's OpenRouter provider reads OPENROUTER_API_KEY directly.  We also
    set OpenRouter attribution headers when absent; this keeps auth local to the
    workflow and avoids requiring callers to source .env manually.
    """
    load_dotenv(env_file, override=False)
    if os.environ.get("OPENROUTER_API_KEY"):
        os.environ.setdefault("OR_SITE_URL", "https://github.com/havriil/smellai")
        os.environ.setdefault("OR_APP_NAME", "smellai-composite-workflow")


def _get_refactor_model(model_override: str | None = None) -> str:
    return model_override or os.environ.get("COMPOSITE_REFACTOR_MODEL", DEFAULT_REFRACTOR_MODEL)


def _close_eliot_log_handles() -> None:
    """Close Eliot file sinks opened by _ensure_eliot_logging()."""
    for handle, destination in list(_ELIOT_LOG_DESTINATIONS):
        try:
            Logger._destinations.remove(destination)
        except Exception:
            pass
        try:
            handle.flush()
            handle.close()
        except Exception:
            pass
        _ELIOT_LOG_DESTINATIONS.remove((handle, destination))


def _enable_litellm_autologging(mlflow_module: Any) -> None:
    global _LITELLM_AUTLOGGED
    if _LITELLM_AUTLOGGED:
        return
    try:
        litellm.autolog()
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.warning("Failed to enable LiteLLM autologging: %s", exc)
    else:
        _LITELLM_AUTLOGGED = True
        LOGGER.info("PHASE mlflow_litellm_autolog enabled")


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


@dataclass
class WorkflowRunArgs:
    planner: str = "befs"
    repos_root: str = str(DEFAULT_REPOS_ROOT)
    repo_url: str = ""
    start_commit_hash: str = ""
    worktree_suffix: str = ""
    detector_backend: str = "organic"
    locality: str = "none"
    max_steps: int = 5
    max_no_progress: int = 2
    retry_budget: int = 1
    timeout: int = 300
    organic_dir: str | None = None
    sonar_url: str = "http://localhost:9000"
    experiment: str = "composite_workflow_full"
    mlflow_healthcheck_experiment: str = "planner-eval"
    skip_mlflow_healthcheck: bool = False
    run_name: str | None = None
    model: str | None = None
    refactor_scope: str = "file"
    project: str = ""
    elements: str = ""
    h_reduction_threshold: float = 0.7
    eval_patch_script: str = "scripts/apply_eval_project_patches.sh"
    targeted_testing: bool = True
    eliot_log_path: str | None = None
    verbose: bool = False


def _build_run_args(
    planner: str = "befs",
    repos_root: str = str(DEFAULT_REPOS_ROOT),
    repo_url: str = "",
    start_commit_hash: str = "",
    worktree_suffix: str = "",
    detector_backend: str = "organic",
    locality: str = "none",
    max_steps: int = 5,
    max_no_progress: int = 2,
    retry_budget: int = 1,
    timeout: int = 300,
    organic_dir: str | None = None,
    sonar_url: str = "http://localhost:9000",
    experiment: str = "composite_workflow_full",
    mlflow_healthcheck_experiment: str = "planner-eval",
    skip_mlflow_healthcheck: bool = False,
    run_name: str | None = None,
    model: str | None = None,
    refactor_scope: str = "file",
    project: str = "",
    elements: str = "",
    h_reduction_threshold: float = 0.7,
    eval_patch_script: str = "scripts/apply_eval_project_patches.sh",
    targeted_testing: bool = True,
    eliot_log_path: str | None = None,
    verbose: bool = False,
) -> WorkflowRunArgs:
    if planner not in {"greedy", "befs"}:
        raise typer.BadParameter("--planner must be one of: greedy, befs")
    if detector_backend not in {"organic", "sonar", "dummy", "static"}:
        raise typer.BadParameter("--detector-backend must be one of: organic, sonar, dummy, static")
    if locality not in {"none", "class", "file"}:
        raise typer.BadParameter("--locality must be one of: none, class, file")
    if refactor_scope != "file":
        raise typer.BadParameter("--refactor-scope currently supports only: file")
    if start_commit_hash.strip() == "":
        raise typer.BadParameter("--start-commit-hash is required")

    return WorkflowRunArgs(
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
        run_name=run_name,
        model=model,
        refactor_scope=refactor_scope,
        project=project,
        elements=elements,
        h_reduction_threshold=h_reduction_threshold,
        eval_patch_script=eval_patch_script,
        targeted_testing=targeted_testing,
        eliot_log_path=eliot_log_path,
        verbose=verbose,
    )


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
                        try:
                            tracker.log_timing(phase, elapsed_ms, step=step)
                        except Exception:
                            pass
                    if action_ok:
                        try:
                            action.addSuccessFields(duration_ms=elapsed_ms, step=step)
                        except Exception:
                            pass

        return _wrapper

    return _decorator


def _case_args_from_batch_case(
    cfg: dict[str, Any],
    case_record: dict[str, Any],
    run_name: str,
) -> WorkflowRunArgs:
    refactor_scope = cfg.get("refactor_scope", "file")
    assert refactor_scope == "file", "full workflow currently supports only file-only refactoring"

    nested_meta = case_record.get("meta") if isinstance(case_record.get("meta"), dict) else {}

    def _field(*keys: str):
        for source in (case_record, nested_meta):
            if not isinstance(source, dict):
                continue
            for key in keys:
                value = source.get(key)
                if value is not None:
                    return value
        return None

    project = case_record.get("project") or cfg.get("project")
    assert project, "batch case is missing project"

    repo_url = case_record.get("repo_url") or _resolve_repo_url(project, "")
    assert repo_url, f"batch case {case_record.get('case_id')!r} is missing repo_url and project {project!r} is unknown"

    start_commit_hash = _field("start_commit_hash", "start_commit", "commit_hash")
    assert start_commit_hash, f"batch case {case_record.get('case_id')!r} is missing start_commit_hash"
    assert isinstance(start_commit_hash, str) and start_commit_hash.strip(), (
        f"batch case {case_record.get('case_id')!r} has invalid start_commit_hash"
    )

    verification = case_record.get("baseline_verification")
    if verification is not None:
        assert verification.get("status") == "passed", (
            f"batch case {case_record.get('case_id')!r} is not baseline-verified"
        )

    elements = _normalize_case_elements(_field("elements") or case_record.get("elements") or [])
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


def _workflow_cli_args(case_args: WorkflowRunArgs) -> list[str]:
    cli_args: list[str] = []
    for field in fields(WorkflowRunArgs):
        value = getattr(case_args, field.name)
        option = f"--{field.name.replace('_', '-') }"
        if isinstance(value, bool):
            if value:
                cli_args.append(option)
            elif field.name == "targeted_testing":
                cli_args.append(f"--no-{option[2:]}")
            continue
        if value is None:
            continue
        cli_args.extend([option, str(value)])
    return cli_args


def _run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


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
        subprocess.run(["rm", "-rf", str(lock_dir)], check=False)


def _resolve_repo_url(project: str, override: str) -> str:
    if override:
        return override
    url = KNOWN_REPO_URLS.get(project)
    assert url, f"Unknown project={project!r}. Pass --repo-url explicitly."
    return url


@_profile_phase("prepare_repo_checkout")
def _prepare_repo_checkout(
    project: str, repo_url: str, repos_root: Path, commit_hash: str, worktree_suffix: str = ""
) -> Path:
    assert commit_hash, "--start-commit-hash is required"
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
            r = _run_git(["clone", "--bare", "--filter=blob:none", repo_url, str(bare_dir)])
            assert r.returncode == 0, f"bare clone failed: {r.stderr[-500:]}"
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
        ])
        if fetched.returncode != 0:
            # Fallback for servers/commits where direct SHA fetch is restricted.
            fb = _run_git(
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
            assert fb.returncode == 0, f"fallback fetch failed: {fb.stderr[-500:]}"

        if worktree_dir.exists():
            _run_git(["--git-dir", str(bare_dir), "worktree", "remove", "--force", str(worktree_dir)])
            if worktree_dir.exists():
                subprocess.run(["bash", "-lc", f"rm -rf '{worktree_dir}'"], check=False)

        add = _run_git([
            "--git-dir",
            str(bare_dir),
            "worktree",
            "add",
            "--detach",
            str(worktree_dir),
            commit_hash,
        ])
        assert add.returncode == 0, f"worktree add failed: {add.stderr[-500:]}"
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
    assert isinstance(smells_all, list), "Detector must return list[SmellEvent]"
    for s in smells_all:
        assert s.smell_id, "SmellEvent.smell_id must be non-empty"
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


def _extract_llm_code(response_text: str) -> str:
    text = response_text.strip()
    fence = re.search(r"```(?:java)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip() + "\n"
    return text + ("\n" if not text.endswith("\n") else "")


def _resolve_smell_file(repo_path: Path, smell) -> Path | None:
    candidates: list[Path] = []
    if getattr(smell, "file_path", ""):
        candidates.append(repo_path / smell.file_path)
    match = re.search(r":(?P<path>[^:\n]+\.java):\d+", smell.smell_id)
    if match:
        candidates.append(repo_path / match.group("path"))
    if getattr(smell, "class_name", None):
        candidates.append(repo_path / (smell.class_name.replace(".", "/") + ".java"))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    basename = Path(getattr(smell, "file_path", "") or "").name
    if basename:
        matches = sorted(repo_path.glob(f"**/{basename}"))
        if matches:
            return matches[0].resolve()
    return None


@_profile_phase("llm_refactor")
def _execute_refactor_action(
    repo_path: Path,
    step_idx: int,
    smell,
    ref_type: str,
    model_name: str,
) -> tuple[bool, Path | None]:
    rel = None
    try:
        with start_action(action_type="resolve_refactor_target", step=step_idx):
            target_file = _resolve_smell_file(repo_path, smell)
        if target_file is None:
            LOGGER.error(
                "Could not resolve source file for smell_id=%s file_path=%s",
                smell.smell_id,
                getattr(smell, "file_path", None),
            )
            return False, None

        rel = target_file.relative_to(repo_path) if target_file.is_relative_to(repo_path) else target_file

        with start_action(
            action_type="import_llm_client",
            step=step_idx,
            model_name=model_name,
            file=str(rel),
        ):
            try:
                from langchain_litellm import ChatLiteLLM
            except Exception as exc:
                LOGGER.error("Cannot run mandatory LLM refactoring: ChatLiteLLM unavailable: %s", exc)
                return False, None

        with start_action(action_type="read_refactor_source", step=step_idx, file=str(rel)):
            try:
                before = target_file.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                LOGGER.error("Cannot read target Java file as UTF-8: %s: %s", target_file, exc)
                return False, None

        line = getattr(smell, "line_number", 0) or 0
        prompt = f"""Apply the requested Java refactoring directly to this file.

Constraints:
- Return ONLY the complete updated Java file content.
- Do not include explanations or Markdown unless wrapping the code in one java fence.
- Preserve behavior, package, imports, tests, and public API unless the refactoring requires a local change.
- Make a small, compilable edit focused on the target smell.
- If the exact ideal refactoring would require many files, make the safest local improvement in this file that addresses the smell.

Repository: {repo_path}
Step: {step_idx}
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
            "PHASE llm_refactor start step=%d file=%s smell_id=%s ref_type=%s model=%s",
            step_idx,
            rel,
            smell.smell_id,
            ref_type,
            model_name,
        )

        with start_action(
            action_type="llm_refactor_call",
            step=step_idx,
            file=str(rel),
            smell_id=getattr(smell, "smell_id", ""),
            model_name=model_name,
            ref_type=ref_type,
        ):
            try:
                response = ChatLiteLLM(model=model_name).invoke([
                    {
                        "role": "system",
                        "content": "You are a careful Java refactoring agent. Produce complete compilable file contents only.",
                    },
                    {"role": "user", "content": prompt},
                ])
            except Exception as exc:
                LOGGER.error("LLM refactoring failed: %s", exc)
                return False, None

        with start_action(action_type="parse_refactor_llm_output", step=step_idx, file=str(rel)):
            after = _extract_llm_code(response.content if hasattr(response, "content") else str(response))
            if not after.strip():
                LOGGER.error("LLM refactoring produced empty output for %s", rel)
                return False, None
            if after.strip() == before.strip():
                LOGGER.error("LLM refactoring produced no source change for %s", rel)
                return False, None
            if "class " not in after and "interface " not in after and "enum " not in after:
                LOGGER.error("LLM refactoring output did not look like a Java compilation unit for %s", rel)
                return False, None

        with start_action(action_type="write_refactor_output", step=step_idx, file=str(rel)):
            target_file.write_text(after, encoding="utf-8")

        with start_action(action_type="verify_refactor_diff", step=step_idx, file=str(rel)):
            diff = subprocess.run(["git", "diff", "--", str(rel)], cwd=repo_path, capture_output=True, text=True)
            if diff.returncode != 0 or not diff.stdout.strip():
                LOGGER.error("No git diff after LLM refactoring for %s", rel)
                return False, None
            LOGGER.info("PHASE llm_refactor done step=%d file=%s diff_chars=%d", step_idx, rel, len(diff.stdout))

        return True, target_file

    except Exception as exc:
        LOGGER.error("Unexpected error during refactor execution step=%d file=%s: %s", step_idx, rel, exc)
        return False, None


@_profile_phase("eval_patch")
def _run_eval_patch_script(script: str, repo_path: Path, project: str, timeout: int) -> None:
    if not script:
        return
    script_path = Path(script).expanduser()
    if not script_path.is_absolute() and not script_path.exists():
        project_root_candidate = Path(__file__).resolve().parent.parent / script_path
        if project_root_candidate.exists():
            script_path = project_root_candidate
    cmd = [str(script_path), str(repo_path), project]
    LOGGER.info("PHASE eval_patch start script=%s", script)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        LOGGER.info("eval patch stdout: %s", result.stdout[-1000:].strip())
    if result.stderr:
        LOGGER.warning("eval patch stderr: %s", result.stderr[-1000:].strip())
    assert result.returncode == 0, f"eval patch script failed with {result.returncode}: {result.stderr[-500:]}"
    LOGGER.info("PHASE eval_patch done")


def _rollback_repo(repo_path: Path) -> bool:
    r = subprocess.run(["bash", "-lc", "git reset --hard && git clean -fd"], cwd=repo_path, capture_output=True, text=True)
    ok = r.returncode == 0
    if not ok:
        LOGGER.error("Rollback failed: %s", r.stderr[-800:])
    return ok


@_profile_phase("mlflow_healthcheck")
def _run_mlflow_healthcheck(tracking_uri: str, experiment_name: str, timeout: int) -> None:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "check_mlflow_health.sh"
    assert script_path.exists(), f"MLflow healthcheck script not found: {script_path}"
    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": tracking_uri,
        "EXPERIMENT_NAME": experiment_name,
    }
    LOGGER.info(
        "PHASE mlflow_healthcheck start tracking_uri=%s experiment_name=%s script=%s",
        tracking_uri,
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
    assert result.returncode == 0, f"MLflow healthcheck failed with {result.returncode}: {(result.stderr or result.stdout)[-1000:]}"
    LOGGER.info("PHASE mlflow_healthcheck done")


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
    assert isinstance(compile_passed, bool)
    assert isinstance(tests_passed, bool)
    return compile_passed, tests_passed


def _parse_elements_arg(elements_arg: str) -> set[str]:
    raw = (elements_arg or "").strip()
    if not raw:
        return set()
    if raw.startswith("["):
        arr = json.loads(raw)
        assert isinstance(arr, list), "elements JSON must be a list"
        return {str(x).strip() for x in arr if str(x).strip()}
    return {e.strip() for e in raw.split(",") if e.strip()}



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
    except Exception:
        return None


def main(args: WorkflowRunArgs) -> int:
    stack_heartbeat = int(os.environ.get("EVAL_STACK_HEARTBEAT_SECONDS", "0") or "0")
    if stack_heartbeat > 0:
        faulthandler.enable(file=sys.stderr)
        faulthandler.dump_traceback_later(stack_heartbeat, repeat=True, file=sys.stderr)

    _load_workflow_env()
    _ensure_eliot_logging(verbose=args.verbose, eliot_log_path=args.eliot_log_path)

    LOGGER.info("PHASE workflow_start project=%s planner=%s detector=%s model=%s", args.project, args.planner, args.detector_backend, _get_refactor_model(args.model))
    assert args.project, "--project is required"
    if not args.skip_mlflow_healthcheck:
        _run_mlflow_healthcheck(
            tracking_uri=MLFLOW_TRACKING_URI,
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
    _run_eval_patch_script(args.eval_patch_script, repo_path, args.project, args.timeout)

    try:
        import mlflow
    except ImportError:
        LOGGER.error("mlflow is not installed. Run: uv add mlflow")
        return 1

    _enable_litellm_autologging(mlflow)

    detector = _select_detector(args)
    assert detector is not None

    try:
        smells = _detect(detector, repo_path, args.elements)
    except SmellDetectionError as e:
        LOGGER.error("Initial smell detection failed: %s", e)
        return 1

    h_trace: list[float] = []
    step_logs: list[StepLog] = []
    retries_used = 0
    no_progress = 0
    stop_reason = "max_steps"

    initial_count = len(smells)
    if initial_count == 0:
        stop_reason = "smells_zero"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(args.experiment)
    run_name = args.run_name or f"online/{repo_path.name}/{int(time.time())}"

    with mlflow.start_run(run_name=run_name):
        try:
            import pandas as pd
            from mlflow.data import from_pandas

            dataset_df = pd.DataFrame(
                [
                    {
                        "repo_path": str(repo_path),
                        "repo_url": repo_url,
                        "start_commit_hash": args.start_commit_hash,
                        "planner": args.planner,
                        "detector_backend": args.detector_backend,
                        "locality": args.locality,
                        "max_steps": args.max_steps,
                        "max_no_progress": args.max_no_progress,
                        "retry_budget": args.retry_budget,
                        "eval_patch_script": args.eval_patch_script,
                        "model": _get_refactor_model(args.model),
                        "refactor_scope": args.refactor_scope,
                        "mlflow_healthcheck_experiment": args.mlflow_healthcheck_experiment,
                        "skip_mlflow_healthcheck": args.skip_mlflow_healthcheck,
                        "project": args.project,
                        "elements": args.elements,
                        "java_test_repair_model": JAVA_TEST_CODE_AGENT_MODEL,
                        "java_test_repair_steps": JAVA_TEST_CODE_AGENT_MAX_STEPS,
                        "java_test_repair_attempts": JAVA_TEST_CODE_AGENT_MAX_ATTEMPTS,
                    }
                ]
            )
            ds = from_pandas(
                dataset_df,
                source=f"repo:{repo_path}",
                name="composite_workflow_input",
            )
            mlflow.log_input(ds, context="evaluation")
        except Exception as exc:
            LOGGER.warning("mlflow.log_input skipped: %s", exc)

        mlflow.log_params(
            {
                "repo_path": str(repo_path),
                "repo_url": repo_url,
                "start_commit_hash": args.start_commit_hash,
                "planner": args.planner,
                "detector_backend": args.detector_backend,
                "locality": args.locality,
                "max_steps": args.max_steps,
                "max_no_progress": args.max_no_progress,
                "retry_budget": args.retry_budget,
                "h_reduction_threshold": args.h_reduction_threshold,
                "eval_patch_script": args.eval_patch_script,
                "model": _get_refactor_model(args.model),
                "refactor_scope": args.refactor_scope,
                "mlflow_healthcheck_experiment": args.mlflow_healthcheck_experiment,
                "skip_mlflow_healthcheck": args.skip_mlflow_healthcheck,
                "java_test_repair_model": JAVA_TEST_CODE_AGENT_MODEL,
                "java_test_repair_steps": JAVA_TEST_CODE_AGENT_MAX_STEPS,
                "java_test_repair_attempts": JAVA_TEST_CODE_AGENT_MAX_ATTEMPTS,
            }
        )

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
                assert action.smell_id, "Action smell_id must be non-empty"

                target_smell = next((s for s in smells if s.smell_id == action.smell_id), None)
                if target_smell is None:
                    LOGGER.error("Selected action references missing smell_id=%s", action.smell_id)
                    stop_reason = "missing_action_smell"
                    break

                LOGGER.info("PHASE action_selected step=%d smell_id=%s ref_type=%s h_before=%.3f", step_idx, action.smell_id, action.ref_type, h_before)
                execution_ok, modified_file = _execute_refactor_action(
                    repo_path,
                    step_idx,
                    target_smell,
                    action.ref_type,
                    _get_refactor_model(args.model),
                )

                compile_passed = False
                tests_passed = False
                target_files = [str(modified_file)] if args.targeted_testing and modified_file else None
                if execution_ok:
                    compile_passed, tests_passed = _run_java_test_bool(
                        repo_path,
                        args.timeout,
                        target_files=target_files,
                    )

                if not execution_ok or not compile_passed:
                    if retries_used < args.retry_budget:
                        retries_used += 1
                        rolled_back = _rollback_repo(repo_path)
                        assert rolled_back, "Rollback must succeed"
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
                                compile_passed=False,
                                tests_passed=False,
                                execution_ok=execution_ok,
                                stop_reason="retry",
                            )
                        )
                        continue
                    stop_reason = "compile_fail_limit"
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
                            stop_reason=stop_reason,
                        )
                    )
                    break

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
                        f"step_{step_idx}_smells_before": float(len(state.active)),
                        f"step_{step_idx}_smells_after": float(len(smells_after)),
                        f"step_{step_idx}_h_before": float(h_before),
                        f"step_{step_idx}_h_after": float(h_after),
                        f"step_{step_idx}_compile_passed": float(compile_passed),
                        f"step_{step_idx}_tests_passed": float(tests_passed),
                    }
                )

                smells = smells_after
                if no_progress >= args.max_no_progress:
                    stop_reason = "no_progress"
                    break
            finally:
                _CURRENT_STEP.reset(step_token)

        if smells and stop_reason == "max_steps" and no_progress >= args.max_no_progress:
            stop_reason = "no_progress"

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
                    f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

            h_path = Path(td) / "h_trace.json"
            h_path.write_text(json.dumps({"h_trace": h_trace}, indent=2), encoding="utf-8")

            mlflow.log_artifact(str(step_path), artifact_path="online")
            mlflow.log_artifact(str(h_path), artifact_path="online")

    summary = f"Done: initial={initial_count} final={len(smells)} steps={len(step_logs)} stop_reason={stop_reason}"
    LOGGER.info(summary)
    print(summary)
    return 0


def _read_config(config_path: Path) -> dict[str, Any]:
    path = config_path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), "config JSON must be an object"
    cfg = raw.get("workflow", raw)
    assert isinstance(cfg, dict), "workflow config must be an object"
    return cfg



_WORKFLOW_ARG_FIELDS = {field.name for field in fields(WorkflowRunArgs)}


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
    refactor_scope = cfg.get("refactor_scope", "file")
    assert refactor_scope == "file", "full workflow currently supports only file-only refactoring"
    meta = ep.get("meta") or {}
    start_commit_hash = cfg.get("start_commit_hash") or meta.get("start_commit_hash")
    assert start_commit_hash, "start_commit_hash missing in config or episode.meta"

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
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = main(case_args)
    finally:
        _close_eliot_log_handles()
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
    run_name: str | None = typer.Option(None, "--run-name", help="Optional MLflow run name."),
    model: str | None = typer.Option(None, "--model", help="LLM model used for Java refactoring; defaults to COMPOSITE_REFACTOR_MODEL."),
    refactor_scope: str = typer.Option("file", "--refactor-scope", help="Refactoring scope (currently only file)."),
    project: str = typer.Option("", "--project", help="Dataset project name."),
    elements: str = typer.Option("", "--elements", help="CSV element FQNs to constrain the workflow."),
    h_reduction_threshold: float = typer.Option(0.7, "--h-reduction-threshold", help="Minimum relative h reduction to count as success."),
    eval_patch_script: str = typer.Option(
        "scripts/apply_eval_project_patches.sh",
        "--eval-patch-script",
        help="Compatibility patch script run after checkout.",
    ),
    targeted_testing: bool = typer.Option(
        True,
        "--targeted-testing/--no-targeted-testing",
        help="Run post-refactor tests scoped to changed file when possible.",
    ),
    eliot_log_path: str | None = typer.Option(
        None,
        "--eliot-log-path",
        help="Optional path to write Eliot JSON logs for this run.",
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
        run_name=run_name,
        model=model,
        refactor_scope=refactor_scope,
        project=project,
        elements=elements,
        h_reduction_threshold=h_reduction_threshold,
        eval_patch_script=eval_patch_script,
        targeted_testing=targeted_testing,
        eliot_log_path=eliot_log_path,
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
    num_cases: int | None = typer.Option(None, "--num-cases", "--limit", help="Optional number of manifest episodes to run."),
    start_index: int = typer.Option(1, min=1, help="1-based episode index to start from."),
    concurrency: int | None = typer.Option(None, min=1, help="Case-level parallelism. Overrides config.concurrency."),
) -> None:
    """Run the full workflow over a manifest using a JSON config."""
    cfg = _read_config(config)

    _load_workflow_env(".env")

    manifest_path = Path(cfg.get("batch_list") or cfg["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = list(manifest.get("episodes", []))
    if episodes:
        source = "episodes"
    else:
        episodes = list(manifest.get("cases", []))
        source = "cases"
    assert episodes, f"No episodes/cases found in {manifest_path}"
    selected = episodes[start_index - 1:]
    if num_cases is not None:
        selected = selected[:num_cases]

    ready_repos_csv = Path(cfg.get("ready_repos_csv", "evals/helper/filer_ready_repos/maven_only_14_ready_commands.csv"))
    repo_urls = {r["project"]: r["repo_url"] for r in csv.DictReader(ready_repos_csv.open())}
    run_name_prefix = cfg.get("run_name_prefix", f"full-config-{config.stem}")

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
    assert max_workers >= 1, "concurrency must be >= 1"
    show_case_output = bool(cfg.get("verbose", False))
    typer.echo(f"Running {len(case_jobs)} case(s) with case-level concurrency={max_workers}")

    def run_case(job: tuple[int, str, WorkflowRunArgs]) -> tuple[int, str, int, str]:
        offset, case_id, case_args = job
        if case_args.eliot_log_path is None:
            case_args.eliot_log_path = str(
                _case_log_path(Path("outputs/evals/full_logs/eliot"), offset, case_id, case_args).with_suffix(".jsonl")
            )
        code, output = _run_case_with_capture(case_args)
        header = "\n" + "=" * 80 + f"\nCASE {offset} {case_id}\nRUN: {case_args.project}::{case_args.start_commit_hash[:12]}\n" + "=" * 80 + "\n"
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
    assert project, "project is required in config or episode.project"
    repo_url = cfg.get("repo_url") or _resolve_repo_url(project, "")
    run_name = cfg.get("run_name") or f"full-config-{config.stem}"
    case_args = _case_args_from_config(cfg, ep, repo_url, run_name)

    code, output = _run_case_with_capture(case_args)
    if output:
        typer.echo(output)
    raise typer.Exit(code)


if __name__ == "__main__":
    app()
