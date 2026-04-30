"""mini-swe-agent ablation wrapper.

Mirrors the swe_eval agent interface so it can be swapped in via --agent mini-swe
in workflows/eval_workflow.py without changing scorers or data loading.

Pipeline:
  a0_setup  (clone, checkout parent commit, switch JDK — same as swe_eval)
  a5_generate  (mini-swe-agent DefaultAgent with LocalEnvironment)
  a6_verify  (compile + run_tests — same primitives as swe_eval, skip file replace)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smellai_datasets.schema import EvalSample
from swe_refactor.adapters import sample_to_refactoring_record
from swe_refactor.dataset import RefactoringRecord
from swe_refactor.runtime import setup_project_workspace, verify_refactoring

from evals.ablation.mini_swe_agent.config import DEFAULT_MINI_CONFIG
from evals.ablation.mini_swe_agent.prompts import build_refactoring_task

LOGGER = logging.getLogger(__name__)


@dataclass
class MiniSWEAgentHandle:
    """Lightweight config handle returned by create_agent.

    Agents are re-instantiated per invoke_agent call because LocalEnvironment
    is workspace-scoped and DefaultAgent holds mutable message state.
    """
    model_name: str
    config: dict = field(default_factory=dict)


def create_agent(
    model_name: str,
    *,
    step_limit: int = 80,
    cost_limit: float = 2.0,
    config_overrides: dict | None = None,
) -> MiniSWEAgentHandle:
    """Create a mini-swe-agent handle (no side effects).

    Args:
        model_name: LiteLLM model name (e.g. "claude-sonnet-4-5-20250929").
        step_limit: Max LLM steps per task.
        cost_limit: Max USD cost per task.
        config_overrides: Extra keys merged on top of DEFAULT_MINI_CONFIG.

    Returns:
        MiniSWEAgentHandle with merged config.
    """
    from minisweagent.utils.serialize import recursive_merge

    cfg = recursive_merge(
        DEFAULT_MINI_CONFIG,
        {"agent": {"step_limit": step_limit, "cost_limit": cost_limit}},
    )
    if config_overrides:
        cfg = recursive_merge(cfg, config_overrides)

    return MiniSWEAgentHandle(model_name=model_name, config=cfg)


def invoke_agent(
    handle: MiniSWEAgentHandle,
    sample: EvalSample,
    workspace_path: str | Path,
    **_kwargs: Any,
) -> dict:
    """Run mini-swe-agent on a single SWE EvalSample.

    Replicates the a0 → a5 → a6 pipeline of swe_eval/agent.py using the
    same swe_refactor.utils primitives so results are directly comparable.

    Args:
        handle: Agent handle from create_agent.
        sample: EvalSample with source="swe".
        workspace_path: Base directory where repos are cloned.
        **_kwargs: Ignored extra kwargs for interface compatibility.

    Returns:
        Dict with keys: project, commit, type, compile_success, test_success,
        error, mini_cost, mini_n_calls, mini_exit, mini_submission.
    """
    if sample.source != "swe":
        raise ValueError(f"mini-swe-agent wrapper expects source='swe', got {sample.source!r}")

    record = sample_to_refactoring_record(sample)

    workspace_path = Path(workspace_path)
    workspace_path.mkdir(parents=True, exist_ok=True)

    # --- a0: setup (clone, checkout, jdk) ---
    setup = setup_project_workspace(record, workspace_path)
    if not setup.success:
        return _failure(record, setup.error or "Setup failed")
    project_path = setup.project_path

    # --- a5: generate (mini-swe-agent) ---
    gen_result = _a5_generate(handle, record, project_path)
    agent_obj = gen_result.pop("_agent", None)  # pull agent for cost stats

    if gen_result.get("error"):
        mini_stats = _extract_stats(agent_obj)
        return {**_failure(record, gen_result["error"]), **mini_stats}

    # --- a6: verify (compile + test — no file replace, agent already mutated fs) ---
    verify_result = _a6_verify(record, project_path)

    mini_stats = _extract_stats(agent_obj)
    return {
        "project": record.projectName,
        "commit": record.commitId,
        "type": record.type,
        "compile_success": verify_result["compile_success"],
        "test_success": verify_result["test_success"],
        "error": verify_result.get("error"),
        "mini_exit": gen_result.get("exit_status", ""),
        "mini_submission": gen_result.get("submission", ""),
        **mini_stats,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _a5_generate(
    handle: MiniSWEAgentHandle,
    record: RefactoringRecord,
    project_path: Path,
) -> dict:
    """Instantiate and run mini-swe-agent. Returns result dict + '_agent' key."""
    from minisweagent.agents import get_agent
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model

    cfg = handle.config

    model_config = {"model_name": handle.model_name, **cfg.get("model", {})}
    env_config = {
        "environment_class": "local",
        **cfg.get("environment", {}),
        "cwd": str(project_path),  # always override with actual project path
    }

    try:
        model = get_model(config=model_config)
        env = get_environment(env_config)
        agent = get_agent(model, env, cfg.get("agent", {}), default_type="default")

        task = build_refactoring_task(record, project_path=str(project_path))
        result = agent.run(task)

        LOGGER.info(
            "a5: mini-swe exit=%s cost=%.4f calls=%s",
            result.get("exit_status"),
            getattr(agent, "cost", 0.0) or 0.0,
            getattr(agent, "n_calls", "?"),
        )
        return {**result, "_agent": agent}

    except Exception as exc:
        LOGGER.error("a5: mini-swe-agent failed: %s", exc)
        return {"error": str(exc), "_agent": None}


def _a6_verify(record: RefactoringRecord, project_path: Path) -> dict:
    """Compile and optionally run tests on the mutated working tree."""
    verification = verify_refactoring(record, project_path)
    return {
        "compile_success": verification.compile_success,
        "test_success": verification.test_success,
        "error": verification.error,
    }


def _extract_stats(agent_obj: Any) -> dict:
    return {
        "mini_cost": float(getattr(agent_obj, "cost", None) or 0.0),
        "mini_n_calls": int(getattr(agent_obj, "n_calls", None) or 0),
    }


def _failure(record: RefactoringRecord, error: str) -> dict:
    return {
        "project": record.projectName,
        "commit": record.commitId,
        "type": record.type,
        "compile_success": False,
        "test_success": False,
        "error": error,
        "mini_exit": "",
        "mini_submission": "",
        "mini_cost": 0.0,
        "mini_n_calls": 0,
    }
