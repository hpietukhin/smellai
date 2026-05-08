"""MLflow predict_fn adapters for evaluation workflows.

MLflow GenAI calls predict functions with flattened EvalSample input fields. This
module reconstructs canonical EvalSample objects and delegates to concrete agent
invoke functions.
"""

from __future__ import annotations

from workflows.common import (
    invoke_swe_agent,
    make_rminer_eval_sample,
    make_swe_eval_sample,
)


def make_rminer_predict_fn(agent):
    """Build an MLflow predict_fn for the RMiner mapping agent."""
    from agents.rminer_eval import invoke_agent as rminer_invoke

    def predict_fn(
        pair_id: str,
        before_code: str,
        file_path: str,
        refactoring_types: list,
        refactoring_descriptions: list,
        diff_hunks: list,
        sonar_issues: list | None = None,
    ) -> dict:
        sample = make_rminer_eval_sample(
            pair_id,
            before_code,
            file_path,
            refactoring_types,
            refactoring_descriptions,
            diff_hunks,
            sonar_issues,
        )
        return rminer_invoke(agent, sample)

    return predict_fn


def make_swe_predict_fn(agent, args):
    """Build an MLflow predict_fn for the standard/composite SWE agent."""
    from agents.swe_eval import invoke_agent as swe_invoke

    analytics_db = None

    def predict_fn(
        project_name: str,
        commit_id: str,
        refactoring_type: str,
        file_path_before: str,
        file_path_after: str,
        class_before: str,
        source_before: str,
        jdk_version: int,
        compile_command: str,
    ) -> dict:
        sample = make_swe_eval_sample(
            project_name,
            commit_id,
            refactoring_type,
            file_path_before,
            file_path_after,
            class_before,
            source_before,
            jdk_version,
            compile_command,
        )
        return invoke_swe_agent(swe_invoke, agent, sample, args, analytics_db)

    return predict_fn


def make_mini_swe_predict_fn(handle, args):
    """Build an MLflow predict_fn for the mini-swe-agent ablation."""
    from evals.ablation.mini_swe_agent import invoke_agent as mini_invoke

    def predict_fn(
        project_name: str,
        commit_id: str,
        refactoring_type: str,
        file_path_before: str,
        file_path_after: str,
        class_before: str,
        source_before: str,
        jdk_version: int,
        compile_command: str,
    ) -> dict:
        sample = make_swe_eval_sample(
            project_name,
            commit_id,
            refactoring_type,
            file_path_before,
            file_path_after,
            class_before,
            source_before,
            jdk_version,
            compile_command,
        )
        return mini_invoke(handle, sample, args.workspace)

    return predict_fn


__all__ = [
    "make_mini_swe_predict_fn",
    "make_rminer_predict_fn",
    "make_swe_predict_fn",
]
