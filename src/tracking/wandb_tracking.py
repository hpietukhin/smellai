from __future__ import annotations

import os
from typing import Any, Dict, Optional

import wandb


def init_run(
    project: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    settings: Optional[wandb.Settings] = None,
    job_type: Optional[str] = None,
) -> wandb.sdk.wandb_run.Run:
    project_name = project or os.getenv("WANDB_PROJECT", "mt")
    run = wandb.init(project=project_name, settings=settings, job_type=job_type)
    if config:
        wandb.config.update(config)
    return run


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    wandb.log(metrics, step=step)


def log_artifact_file(path: str, name: Optional[str] = None, type: Optional[str] = None) -> wandb.Artifact:
    artifact_name = name or os.path.basename(path)
    artifact = wandb.Artifact(artifact_name, type=type)
    artifact.add_file(path)
    wandb.log_artifact(artifact)
    return artifact


def use_artifact(ref: str, type: Optional[str] = None, alias: str = "latest") -> str:
    art = wandb.use_artifact(ref if ":" in ref else f"{ref}:{alias}", type=type)
    return art.download()


def finish_run() -> None:
    run = wandb.run
    if run is not None:
        run.finish()

