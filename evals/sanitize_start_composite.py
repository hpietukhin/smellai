#!/usr/bin/env python3
"""Sanity-check one start-composite candidate with java_test_agent only.

This script checks out only the candidate boundary commits (start/end by
default), optionally applies eval compatibility patches, and runs
``agents.java_test.agent.run_java_test_analysis``.  It does not run planner,
intermediate composite steps, or LLM refactoring.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from agents.java_test.agent import run_java_test_analysis
from agents.tools.java_test_tools import TestRunSummary, test_summary_to_dict
from workflows.composite_workflow_full import _prepare_repo_checkout

JsonObject = dict[str, object]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize one start-composite candidate")
    parser.add_argument("--case-file", required=True, help="JSON file containing one case or a {cases:[...]} batch")
    parser.add_argument("--case-id", default="", help="Case id to select when --case-file contains multiple cases")
    parser.add_argument("--repo-url", default="", help="Override repository URL")
    parser.add_argument("--repo-map", default="evals/neo4j_projects.csv", help="CSV with project,repo_url columns")
    parser.add_argument("--repos-root", default="temp/start_composite_sanitation_repos", help="Checkout/worktree root")
    parser.add_argument("--output", required=True, help="Output JSON result path")
    parser.add_argument("--timeout", type=int, default=300, help="java_test_agent command timeout in seconds")
    parser.add_argument("--clean", action="store_true", help="Run clean test command where supported")
    parser.add_argument(
        "--points",
        default="start,end",
        help="Comma-separated boundary points to check: start,end. Default checks both; no intermediate commits are checked.",
    )
    parser.add_argument(
        "--enable-code-agent-repair",
        action="store_true",
        help="Allow java_test_agent repair fallback. Default is off for non-mutating baseline sanitation.",
    )
    parser.add_argument(
        "--eval-patch-script",
        default="scripts/apply_eval_project_patches.sh",
        help="Optional compatibility patch script. Pass empty string to disable.",
    )
    return parser.parse_args()


def _load_repo_urls(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            str(row["project"]): str(row["repo_url"])
            for row in rows
            if row.get("project") and row.get("repo_url")
        }


def _load_cases(path: Path) -> list[JsonObject]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return [case for case in data["cases"] if isinstance(case, dict)]
    if isinstance(data, dict) and data.get("case_id"):
        return [data]
    raise ValueError(f"No case object(s) found in {path}")


def _select_case(cases: list[JsonObject], case_id: str) -> JsonObject:
    if case_id:
        for case in cases:
            if str(case.get("case_id", "")) == case_id:
                return case
        raise ValueError(f"case_id not found: {case_id}")
    if len(cases) == 1:
        return cases[0]
    raise ValueError("--case-id is required when --case-file contains multiple cases")


def _summary_payload(summary: object) -> JsonObject | None:
    if summary is None:
        return None
    if not isinstance(summary, TestRunSummary):
        raise TypeError(f"Unexpected summary type: {type(summary).__name__}")
    payload = test_summary_to_dict(summary)
    payload["stdout_tail"] = summary.stdout[-4000:]
    payload["stderr_tail"] = summary.stderr[-4000:]
    return payload


def _configure_sdkman_toolchain() -> None:
    """Expose the Java 8 / Maven 3.6.3 SDKMAN toolchain to subprocesses.

    Some non-curated repositories fall through to the default java_test_agent
    Maven path, which invokes `mvn` directly.  The full workflow uses a bash
    SDKMAN wrapper for curated repos; sanitation needs the same binaries in
    PATH for broader start-composite candidates.
    """
    home = Path.home()
    java_home = home / ".sdkman" / "candidates" / "java" / "8.0.442-amzn"
    maven_bin = home / ".sdkman" / "candidates" / "maven" / "3.6.3" / "bin"
    path_parts = []
    if maven_bin.exists():
        path_parts.append(str(maven_bin))
    if (java_home / "bin").exists():
        path_parts.append(str(java_home / "bin"))
        os.environ["JAVA_HOME"] = str(java_home)
    existing_path = os.environ.get("PATH", "")
    os.environ["PATH"] = ":".join([*path_parts, existing_path]) if path_parts else existing_path


def _run_patch_script(script: str, repo_path: Path, project: str, timeout: int) -> JsonObject:
    if not script.strip():
        return {"enabled": False, "status": "skipped"}
    script_path = Path(script)
    if not script_path.exists():
        return {"enabled": True, "status": "missing", "script": script}
    result = subprocess.run(
        [str(script_path), str(repo_path), project],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "enabled": True,
        "status": "passed" if result.returncode == 0 else "failed",
        "script": script,
        "exit_code": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def _parse_points(raw_points: str) -> list[str]:
    points = [point.strip() for point in raw_points.split(",") if point.strip()]
    if not points:
        raise ValueError("--points must contain at least one of: start,end")
    invalid = [point for point in points if point not in {"start", "end"}]
    if invalid:
        raise ValueError(f"Unsupported --points values: {invalid}; expected start,end")
    return list(dict.fromkeys(points))


def _run_boundary_check(
    *,
    args: argparse.Namespace,
    project: str,
    repo_url: str,
    repos_root: Path,
    case_id: str,
    point: str,
    commit_hash: str,
) -> JsonObject:
    repo_path = _prepare_repo_checkout(
        project=project,
        repo_url=repo_url,
        repos_root=repos_root,
        commit_hash=commit_hash,
        worktree_suffix=f"sanitize-{point}-{case_id.replace(':', '-')}",
    )
    patch_result = _run_patch_script(str(args.eval_patch_script), repo_path, project, int(args.timeout))
    result = run_java_test_analysis(
        str(repo_path),
        clean=bool(args.clean),
        timeout=int(args.timeout),
        enable_code_agent_repair=bool(args.enable_code_agent_repair),
    )
    summary_payload = _summary_payload(result.get("summary"))
    success = bool(summary_payload and summary_payload.get("success") is True)
    return {
        "point": point,
        "commit_hash": commit_hash,
        "repo_path": str(repo_path),
        "patch": patch_result,
        "java_test_agent": {
            "success": success,
            "build_system": result.get("build_system"),
            "command": result.get("command"),
            "command_source": result.get("command_source"),
            "summary": summary_payload,
            "error": result.get("error"),
            "code_agent_repair": result.get("code_agent_repair"),
            "pre_code_agent_exit_code": result.get("pre_code_agent_exit_code"),
        },
    }


def _sanitize_case(args: argparse.Namespace) -> JsonObject:
    cases = _load_cases(Path(args.case_file))
    case = _select_case(cases, str(args.case_id))
    project = str(case.get("project", ""))
    case_id = str(case.get("case_id", ""))
    start_commit_hash = str(case.get("start_commit_hash", ""))
    end_commit_hash = str(case.get("end_commit_hash", ""))
    if not project or not case_id or not start_commit_hash:
        raise ValueError("case must contain project, case_id, and start_commit_hash")

    requested_points = _parse_points(str(args.points))
    commits_by_point = {"start": start_commit_hash, "end": end_commit_hash}
    missing_points = [point for point in requested_points if not commits_by_point.get(point)]
    if missing_points:
        raise ValueError(f"case is missing commit hash for requested point(s): {missing_points}")

    repo_urls = _load_repo_urls(Path(args.repo_map))
    repo_url = str(args.repo_url or case.get("repo_url") or repo_urls.get(project, ""))
    if not repo_url:
        raise ValueError(f"No repo_url found for project={project!r}")

    repos_root = Path(args.repos_root)
    repos_root.mkdir(parents=True, exist_ok=True)
    _configure_sdkman_toolchain()

    checks = [
        _run_boundary_check(
            args=args,
            project=project,
            repo_url=repo_url,
            repos_root=repos_root,
            case_id=case_id,
            point=point,
            commit_hash=commits_by_point[point],
        )
        for point in requested_points
    ]
    def check_passed(check: JsonObject) -> bool:
        java_test_agent = check.get("java_test_agent")
        return isinstance(java_test_agent, dict) and java_test_agent.get("success") is True

    success = all(check_passed(check) for check in checks)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "case_id": case_id,
        "project": project,
        "repo_url": repo_url,
        "points_checked": requested_points,
        "start_commit_hash": start_commit_hash,
        "end_commit_hash": end_commit_hash,
        "stratification": case.get("stratification", {}),
        "range_metadata": case.get("range_metadata", {}),
        "success": success,
        "checks": checks,
        # Backward-compatible summary used by the batch wrapper: success is the
        # conjunction of requested boundary checks.  Detailed per-point data is
        # in `checks`.
        "java_test_agent": {"success": success},
    }


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = _sanitize_case(args)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, RuntimeError) as exc:
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "case_id": str(args.case_id),
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ok = bool(payload["java_test_agent"]["success"])  # type: ignore[index]
    print(f"{'PASS' if ok else 'FAIL'} {payload['case_id']} -> {output_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
