#!/usr/bin/env python3
"""Batch sanitation for stratified start-composite candidates.

The batch runner intentionally delegates every case to
``evals/sanitize_start_composite.py`` so the single-run path is the source of
truth.  By default it consumes ``evals/start_composites/stratified_3x3.json``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

JsonObject = dict[str, object]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch sanitize start-composite candidates")
    parser.add_argument("--case-file", default="evals/start_composites/stratified_3x3.json")
    parser.add_argument("--output-dir", default="evals/start_composites/sanitation")
    parser.add_argument("--single-script", default="evals/sanitize_start_composite.py")
    parser.add_argument("--repo-map", default="evals/neo4j_projects.csv")
    parser.add_argument("--repos-root", default="temp/start_composite_sanitation_repos")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--enable-code-agent-repair", action="store_true")
    parser.add_argument("--eval-patch-script", default="scripts/apply_eval_project_patches.sh")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of cases; 0 means all")
    return parser.parse_args()


def _load_cases(path: Path) -> list[JsonObject]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError(f"Expected {path} to contain a top-level cases list")
    return [case for case in data["cases"] if isinstance(case, dict)]


def _safe_name(case_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in case_id)


def _run_single(args: argparse.Namespace, case: JsonObject, output_path: Path) -> JsonObject:
    case_id = str(case.get("case_id", ""))
    cmd = [
        sys.executable,
        str(args.single_script),
        "--case-file",
        str(args.case_file),
        "--case-id",
        case_id,
        "--repo-map",
        str(args.repo_map),
        "--repos-root",
        str(args.repos_root),
        "--output",
        str(output_path),
        "--timeout",
        str(args.timeout),
        "--eval-patch-script",
        str(args.eval_patch_script),
    ]
    if bool(args.clean):
        cmd.append("--clean")
    if bool(args.enable_code_agent_repair):
        cmd.append("--enable-code-agent-repair")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    payload: JsonObject
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        payload = loaded if isinstance(loaded, dict) else {"case_id": case_id, "status": "bad_output"}
    else:
        payload = {"case_id": case_id, "status": "missing_output"}

    java_test_agent = payload.get("java_test_agent")
    success = bool(isinstance(java_test_agent, dict) and java_test_agent.get("success") is True)
    return {
        "case_id": case_id,
        "project": str(case.get("project", "")),
        "stratification": case.get("stratification", {}),
        "output": str(output_path),
        "exit_code": result.returncode,
        "success": success,
        "stdout_tail": result.stdout[-3000:],
        "stderr_tail": result.stderr[-3000:],
    }


def main() -> int:
    args = _parse_args()
    try:
        cases = _load_cases(Path(args.case_file))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if int(args.limit) > 0:
        cases = cases[: int(args.limit)]

    output_dir = Path(args.output_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    results: list[JsonObject] = []
    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("case_id", f"case-{index}"))
        output_path = runs_dir / f"{index:02d}-{_safe_name(case_id)}.json"
        result = _run_single(args, case, output_path)
        results.append(result)
        status = "PASS" if result["success"] else "FAIL"
        print(f"{index:02d}/{len(cases)} {status} {case_id}")

    passed = sum(1 for result in results if result.get("success") is True)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "case_file": str(args.case_file),
        "single_script": str(args.single_script),
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"summary -> {summary_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
