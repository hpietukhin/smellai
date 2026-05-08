#!/usr/bin/env python3
"""Validate build/test-system audit using Renovate CLI dry-run extraction.

Reads:
  evals/neo4j_project_build_test_systems.csv
  .env.token (raw token or KEY=value)
Writes:
  evals/neo4j_project_build_test_systems.renovate_validation.csv
  evals/neo4j_project_build_test_systems.renovate_validation.md
  outputs/renovate-validation/*.log
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "evals/neo4j_project_build_test_systems.csv"
OUT_CSV = ROOT / "evals/neo4j_project_build_test_systems.renovate_validation.csv"
OUT_MD = ROOT / "evals/neo4j_project_build_test_systems.renovate_validation.md"
LOG_DIR = ROOT / "outputs/renovate-validation"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MANAGER_TO_SYSTEM = {
    "maven": "Maven",
    "maven-wrapper": "Maven",
    "gradle": "Gradle",
    "gradle-wrapper": "Gradle",
    "bazel": "Bazel",
    "bazel-module": "Bazel",
    "npm": "npm",
}
RELEVANT_MANAGERS = sorted(MANAGER_TO_SYSTEM)


def load_token() -> str:
    p = ROOT / ".env.token"
    if not p.exists():
        return ""
    s = p.read_text().strip()
    if "=" in s:
        k, v = s.split("=", 1)
        if k.strip() in {"GITHUB_TOKEN", "RENOVATE_TOKEN", "GH_TOKEN"}:
            return v.strip().strip('"').strip("'")
    return s


def repo_slug(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return path


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")[:80]


def parse_renovate(log_text: str) -> tuple[set[str], dict[str, list[str]]]:
    managers: set[str] = set()
    files_by_manager: dict[str, list[str]] = {}

    # DEBUG: Matched N file(s) for manager maven: a, b, c (repository=...)
    for m in re.finditer(r"Matched \d+ file\(s\) for manager ([\w-]+): (.*?)(?: \(repository=|\n)", log_text):
        mgr = m.group(1)
        if mgr in RELEVANT_MANAGERS:
            managers.add(mgr)
            files = [x.strip() for x in m.group(2).split(",") if x.strip()]
            files_by_manager.setdefault(mgr, []).extend(files[:30])

    # manager extract durations block: "managers": {"maven": 19, "html": 35}
    for m in re.finditer(r'"managers"\s*:\s*(\{[^\n]+\})', log_text):
        try:
            obj = json.loads(m.group(1))
            for mgr in obj:
                if mgr in RELEVANT_MANAGERS:
                    managers.add(mgr)
        except Exception:
            pass

    return managers, files_by_manager


def expected_systems(build_system: str) -> set[str]:
    s = (build_system or "").lower()
    out: set[str] = set()
    if "maven" in s:
        out.add("Maven")
    if "gradle" in s:
        out.add("Gradle")
    if "bazel" in s:
        out.add("Bazel")
    if "ant" in s:
        out.add("Ant")
    if "mixed" in s:
        out.update({"Maven", "Gradle", "Ant", "Bazel"})
    return out


def renovate_systems(managers: set[str]) -> set[str]:
    return {MANAGER_TO_SYSTEM[m] for m in managers if m in MANAGER_TO_SYSTEM}


def status_for(expected: set[str], detected: set[str], audit_build: str) -> str:
    if expected & detected:
        return "match"
    if expected == {"Ant"} and not detected:
        return "renovate_no_ant_support"
    if expected == {"Ant"} and detected:
        return "secondary_managers_detected_for_ant_project"
    if not expected and detected:
        return "renovate_found_unexpected"
    if expected and not detected:
        return "not_detected_by_renovate"
    return "unknown"


def main() -> int:
    token = load_token()
    rows = list(csv.DictReader(INPUT.open()))
    results: list[dict[str, str]] = []
    env = {**os.environ}
    if token:
        env["GITHUB_TOKEN"] = token
        env["RENOVATE_TOKEN"] = token
    env["LOG_LEVEL"] = "debug"

    fieldnames = [
        "project", "repo", "audit_build_system", "audit_test_system",
        "renovate_managers", "renovate_systems", "renovate_package_files_sample",
        "validation_status", "renovate_exit_code", "elapsed_s", "log_path",
    ]

    def write_outputs() -> None:
        with OUT_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(results)
        counts: dict[str, int] = {}
        for rr in results:
            counts[rr["validation_status"]] = counts.get(rr["validation_status"], 0) + 1
        mismatches = [rr for rr in results if rr["validation_status"] not in {"match", "renovate_no_ant_support", "secondary_managers_detected_for_ant_project"}]
        lines = [
            "# Renovate validation of Neo4j project build/test-system audit",
            "",
            f"Progress: {len(results)}/{len(rows)} rows validated.",
            f"Input audit: `{INPUT.relative_to(ROOT)}`",
            f"Output CSV: `{OUT_CSV.relative_to(ROOT)}`",
            f"Logs: `{LOG_DIR.relative_to(ROOT)}/`",
            "",
            "Renovate CLI was run with `--dry-run=extract --platform=github --require-config=ignored` and relevant managers enabled (maven/gradle/wrappers/bazel/npm). Renovate does not support Ant as a build manager, so Ant-primary projects are validated only negatively (no Maven/Gradle/Bazel found) or marked as having secondary package descriptors.",
            "",
            "## Status counts",
            "",
        ]
        for k, v in sorted(counts.items()):
            lines.append(f"- `{k}`: {v}")
        lines += ["", "## Rows needing review", ""]
        if not mismatches:
            lines.append("None beyond expected Ant limitations/secondary descriptors.")
        else:
            for rr in mismatches:
                lines.append(f"- **{rr['project']}**: audit={rr['audit_build_system']}, renovate={rr['renovate_systems'] or 'none'}, status={rr['validation_status']}, log={rr['log_path']}")
        OUT_MD.write_text("\n".join(lines) + "\n")

    for idx, row in enumerate(rows, 1):
        slug = repo_slug(row["repo_url"])
        log_path = LOG_DIR / f"{idx:02d}-{safe_name(slug.replace('/', '-'))}.log"
        cmd = [
            "npx", "--yes", "renovate",
            "--dry-run=extract",
            "--platform=github",
            "--require-config=ignored",
            "--enabled-managers=maven,gradle,gradle-wrapper,maven-wrapper,bazel,bazel-module,npm",
            slug,
        ]
        started = time.monotonic()
        print(f"[{idx:02d}/{len(rows)}] renovate {slug}", flush=True)
        try:
            cp = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
            log = cp.stdout
            rc = cp.returncode
        except subprocess.TimeoutExpired as e:
            log = (e.stdout or "") if isinstance(e.stdout, str) else ""
            log += f"\nTIMEOUT after {e.timeout}s\n"
            rc = -1
        log_path.write_text(log)
        managers, files_by_manager = parse_renovate(log)
        det_systems = renovate_systems(managers)
        exp = expected_systems(row["build_system"])
        status = status_for(exp, det_systems, row["build_system"])
        results.append({
            "project": row["project"],
            "repo": slug,
            "audit_build_system": row["build_system"],
            "audit_test_system": row["test_system"],
            "renovate_managers": ";".join(sorted(managers)),
            "renovate_systems": ";".join(sorted(det_systems)),
            "renovate_package_files_sample": json.dumps(files_by_manager, ensure_ascii=False)[:1000],
            "validation_status": status,
            "renovate_exit_code": str(rc),
            "elapsed_s": f"{time.monotonic()-started:.1f}",
            "log_path": str(log_path.relative_to(ROOT)),
        })
        write_outputs()
        print(f"[{idx:02d}/{len(rows)}] wrote incremental validation row status={status} managers={';'.join(sorted(managers)) or 'none'}", flush=True)

    write_outputs()
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    print(f"rows={len(results)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
