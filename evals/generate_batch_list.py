#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import tempfile
from bisect import bisect_left
from datetime import UTC, datetime
from pathlib import Path
import time

from agents.java_test.agent import run_java_test_analysis
from evals.commit_build_system_filter import classify_commit_window_build_system
import orjson
from py2neo import Graph
from workflows.composite_workflow_full import _prepare_repo_checkout


# Start-condition defaults (strict, paper-aligned practical filter)
DEFAULT_MIN_REFS = 3
DEFAULT_MAX_REFS = 20
DEFAULT_MIN_SCOPE = 2
DEFAULT_MAX_SCOPE = 10
# Candidate generation accepts only repos from this curated safe set.
REQUESTABLE_EVAL_PROJECTS = {
    "JUnit4",  # junit-team/junit4
    "Lyra",  # jhalterman/lyra
    "OkHttp",  # square/okhttp
    "PhiCode Philib",  # PhiCode/philib
    "Tap4j",  # tupilabs/tap4j
}

# Temporarily excluded from runnable batch generation despite being requestable,
# with explicit reasons recorded in the candidate report.
EXCLUDED_EVAL_PROJECTS = {
    "OkHttp": "excluded from batch generation: historical compatibility/patch target",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a runnable batch list from range-based Neo4j candidates")
    p.add_argument(
        "--projects",
        required=True,
        help="Comma-separated Neo4j project names. Requestable: JUnit4,Lyra,OkHttp,PhiCode Philib,Tap4j (currently excluded: OkHttp)",
    )
    p.add_argument("--uri", default="http://localhost:7474")
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", default="boil2.eat")
    p.add_argument("--limit-per-project", type=int, default=5)
    p.add_argument("--heuristic", choices=["range"], default="range")
    p.add_argument("--min-ref-count", type=int, default=DEFAULT_MIN_REFS)
    p.add_argument("--max-ref-count", type=int, default=DEFAULT_MAX_REFS)
    p.add_argument("--min-scope-size", type=int, default=DEFAULT_MIN_SCOPE)
    p.add_argument("--max-scope-size", type=int, default=DEFAULT_MAX_SCOPE)
    p.add_argument("--output-batch-list", required=True, help="Output batch-list JSON")
    p.add_argument("--output-report", required=True, help="Output candidate-report JSON")
    p.add_argument(
        "--ready-repos-csv",
        default="evals/helper/filer_ready_repos/maven_only_14_ready_commands.csv",
        help="CSV with project -> repo_url mappings for runnable eval repos",
    )
    p.add_argument(
        "--repo-cache-root",
        default="temp/eval_repos",
        help="Local git cache used to inspect historical commit trees before baseline verification",
    )
    p.add_argument(
        "--eval-patch-script",
        default="",
        help="Deprecated compatibility option; ignored. Baseline failures are handled by CodeAgent repair.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for patching and baseline test execution",
    )
    p.add_argument(
        "--start-elements-count",
        type=int,
        default=None,
        help="If set, use exactly this number of start elements per case",
    )
    p.add_argument(
        "--outlier-percentile",
        type=float,
        default=0.95,
        help="Upper percentile cutoff for outlier removal (applied to ref_count, scope_size, and smell_delta_abs)",
    )
    p.add_argument(
        "--min-start-elements-with-smell",
        type=int,
        default=1,
        help="Minimum number of selected start elements that must have >=1 smell at start_commit_hash",
    )
    p.add_argument(
        "--max-case-elements",
        type=int,
        default=None,
        help="Hard cap on selected start elements per case (for quicker, smaller runs).",
    )
    p.add_argument(
        "--max-case-smells",
        type=int,
        default=None,
        help="Hard cap on start-state smells_per_case (sum over selected elements).",
    )
    p.add_argument(
        "--skip-baseline-verification",
        action="store_true",
        help="Skip per-case build/test verification to speed up candidate generation for dry runs.",
    )
    return p.parse_args()


def _case_id(heuristic: str, project: str, elements: list[str]) -> str:
    key = "|".join(sorted(elements))
    h = hashlib.sha1(f"{heuristic}|{project}|{key}".encode("utf-8")).hexdigest()[:12]
    return f"{heuristic}:{project}:{h}"


def _episode_id(project: str, elements: list[str]) -> str:
    return f"{project}::{'|'.join(sorted(elements))}"


def _commits_by_developer(g: Graph, project: str) -> dict[str, list[int]]:
    rows = g.run(
        """
        MATCH (c:Commit)-[:BELONGS_TO]->(:Project {name: $project})
        RETURN c.author_email AS developer, c.order AS commit_order
        ORDER BY developer, commit_order
        """,
        project=project,
    ).data()
    commits: dict[str, list[int]] = {}
    for row in rows:
        commits.setdefault(str(row["developer"] or ""), []).append(int(row["commit_order"]))
    return commits


def _worked_between(commits_by_developer: dict[str, list[int]], developer: str, start_order: int, end_order: int) -> bool:
    """Original range/scope script's temporal guard.

    `detect_scope_based.py` keeps refactorings in one batch only when the same
    developer has not worked in another commit between the current batch and
    the candidate refactoring.  The script implements this as developer-commit
    index distance > 1.
    """
    commits = commits_by_developer.get(developer or "", [])
    start_index = bisect_left(commits, int(start_order))
    end_index = bisect_left(commits, int(end_order))
    if start_index == len(commits) or commits[start_index] != int(start_order):
        return True
    if end_index == len(commits) or commits[end_index] != int(end_order):
        return True
    return abs(end_index - start_index) > 1


def _load_project_refactorings(g: Graph, project: str) -> list[dict]:
    """Load refactorings in the same shape used by the range/scope script."""
    rows = g.run(
        """
        MATCH (r:Refactoring)-[:STARTED_AT]->(cs:Commit)-[:BELONGS_TO]->(p:Project {name: $project})
        MATCH (r)-[:ENDED_AT]->(ce:Commit)-[:BELONGS_TO]->(p)
        OPTIONAL MATCH (r)-[:CHANGED]->(ec:Element)
        WITH r, cs, ce, collect(DISTINCT ec.name) AS changed_names
        OPTIONAL MATCH (r)-[:PRODUCED]->(ep:Element)
        WITH r, cs, ce, changed_names, collect(DISTINCT ep.name) AS produced_names
        RETURN r.hash_id AS ref_id,
               r.type AS ref_type,
               r.classification AS classification,
               r.degradation_level AS degradation_level,
               cs.hash AS start_commit_hash,
               cs.hash_id AS start_dataset_commit_id,
               cs.order AS start_commit_order,
               ce.author_email AS developer,
               ce.hash AS end_commit_hash,
               ce.hash_id AS end_dataset_commit_id,
               ce.order AS end_commit_order,
               changed_names,
               produced_names
        ORDER BY end_commit_order, ref_id
        """,
        project=project,
    ).data()

    refs: list[dict] = []
    for row in rows:
        changed = [x for x in (row.get("changed_names") or []) if x]
        produced = [x for x in (row.get("produced_names") or []) if x]
        elements = sorted(set(changed + produced))
        if not elements:
            continue
        refs.append({
            "ref_id": str(row["ref_id"] or ""),
            "ref_type": str(row["ref_type"] or ""),
            "classification": str(row["classification"] or ""),
            "degradation_level": str(row["degradation_level"] or ""),
            "developer": str(row["developer"] or ""),
            "start_commit_hash": str(row["start_commit_hash"]),
            "start_dataset_commit_id": str(row["start_dataset_commit_id"]),
            "start_commit_order": int(row["start_commit_order"]),
            "end_commit_hash": str(row["end_commit_hash"]),
            "end_dataset_commit_id": str(row["end_dataset_commit_id"]),
            "end_commit_order": int(row["end_commit_order"]),
            # Original detect_scope_based.py batches by row["commit"], i.e. the
            # commit where RefactoringMiner reports the refactoring.  In this
            # graph that is ENDED_AT (the child/refactoring commit), while
            # STARTED_AT is its parent/before state.
            "batch_commit_order": int(row["end_commit_order"]),
            "changed_elements": changed,
            "produced_elements": produced,
            "elements": elements,
        })
    return refs


def _synthesize_range_batches(refactorings: list[dict], commits_by_developer: dict[str, list[int]]) -> list[dict]:
    """Python 3 port of the dataset's `detect_scope_based.py` batching logic."""
    batches: list[dict] = []

    def add_refactoring(batch: dict, refactoring: dict) -> None:
        if any(r["ref_id"] == refactoring["ref_id"] for r in batch["refactorings"]):
            return
        batch["refactorings"].append(refactoring)
        batch["elements"].update(refactoring["elements"])
        batch["last_commit_order"] = max(batch["last_commit_order"], refactoring["batch_commit_order"])

    def merge_into(target: dict, source: dict) -> None:
        target["last_commit_order"] = max(target["last_commit_order"], source["last_commit_order"])
        for refactoring in source["refactorings"]:
            add_refactoring(target, refactoring)

    for refactoring in refactorings:
        ref_elements = set(refactoring["elements"])
        candidates = [
            batch for batch in batches
            if batch["developer"] == refactoring["developer"]
            and not _worked_between(
                commits_by_developer,
                batch["developer"],
                batch["last_commit_order"],
                refactoring["batch_commit_order"],
            )
            and bool(batch["elements"] & ref_elements)
        ]

        if len(candidates) > 1:
            retained = candidates[0]
            for batch in candidates[1:]:
                merge_into(retained, batch)
                batches.remove(batch)
            add_refactoring(retained, refactoring)
        elif len(candidates) == 1:
            add_refactoring(candidates[0], refactoring)
        else:
            batches.append({
                "developer": refactoring["developer"],
                "refactorings": [refactoring],
                "elements": set(refactoring["elements"]),
                "last_commit_order": refactoring["batch_commit_order"],
            })

    return [batch for batch in batches if len(batch["refactorings"]) > 1]


def _candidate_from_range_batch(batch: dict) -> dict:
    refs = batch["refactorings"]
    start_ref = min(refs, key=lambda r: (r["start_commit_order"], r["ref_id"]))
    end_ref = max(refs, key=lambda r: (r["end_commit_order"], r["ref_id"]))
    element_ref_counts = {element: 0 for element in batch["elements"]}
    for ref in refs:
        for element in ref["elements"]:
            element_ref_counts[element] = element_ref_counts.get(element, 0) + 1
    anchor_element = sorted(element_ref_counts, key=lambda e: (-element_ref_counts[e], e))[0]
    return {
        "element": anchor_element,
        "developer": batch["developer"],
        "ref_count": len(refs),
        "scope_size": len(batch["elements"]),
        "scope_names": sorted(batch["elements"]),
        "ref_ids": [r["ref_id"] for r in refs],
        "ref_types": [r["ref_type"] for r in refs],
        "element_ref_counts": element_ref_counts,
        "start_commit_hash": start_ref["start_commit_hash"],
        "start_commit_order": start_ref["start_commit_order"],
        "start_dataset_commit_id": start_ref["start_dataset_commit_id"],
        "end_commit_hash": end_ref["end_commit_hash"],
        "end_commit_order": end_ref["end_commit_order"],
        "end_dataset_commit_id": end_ref["end_dataset_commit_id"],
    }


def _extreme_commit_snapshot(g: Graph, project: str, element: str, latest: bool) -> dict:
    order_clause = "DESC" if latest else "ASC"
    row = g.run(
        f"""
        MATCH (p:Project {{name: $project}})
        MATCH (e:Element {{name: $element}})-[:COMMITTED_IN]->(c:Commit)-[:BELONGS_TO]->(p)
        OPTIONAL MATCH (e)-[:AFFECTED_BY]->(s:Smell)
        WITH c.hash AS commit_hash, c.hash_id AS dataset_commit_id, c.order AS commit_order, count(s) AS smell_count
        ORDER BY commit_order {order_clause}
        LIMIT 1
        RETURN commit_hash, dataset_commit_id, commit_order, smell_count
        """,
        project=project,
        element=element,
    ).data()
    if not row:
        raise ValueError(f"No commit snapshots found for element={element!r} in project={project!r}")
    return {
        "commit_hash": str(row[0]["commit_hash"]),
        "commit_order": int(row[0]["commit_order"]),
        "dataset_commit_id": str(row[0]["dataset_commit_id"]),
        "smell_count": int(row[0]["smell_count"]),
    }


def _refactoring_window_snapshot(g: Graph, project: str, element: str, *, end: bool) -> dict:
    """Endpoint of the refactoring window for an anchor element.

    Sousa et al. classify a composite by comparing smells before the first
    refactoring and after the last refactoring in the composite scope.  For this
    selector's anchor-derived candidate pool, approximate that window as:

    - start: earliest STARTED_AT commit among refactorings touching the anchor;
    - end: latest ENDED_AT commit among refactorings touching the anchor.
    """
    rel = "ENDED_AT" if end else "STARTED_AT"
    order_clause = "DESC" if end else "ASC"
    row = g.run(
        f"""
        MATCH (p:Project {{name: $project}})
        MATCH (e:Element {{name: $element}})
        MATCH (r:Refactoring)-[:CHANGED|PRODUCED]->(e)
        MATCH (r)-[:{rel}]->(c:Commit)-[:BELONGS_TO]->(p)
        RETURN c.hash AS commit_hash,
               c.hash_id AS dataset_commit_id,
               c.order AS commit_order
        ORDER BY commit_order {order_clause}
        LIMIT 1
        """,
        project=project,
        element=element,
    ).data()
    if not row:
        raise ValueError(f"No refactoring window found for element={element!r} in project={project!r}")
    return {
        "commit_hash": str(row[0]["commit_hash"]),
        "commit_order": int(row[0]["commit_order"]),
        "dataset_commit_id": str(row[0]["dataset_commit_id"]),
    }


def _pick_start_elements(g: Graph, project: str, anchor_element: str, scope_elements: list[str], k: int) -> list[str]:
    assert scope_elements, "scope_elements must be non-empty"
    assert 1 <= k <= len(scope_elements)

    rows = g.run(
        """
        MATCH (p:Project {name: $project})
        UNWIND $scope_elements AS scope_name
        MATCH (e:Element {name: scope_name})
        OPTIONAL MATCH (ea:Element {name: $anchor_element})<-[:CHANGED|PRODUCED]-(r:Refactoring)-[:STARTED_AT]->(:Commit)-[:BELONGS_TO]->(p)
        WHERE (r)-[:CHANGED|PRODUCED]->(e)
        WITH e.name AS element,
             count(DISTINCT r) AS ref_coverage,
             CASE WHEN e.name = $anchor_element THEN 1 ELSE 0 END AS is_anchor
        ORDER BY is_anchor DESC, ref_coverage DESC, element ASC
        LIMIT $k
        RETURN element
        """,
        project=project,
        anchor_element=anchor_element,
        scope_elements=scope_elements,
        k=k,
    ).data()

    return [str(r["element"]) for r in rows]


def _smells_per_element(g: Graph, project: str, elements: list[str], commit_hash: str) -> dict[str, int]:
    if not elements:
        return {}
    rows = g.run(
        """
        MATCH (e:Element)-[:COMMITTED_IN]->(c:Commit {hash: $commit_hash})-[:BELONGS_TO]->(:Project {name: $project})
        WHERE e.name IN $elements
        OPTIONAL MATCH (e)-[:AFFECTED_BY]->(s:Smell)
        RETURN e.name AS element, count(s) AS smell_count
        """,
        project=project,
        commit_hash=commit_hash,
        elements=elements,
    ).data()
    out = {e: 0 for e in elements}
    for r in rows:
        out[str(r["element"])] = int(r["smell_count"])
    return out


def _smell_ids_for_elements(g: Graph, project: str, elements: list[str], commit_hash: str) -> set[str]:
    """Stable smell identity set for a scope at one commit.

    The dataset paper classifies composite effects by smell incidence before
    vs. after the composite scope.  Candidate selection therefore compares the
    smell *set* over the whole scope, not only the smell count of the anchor
    element.  `hash_id` is normally present; the fallback keeps tests/odd rows
    deterministic if a smell node lacks it.
    """
    if not elements:
        return set()

    rows = g.run(
        """
        MATCH (e:Element)-[:COMMITTED_IN]->(c:Commit {hash: $commit_hash})-[:BELONGS_TO]->(:Project {name: $project})
        WHERE e.name IN $elements
        OPTIONAL MATCH (e)-[:AFFECTED_BY]->(s:Smell)
        RETURN e.name AS element,
               e.path AS element_path,
               s.hash_id AS smell_hash,
               s.type AS smell_type,
               s.starting_line AS starting_line
        """,
        project=project,
        commit_hash=commit_hash,
        elements=elements,
    ).data()

    smell_ids: set[str] = set()
    for row in rows:
        smell_hash = row.get("smell_hash")
        smell_type = row.get("smell_type")
        if smell_hash:
            smell_ids.add(str(smell_hash))
        elif smell_type:
            smell_ids.add(
                f"{smell_type}:{row.get('element') or ''}:{row.get('element_path') or ''}:{row.get('starting_line') or 0}"
            )
    return smell_ids


def _smell_types_for_elements(g: Graph, project: str, elements: list[str], commit_hash: str) -> list[str]:
    if not elements:
        return []
    rows = g.run(
        """
        MATCH (e:Element)-[:COMMITTED_IN]->(c:Commit {hash: $commit_hash})-[:BELONGS_TO]->(:Project {name: $project})
        WHERE e.name IN $elements
        MATCH (e)-[:AFFECTED_BY]->(s:Smell)
        RETURN DISTINCT s.type AS smell_type
        ORDER BY smell_type
        """,
        project=project,
        commit_hash=commit_hash,
        elements=elements,
    ).data()
    return [str(r["smell_type"]) for r in rows if r.get("smell_type")]


def _percentile(values: list[int], q: float) -> int:
    assert values, "values must be non-empty"
    assert 0.0 < q <= 1.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, math.ceil(q * len(s)) - 1))
    return int(s[idx])


def _load_repo_urls(path: str) -> dict[str, str]:
    csv_path = Path(path)
    rows = csv.DictReader(csv_path.open())
    mapping = {str(r["project"]): str(r["repo_url"]) for r in rows if r.get("project") and r.get("repo_url")}
    assert mapping, f"No repo URLs found in {csv_path}"
    return mapping


def _select_project_candidates(g: Graph, project: str, args: argparse.Namespace) -> list[dict]:
    refactorings = _load_project_refactorings(g, project)
    batches = _synthesize_range_batches(refactorings, _commits_by_developer(g, project))
    rows = [
        _candidate_from_range_batch(batch)
        for batch in batches
        if args.min_ref_count <= len(batch["refactorings"]) <= args.max_ref_count
        and args.min_scope_size <= len(batch["elements"]) <= args.max_scope_size
    ]
    rows.sort(key=lambda r: (-r["ref_count"], -r["scope_size"], r["element"], r["start_commit_order"]))

    provisional: list[dict] = []
    for r in rows:
        scope_names = sorted([x for x in (r.get("scope_names") or []) if x])
        smells_before = _smell_ids_for_elements(g, project, scope_names, r["start_commit_hash"])
        smells_after = _smell_ids_for_elements(g, project, scope_names, r["end_commit_hash"])
        if smells_before == smells_after:
            continue
        provisional.append(
            {
                **r,
                "smells_before": len(smells_before),
                "smells_after": len(smells_after),
                "smell_delta_abs": abs(len(smells_after) - len(smells_before)),
                "smell_set_delta_size": len(smells_before.symmetric_difference(smells_after)),
            }
        )

    selected: list[dict] = []
    if not provisional:
        return selected

    ref_cutoff = _percentile([x["ref_count"] for x in provisional], args.outlier_percentile)
    scope_cutoff = _percentile([x["scope_size"] for x in provisional], args.outlier_percentile)
    delta_cutoff = _percentile([x["smell_delta_abs"] for x in provisional], args.outlier_percentile)

    for c in provisional:
        if c["ref_count"] > ref_cutoff or c["scope_size"] > scope_cutoff or c["smell_delta_abs"] > delta_cutoff:
            continue

        anchor_element = c["element"]
        scope_elements = sorted([x for x in (c.get("scope_names") or []) if x])
        assert len(scope_elements) > 0, "scope_names must not be empty for selected candidate"
        assert args.min_scope_size <= len(scope_elements) <= args.max_scope_size, "scope elements size must satisfy start conditions"

        selected_elements = scope_elements
        if args.start_elements_count is not None:
            if len(scope_elements) < args.start_elements_count:
                continue
            if len(scope_elements) > args.start_elements_count:
                element_ref_counts = c.get("element_ref_counts") or {}
                selected_elements = sorted(
                    scope_elements,
                    key=lambda e: (e != anchor_element, -int(element_ref_counts.get(e, 0)), e),
                )[:args.start_elements_count]

        assert 1 <= len(selected_elements) <= len(scope_elements), "selected_elements must be non-empty and bounded by scope"
        if args.max_case_elements is not None:
            assert args.max_case_elements >= args.min_scope_size
            if len(selected_elements) > args.max_case_elements:
                continue

        selected_smells_before = _smell_ids_for_elements(g, project, selected_elements, c["start_commit_hash"])
        selected_smells_after = _smell_ids_for_elements(g, project, selected_elements, c["end_commit_hash"])

        if selected_smells_before == selected_smells_after:
            continue

        per_element_smells = _smells_per_element(g, project, selected_elements, c["start_commit_hash"])
        start_smell_count = sum(per_element_smells.values())
        assert start_smell_count >= 0, "start_smell_count cannot be negative"
        if args.max_case_smells is not None and start_smell_count > args.max_case_smells:
            continue

        elements_with_start_smell = sum(1 for _, cnt in per_element_smells.items() if cnt > 0)
        if elements_with_start_smell < args.min_start_elements_with_smell:
            continue
        selected.append(
            {
                "case_id": _case_id(args.heuristic, project, selected_elements),
                "project": project,
                "elements": selected_elements,
                "start_commit_hash": c["start_commit_hash"],
                "start_commit_order": c["start_commit_order"],
                "end_commit_hash": c["end_commit_hash"],
                "end_commit_order": c["end_commit_order"],
                "start_state": {
                    "smells_total": sum(per_element_smells.values()),
                    "elements_with_smell": elements_with_start_smell,
                    "smell_types": _smell_types_for_elements(g, project, selected_elements, c["start_commit_hash"]),
                },
                "range_metadata": {
                    "heuristic": args.heuristic,
                    "anchor_element": anchor_element,
                    "developer": c.get("developer", ""),
                    "ref_count": c["ref_count"],
                    "scope_size": len(scope_elements),
                    "ref_ids": c.get("ref_ids", []),
                    "ref_types": sorted(set(c.get("ref_types", []))),
                    "selected_start_elements_count": len(selected_elements),
                    "selected_start_smells": sum(per_element_smells.values()),
                    "selected_start_elements_with_smell": elements_with_start_smell,
                    "smells_after": len(selected_smells_after),
                    "smell_delta_abs": abs(len(selected_smells_after) - len(selected_smells_before)),
                    "smell_set_delta_size": len(selected_smells_before.symmetric_difference(selected_smells_after)),
                    "scope_smells_before": c["smells_before"],
                    "scope_smells_after": c["smells_after"],
                    "scope_smell_delta_abs": c["smell_delta_abs"],
                    "scope_smell_set_delta_size": c["smell_set_delta_size"],
                    "start_dataset_commit_id": c["start_dataset_commit_id"],
                    "end_dataset_commit_id": c["end_dataset_commit_id"],
                    "outlier_percentile": args.outlier_percentile,
                    "outlier_cutoffs": {
                        "ref_count": ref_cutoff,
                        "scope_size": scope_cutoff,
                        "smell_delta_abs": delta_cutoff,
                    },
                },
            }
        )
        if len(selected) >= args.limit_per_project:
            break

    return selected


def _classify_baseline_failure(result: dict) -> tuple[str, str]:
    summary = result.get("summary")
    if summary is None:
        return "toolchain_fail", str(result.get("error") or "no test summary returned")
    if summary.exit_code == 0:
        return "passed", "baseline build/test passed"
    if summary.counts.failed or summary.counts.errors:
        return "test_fail", f"tests failed: failed={summary.counts.failed} errors={summary.counts.errors}"
    return "build_fail", f"command exited with code {summary.exit_code}"


def _verify_candidate(candidate: dict, repo_url: str, args: argparse.Namespace) -> dict:
    assert candidate["project"], "candidate.project must be set"
    assert candidate["start_commit_hash"], "candidate.start_commit_hash must be set"
    assert candidate["case_id"], "candidate.case_id must be set"
    with tempfile.TemporaryDirectory(prefix="batch-list-verify-") as td:
        try:
            repo_path = _prepare_repo_checkout(
                project=candidate["project"],
                repo_url=repo_url,
                repos_root=Path(td),
                commit_hash=candidate["start_commit_hash"],
                worktree_suffix=candidate["case_id"],
            )
        except Exception as exc:
            return {"status": "checkout_fail", "details": str(exc)}

        result = run_java_test_analysis(str(repo_path), clean=False, timeout=args.timeout)
        status, details = _classify_baseline_failure(result)
        summary = result.get("summary")
        return {
            "status": status,
            "details": details,
            "build_system": result.get("build_system"),
            "command": result.get("command"),
            "command_source": result.get("command_source"),
            "llm_repair": result.get("llm_repair"),
            "pre_repair_exit_code": result.get("pre_repair_exit_code"),
            "exit_code": None if summary is None else int(summary.exit_code),
            "tests_total": None if summary is None else int(summary.counts.total),
            "tests_failed": None if summary is None else int(summary.counts.failed),
            "tests_errors": None if summary is None else int(summary.counts.errors),
            "tests_skipped": None if summary is None else int(summary.counts.skipped),
            "patches_applied": False,
        }


def _partition_requested_projects(projects: list[str]) -> tuple[list[str], dict[str, str]]:
    active = [p for p in projects if p not in EXCLUDED_EVAL_PROJECTS]
    excluded = {p: EXCLUDED_EVAL_PROJECTS[p] for p in projects if p in EXCLUDED_EVAL_PROJECTS}
    return active, excluded


def main() -> int:
    args = _parse_args()
    t_run_start = time.perf_counter()

    assert args.limit_per_project >= 1
    assert 1 <= args.min_ref_count <= args.max_ref_count
    assert 1 <= args.min_scope_size <= args.max_scope_size
    assert 0.0 < args.outlier_percentile <= 1.0
    assert args.timeout > 0, "--timeout must be positive"
    if args.start_elements_count is not None:
        assert args.start_elements_count >= 1
    if args.max_case_elements is not None:
        assert args.max_case_elements >= args.min_scope_size
    if args.max_case_smells is not None:
        assert args.max_case_smells >= 0
    assert args.min_start_elements_with_smell >= 0

    requested_projects = [p.strip() for p in args.projects.split(",") if p.strip()]
    assert requested_projects, "--projects must contain at least one project"
    invalid = [p for p in requested_projects if p not in REQUESTABLE_EVAL_PROJECTS]
    assert not invalid, (
        f"Projects not in safe Maven eval set: {invalid}. "
        f"Allowed: {sorted(REQUESTABLE_EVAL_PROJECTS)}"
    )
    projects, excluded_projects = _partition_requested_projects(requested_projects)

    print(f"[DBG] selection start: projects={projects}, excluded={list(excluded_projects)}")
    print(
        "[DBG] filters="
        f"min_ref={args.min_ref_count}, max_ref={args.max_ref_count}, "
        f"scope=[{args.min_scope_size},{args.max_scope_size}], "
        f"max_case_elements={args.max_case_elements}, max_case_smells={args.max_case_smells}, "
        f"start_elements_count={args.start_elements_count}"
    )

    g = Graph(args.uri, auth=(args.user, args.password))
    repo_urls = _load_repo_urls(args.ready_repos_csv)
    assert repo_urls, "Repo URL map from csv must be non-empty"

    accepted_cases: list[dict] = []
    accepted_report: list[dict] = []
    rejected_cases: list[dict] = []
    raw_candidates = 0

    for project, reason in excluded_projects.items():
        rejected_cases.append(
            {
                "case_id": None,
                "project": project,
                "start_commit_hash": None,
                "reason": "project_excluded",
                "details": reason,
            }
        )

    for project in projects:
        project_start = time.perf_counter()
        project_candidates = _select_project_candidates(g, project, args)
        raw_candidates += len(project_candidates)
        print(f"[DBG] project={project}: {len(project_candidates)} candidate cases after selection")

        repo_url = repo_urls.get(project)
        assert repo_url, f"Missing repo_url for project={project!r} in {args.ready_repos_csv}"

        for candidate in project_candidates:
            print(f"[DBG] verifying case={candidate['case_id']} elems={len(candidate['elements'])} smells={candidate['start_state']['smells_total']}")
            commit_build_window = classify_commit_window_build_system(
                g,
                project,
                repo_url,
                candidate["start_commit_order"],
                candidate["end_commit_order"],
                cache_root=args.repo_cache_root,
            )
            if not commit_build_window.all_maven:
                rejected_cases.append(
                    {
                        "case_id": candidate["case_id"],
                        "project": project,
                        "start_commit_hash": candidate["start_commit_hash"],
                        "reason": "unsupported_commit_build_system_window",
                        "details": (
                            "candidate window is not fully maven; "
                            f"first non-maven commit at order {commit_build_window.first_non_maven_order} "
                            f"uses {commit_build_window.first_non_maven_primary}"
                        ),
                        "commit_build_system_window": commit_build_window.to_dict(),
                    }
                )
                continue

            if args.skip_baseline_verification:
                verification = {
                    "status": "passed",
                    "details": "baseline verification skipped",
                    "build_system": "maven",
                    "command": None,
                    "command_source": None,
                    "llm_repair": False,
                    "pre_repair_exit_code": None,
                    "exit_code": None,
                    "tests_total": None,
                    "tests_failed": None,
                    "tests_errors": None,
                    "tests_skipped": None,
                    "patches_applied": False,
                }
            else:
                verification = _verify_candidate(candidate, repo_url, args)

            if verification["status"] == "passed":
                case = {
                    **candidate,
                    "repo_url": repo_url,
                    "commit_build_system_window": commit_build_window.to_dict(),
                    "baseline_verification": verification,
                }
                accepted_cases.append(case)
                accepted_report.append(
                    {
                        "case_id": candidate["case_id"],
                        "project": project,
                        "start_commit_hash": candidate["start_commit_hash"],
                        "reason": "baseline_verified",
                    }
                )
            else:
                rejected_cases.append(
                    {
                        "case_id": candidate["case_id"],
                        "project": project,
                        "start_commit_hash": candidate["start_commit_hash"],
                        "reason": verification["status"],
                        "details": verification.get("details", ""),
                        "commit_build_system_window": commit_build_window.to_dict(),
                    }
                )
        print(f"[DBG] project={project} done in {time.perf_counter() - project_start:.2f}s")

    batch_name = f"safe-maven-range-{datetime.now(UTC).date().isoformat()}"
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    selection_policy = {
        "heuristic": args.heuristic,
        "requested_projects": requested_projects,
        "projects": projects,
        "excluded_projects": excluded_projects,
        "limit_per_project": args.limit_per_project,
        "max_case_elements": args.max_case_elements,
        "max_case_smells": args.max_case_smells,
        "start_elements_count": args.start_elements_count,
        "requires_baseline_verification": not args.skip_baseline_verification,
    }

    batch_list = {
        "batch_name": batch_name,
        "generated_at": generated_at,
        "selection_policy": selection_policy,
        "cases": accepted_cases,
    }
    report = {
        "batch_name": batch_name,
        "generated_at": generated_at,
        "selection_policy": selection_policy,
        "summary": {
            "raw_candidates": raw_candidates,
            "accepted": len(accepted_cases),
            "rejected": len(rejected_cases),
        },
        "accepted_cases": accepted_report,
        "rejected_cases": rejected_cases,
    }

    batch_path = Path(args.output_batch_list)
    report_path = Path(args.output_report)
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_bytes(orjson.dumps(batch_list, option=orjson.OPT_INDENT_2))
    report_path.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))

    elapsed = time.perf_counter() - t_run_start
    print(f"[DBG] generation complete in {elapsed:.2f}s")
    print(f"accepted {len(accepted_cases)} runnable cases from {raw_candidates} raw candidates")
    print(f"batch-list: {batch_path}")
    print(f"candidate-report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
