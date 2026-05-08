"""Composite Refactorings 2020 dataset access layer.

Provides a thin wrapper around the legacy Neo4j 3.1.0 REST API
for querying the composite refactorings graph database, plus
high-level episode extraction for commit-based composites.

Primary usage — on-demand, no intermediate files needed:

    from smellai_datasets.composite_dataset import extract_commit_episodes

    episodes = extract_commit_episodes()                   # all ~4800, ~4.5s
    episodes = extract_commit_episodes(project="JUnit4")   # one project
    episodes = extract_commit_episodes(classification="positive", min_size=2)

    for ep in episodes:
        graph = ep.to_smell_graph()          # SmellGraph, ready for planner
        plan  = graph.calculate_priorities() # greedy ordered plan

Optional JSONL snapshot (reproducibility / offline / sharing only):

    uv run python smellai_datasets/composite_dataset.py extract \
        --output outputs/snapshot.jsonl
    uv run python smellai_datasets/composite_dataset.py query \
        "MATCH (s:Smell) RETURN DISTINCT s.type, count(*) AS cnt ORDER BY cnt DESC"
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

NEO4J_URL = "http://localhost:7474/db/data/cypher"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "boil2.eat"


def _post_cypher(cypher: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": cypher}
    if params:
        payload["params"] = params

    # Send payload on stdin (`-d @-`) so large batched params do not hit
    # the OS argument length limit.
    result = subprocess.run(
        [
            "curl", "-s", "-X", "POST", NEO4J_URL,
            "-H", "Content-Type: application/json",
            "-u", f"{NEO4J_USER}:{NEO4J_PASSWORD}",
            "-d", "@-",
        ],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr}")

    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON response: {result.stdout[:200]}") from e

    if "message" in response and "columns" not in response:
        raise RuntimeError(f"Neo4j error: {response.get('message', response)}")
    return response


def query(cypher: str, *, params: dict[str, Any] | None = None) -> list[list[Any]]:
    """Execute Cypher and return raw row lists."""
    return _post_cypher(cypher, params).get("data", [])


def query_full(cypher: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Like query() but returns full response with columns + data."""
    return _post_cypher(cypher, params)


def query_table(cypher: str, *, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Execute Cypher and return rows as list of dicts (column_name -> value)."""
    resp = query_full(cypher, params=params)
    columns = resp.get("columns", [])
    rows = resp.get("data", [])
    return [dict(zip(columns, row)) for row in rows]


def load_episodes_jsonl(path: str) -> list["CompositeEpisode"]:
    """Load exported CompositeEpisode JSONL without requiring Neo4j."""
    from smellai_datasets.composite_models import episode_from_dict

    episodes = []
    with open(path) as f:
        for line in f:
            if line.strip():
                episodes.append(episode_from_dict(json.loads(line)))
    return episodes


def is_available() -> bool:
    """Check if Neo4j is reachable."""
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             "http://localhost:7474/db/data/", "-u", f"{NEO4J_USER}:{NEO4J_PASSWORD}"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() == "200"
    except (subprocess.TimeoutExpired, OSError):
        return False


# ---------------------------------------------------------------------------
# Bulk episode extraction (4 queries instead of N*6)
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _smell_instance_from_row(row: dict[str, Any], element_name: str) -> "SmellInstance":
    from smellai_datasets.composite_models import SmellInstance

    return SmellInstance(
        smell_type=row.get("smell_type", ""),
        hash_id=row.get("smell_hash", ""),
        reason=row.get("reason", "") or "",
        starting_line=row.get("starting_line", 0) or 0,
        ending_line=row.get("ending_line", 0) or 0,
        element_name=element_name,
        element_path=row.get("element_path", "") or "",
    )


def _append_unique_smells(
    target: list["SmellInstance"],
    seen_ids: set[str],
    rows: list[dict[str, Any]],
    element_name: str,
) -> None:
    for row in rows:
        smell_id = row.get("smell_hash", "")
        if smell_id and smell_id not in seen_ids:
            seen_ids.add(smell_id)
            target.append(_smell_instance_from_row(row, element_name))


def extract_commit_episodes(
    *,
    project: str | None = None,
    min_size: int = 2,
    max_size: int = 10,
    classification: str | None = None,
    with_smells: bool = True,
) -> list["CompositeEpisode"]:
    """Extract commit-based composite episodes from Neo4j.

    Uses 4 bulk Cypher queries (composites, elements, end-commits, smells)
    then joins everything in Python. Fits in <1GB RAM.

    Args:
        project: filter to a single project name (e.g. "JUnit4").
        min_size: minimum number of refactorings in a composite (default 2).
        max_size: maximum number of refactorings in a composite (default 10).
        classification: if set, return only episodes with this majority
            classification ("positive", "negative", or "neutral").
        with_smells: whether to load before/after smell state (default True).
    """
    from collections import defaultdict
    from smellai_datasets.composite_models import (
        CodeElement,
        CompositeEpisode,
        RefactoringStep,
    )

    # --- Query 1: all composites with refactoring details ---
    _log("[1/4] Loading composites...")
    project_filter = "AND p.name = {project}" if project else ""
    params: dict[str, Any] = {"min_size": min_size, "max_size": max_size}
    if project:
        params["project"] = project

    composites_raw = query_table(f"""
        MATCH (r:Refactoring)-[:ENDED_AT]->(c_end:Commit)-[:BELONGS_TO]->(p:Project),
              (r)-[:STARTED_AT]->(c_start:Commit)-[:BELONGS_TO]->(p)
        WHERE 1=1 {project_filter}
        WITH p, c_start, c_end, collect(r) AS refs
        WHERE size(refs) >= {{min_size}} AND size(refs) <= {{max_size}}
        RETURN
            p.name AS project,
            c_start.hash AS commit_hash,
            c_start.order AS commit_order,
            c_start.message AS commit_message,
            c_end.hash AS end_commit_hash,
            c_end.order AS end_commit_order,
            [r IN refs | {{
                ref_type: r.type,
                hash_id: r.hash_id,
                classification: r.classification,
                degradation_level: r.degradation_level,
                smelly: r.smelly,
                parameters: r.parameters,
                commit_hash: c_end.hash,
                commit_order: c_end.order
            }}] AS refactorings
        ORDER BY p.name, c_end.order
    """, params=params)
    _log(f"  -> {len(composites_raw)} composites")

    # Collect all ref hash_ids and commit hashes
    all_ref_ids: list[str] = []
    all_start_commits: set[str] = set()
    all_end_commits: set[str] = set()
    for row in composites_raw:
        all_start_commits.add(row["commit_hash"])
        if row.get("end_commit_hash"):
            all_end_commits.add(row["end_commit_hash"])
        for r in row["refactorings"]:
            hid = r.get("hash_id", "")
            if hid:
                all_ref_ids.append(hid)

    # --- Query 2: all element relationships (CHANGED + PRODUCED) in bulk ---
    _log(f"[2/4] Loading elements for {len(all_ref_ids)} refactorings...")
    # ref_id -> [(element_name, element_type, element_path, relation)]
    ref_elements: dict[str, list[dict[str, str]]] = defaultdict(list)
    all_element_names: set[str] = set()

    for relation in ("CHANGED", "PRODUCED"):
        rows = query_table(f"""
            MATCH (r:Refactoring)-[:{relation}]->(e:Element)
            WHERE r.hash_id IN {{ref_ids}}
            RETURN r.hash_id AS ref_id,
                   e.name AS name,
                   e.type AS etype,
                   e.path AS path
        """, params={"ref_ids": all_ref_ids})
        for r in rows:
            name = r["name"] or ""
            ref_elements[r["ref_id"]].append({
                "name": name,
                "etype": r.get("etype", "") or "",
                "path": r.get("path", "") or "",
                "relation": relation,
            })
            if name:
                all_element_names.add(name)
    _log(f"  -> {len(all_element_names)} unique elements")

    # --- Query 3: end commits for all start commits ---
    _log("[3/4] Loading end commits...")
    start_to_end: dict[str, str] = {}
    if with_smells:
        rows = query_table("""
            MATCH (r:Refactoring)-[:STARTED_AT]->(c_start:Commit),
                  (r)-[:ENDED_AT]->(c_end:Commit)
            WHERE c_start.hash IN {hashes}
            RETURN DISTINCT c_start.hash AS start_hash, c_end.hash AS end_hash
        """, params={"hashes": list(all_start_commits)})
        for r in rows:
            start_to_end[r["start_hash"]] = r["end_hash"]
    _log(f"  -> {len(start_to_end)} mappings")

    # --- Query 4: all smells on relevant elements at relevant commits ---
    smells_by_commit_element: dict[tuple[str, str], list[dict]] = defaultdict(list)
    if with_smells and all_element_names:
        all_relevant_commits = list(all_start_commits | all_end_commits | set(start_to_end.values()))
        _log(f"[4/4] Loading smells for {len(all_element_names)} elements "
             f"at {len(all_relevant_commits)} commits...")
        rows = query_table("""
            MATCH (e:Element)-[:COMMITTED_IN]->(c:Commit),
                  (e)-[:AFFECTED_BY]->(s:Smell)
            WHERE c.hash IN {commits} AND e.name IN {elements}
            RETURN c.hash AS commit_hash,
                   e.name AS element_name,
                   e.path AS element_path,
                   s.type AS smell_type,
                   s.hash_id AS smell_hash,
                   s.reason AS reason,
                   s.starting_line AS starting_line,
                   s.ending_line AS ending_line
        """, params={
            "commits": all_relevant_commits,
            "elements": list(all_element_names),
        })
        for r in rows:
            key = (r["commit_hash"], r["element_name"] or "")
            smells_by_commit_element[key].append(r)
        _log(f"  -> {len(rows)} smell-element-commit records")
    else:
        _log("[4/4] Skipping smells (disabled or no elements)")

    # --- Assemble episodes in Python ---
    _log("Assembling episodes...")
    episodes: list[CompositeEpisode] = []

    for row in composites_raw:
        proj = row["project"]
        commit = row["commit_hash"]
        order = row["commit_order"] or 0
        message = row.get("commit_message") or ""

        steps: list[RefactoringStep] = []
        n_pos = n_neg = n_neu = 0
        episode_elements: dict[str, CodeElement] = {}

        for r in row["refactorings"]:
            cls = r.get("classification", "neutral") or "neutral"
            if cls == "positive":
                n_pos += 1
            elif cls == "negative":
                n_neg += 1
            else:
                n_neu += 1

            hid = r.get("hash_id", "")
            elem_info = ref_elements.get(hid, [])
            changed = [e["name"] for e in elem_info if e["relation"] == "CHANGED"]
            produced = [e["name"] for e in elem_info if e["relation"] == "PRODUCED"]

            for e in elem_info:
                name = e["name"]
                if name and name not in episode_elements:
                    episode_elements[name] = CodeElement(
                        name=name,
                        element_type=e["etype"],
                        file_path=e["path"],
                    )

            steps.append(RefactoringStep(
                ref_type=r.get("ref_type", ""),
                hash_id=hid,
                classification=cls,
                degradation_level=r.get("degradation_level", "") or "",
                smelly=bool(r.get("smelly", False)),
                commit_hash=r.get("commit_hash", commit) or commit,
                commit_order=r.get("commit_order", order) or order,
                parameters=r.get("parameters", "") or "",
                changed_elements=changed,
                produced_elements=produced,
            ))

        if n_pos > n_neg:
            episode_classification = "positive"
        elif n_neg > n_pos:
            episode_classification = "negative"
        else:
            episode_classification = "neutral"

        # Collect smells
        smells_before: list[SmellInstance] = []
        smells_after: list[SmellInstance] = []

        if with_smells:
            seen_before: set[str] = set()
            seen_after: set[str] = set()
            end_hash = row.get("end_commit_hash") or start_to_end.get(commit)

            for ename in episode_elements:
                _append_unique_smells(
                    smells_before,
                    seen_before,
                    smells_by_commit_element.get((commit, ename), []),
                    ename,
                )

                if end_hash:
                    _append_unique_smells(
                        smells_after,
                        seen_after,
                        smells_by_commit_element.get((end_hash, ename), []),
                        ename,
                    )

        episodes.append(CompositeEpisode(
            episode_id=f"{proj}:{commit}:{order}",
            project=proj,
            commit_hash=commit,
            commit_order=order,
            commit_message=message,
            refactorings=steps,
            scope_elements=list(episode_elements.values()),
            smells_before=smells_before,
            smells_after=smells_after,
            classification=episode_classification,
            n_positive=n_pos,
            n_negative=n_neg,
            n_neutral=n_neu,
        ))

    if classification:
        episodes = [e for e in episodes if e.classification == classification]

    _log(f"Done: {len(episodes)} episodes")
    return episodes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_extract(args: Any) -> None:
    """CLI handler for --extract."""
    import pathlib

    episodes = extract_commit_episodes(
        project=args.project,
        min_size=args.min_size,
        max_size=args.max_size,
        with_smells=not args.no_smells,
    )

    if args.output:
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for ep in episodes:
                f.write(json.dumps(ep.to_dict()) + "\n")
        print(f"Wrote {len(episodes)} episodes to {out}")
    else:
        for ep in episodes:
            print(json.dumps(ep.to_dict()))

    # Summary
    n_pos = sum(1 for e in episodes if e.is_positive)
    n_neg = sum(1 for e in episodes if e.classification == "negative")
    n_neu = sum(1 for e in episodes if e.classification == "neutral")
    n_agg = sum(1 for e in episodes if e.is_agglomeration)
    print(f"\nSummary: {len(episodes)} episodes "
          f"(positive={n_pos}, negative={n_neg}, neutral={n_neu}, "
          f"agglomeration={n_agg})", file=sys.stderr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Composite Refactorings 2020 dataset tools")
    sub = parser.add_subparsers(dest="command")

    # Raw Cypher query
    q_parser = sub.add_parser("query", help="Run a raw Cypher query")
    q_parser.add_argument("cypher", nargs="+", help="Cypher query")

    # Episode extraction
    e_parser = sub.add_parser("extract", help="Extract commit-based composite episodes")
    e_parser.add_argument("--project", help="Filter to specific project name")
    e_parser.add_argument("--min-size", type=int, default=2, help="Minimum composite size")
    e_parser.add_argument("--max-size", type=int, default=10, help="Maximum composite size")
    e_parser.add_argument("--output", "-o", help="Output JSONL file path")
    e_parser.add_argument("--no-smells", action="store_true", help="Skip smell state queries (faster)")

    args = parser.parse_args()

    if args.command == "extract":
        _cli_extract(args)
    elif args.command == "query":
        cypher_input = " ".join(args.cypher)
        resp = query_full(cypher_input)
        print(json.dumps(resp, indent=2))
    else:
        # Backwards compat: bare args = cypher query
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            cypher_input = " ".join(sys.argv[1:])
            resp = query_full(cypher_input)
            print(json.dumps(resp, indent=2))
        else:
            parser.print_help()
