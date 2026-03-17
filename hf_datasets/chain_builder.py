"""Reconstruct consecutive refactoring chains from HF Datasets.

Each source dataset has different chain semantics:
- rminer: group by repository, order by time (no parent_sha in oracle)
- swe: group by (project_name, commit_id) → co-located refactorings
- tdd: walk parent_sha linkage per project for smell lifecycle chains
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from datasets import Dataset


def build_commit_chains(
    ds: Dataset,
    source: Literal["rminer", "swe", "tdd"],
) -> list[list[dict]]:
    """Reconstruct consecutive commit chains from a HF Dataset.

    Args:
        ds: HF Dataset produced by one of the converter functions
        source: Which dataset schema to use for chain reconstruction

    Returns:
        List of chains.  Each chain is an ordered list of row dicts.
        - rminer/swe: each chain = consecutive commits in one repo/project
        - tdd: each chain = smell lifecycle (introduced→persistent→resolved)
    """
    rows = ds.to_list()

    if source == "rminer":
        return _rminer_chains(rows)
    if source == "swe":
        return _swe_chains(rows)
    if source == "tdd":
        return _tdd_chains(rows)
    raise ValueError(f"Unknown source: {source!r}. Expected 'rminer', 'swe', or 'tdd'.")


# ---------------------------------------------------------------------------
# RMiner chains
# ---------------------------------------------------------------------------

def _rminer_chains(rows: list[dict]) -> list[list[dict]]:
    """Group by repository, sort by time, split into consecutive commit runs."""
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_repo[row.get("repository", "")].append(row)

    chains: list[list[dict]] = []
    for repo_rows in by_repo.values():
        # Sort chronologically using the time string (ISO-like in oracle)
        sorted_rows = sorted(repo_rows, key=lambda r: r.get("time", ""))
        # Group consecutive rows that share the same commit_sha into one "commit"
        chain: list[dict] = []
        prev_sha: str | None = None
        for row in sorted_rows:
            sha = row.get("commit_sha", "")
            if sha != prev_sha and prev_sha is not None:
                # New commit boundary — the existing chain continues (same repo)
                pass
            chain.append(row)
            prev_sha = sha
        if chain:
            chains.append(chain)

    return chains


# ---------------------------------------------------------------------------
# SWE-Refactor chains
# ---------------------------------------------------------------------------

def _swe_chains(rows: list[dict]) -> list[list[dict]]:
    """Group by (project_name, commit_id); order groups by commit_id."""
    by_project: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = row.get("project_name", "")
        by_project[key].append(row)

    chains: list[list[dict]] = []
    for project_rows in by_project.values():
        # Sort by commit_id (lexicographic proxy for order)
        sorted_rows = sorted(project_rows, key=lambda r: r.get("commit_id", ""))
        # Each distinct commit_id is one "node"; collect into a single chain per project
        commit_groups: dict[str, list[dict]] = defaultdict(list)
        for row in sorted_rows:
            commit_groups[row.get("commit_id", "")].append(row)

        # Flatten commit groups in order → one chain per project
        chain: list[dict] = []
        for commit_id in sorted(commit_groups):
            chain.extend(commit_groups[commit_id])
        if chain:
            chains.append(chain)

    return chains


# ---------------------------------------------------------------------------
# TDD smell-lifecycle chains
# ---------------------------------------------------------------------------

def _tdd_chains(rows: list[dict]) -> list[list[dict]]:
    """Walk parent_sha links per project to build smell lifecycle chains.

    A SmellChain groups rows of the same (project, smell_type, file_path)
    in commit-graph order: introduced → persistent* → resolved.
    """
    # Build parent→child mapping per project
    by_project: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_project[row.get("project", "")].append(row)

    chains: list[list[dict]] = []
    for project, p_rows in by_project.items():
        # Build commit ordering via parent_sha chain
        parent_of: dict[str, str] = {}  # sha → parent_sha
        for row in p_rows:
            sha = row.get("commit_sha", "")
            parent = row.get("parent_sha", "")
            if sha and parent:
                parent_of[sha] = parent

        ordered_shas = _topological_sort(parent_of)

        sha_index = {sha: i for i, sha in enumerate(ordered_shas)}

        # Group rows by smell identity (project, smell_type, file_path)
        by_smell: dict[tuple, list[dict]] = defaultdict(list)
        for row in p_rows:
            key = (project, row.get("smell_type", ""), row.get("file_path", ""))
            by_smell[key].append(row)

        for smell_rows in by_smell.values():
            ordered = sorted(
                smell_rows,
                key=lambda r: sha_index.get(r.get("commit_sha", ""), 0),
            )
            if ordered:
                chains.append(ordered)

    return chains


def _topological_sort(parent_of: dict[str, str]) -> list[str]:
    """Return commits in topological order (parents before children)."""
    all_nodes = set(parent_of) | set(parent_of.values())
    children: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = defaultdict(int)

    for child, parent in parent_of.items():
        children[parent].append(child)
        in_degree[child] += 1

    roots = [n for n in all_nodes if in_degree[n] == 0]
    order: list[str] = []
    queue = list(roots)
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # Append any remaining nodes (disconnected components)
    seen = set(order)
    for n in all_nodes:
        if n not in seen:
            order.append(n)

    return order
