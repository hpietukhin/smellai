"""
RefactoringMiner data processing utilities.

This module provides functions for parsing, analyzing, and clustering
RefactoringMiner benchmark data. It supports the simplified oracle data.json
format and includes extensibility for full RefactoringMiner output.

Key functions:
- load_rminer_data: Parse data.json into Pydantic models
- group_by_repository: Organize commits by repository
- find_consecutive_commits: Detect clusters based on ID proximity
- calculate_semantic_similarity: Compute relatedness scores
- compute_statistics: Generate summary statistics

# TODO SPEC-017: Verify manifest format matches raw RefactoringMiner 2.0 output.
# Need to verify if intermediate processing is needed between RM output and manifest format.
# LOW priority - verification task.
# (See TECHNICAL_SPECIFICATION.md §6.5)
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from rminer.diff_hunk import DiffHunk
from models.refactoring import (
    CommitCluster,
    RefactoringStats,
    RMinerCommit,
)

LOGGER = logging.getLogger(__name__)

_HUNK_PATTERN = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def get_diff(
    before_file: str,
    after_file: str,
    *,
    repo_dir: Optional[str] = None,
    function_context: bool = False,
) -> str:
    """Generate unified diff between two files using diff command."""
    import subprocess

    cmd = ["diff", "-u"]
    if function_context:
        cmd.append("-p")
    cmd.extend([before_file, after_file])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=repo_dir if repo_dir else None,
        )
        # diff returns 1 when files differ, which is expected
        return result.stdout
    except Exception as e:
        LOGGER.error(f"Error running diff: {e}")
        return ""


def _new_hunk_data(match: re.Match[str]) -> Dict[str, Any]:
    return {
        "old_start": int(match.group(1)),
        "old_count": int(match.group(2) or 1),
        "new_start": int(match.group(3)),
        "new_count": int(match.group(4) or 1),
        "removed_lines": [],
        "added_lines": [],
        "context_lines": [],
    }


def _add_diff_line(current_data: Dict[str, Any], line: str) -> None:
    if line.startswith(("---", "+++", "\\ No newline at end of file")):
        return
    if line.startswith("-"):
        current_data["removed_lines"].append(line[1:])
    elif line.startswith("+"):
        current_data["added_lines"].append(line[1:])
    elif line.startswith(" "):
        current_data["context_lines"].append(line[1:])


def parse_diff_hunks(diff_text: str) -> List[DiffHunk]:
    """Parse unified diff text into structured hunks."""
    hunks: List[DiffHunk] = []
    current_data: Optional[Dict[str, Any]] = None

    for line in diff_text.splitlines():
        match = _HUNK_PATTERN.match(line)
        if match:
            if current_data:
                hunks.append(DiffHunk(**current_data))
            current_data = _new_hunk_data(match)
        elif current_data:
            _add_diff_line(current_data, line)

    if current_data:
        hunks.append(DiffHunk(**current_data))
    return hunks


def compute_diff_hunks_from_files(
    before_file: str | Path,
    after_file: str | Path,
    *,
    repo_dir: Optional[str | Path] = None,
    function_context: bool = False,
) -> List[DiffHunk]:
    """Compute unified diff hunks for two files using git_utils.get_diff."""

    before_path = Path(before_file)
    after_path = Path(after_file)

    if not before_path.exists():
        raise FileNotFoundError(f"Before file not found: {before_path}")
    if not after_path.exists():
        raise FileNotFoundError(f"After file not found: {after_path}")

    repo_dir_str = str(repo_dir) if repo_dir is not None else None
    LOGGER.debug(
        "Computing diff hunks between %s and %s (repo_dir=%s, function_context=%s)",
        before_path,
        after_path,
        repo_dir_str,
        function_context,
    )
    diff_text = get_diff(
        str(before_path),
        str(after_path),
        repo_dir=repo_dir_str,
        function_context=function_context,
    )

    if not diff_text.strip():
        return []

    return parse_diff_hunks(diff_text)


def load_rminer_data(file_path: str) -> List[RMinerCommit]:
    """
    Load and parse RefactoringMiner benchmark data.json.

    Reads the JSON file and converts each commit entry into a RMinerCommit
    Pydantic model with validation. Handles both camelCase (from JSON) and
    snake_case field names.

    Args:
        file_path: Absolute path to data.json file

    Returns:
        List of RMinerCommit objects sorted by commit ID

    Raises:
        FileNotFoundError: If file doesn't exist at the specified path
        ValueError: If JSON is malformed or doesn't match expected schema
        json.JSONDecodeError: If file contains invalid JSON

    Examples:
        >>> commits = load_rminer_data("/path/to/data.json")
        >>> len(commits)
        549
        >>> commits[0].repository
        'https://github.com/realm/realm-java.git'
    """
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}. "
            "Please verify the path to RefactoringMiner data.json."
        )

    LOGGER.debug(f"Reading RefactoringMiner data from {file_path}")

    with open(file_path_obj, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected JSON array at root level, got {type(data).__name__}. "
            "Data file should contain an array of commit objects."
        )

    commits = []
    for i, commit_data in enumerate(data):
        try:
            commit = RMinerCommit(**commit_data)
            commits.append(commit)
        except Exception as e:
            LOGGER.warning(
                f"Failed to parse commit at index {i} (id={commit_data.get('id', 'unknown')}): {e}"
            )
            continue

    LOGGER.info(
        f"Loaded {len(commits)} commits from {file_path} "
        f"({len(data) - len(commits)} failed to parse)"
    )

    return sorted(commits, key=lambda c: c.id)


def group_by_repository(commits: List[RMinerCommit]) -> Dict[str, List[RMinerCommit]]:
    """
    Group commits by repository URL.

    Organizes commits into a dictionary keyed by repository URL,
    with each value being a list of commits sorted by commit ID.

    Args:
        commits: List of commits to group

    Returns:
        Dictionary mapping repository URL to sorted list of commits

    Examples:
        >>> repo_groups = group_by_repository(commits)
        >>> len(repo_groups)
        188
        >>> len(repo_groups['https://github.com/JetBrains/intellij-community.git'])
        37
    """
    grouped = defaultdict(list)

    for commit in commits:
        grouped[commit.repository].append(commit)

    for repo_url in grouped:
        grouped[repo_url].sort(key=lambda c: c.id)

    LOGGER.debug(f"Grouped {len(commits)} commits into {len(grouped)} repositories")

    return dict(grouped)


def find_consecutive_commits(
    commits: List[RMinerCommit],
    *,
    max_id_gap: int = 10,
    min_cluster_size: int = 2,
) -> List[CommitCluster]:
    """
    Identify clusters of consecutive commits based on ID proximity.

    Uses a sliding window algorithm to detect groups of commits where
    the gap between consecutive IDs is at most max_id_gap. Each cluster
    is scored based on ID proximity, semantic similarity, and size.

    Algorithm:
    1. Sort commits by ID
    2. Scan through commits, grouping when gap <= max_id_gap
    3. Finalize clusters that meet min_cluster_size
    4. Calculate cluster_score for each cluster

    Args:
        commits: List of commits from the same repository
        max_id_gap: Maximum gap between IDs to consider consecutive (default: 10)
        min_cluster_size: Minimum commits required for a cluster (default: 2)

    Returns:
        List of CommitCluster objects sorted by cluster_score (descending)

    Examples:
        >>> commits = [commit1, commit2, commit3]  # IDs: [1100842, 1100856, 1100868]
        >>> clusters = find_consecutive_commits(commits, max_id_gap=15)
        >>> len(clusters)
        1
        >>> clusters[0].commit_ids
        [1100842, 1100856, 1100868]
        >>> clusters[0].cluster_score
        0.87
    """
    if not commits:
        return []

    sorted_commits = sorted(commits, key=lambda c: c.id)

    if len(sorted_commits) == 1:
        LOGGER.debug("Only one commit provided, no clusters formed")
        return []

    clusters = []
    current_cluster_commits = [sorted_commits[0]]

    for i in range(1, len(sorted_commits)):
        gap = sorted_commits[i].id - sorted_commits[i - 1].id

        if gap <= max_id_gap:
            current_cluster_commits.append(sorted_commits[i])
        else:
            if len(current_cluster_commits) >= min_cluster_size:
                cluster = _create_cluster(current_cluster_commits)
                clusters.append(cluster)

            current_cluster_commits = [sorted_commits[i]]

    if len(current_cluster_commits) >= min_cluster_size:
        cluster = _create_cluster(current_cluster_commits)
        clusters.append(cluster)

    clusters.sort(key=lambda c: c.cluster_score, reverse=True)

    LOGGER.debug(
        f"Found {len(clusters)} clusters from {len(commits)} commits "
        f"(max_gap={max_id_gap}, min_size={min_cluster_size})"
    )

    return clusters


def _create_cluster(commits: List[RMinerCommit]) -> CommitCluster:
    """
    Create a CommitCluster with calculated score.

    Args:
        commits: List of commits to cluster

    Returns:
        CommitCluster with populated fields and calculated score
    """
    cluster = CommitCluster(
        repository=commits[0].repository,
        commit_ids=[c.id for c in commits],
        commits=commits,
    )
    cluster.cluster_score = _calculate_cluster_score(cluster)
    return cluster


def _calculate_cluster_score(cluster: CommitCluster) -> float:
    """
    Calculate cluster quality score (0.0 to 1.0).

    Combines three factors:
    1. ID proximity: Smaller gaps = higher score (50% weight)
    2. Semantic similarity: File/type overlap = higher score (40% weight)
    3. Size bonus: Logarithmic bonus for larger clusters (10% weight, capped at 0.2)

    Args:
        cluster: CommitCluster to score

    Returns:
        Score from 0.0 to 1.0 indicating cluster quality
    """
    avg_gap = cluster.avg_id_gap()
    proximity_score = 1.0 / (1.0 + avg_gap / 10.0)

    semantic_score = calculate_semantic_similarity(cluster)

    size_bonus = math.log(len(cluster.commits)) / math.log(10.0)
    size_bonus = min(size_bonus, 0.2)

    final_score = 0.5 * proximity_score + 0.4 * semantic_score + 0.1 * size_bonus

    return min(final_score, 1.0)


def _average_jaccard_similarity(sets: list[set[str]]) -> float:
    similarities = []
    for i, set_a in enumerate(sets):
        for set_b in sets[i + 1:]:
            if set_a or set_b:
                similarities.append(len(set_a & set_b) / len(set_a | set_b))
    return sum(similarities) / len(similarities) if similarities else 0.0


def calculate_semantic_similarity(cluster: CommitCluster) -> float:
    """
    Calculate semantic similarity score based on refactoring overlap.

    Analyzes three aspects of semantic relatedness:
    1. Refactoring type overlap (Jaccard similarity)
    2. File path overlap extracted from descriptions
    3. Author consistency
    """
    if len(cluster.commits) < 2:
        return 1.0

    refactoring_types = [
        {ref.type for ref in commit.refactorings} for commit in cluster.commits
    ]
    file_paths = [_extract_file_paths(commit) for commit in cluster.commits]
    author_score = 1.0 if len({commit.author for commit in cluster.commits}) == 1 else 0.5

    type_score = _average_jaccard_similarity(refactoring_types)
    path_score = _average_jaccard_similarity(file_paths)
    return 0.4 * type_score + 0.4 * path_score + 0.2 * author_score


def _extract_file_paths(commit: RMinerCommit) -> Set[str]:
    """
    Extract file paths mentioned in refactoring descriptions.

    Uses regex to find Java package/class patterns like "com.example.Foo"
    and treats them as pseudo file paths for similarity analysis.

    Args:
        commit: Commit to extract paths from

    Returns:
        Set of file path patterns found in descriptions
    """
    file_paths = set()

    class_pattern = re.compile(r"\b([a-z][a-z0-9]*\.)+[A-Z][a-zA-Z0-9]*\b")

    for ref in commit.refactorings:
        matches = class_pattern.findall(ref.description)
        for match in matches:
            file_paths.add(match)

    return file_paths


def compute_statistics(commits: List[RMinerCommit]) -> RefactoringStats:
    """
    Generate statistical summary of refactoring data.

    Calculates aggregated metrics including:
    - Total counts (commits, repositories, refactorings)
    - Refactoring type distribution
    - Validation status distribution
    - Top repositories by commit count

    Args:
        commits: All commits to analyze

    Returns:
        RefactoringStats object with computed metrics

    Examples:
        >>> stats = compute_statistics(commits)
        >>> stats.total_commits
        549
        >>> stats.total_repositories
        188
        >>> stats.refactoring_type_counts['Extract Method']
        267
    """
    total_commits = len(commits)

    repositories = {commit.repository for commit in commits}
    total_repositories = len(repositories)

    all_refactorings = [
        refactoring for commit in commits for refactoring in commit.refactorings
    ]
    total_refactorings = len(all_refactorings)

    refactoring_type_counter = Counter(ref.type for ref in all_refactorings)
    refactoring_type_counts = dict(refactoring_type_counter)

    validation_counter = Counter()
    for ref in all_refactorings:
        if ref.validation in ("TP", "FP"):
            validation_counter[ref.validation] += 1
        else:
            validation_counter["Other/null"] += 1

    validation_counts = dict(validation_counter)

    repo_commit_counts = Counter(commit.repository for commit in commits)
    top_repositories = [
        {"repository": repo, "commit_count": count}
        for repo, count in repo_commit_counts.most_common(10)
    ]

    stats = RefactoringStats(
        total_commits=total_commits,
        total_repositories=total_repositories,
        total_refactorings=total_refactorings,
        refactoring_type_counts=refactoring_type_counts,
        validation_counts=validation_counts,
        top_repositories=top_repositories,
        clusters_found=0,
        clusters_detail=[],
    )

    LOGGER.debug(
        f"Computed statistics: {total_commits} commits, "
        f"{total_repositories} repos, {total_refactorings} refactorings"
    )

    return stats


def fetch_full_refactoring_json(
    repo_url: str, commit_sha: str, *, rminer_jar_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute RefactoringMiner to get full JSON with location data.

    This is a placeholder for Phase 2. Full implementation will:
    1. Clone repository (or use existing clone)
    2. Run RefactoringMiner CLI: java -jar RefactoringMiner.jar -c repo commit
    3. Parse JSON output with leftSideLocations/rightSideLocations
    4. Return structured data

    Args:
        repo_url: Git repository URL
        commit_sha: Commit SHA to analyze
        rminer_jar_path: Path to RefactoringMiner JAR (default: env var RMINER_JAR)

    Returns:
        Full RefactoringMiner JSON output as dict

    Raises:
        NotImplementedError: This feature is not yet implemented
    """
    raise NotImplementedError(
        "Phase 2 feature: RefactoringMiner CLI integration not yet implemented. "
        "This requires building RefactoringMiner JAR and configuring RMINER_JAR env var. "
        f"Requested: {repo_url} @ {commit_sha}"
    )


def build_dependency_graph(commits: List[RMinerCommit]) -> Dict[str, Any]:
    """
    Build dependency graph from refactoring location data.

    This is a placeholder for Phase 2. Full implementation will:
    1. Parse leftSideLocations/rightSideLocations from all commits
    2. Build graph where nodes are code elements (methods, classes)
    3. Edges represent refactoring dependencies
    4. Calculate graph metrics (connected components, centrality, etc.)

    Args:
        commits: Commits with location data populated

    Returns:
        Dictionary with graph structure and metrics

    Raises:
        ValueError: If commits lack location data
        NotImplementedError: This feature is not yet implemented
    """
    for commit in commits:
        for ref in commit.refactorings:
            if not ref.left_side_locations and not ref.right_side_locations:
                raise ValueError(
                    f"Dependency graph requires location data. "
                    f"Commit {commit.id} lacks location information. "
                    "Use fetch_full_refactoring_json() to populate locations first."
                )

    raise NotImplementedError(
        "Phase 2 feature: Dependency graph construction not yet implemented. "
        "This will use networkx to build and analyze refactoring dependency graphs."
    )


def parse_refactoring_info(pair: dict) -> tuple[List[str], List[str]]:
    """Extract refactoring types and descriptions from a pair dictionary.

    Args:
        pair: Dictionary containing 'refactoring_type' and 'refactoring_description' keys

    Returns:
        Tuple of (types list, descriptions list)
    """
    ref_type = pair.get("refactoring_type", "")
    ref_desc = pair.get("refactoring_description", "")

    types = [t.strip() for t in ref_type.split("|")] if ref_type else []
    descriptions = [d.strip() for d in ref_desc.split("\n")] if ref_desc else []

    return types, descriptions


