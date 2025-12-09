"""Exploratory script for RefactoringMiner benchmark analysis.

This script analyzes refactoring operations from RefactoringMiner's benchmark
data.json file. It identifies repositories with multiple commits, finds
potentially consecutive commit clusters based on ID proximity and semantic
similarity, and generates statistical reports.

Typical usage (with uv):

    uv run tools/rminer_explore.py /path/to/data.json

    uv run tools/rminer_explore.py /path/to/data.json --max-commits 20

    uv run tools/rminer_explore.py /path/to/data.json \\
        --max-commits 50 \\
        --max-id-gap 5 \\
        --min-cluster-size 3 \\
        --output analysis_results.json \\
        --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from smellai.rminer.rminer_utils import (
    compute_statistics,
    find_consecutive_commits,
    group_by_repository,
    load_rminer_data,
)
from smellai.models.refactoring import CommitCluster, RefactoringStats, RMinerCommit

LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Configure logging level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _select_top_repositories(
    repo_groups: Dict[str, List[RMinerCommit]], *, max_commits: int
) -> Dict[str, List[RMinerCommit]]:
    """
    Select repositories with multiple commits, up to max_commits total.

    Strategy:
    1. Filter repositories with >= 2 commits
    2. Sort by commit count (descending)
    3. Accumulate commits until max_commits reached

    Args:
        repo_groups: Dictionary mapping repo URL to list of commits
        max_commits: Maximum total commits to process

    Returns:
        Filtered dictionary with selected repositories
    """
    multi_commit_repos = [
        (repo_url, commits_list)
        for repo_url, commits_list in repo_groups.items()
        if len(commits_list) >= 2
    ]

    multi_commit_repos.sort(key=lambda x: len(x[1]), reverse=True)

    selected_repos = {}
    total_selected = 0

    for repo_url, commits_list in multi_commit_repos:
        if total_selected >= max_commits:
            break

        remaining = max_commits - total_selected
        selected_commits = commits_list[:remaining]

        selected_repos[repo_url] = selected_commits
        total_selected += len(selected_commits)

    LOGGER.debug(
        f"Selected {len(selected_repos)} repositories with {total_selected} commits"
    )

    return selected_repos


def _analyze_clusters(
    repo_groups: Dict[str, List[RMinerCommit]],
    *,
    max_id_gap: int,
    min_cluster_size: int,
) -> List[CommitCluster]:
    """
    Find and analyze commit clusters across all repositories.

    Args:
        repo_groups: Dictionary mapping repo URL to list of commits
        max_id_gap: Maximum gap between IDs for clustering
        min_cluster_size: Minimum commits required for a cluster

    Returns:
        List of CommitCluster objects sorted by score (descending)
    """
    all_clusters = []

    for repo_url, repo_commits in repo_groups.items():
        LOGGER.debug(f"Analyzing {len(repo_commits)} commits from {repo_url}")

        clusters = find_consecutive_commits(
            repo_commits, max_id_gap=max_id_gap, min_cluster_size=min_cluster_size
        )

        all_clusters.extend(clusters)

    all_clusters.sort(key=lambda c: c.cluster_score, reverse=True)

    LOGGER.debug(f"Found {len(all_clusters)} total clusters across all repositories")

    return all_clusters


def _export_results(
    stats: RefactoringStats,
    clusters: List[CommitCluster],
    *,
    output_path: Optional[Path] = None,
    format: str = "json",
) -> None:
    """
    Export analysis results to file.

    Args:
        stats: Statistical summary
        clusters: List of commit clusters
        output_path: Output file path (if None, print to stdout)
        format: Output format ('json' or 'csv')
    """
    if format == "json":
        output_data = {
            "statistics": stats.model_dump(),
            "clusters": [c.model_dump() for c in clusters],
        }

        output_json = json.dumps(output_data, indent=2)

        if output_path:
            output_path.write_text(output_json, encoding="utf-8")
            LOGGER.info(f"Results exported to {output_path}")
        else:
            print(output_json)

    elif format == "csv":
        import csv
        from io import StringIO

        if not output_path:
            LOGGER.warning(
                "CSV format requires --output parameter. Skipping CSV export."
            )
            return

        output_buffer = StringIO()
        writer = csv.writer(output_buffer)

        writer.writerow(
            [
                "repository",
                "cluster_size",
                "cluster_score",
                "commit_ids",
                "max_id_gap",
                "avg_id_gap",
                "total_refactorings",
            ]
        )

        for cluster in clusters:
            writer.writerow(
                [
                    cluster.repository,
                    len(cluster.commits),
                    f"{cluster.cluster_score:.3f}",
                    ",".join(str(cid) for cid in cluster.commit_ids),
                    cluster.max_id_gap(),
                    f"{cluster.avg_id_gap():.2f}",
                    cluster.total_refactorings(),
                ]
            )

        output_path.write_text(output_buffer.getvalue(), encoding="utf-8")
        LOGGER.info(f"CSV results exported to {output_path}")


def _print_summary(stats: RefactoringStats, clusters: List[CommitCluster]) -> None:
    """Print human-readable summary to console."""
    LOGGER.info("=" * 60)
    LOGGER.info("RefactoringMiner Benchmark Analysis Summary")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Total commits: {stats.total_commits}")
    LOGGER.info(f"Total repositories: {stats.total_repositories}")
    LOGGER.info(f"Total refactorings: {stats.total_refactorings}")
    LOGGER.info("")

    LOGGER.info("Top 10 refactoring types:")
    sorted_types = sorted(
        stats.refactoring_type_counts.items(), key=lambda x: x[1], reverse=True
    )
    for rtype, count in sorted_types[:10]:
        LOGGER.info(f"  {count:5d} {rtype}")

    LOGGER.info("")
    LOGGER.info("Validation distribution:")
    for validation, count in sorted(stats.validation_counts.items()):
        LOGGER.info(f"  {validation}: {count}")

    LOGGER.info("")
    LOGGER.info(f"Consecutive commit clusters found: {len(clusters)}")

    if clusters:
        LOGGER.info("Top 5 clusters by score:")
        for i, cluster in enumerate(clusters[:5], 1):
            LOGGER.info(
                f"  {i}. {cluster.repository.split('/')[-1].replace('.git', '')} "
                f"({len(cluster.commit_ids)} commits, "
                f"score={cluster.cluster_score:.2f}, "
                f"max_gap={cluster.max_id_gap()}, "
                f"refactorings={cluster.total_refactorings()})"
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "data_json", type=Path, help="Path to RefactoringMiner data.json file"
    )

    parser.add_argument(
        "--max-commits",
        type=int,
        default=100,
        help="Maximum number of commits to process (default: 100)",
    )

    parser.add_argument(
        "--max-id-gap",
        type=int,
        default=10,
        help="Maximum ID gap for consecutive commits (default: 10)",
    )

    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum commits in a cluster (default: 2)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: print to stdout)",
    )

    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main entry point."""
    args = parse_args(argv)
    _configure_logging(args.verbose)

    if not args.data_json.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_json}")

    LOGGER.info(f"Loading RefactoringMiner data from {args.data_json}")
    commits = load_rminer_data(str(args.data_json.resolve()))

    unique_repos = {c.repository for c in commits}
    LOGGER.info(f"Loaded {len(commits)} commits from {len(unique_repos)} repositories")

    LOGGER.info("Grouping commits by repository...")
    repo_groups = group_by_repository(commits)

    multi_commit_repos = {
        repo: commits_list
        for repo, commits_list in repo_groups.items()
        if len(commits_list) >= 2
    }

    total_multi_commit = sum(len(c) for c in multi_commit_repos.values())
    LOGGER.info(
        f"Found {len(multi_commit_repos)} repositories with multiple commits "
        f"(total: {total_multi_commit} commits)"
    )

    selected_repos = _select_top_repositories(
        multi_commit_repos, max_commits=args.max_commits
    )

    total_selected = sum(len(c) for c in selected_repos.values())
    LOGGER.info(
        f"Selected {len(selected_repos)} repositories "
        f"with {total_selected} commits (limit: {args.max_commits})"
    )

    LOGGER.info("Analyzing commit clusters...")
    all_clusters = _analyze_clusters(
        selected_repos,
        max_id_gap=args.max_id_gap,
        min_cluster_size=args.min_cluster_size,
    )
    LOGGER.info(f"Found {len(all_clusters)} commit clusters")

    LOGGER.info("Computing statistics...")
    selected_commits = [
        commit for commits_list in selected_repos.values() for commit in commits_list
    ]
    stats = compute_statistics(selected_commits)

    stats.clusters_found = len(all_clusters)
    stats.clusters_detail = [
        {
            "repository": cluster.repository,
            "size": len(cluster.commits),
            "score": round(cluster.cluster_score, 3),
            "commit_ids": cluster.commit_ids,
            "max_gap": cluster.max_id_gap(),
            "avg_gap": round(cluster.avg_id_gap(), 2),
        }
        for cluster in all_clusters
    ]

    _print_summary(stats, all_clusters)

    if args.output:
        _export_results(
            stats, all_clusters, output_path=args.output, format=args.format
        )


if __name__ == "__main__":
    main()
