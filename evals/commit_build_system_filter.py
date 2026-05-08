from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import git
from py2neo import Graph

from repo_utils.operations import RepositoryError, clone_repository


@dataclass(frozen=True)
class CommitBuildSystemInfo:
    repo_url: str
    commit_hash: str
    has_maven: bool
    has_gradle: bool
    has_ant: bool
    primary: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CommitWindowBuildSystemInfo:
    repo_url: str
    project: str
    start_commit_order: int
    end_commit_order: int
    commit_count: int
    all_maven: bool
    first_non_maven_order: int | None = None
    first_non_maven_hash: str | None = None
    first_non_maven_primary: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _repo_slug(repo_url: str) -> str:
    raw = repo_url.strip().removesuffix(".git")
    if raw.endswith(":"):
        raw = raw[:-1]
    if "github.com:" in raw:
        raw = raw.split("github.com:", 1)[1]
    elif "github.com/" in raw:
        raw = raw.split("github.com/", 1)[1]
    else:
        raw = Path(raw).name or raw
    raw = raw.strip("/")
    if not raw:
        raise RepositoryError(f"Could not derive repo slug from {repo_url!r}")
    return raw.lower().replace("/", "__")


def classify_commit_tree(repo_url: str, commit_hash: str, paths: list[str]) -> CommitBuildSystemInfo:
    path_set = {p.strip() for p in paths if p.strip()}
    has_maven = any(p.endswith("pom.xml") for p in path_set)
    has_gradle = any(p.endswith("build.gradle") or p.endswith("build.gradle.kts") for p in path_set)
    has_ant = any(p.endswith("build.xml") for p in path_set)

    if has_maven:
        primary = "maven"
    elif has_gradle:
        primary = "gradle"
    elif has_ant:
        primary = "ant"
    else:
        primary = "unknown"

    return CommitBuildSystemInfo(
        repo_url=repo_url,
        commit_hash=commit_hash,
        has_maven=has_maven,
        has_gradle=has_gradle,
        has_ant=has_ant,
        primary=primary,
    )


def _ensure_repo(repo_url: str, cache_root: str | Path) -> git.Repo:
    cache_dir = Path(cache_root) / _repo_slug(repo_url)
    repo = clone_repository(repo_url, cache_dir, shallow=False)
    try:
        repo.git.fetch("--all", "--tags", prune=True)
    except git.GitCommandError:
        # Best effort only; existing local history may already contain the commit.
        pass
    return repo


def classify_commit_build_system(
    repo_url: str,
    commit_hash: str,
    *,
    cache_root: str | Path = "temp/eval_repos",
) -> CommitBuildSystemInfo:
    repo = _ensure_repo(repo_url, cache_root)
    try:
        output = repo.git.ls_tree("-r", "--name-only", commit_hash)
    except git.GitCommandError as exc:
        raise RepositoryError(f"Failed to inspect commit tree for {commit_hash}: {exc}") from exc
    paths = output.splitlines()
    return classify_commit_tree(repo_url, commit_hash, paths)


def _commit_hashes_for_order_window(
    graph: Graph,
    project: str,
    start_commit_order: int,
    end_commit_order: int,
) -> list[tuple[int, str]]:
    rows = graph.run(
        """
        MATCH (c:Commit)-[:BELONGS_TO]->(:Project {name: $project})
        WHERE c.order >= $start_commit_order AND c.order <= $end_commit_order
        RETURN c.order AS commit_order, c.hash AS commit_hash
        ORDER BY commit_order ASC
        """,
        project=project,
        start_commit_order=int(start_commit_order),
        end_commit_order=int(end_commit_order),
    ).data()
    return [(int(r["commit_order"]), str(r["commit_hash"])) for r in rows]


def summarize_commit_window_build_system(
    *,
    repo_url: str,
    project: str,
    start_commit_order: int,
    end_commit_order: int,
    commits: list[tuple[int, CommitBuildSystemInfo]],
) -> CommitWindowBuildSystemInfo:
    first_non_maven: tuple[int, CommitBuildSystemInfo] | None = next(
        ((order, info) for order, info in commits if info.primary != "maven"),
        None,
    )
    return CommitWindowBuildSystemInfo(
        repo_url=repo_url,
        project=project,
        start_commit_order=int(start_commit_order),
        end_commit_order=int(end_commit_order),
        commit_count=len(commits),
        all_maven=first_non_maven is None,
        first_non_maven_order=None if first_non_maven is None else int(first_non_maven[0]),
        first_non_maven_hash=None if first_non_maven is None else first_non_maven[1].commit_hash,
        first_non_maven_primary=None if first_non_maven is None else first_non_maven[1].primary,
    )


def classify_commit_window_build_system(
    graph: Graph,
    project: str,
    repo_url: str,
    start_commit_order: int,
    end_commit_order: int,
    *,
    cache_root: str | Path = "temp/eval_repos",
) -> CommitWindowBuildSystemInfo:
    commit_rows = _commit_hashes_for_order_window(graph, project, start_commit_order, end_commit_order)
    if not commit_rows:
        raise RepositoryError(
            f"No commit rows found for project={project!r} order window {start_commit_order}..{end_commit_order}"
        )
    commits = [
        (order, classify_commit_build_system(repo_url, commit_hash, cache_root=cache_root))
        for order, commit_hash in commit_rows
    ]
    return summarize_commit_window_build_system(
        repo_url=repo_url,
        project=project,
        start_commit_order=start_commit_order,
        end_commit_order=end_commit_order,
        commits=commits,
    )


def supports_maven_eval(
    repo_url: str,
    commit_hash: str,
    *,
    cache_root: str | Path = "temp/eval_repos",
) -> CommitBuildSystemInfo:
    return classify_commit_build_system(repo_url, commit_hash, cache_root=cache_root)
