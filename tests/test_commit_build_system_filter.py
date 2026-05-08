from __future__ import annotations

import subprocess
from pathlib import Path

from evals.commit_build_system_filter import (
    CommitBuildSystemInfo,
    classify_commit_build_system,
    classify_commit_tree,
    summarize_commit_window_build_system,
)


def _git(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(["git", *cmd], cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def _make_history(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init"], repo)
    _git(["config", "user.email", "test@example.com"], repo)
    _git(["config", "user.name", "Test User"], repo)

    (repo / "build.xml").write_text("<project name='demo' default='build'></project>\n", encoding="utf-8")
    _git(["add", "build.xml"], repo)
    _git(["commit", "-m", "ant era"], repo)
    ant_commit = _git(["rev-parse", "HEAD"], repo)

    (repo / "pom.xml").write_text("<project></project>\n", encoding="utf-8")
    _git(["add", "pom.xml"], repo)
    _git(["commit", "-m", "maven era"], repo)
    maven_commit = _git(["rev-parse", "HEAD"], repo)

    return repo, ant_commit, maven_commit


def test_classify_commit_tree_prefers_maven_over_ant():
    info = classify_commit_tree(
        "https://github.com/junit-team/junit4.git",
        "deadbeef",
        ["build.xml", "pom.xml"],
    )

    assert info.has_ant is True
    assert info.has_maven is True
    assert info.primary == "maven"


def test_classify_commit_build_system_detects_ant_and_maven_eras(tmp_path):
    repo, ant_commit, maven_commit = _make_history(tmp_path)

    ant_info = classify_commit_build_system(str(repo), ant_commit, cache_root=tmp_path / "cache")
    maven_info = classify_commit_build_system(str(repo), maven_commit, cache_root=tmp_path / "cache")

    assert ant_info.primary == "ant"
    assert ant_info.has_ant is True
    assert ant_info.has_maven is False

    assert maven_info.primary == "maven"
    assert maven_info.has_ant is True
    assert maven_info.has_maven is True


def test_summarize_commit_window_build_system_requires_full_maven_window():
    commits = [
        (10, CommitBuildSystemInfo("repo", "a", True, False, True, "maven")),
        (11, CommitBuildSystemInfo("repo", "b", False, False, True, "ant")),
        (12, CommitBuildSystemInfo("repo", "c", True, False, True, "maven")),
    ]

    info = summarize_commit_window_build_system(
        repo_url="repo",
        project="JUnit4",
        start_commit_order=10,
        end_commit_order=12,
        commits=commits,
    )

    assert info.all_maven is False
    assert info.commit_count == 3
    assert info.first_non_maven_order == 11
    assert info.first_non_maven_hash == "b"
    assert info.first_non_maven_primary == "ant"
