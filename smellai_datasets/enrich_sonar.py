"""Optional SonarQube enrichment for SWE-Refactor EvalSamples.

This module is deliberately NOT called from the base loaders — SonarQube
scanning is a heavy I/O operation that requires a running server and a git
clone. Call enrich_swe_with_sonar() explicitly after loading base samples.

Mirrors the playbook in notebooks/swe_to_evalsample.py.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from .schema import EvalSample

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static lookup tables (conf.tex §III-A Table I)
# ---------------------------------------------------------------------------

PROJECT_TO_REPO_URL: dict[str, str] = {
    "checkstyle": "https://github.com/checkstyle/checkstyle.git",
    "guava": "https://github.com/google/guava.git",
    "junit5": "https://github.com/junit-team/junit5.git",
    "hibernate-orm": "https://github.com/hibernate/hibernate-orm.git",
    "mockito": "https://github.com/mockito/mockito.git",
    "spring-framework": "https://github.com/spring-projects/spring-framework.git",
    "RxJava": "https://github.com/ReactiveX/RxJava.git",
    "commons-lang": "https://github.com/apache/commons-lang.git",
    "elasticsearch": "https://github.com/elastic/elasticsearch.git",
    "flink": "https://github.com/apache/flink.git",
}

# Refactoring type → SonarQube rules expected to be violated (pre-refactoring)
JAVA_RULE_CLASS_DEPENDENCY = "java:S1200"

REFACTORING_TO_RULES: dict[str, set[str]] = {
    "Extract Method": {"java:S138", "java:S1541", "java:S1067"},
    "Extract Class": {JAVA_RULE_CLASS_DEPENDENCY, "java:S110"},
    "Move Method": {JAVA_RULE_CLASS_DEPENDENCY},
    "Extract And Move Method": {JAVA_RULE_CLASS_DEPENDENCY, "java:S138", "java:S1541"},
    "Move Attribute": {JAVA_RULE_CLASS_DEPENDENCY},
    "Introduce Parameter Object": {"java:S107"},
    "Consolidate Conditional Expression": {"java:S1871"},
}


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def enrich_swe_with_sonar(
    samples: list[EvalSample],
    *,
    sonar_url: str,
    sonar_token: str,
    cache_dir: Path | str,
    skip_compile: bool = True,
    work_dir: Path | str | None = None,
) -> list[EvalSample]:
    """Clone repos, run SonarQube scans, return enriched EvalSamples.

    Adds to each sample:
      inputs:  ``repo_url``
      tags:    ``sonar_smells_count``, ``dataset_rules_covered``,
               ``expected_rules``, ``found_rules``

    Samples whose ``project_name`` is not in PROJECT_TO_REPO_URL are returned
    unchanged with a warning.

    EvalSample is frozen — enriched copies are new instances.

    Args:
        samples: Base SWE EvalSamples (source must be "swe").
        sonar_url: SonarQube server URL (e.g. "http://localhost:9000").
        sonar_token: SonarQube auth token.
        cache_dir: Directory to cache scan results (re-runs are instant).
        skip_compile: Pass skip_compile=True to sonar scanner for speed.
        work_dir: Directory for temporary git clones. Uses tempfile if None.

    Returns:
        New list of EvalSamples with sonar tags/inputs populated.
    """
    from sonarqube.commit_scan import scan_commit  # type: ignore[import]

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _tmp_ctx = None
    if work_dir is None:
        _tmp_ctx = tempfile.TemporaryDirectory(prefix="smellai_enrich_")
        work_dir = Path(_tmp_ctx.name)
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    enriched: list[EvalSample] = []

    try:
        for sample in samples:
            assert sample.source == "swe", f"enrich_swe_with_sonar expects source='swe', got {sample.source!r}"

            project_name = sample.inputs.get("project_name", "")
            repo_url = PROJECT_TO_REPO_URL.get(project_name)
            if repo_url is None:
                LOGGER.warning(
                    "project_name %r not in PROJECT_TO_REPO_URL — skipping sonar scan",
                    project_name,
                )
                enriched.append(sample)
                continue

            commit_id = sample.inputs.get("commit_id", "")
            refactoring_type = sample.inputs.get("refactoring_type", "")

            try:
                smells_by_file = scan_commit(
                    repo_url=repo_url,
                    commit_sha=commit_id,
                    sonar_url=sonar_url,
                    sonar_token=sonar_token,
                    cache_dir=cache_dir,
                    skip_compile=skip_compile,
                )
            except Exception as exc:
                LOGGER.warning(
                    "SonarQube scan failed for %s@%s: %s — skipping enrichment",
                    project_name,
                    commit_id[:8],
                    exc,
                )
                enriched.append(sample)
                continue

            total_smells = sum(len(v) for v in smells_by_file.values())
            found_rules: set[str] = {
                issue["rule"]
                for issues in smells_by_file.values()
                for issue in issues
                if "rule" in issue
            }
            expected_rules: set[str] = REFACTORING_TO_RULES.get(refactoring_type, set())
            covered = bool(expected_rules & found_rules)

            # EvalSample is frozen — build a new one with enriched fields
            new_inputs = dict(sample.inputs) | {"repo_url": repo_url}
            new_tags = dict(sample.tags) | {
                "sonar_smells_count": total_smells,
                "dataset_rules_covered": covered,
                "expected_rules": sorted(expected_rules),
                "found_rules": sorted(found_rules),
            }

            enriched.append(
                EvalSample(
                    source=sample.source,
                    sample_id=sample.sample_id,
                    inputs=new_inputs,
                    expectations=sample.expectations,
                    tags=new_tags,
                )
            )

    finally:
        if _tmp_ctx is not None:
            _tmp_ctx.cleanup()

    return enriched


__all__ = [
    "PROJECT_TO_REPO_URL",
    "REFACTORING_TO_RULES",
    "enrich_swe_with_sonar",
]
