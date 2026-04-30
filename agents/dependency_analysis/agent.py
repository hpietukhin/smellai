"""Dependency analysis agent for refactoring.

This module exposes smell-dependency analysis over SonarQube issues while using
``SmellGraph`` as the canonical in-memory representation. When a LangGraph
``BaseStore`` is provided, the built graph and computed priorities can be
persisted through ``SmellStore`` for later workflow steps.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from domain.graph import SmellGraph
from domain.rules import DEPENDENCY_RULES
from store.smell_store import SmellStore
from domain.models import SmellAction, SmellEvent


class DependencyAnalysis(BaseModel):
    """Analysis of dependencies for a specific code smell."""

    smell_type: str
    rule_id: str
    positive_dependencies: List[str] = Field(
        description="Smells that might be solved/removed"
    )
    negative_dependencies: List[str] = Field(
        description="Smells that might be caused/created"
    )


# TODO SPEC-009: Create comprehensive map of dependency rules with detailed citations.
# Rules are based on Markovič & Polášek research.
# Need comprehensive mapping with paper references and detailed citations.
# MEDIUM priority.
# (See TECHNICAL_SPECIFICATION.md §4.4)


def issue_to_smell_event(issue: Dict[str, Any]) -> SmellEvent | None:
    """Convert one SonarQube issue dict into a ``SmellEvent``.

    Returns ``None`` when the issue does not contain enough data to locate the
    smell in a source file.
    """
    component = issue.get("component", "")
    if ":" not in component:
        return None

    from sonarqube.commit_scan import normalize_issue

    file_path = component.split(":", 1)[1]
    n = normalize_issue(issue)
    smell_type = n["smell_type"]
    severity = n["severity"]
    line = n.get("line") or 0

    return SmellEvent(
        smell_id=f"{smell_type}:{file_path}:{line}",
        smell_type=smell_type,
        severity=severity,
        file_path=file_path,
        line_number=line,
        action=SmellAction.DETECTED,
    )


def issues_to_smell_events(sonar_issues: List[Dict[str, Any]]) -> List[SmellEvent]:
    """Convert SonarQube issues into canonical ``SmellEvent`` objects."""
    return [
        event
        for issue in sonar_issues
        if (event := issue_to_smell_event(issue)) is not None
    ]


def build_smell_graph(
    sonar_issues: List[Dict[str, Any]],
    *,
    store: BaseStore | None = None,
    session_id: str | None = None,
    iteration: int = 0,
) -> SmellGraph:
    """Build a ``SmellGraph`` from SonarQube issues and optionally persist it."""
    graph = SmellGraph.from_smells(issues_to_smell_events(sonar_issues))

    if store is not None and session_id:
        SmellStore(store).save_graph(session_id, graph, iteration=iteration)

    return graph


def prioritize_smells(
    sonar_issues: List[Dict[str, Any]],
    *,
    store: BaseStore | None = None,
    session_id: str | None = None,
    iteration: int = 0,
) -> List[dict[str, Any]]:
    """Prioritize smells by building a graph and applying greedy scoring.

    When ``store`` and ``session_id`` are provided, both the graph snapshot and
    the computed priority queue are persisted via ``SmellStore``.
    """
    graph = build_smell_graph(
        sonar_issues,
        store=store,
        session_id=session_id,
        iteration=iteration,
    )
    priorities = graph.calculate_priorities()

    if store is not None and session_id:
        SmellStore(store).save_priorities(session_id, priorities)

    return priorities


def analyze_dependencies(
    sonar_issues: List[Dict[str, Any]],
    *,
    store: BaseStore | None = None,
    session_id: str | None = None,
    iteration: int = 0,
) -> List[DependencyAnalysis]:
    """Analyze dependencies for a list of SonarQube issues.

    This keeps the original smell-type-level response shape for callers, but now
    builds the canonical ``SmellGraph`` first. If a store/session is provided,
    the graph is persisted for downstream LangGraph steps.

    Args:
        sonar_issues: List of issues from SonarQube.
        store: Optional LangGraph store for persistence.
        session_id: Session key used when persisting to ``SmellStore``.
        iteration: Iteration metadata used for persisted snapshots.

    Returns:
        List of ``DependencyAnalysis`` objects.
    """
    graph = build_smell_graph(
        sonar_issues,
        store=store,
        session_id=session_id,
        iteration=iteration,
    )

    from sonarqube.commit_scan import normalize_issue

    first_rule_by_type: dict[str, str] = {}
    for issue in sonar_issues:
        rule = issue.get("rule")
        if rule:
            smell_type = normalize_issue(issue)["smell_type"]
            first_rule_by_type.setdefault(smell_type, rule)

    results: List[DependencyAnalysis] = []
    seen_smell_types: set[str] = set()
    for smell_id in graph.all_smell_ids():
        smell_type = graph.node_data(smell_id).get("smell_type")
        if not smell_type or smell_type in seen_smell_types:
            continue
        seen_smell_types.add(smell_type)

        deps = DEPENDENCY_RULES.get(smell_type)
        rule_id = first_rule_by_type.get(smell_type)
        if deps and rule_id:
            results.append(
                DependencyAnalysis(
                    smell_type=smell_type,
                    rule_id=rule_id,
                    positive_dependencies=deps["positive"],
                    negative_dependencies=deps["negative"],
                )
            )

    return results
