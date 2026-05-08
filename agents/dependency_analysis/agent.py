"""Dependency analysis agent for refactoring.

This module exposes smell-dependency analysis over SonarQube issues while using
``DependencyGraph`` as the canonical in-memory representation and
``RefactoringTree`` for planning. When a LangGraph ``BaseStore`` is provided,
the built graph and computed priorities can be persisted through ``SmellStore``
for later workflow steps.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from domain.dependency_graph import DependencyGraph
from domain.refactoring_tree import RefactoringTree, State
from domain.rules import DEPENDENCY_RULES
from store.smell_store import SmellStore
from domain.models import SmellEvent


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
        action="detected",
    )


def issues_to_smell_events(sonar_issues: List[Dict[str, Any]]) -> List[SmellEvent]:
    """Convert SonarQube issues into canonical ``SmellEvent`` objects."""
    return [
        event
        for issue in sonar_issues
        if (event := issue_to_smell_event(issue)) is not None
    ]


def build_dependency_graph(
    smell_events: List[SmellEvent],
    *,
    locality: str = "none",
    store: BaseStore | None = None,
    session_id: str | None = None,
    iteration: int = 0,
) -> DependencyGraph:
    """Build a ``DependencyGraph`` from already-normalized ``SmellEvent`` objects."""
    graph = DependencyGraph.from_events(smell_events, locality=locality)

    if store is not None and session_id:
        SmellStore(store).save_graph(session_id, graph, iteration=iteration)

    return graph


def prioritize_smells(
    sonar_issues: List[Dict[str, Any]],
    *,
    locality: str = "none",
    store: BaseStore | None = None,
    session_id: str | None = None,
    iteration: int = 0,
) -> List[dict[str, Any]]:
    """Prioritize smells using greedy planner on DependencyGraph.

    When ``store`` and ``session_id`` are provided, both the graph snapshot and
    the computed priority queue are persisted via ``SmellStore``.
    """
    events = issues_to_smell_events(sonar_issues)
    graph = build_dependency_graph(
        events,
        locality=locality,
        store=store,
        session_id=session_id,
        iteration=iteration,
    )
    initial = State(frozenset(e.smell_id for e in events))
    tree = RefactoringTree(initial, graph)
    plan = tree.greedy()

    # Convert Plan to legacy priority list format
    priorities = []
    for i, action in enumerate(plan.actions):
        smell_type = graph.smell_type_of(action.smell_id)
        priorities.append({
            "order": i + 1,
            "smell_id": action.smell_id,
            "smell_type": smell_type,
            "ref_type": action.ref_type,
            "h_before": plan.h_trace[i],
            "h_after": plan.h_trace[i + 1],
        })

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

    Builds the canonical ``DependencyGraph`` first. If a store/session is
    provided, the graph is persisted for downstream LangGraph steps.
    """
    events = issues_to_smell_events(sonar_issues)
    graph = build_dependency_graph(
        events,
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
        smell_type = graph.smell_type_of(smell_id)
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
