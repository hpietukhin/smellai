"""Dependency analysis agent for refactoring.

This module analyzes positive and negative dependencies for code smells
to determine the optimal sequence for applying refactoring rules.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field

from sonarqube.constants import RULE_NAME_MAP  # noqa: F401 (re-exported for callers)


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
DEPENDENCY_RULES = {
    "Long Method": {
        "positive": [
            "Switch Statement",
            "Feature Envy",
            "Duplicated Code",
            "Divergent Change",
            "Comments",
            "Long Parameter List",
        ],
        "negative": ["Long Method", "Long Parameter List"],
    },
    "Complex Method": {
        "positive": [
            "Switch Statement",
            "Feature Envy",
            "Duplicated Code",
            "Divergent Change",
            "Comments",
            "Long Parameter List",
        ],
        "negative": ["Long Method", "Long Parameter List"],
    },
    "Conditional Complexity": {
        "positive": [
            "Switch Statement",
            "Feature Envy",
            "Duplicated Code",
            "Divergent Change",
            "Comments",
            "Long Parameter List",
        ],
        "negative": ["Long Method", "Long Parameter List"],
    },
    "Long Parameter List": {
        "positive": ["Long Parameter List", "Data Clumps"],
        "negative": ["Data Class"],
    },
    "Large Class": {
        "positive": ["Data Clumps", "Feature Envy", "Bad Class Content"],
        "negative": [
            "Long Method",
            "Data Class",
            "Inappropriate Intimacy",
            "Message Chains",
        ],
    },
    "God Class": {
        "positive": ["Data Clumps", "Feature Envy", "Bad Class Content"],
        "negative": [
            "Long Method",
            "Data Class",
            "Inappropriate Intimacy",
            "Message Chains",
        ],
    },
    "Duplicated Conditions": {
        "positive": ["Divergent Change", "Shotgun Surgery"],
        "negative": ["Large Class", "Bad Inheritance"],
    },
    "Print Statements": {
        "positive": ["Needless Part"],
        "negative": ["Data Class", "Lazy Class"],
    },
}


def analyze_dependencies(
    sonar_issues: List[Dict[str, Any]],
) -> List[DependencyAnalysis]:
    """Analyze dependencies for a list of SonarQube issues.

    Args:
        sonar_issues: List of issues from SonarQube

    Returns:
        List of DependencyAnalysis objects
    """
    results = []
    seen_rules = set()

    for issue in sonar_issues:
        rule = issue.get("rule")
        if not rule or rule in seen_rules:
            continue

        smell_type = RULE_NAME_MAP.get(rule)
        if not smell_type:
            continue

        seen_rules.add(rule)

        deps = DEPENDENCY_RULES.get(smell_type)
        if deps:
            results.append(
                DependencyAnalysis(
                    smell_type=smell_type,
                    rule_id=rule,
                    positive_dependencies=deps["positive"],
                    negative_dependencies=deps["negative"],
                )
            )

    return results
