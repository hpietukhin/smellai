"""Heuristics for incomplete-composite risk checks.

Paper alignment:
- Section on "Incomplete composites" (Feature Envy discussion): partial
  transformations can introduce smells when follow-up steps are missing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompletionRule:
    trigger_ref_type: str
    required_follow_up: str


# Minimal seed rules grounded in paper examples; extend gradually.
DEFAULT_COMPLETION_RULES: tuple[CompletionRule, ...] = (
    CompletionRule("Extract Method", "Move Method"),
    CompletionRule("Move Attribute", "Move Method"),
)


def detect_missing_follow_ups(
    refactoring_types: list[str],
    rules: tuple[CompletionRule, ...] = DEFAULT_COMPLETION_RULES,
) -> list[CompletionRule]:
    present = set(refactoring_types)
    missing: list[CompletionRule] = []
    for rule in rules:
        if rule.trigger_ref_type in present and rule.required_follow_up not in present:
            missing.append(rule)
    return missing
