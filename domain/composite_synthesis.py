"""Composite synthesis strategies aligned with Sousa et al. (MSR 2020).

References in comments use the paper's findings:
- Finding 1 / 6: element-only scope is partial for smell-effect analysis.
- Finding 4: commit-level grouping can capture semantic (task-level) relations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


@dataclass(frozen=True)
class RefactoringOccurrence:
    """Refactoring event projected to synthesis-friendly fields."""

    ref_id: str
    ref_type: str
    commit_hash: str
    commit_order: int
    scope: frozenset[str]


@dataclass(frozen=True)
class CompositeRefactoring:
    """Synthesized composite with strategy metadata."""

    heuristic: str  # element-based | commit-based | range-based
    ref_ids: tuple[str, ...]


class CompositeSynthesizer(Protocol):
    def synthesize(self, occurrences: Iterable[RefactoringOccurrence]) -> list[CompositeRefactoring]: ...


class ElementBasedSynthesizer:
    """Group refactorings by a single shared element.

    Mirrors the element-based heuristic from the paper. Kept as a baseline,
    not as the default, because Findings 1/6 show it can miss cross-element
    effects.
    """

    def synthesize(self, occurrences: Iterable[RefactoringOccurrence]) -> list[CompositeRefactoring]:
        by_element: dict[str, list[str]] = {}
        for occ in occurrences:
            for e in occ.scope:
                by_element.setdefault(e, []).append(occ.ref_id)

        result: list[CompositeRefactoring] = []
        for _, ref_ids in sorted(by_element.items()):
            uniq = tuple(sorted(set(ref_ids)))
            if len(uniq) >= 2:
                result.append(CompositeRefactoring("element-based", uniq))
        return result


class CommitBasedSynthesizer:
    """Group all refactorings performed in the same commit.

    Reflects commit-based heuristic; this supports analysis of semantic
    task-level relations (Finding 4).
    """

    def synthesize(self, occurrences: Iterable[RefactoringOccurrence]) -> list[CompositeRefactoring]:
        by_commit: dict[str, list[str]] = {}
        for occ in occurrences:
            by_commit.setdefault(occ.commit_hash, []).append(occ.ref_id)

        result: list[CompositeRefactoring] = []
        for _, ref_ids in sorted(by_commit.items()):
            uniq = tuple(sorted(set(ref_ids)))
            if len(uniq) >= 2:
                result.append(CompositeRefactoring("commit-based", uniq))
        return result


class RangeBasedSynthesizer:
    """Connected-components synthesis over scope intersections.

    Matches the range-based idea: if scopes intersect transitively, refactorings
    belong to one composite.
    """

    def synthesize(self, occurrences: Iterable[RefactoringOccurrence]) -> list[CompositeRefactoring]:
        items = sorted(list(occurrences), key=lambda o: (o.commit_order, o.ref_id))
        if not items:
            return []

        parent = {o.ref_id: o.ref_id for o in items}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                if a.scope & b.scope:
                    union(a.ref_id, b.ref_id)

        groups: dict[str, list[str]] = {}
        for o in items:
            groups.setdefault(find(o.ref_id), []).append(o.ref_id)

        result: list[CompositeRefactoring] = []
        for _, ref_ids in sorted(groups.items()):
            uniq = tuple(sorted(set(ref_ids)))
            if len(uniq) >= 2:
                result.append(CompositeRefactoring("range-based", uniq))
        return result
