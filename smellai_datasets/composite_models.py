"""Domain models for Composite Refactorings 2020 dataset episodes.

These are pure data structures for representing extracted composite
refactoring episodes with before/after smell states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from domain.models import SmellEvent
from domain.dependency_graph import DependencyGraph
from domain.rules import (
    get_default_severity,
    normalize_dataset_smell_type,
)


@dataclass
class CodeElement:
    """A code element (class or method) from the dataset."""

    name: str              # FQN, e.g. "org.apache.Foo.bar(int)"
    element_type: str      # "Class", "Public Method", "Method", etc.
    file_path: str         # e.g. "org/apache/Foo.java"
    hash_id: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def class_name(self) -> str:
        """Extract class FQN from element name.

        For methods: "org.Foo.bar(int)" -> "org.Foo"
        For classes: "org.Foo" -> "org.Foo"
        """
        if "Method" in self.element_type or "Constructor" in self.element_type:
            paren = self.name.find("(")
            base = self.name[:paren] if paren != -1 else self.name
            dot = base.rfind(".")
            return base[:dot] if dot != -1 else base
        return self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "element_type": self.element_type,
            "file_path": self.file_path,
            "hash_id": self.hash_id,
            "metrics": self.metrics,
            "class_name": self.class_name,
        }


def class_name_from_element_name(element_name: str) -> str:
    """Best-effort class FQN extraction from dataset element FQN."""
    if not element_name:
        return ""
    paren = element_name.find("(")
    base = element_name[:paren] if paren != -1 else element_name
    dot = base.rfind(".")
    if dot == -1:
        return base
    # If there is no method signature, this may already be a class name.
    # Dataset class names usually end with UpperCamelCase; methods are lower.
    tail = base[dot + 1:]
    if tail and tail[0].islower():
        return base[:dot]
    return base


def method_signature_from_element_name(element_name: str) -> str | None:
    """Extract method signature from dataset element FQN when present."""
    paren = element_name.find("(")
    if paren == -1:
        return None
    base = element_name[:paren]
    dot = base.rfind(".")
    if dot == -1:
        return element_name
    return element_name[dot + 1:]


@dataclass
class SmellInstance:
    """A smell instance from the dataset graph (pre-normalisation)."""

    smell_type: str         # Raw type from Neo4j, e.g. "FeatureEnvy"
    hash_id: str
    reason: str = ""        # Detection rule, e.g. "MLOC > 6.87"
    starting_line: int = 0
    ending_line: int = 0
    element_name: str = ""  # FQN of the affected element
    element_path: str = ""  # File path of the affected element

    def to_dict(self) -> dict[str, Any]:
        return {
            "smell_type": self.smell_type,
            "hash_id": self.hash_id,
            "reason": self.reason,
            "element_name": self.element_name,
            "element_path": self.element_path,
            "starting_line": self.starting_line,
            "ending_line": self.ending_line,
        }

    def to_domain_event(
        self,
        *,
        project: str,
        commit_hash: str,
        normalize_type: bool = True,
    ) -> SmellEvent:
        """Convert to domain SmellEvent for SmellGraph/planner use."""
        smell_type = (
            normalize_dataset_smell_type(self.smell_type)
            if normalize_type else self.smell_type
        )
        class_name = class_name_from_element_name(self.element_name)
        return SmellEvent(
            smell_id=self.hash_id or f"{smell_type}:{self.element_path}:{self.starting_line}",
            smell_type=smell_type,
            severity=get_default_severity(smell_type),
            file_path=self.element_path,
            line_number=self.starting_line,
            class_name=class_name or None,
            method_signature=method_signature_from_element_name(self.element_name),
            project=project,
            commit_hash=commit_hash,
            end_line=self.ending_line,
            detection_reason=self.reason,
        )


@dataclass
class RefactoringStep:
    """A single refactoring operation within a composite."""

    ref_type: str               # "Extract Method", "Move Method", etc.
    hash_id: str
    classification: str         # "positive", "negative", "neutral"
    degradation_level: str      # "agglomeration", "smell", "no smells"
    smelly: bool = False
    commit_hash: str = ""
    commit_order: int = 0
    produced_elements: list[str] = field(default_factory=list)  # Element FQNs
    changed_elements: list[str] = field(default_factory=list)   # Element FQNs
    parameters: str = ""        # Raw parameter string from Neo4j

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref_type": self.ref_type,
            "hash_id": self.hash_id,
            "classification": self.classification,
            "degradation_level": self.degradation_level,
            "smelly": self.smelly,
            "commit_hash": self.commit_hash,
            "commit_order": self.commit_order,
            "produced_elements": self.produced_elements,
            "changed_elements": self.changed_elements,
            "parameters": self.parameters,
        }


@dataclass
class CompositeEpisode:
    """A composite refactoring episode extracted from the dataset.

    Represents a group of interrelated refactorings (commit-based heuristic)
    with before/after smell state on affected elements.
    """

    episode_id: str
    project: str
    heuristic: Literal["commit", "range", "element"] = "commit"

    # Commit context
    commit_hash: str = ""
    commit_order: int = 0
    commit_message: str = ""

    # Refactoring steps
    refactorings: list[RefactoringStep] = field(default_factory=list)

    # Affected elements
    scope_elements: list[CodeElement] = field(default_factory=list)

    # Smell state before and after
    smells_before: list[SmellInstance] = field(default_factory=list)
    smells_after: list[SmellInstance] = field(default_factory=list)

    # Pre-computed labels
    classification: str = "neutral"  # majority vote of refactoring classifications
    n_positive: int = 0
    n_negative: int = 0
    n_neutral: int = 0

    @property
    def size(self) -> int:
        """Number of refactoring steps."""
        return len(self.refactorings)

    @property
    def is_positive(self) -> bool:
        return self.classification == "positive"

    @property
    def is_agglomeration(self) -> bool:
        """True if any refactoring targets an agglomeration context."""
        return any(r.degradation_level == "agglomeration" for r in self.refactorings)

    @property
    def ref_types(self) -> list[str]:
        """Ordered list of refactoring types in this episode."""
        return [r.ref_type for r in self.refactorings]

    @property
    def smell_delta(self) -> int:
        """Change in smell count: negative = improvement."""
        return len(self.smells_after) - len(self.smells_before)

    def to_smell_events(
        self,
        which: Literal["before", "after"] = "before",
        *,
        normalize_type: bool = True,
    ) -> list[SmellEvent]:
        """Convert before/after dataset smells into domain SmellEvent objects."""
        smells = self.smells_before if which == "before" else self.smells_after
        return [
            smell.to_domain_event(
                project=self.project,
                commit_hash=self.commit_hash,
                normalize_type=normalize_type,
            )
            for smell in smells
        ]

    def to_dependency_graph(
        self,
        *,
        which: Literal["before", "after"] = "before",
        locality: str = "none",
        normalize_type: bool = True,
    ) -> DependencyGraph:
        """Project this episode's smell state into a DependencyGraph."""
        return DependencyGraph.from_events(
            self.to_smell_events(which, normalize_type=normalize_type),
            locality=locality,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-friendly dict."""
        return {
            "episode_id": self.episode_id,
            "project": self.project,
            "heuristic": self.heuristic,
            "commit_hash": self.commit_hash,
            "commit_order": self.commit_order,
            "commit_message": self.commit_message,
            "classification": self.classification,
            "size": self.size,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_neutral": self.n_neutral,
            "smell_delta": self.smell_delta,
            "refactorings": [r.to_dict() for r in self.refactorings],
            "scope_elements": [e.to_dict() for e in self.scope_elements],
            "smells_before": [s.to_dict() for s in self.smells_before],
            "smells_after": [s.to_dict() for s in self.smells_after],
        }


def _refactoring_from_dict(data: dict[str, Any], episode: dict[str, Any]) -> RefactoringStep:
    return RefactoringStep(
        ref_type=data.get("ref_type", ""),
        hash_id=data.get("hash_id", ""),
        classification=data.get("classification", "neutral"),
        degradation_level=data.get("degradation_level", ""),
        smelly=bool(data.get("smelly", False)),
        commit_hash=data.get("commit_hash", episode.get("commit_hash", "")),
        commit_order=data.get("commit_order", episode.get("commit_order", 0)) or 0,
        produced_elements=list(data.get("produced_elements", [])),
        changed_elements=list(data.get("changed_elements", [])),
        parameters=data.get("parameters", ""),
    )


def _code_element_from_dict(data: dict[str, Any]) -> CodeElement:
    return CodeElement(
        name=data.get("name", ""),
        element_type=data.get("element_type", ""),
        file_path=data.get("file_path", ""),
        hash_id=data.get("hash_id", ""),
        metrics=dict(data.get("metrics", {})),
    )


def _smell_from_dict(data: dict[str, Any]) -> SmellInstance:
    return SmellInstance(
        smell_type=data.get("smell_type", ""),
        hash_id=data.get("hash_id", ""),
        reason=data.get("reason", ""),
        starting_line=data.get("starting_line", 0) or 0,
        ending_line=data.get("ending_line", 0) or 0,
        element_name=data.get("element_name", ""),
        element_path=data.get("element_path", ""),
    )


def episode_from_dict(data: dict[str, Any]) -> CompositeEpisode:
    """Rehydrate CompositeEpisode from JSONL/dict output of to_dict()."""
    return CompositeEpisode(
        episode_id=data["episode_id"],
        project=data["project"],
        heuristic=data.get("heuristic", "commit"),
        commit_hash=data.get("commit_hash", ""),
        commit_order=data.get("commit_order", 0) or 0,
        commit_message=data.get("commit_message", ""),
        refactorings=[
            _refactoring_from_dict(r, data)
            for r in data.get("refactorings", [])
        ],
        scope_elements=[
            _code_element_from_dict(e)
            for e in data.get("scope_elements", [])
        ],
        smells_before=[
            _smell_from_dict(s)
            for s in data.get("smells_before", [])
        ],
        smells_after=[
            _smell_from_dict(s)
            for s in data.get("smells_after", [])
        ],
        classification=data.get("classification", "neutral"),
        n_positive=data.get("n_positive", 0) or 0,
        n_negative=data.get("n_negative", 0) or 0,
        n_neutral=data.get("n_neutral", 0) or 0,
    )
