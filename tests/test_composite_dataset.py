from __future__ import annotations

import json
import pytest

from domain.rules import (
    DATASET_DEFAULT_SEVERITY,
    DATASET_SMELL_TYPE_MAP,
    FEATURE_ENVY,
    LONG_METHOD,
    get_default_severity,
    get_refactoring_types,
    normalize_dataset_smell_type,
)
from smellai_datasets.composite_dataset import load_episodes_jsonl
from smellai_datasets.composite_models import (
    CodeElement,
    CompositeEpisode,
    RefactoringStep,
    SmellInstance,
    episode_from_dict,
)


# ── Concrete samples from Composite Refactorings 2020 Neo4j ──────────────
# These replace Hypothesis strategies to eliminate flakiness while still
# covering the full range of dataset smell types and refactoring types.

ALL_DATASET_SMELL_TYPES = sorted(DATASET_SMELL_TYPE_MAP.keys())

CONCRETE_SMELL_COMBOS: list[list[str]] = [
    ["LongMethod", "FeatureEnvy"],
    ["GodClass", "ComplexClass", "BrainClass"],
    ["LongParameterList", "DataClass", "LazyClass"],
    ["SpaghettiCode", "ClassDataShouldBePrivate", "RefusedBequest"],
    ["IntensiveCoupling", "DispersedCoupling", "ShotgunSurgery", "BrainMethod", "SpeculativeGenerality"],
]

CONCRETE_REF_TYPE_COMBOS: list[list[str]] = [
    ["Extract Method", "Move Method"],
    ["Rename Method", "Rename Method", "Rename Method"],
    ["Extract Method", "Inline Method"],
    ["Pull Up Method", "Rename Class", "Move Class"],
    ["Extract Method", "Extract Method", "Move Method", "Rename Method"],
]


def _episode(
    *,
    smell_types: list[str] | None = None,
    element_name: str = "org.example.Foo.longMethod()",
) -> CompositeEpisode:
    smell_types = smell_types or ["LongMethod", "FeatureEnvy"]
    return CompositeEpisode(
        episode_id="Demo:abc:1",
        project="Demo",
        commit_hash="abc",
        commit_order=1,
        refactorings=[
            RefactoringStep(
                ref_type="Extract Method",
                hash_id="r1",
                classification="positive",
                degradation_level="agglomeration",
                smelly=True,
                changed_elements=[element_name],
            ),
            RefactoringStep(
                ref_type="Move Method",
                hash_id="r2",
                classification="neutral",
                degradation_level="agglomeration",
                smelly=True,
                produced_elements=[element_name],
            ),
        ],
        scope_elements=[
            CodeElement(
                name=element_name,
                element_type="Public Method",
                file_path="src/org/example/Foo.java",
            )
        ],
        smells_before=[
            SmellInstance(
                smell_type=t,
                hash_id=f"s{i}",
                reason="metric > threshold",
                starting_line=10 + i,
                ending_line=20 + i,
                element_name=element_name,
                element_path="src/org/example/Foo.java",
            )
            for i, t in enumerate(smell_types)
        ],
        smells_after=[],
        classification="positive",
        n_positive=1,
        n_neutral=1,
    )


# ── Smell type mapping ───────────────────────────────────────────────────

@pytest.mark.parametrize("raw_smell", ALL_DATASET_SMELL_TYPES)
def test_dataset_smell_mapping_has_default_severity(raw_smell: str) -> None:
    canonical = normalize_dataset_smell_type(raw_smell)
    assert canonical in DATASET_DEFAULT_SEVERITY
    assert get_default_severity(canonical) in {"HIGH", "MEDIUM", "LOW"}


# ── Locality controls edges ──────────────────────────────────────────────

@pytest.mark.parametrize("same_class", [True, False])
@pytest.mark.parametrize("locality", ["class", "file", "none"])
def test_projection_locality_controls_dependency_edges(
    same_class: bool,
    locality: str,
) -> None:
    ep = _episode()
    events = ep.to_smell_events("before")

    if not same_class:
        events[1].class_name = "org.example.Other"

    from domain.dependency_graph import DependencyGraph

    graph = DependencyGraph.from_events(events, locality=locality)

    has_edge = any(
        src == events[0].smell_id
        and dst == events[1].smell_id
        and data.get("relation") == "positive"
        for src, dst, data in graph.graph.edges(data=True)
    )

    expected = locality in {"none", "file"} or same_class
    assert has_edge is expected


# ── Episode JSON roundtrip ───────────────────────────────────────────────

@pytest.mark.parametrize(
    "ref_types, raw_smells",
    list(zip(CONCRETE_REF_TYPE_COMBOS, CONCRETE_SMELL_COMBOS)),
    ids=[f"combo_{i}" for i in range(len(CONCRETE_SMELL_COMBOS))],
)
def test_episode_json_roundtrip_preserves_high_level_shape(
    ref_types: list[str],
    raw_smells: list[str],
) -> None:
    ep = _episode(smell_types=raw_smells)
    ep.refactorings = [
        RefactoringStep(
            ref_type=rt,
            hash_id=f"r{i}",
            classification="positive" if i == 0 else "neutral",
            degradation_level="agglomeration",
            smelly=True,
        )
        for i, rt in enumerate(ref_types)
    ]

    hydrated = episode_from_dict(ep.to_dict())

    assert hydrated.episode_id == ep.episode_id
    assert hydrated.project == ep.project
    assert hydrated.ref_types == ref_types
    assert [s.smell_type for s in hydrated.smells_before] == raw_smells
    assert hydrated.to_dict()["size"] == len(ref_types)


# ── JSONL offline loading ────────────────────────────────────────────────

def test_episode_jsonl_loads_offline(tmp_path) -> None:
    ep = _episode()
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(ep.to_dict()) + "\n")

    loaded = load_episodes_jsonl(str(path))

    assert len(loaded) == 1
    assert loaded[0].episode_id == ep.episode_id
    assert len(loaded[0].to_dependency_graph(locality="class")) == 2


# ── Refactoring catalogue ───────────────────────────────────────────────

def test_refactoring_catalogue_supports_projected_smells() -> None:
    ep = _episode()
    projected = ep.to_smell_events("before")
    smell_types = {e.smell_type for e in projected}

    assert smell_types == {LONG_METHOD, FEATURE_ENVY}
    assert get_refactoring_types(LONG_METHOD)[0] == "Extract Method"
    assert "Move Method" in get_refactoring_types(FEATURE_ENVY)


# ── Real dataset episode (Apache Tomcat DefaultServlet) ──────────────────

def test_real_episode_dependency_graph_and_plan() -> None:
    """Concrete episode from Neo4j: Apache Tomcat DefaultServlet with 11 smells.
    Tests the full pipeline: SmellInstance → SmellEvent → DependencyGraph → Plan."""
    from domain.dependency_graph import DependencyGraph
    from domain.refactoring_tree import RefactoringTree, State

    ep = _episode(smell_types=[
        "BrainClass", "SpaghettiCode", "RefusedBequest", "ComplexClass",
        "ClassDataShouldBePrivate", "GodClass", "BrainMethod",
        "IntensiveCoupling", "LongParameterList", "LongMethod", "FeatureEnvy",
    ])

    events = ep.to_smell_events("before")
    assert len(events) == 11

    dg = DependencyGraph.from_events(events, locality="none")
    assert len(dg) == 11

    initial = State(frozenset(e.smell_id for e in events))
    tree = RefactoringTree(initial, dg)

    plan_greedy = tree.greedy()
    plan_befs = tree.befs()

    # Both plans should resolve all smells (h=0)
    assert plan_greedy.h_trace[-1] == 0
    assert plan_befs.h_trace[-1] == 0

    # BeFS should find plan no longer than greedy
    assert len(plan_befs.actions) <= len(plan_greedy.actions)
