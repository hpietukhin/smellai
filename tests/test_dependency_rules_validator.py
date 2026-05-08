from dataclasses import dataclass

from domain.dependency_rules_validator import DependencyRulesValidator
from domain.models import SmellEvent


@dataclass
class _Ref:
    ref_type: str


@dataclass
class _Step:
    smells: list[SmellEvent]
    refactorings: list[_Ref]


def _smell(smell_type: str, sid: str):
    return SmellEvent(smell_id=sid, smell_type=smell_type, severity="MEDIUM", file_path="A.java")


def test_validator_returns_metrics_and_ablation():
    # before: Long Method + Duplicated Code
    s0 = _Step(
        smells=[_smell("Long Method", "lm1"), _smell("Duplicated Code", "dc1")],
        refactorings=[_Ref("Extract Method")],
    )
    # after: Long Method remains, Duplicated Code resolved
    s1 = _Step(
        smells=[_smell("Long Method", "lm1")],
        refactorings=[],
    )

    res = DependencyRulesValidator().validate_steps([s0, s1])
    assert 0.0 <= res.positive_f1 <= 1.0
    assert 0.0 <= res.negative_f1 <= 1.0
    assert "Long Method" in res.ablation_drop_by_rule
