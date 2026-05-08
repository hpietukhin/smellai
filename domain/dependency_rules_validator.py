"""Empirical validation for Markovič dependency rules.

Implements two validation modes:
1) Predictive validation on dataset trajectories (precision/recall/F1)
2) Leave-one-rule-out ablation (rule contribution to F1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.metrics import precision_recall_fscore_support

from domain.rules import DEPENDENCY_RULES, REFACTORING_CATALOGUE


@dataclass(frozen=True)
class RuleValidationResult:
    positive_precision: float
    positive_recall: float
    positive_f1: float
    negative_precision: float
    negative_recall: float
    negative_f1: float
    support_positive: int
    support_negative: int
    ablation_drop_by_rule: dict[str, float]

    def to_mlflow_metrics(self) -> dict[str, float]:
        out = {
            "rules_positive_precision": self.positive_precision,
            "rules_positive_recall": self.positive_recall,
            "rules_positive_f1": self.positive_f1,
            "rules_negative_precision": self.negative_precision,
            "rules_negative_recall": self.negative_recall,
            "rules_negative_f1": self.negative_f1,
            "rules_support_positive": float(self.support_positive),
            "rules_support_negative": float(self.support_negative),
        }
        for k, v in self.ablation_drop_by_rule.items():
            out[f"ablation_f1_drop::{k}"] = v
        return out


class DependencyRulesValidator:
    """Validate dependency rules against empirical composite steps."""

    def __init__(self, rules: dict[str, dict[str, list[str]]] | None = None) -> None:
        self.rules = rules or DEPENDENCY_RULES

    def validate_steps(self, steps: list[Any]) -> RuleValidationResult:
        y_true_pos, y_pred_pos, y_true_neg, y_pred_neg = self._build_labels(steps, self.rules)

        p_pos, r_pos, f_pos, _ = precision_recall_fscore_support(
            y_true_pos, y_pred_pos, average="binary", zero_division=0
        )
        p_neg, r_neg, f_neg, _ = precision_recall_fscore_support(
            y_true_neg, y_pred_neg, average="binary", zero_division=0
        )

        base_f1 = (f_pos + f_neg) / 2.0
        ablation: dict[str, float] = {}
        for src in sorted(self.rules):
            ablated_rules = {k: v for k, v in self.rules.items() if k != src}
            yp, pp, yn, pn = self._build_labels(steps, ablated_rules)
            _, _, f_pos_a, _ = precision_recall_fscore_support(yp, pp, average="binary", zero_division=0)
            _, _, f_neg_a, _ = precision_recall_fscore_support(yn, pn, average="binary", zero_division=0)
            ablation[src] = base_f1 - ((f_pos_a + f_neg_a) / 2.0)

        return RuleValidationResult(
            positive_precision=float(p_pos),
            positive_recall=float(r_pos),
            positive_f1=float(f_pos),
            negative_precision=float(p_neg),
            negative_recall=float(r_neg),
            negative_f1=float(f_neg),
            support_positive=len(y_true_pos),
            support_negative=len(y_true_neg),
            ablation_drop_by_rule=ablation,
        )

    def _build_labels(self, steps: list[Any], rules: dict[str, dict[str, list[str]]]):
        y_true_pos: list[int] = []
        y_pred_pos: list[int] = []
        y_true_neg: list[int] = []
        y_pred_neg: list[int] = []

        for i in range(len(steps) - 1):
            before = steps[i]
            after = steps[i + 1]
            before_counts = self._count_smell_types(before.smells)
            after_counts = self._count_smell_types(after.smells)

            active_triggers = self._active_triggers(before_counts, before.refactorings)
            for src in active_triggers:
                rr = rules.get(src, {})
                for tgt in rr.get("positive", []):
                    y_pred_pos.append(1)
                    y_true_pos.append(int(after_counts.get(tgt, 0) < before_counts.get(tgt, 0)))
                for tgt in rr.get("negative", []):
                    y_pred_neg.append(1)
                    y_true_neg.append(int(after_counts.get(tgt, 0) > before_counts.get(tgt, 0)))

        if not y_true_pos:
            y_true_pos, y_pred_pos = [0], [0]
        if not y_true_neg:
            y_true_neg, y_pred_neg = [0], [0]

        return y_true_pos, y_pred_pos, y_true_neg, y_pred_neg

    @staticmethod
    def _count_smell_types(smells: list[Any]) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in smells:
            out[s.smell_type] = out.get(s.smell_type, 0) + 1
        return out

    @staticmethod
    def _active_triggers(before_counts: dict[str, int], refactorings: list[Any]) -> set[str]:
        ref_types = {r.ref_type for r in refactorings}
        triggers: set[str] = set()
        for smell_type, count in before_counts.items():
            if count <= 0:
                continue
            ops = {op for op, _ in REFACTORING_CATALOGUE.get(smell_type, [])}
            if ops & ref_types:
                triggers.add(smell_type)
        return triggers
