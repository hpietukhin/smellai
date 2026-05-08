"""Lightweight statistics helpers for evaluation summaries."""

from __future__ import annotations

from sklearn.utils import resample


def bootstrap_ci_mean(values: list[float], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    assert values, "values must be non-empty"
    assert n_boot > 10, "n_boot too small"
    assert 0 < alpha < 1, "alpha must be in (0,1)"

    means: list[float] = []
    n = len(values)
    for i in range(n_boot):
        sample = resample(values, replace=True, n_samples=n, random_state=seed + i)
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2.0) * n_boot)
    hi_idx = int((1.0 - alpha / 2.0) * n_boot) - 1
    lo_idx = max(0, min(lo_idx, n_boot - 1))
    hi_idx = max(0, min(hi_idx, n_boot - 1))
    return means[lo_idx], means[hi_idx]


def cliffs_delta(xs: list[float], ys: list[float]) -> float:
    assert xs and ys, "samples must be non-empty"
    gt = 0
    lt = 0
    for x in xs:
        for y in ys:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    total = len(xs) * len(ys)
    assert total > 0
    return (gt - lt) / total
