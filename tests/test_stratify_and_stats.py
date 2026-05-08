from dataset.stratify import bucket_composite_size, bucket_scope_size
from domain.stats import bootstrap_ci_mean, cliffs_delta


def test_scope_bucket_boundaries():
    assert bucket_scope_size(2) == "2"
    assert bucket_scope_size(3) == "3-5"
    assert bucket_scope_size(5) == "3-5"
    assert bucket_scope_size(6) == "6-10"
    assert bucket_scope_size(11) == ">10"


def test_composite_bucket_boundaries():
    assert bucket_composite_size(1) == "small"
    assert bucket_composite_size(4) == "small"
    assert bucket_composite_size(5) == "medium"
    assert bucket_composite_size(10) == "medium"
    assert bucket_composite_size(11) == "large"


def test_bootstrap_ci_mean_returns_ordered_bounds():
    lo, hi = bootstrap_ci_mean([1.0, 2.0, 3.0, 4.0], n_boot=200, seed=7)
    assert lo <= hi
    assert lo <= 2.5 <= hi


def test_cliffs_delta_sign_and_range():
    d = cliffs_delta([5, 6, 7], [1, 2, 3])
    assert 0.0 < d <= 1.0

    d2 = cliffs_delta([1, 2, 3], [5, 6, 7])
    assert -1.0 <= d2 < 0.0
