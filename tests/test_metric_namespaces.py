from workflows.planner_eval_workflow import _namespace_metrics


def test_namespace_metrics_prefixes_keys():
    src = {"a": 1.0, "b": 2.0}
    out = _namespace_metrics("algo", src)
    assert out == {"algo.a": 1.0, "algo.b": 2.0}
