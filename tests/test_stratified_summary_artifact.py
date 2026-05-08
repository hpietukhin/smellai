from workflows.planner_eval_workflow import _build_stratified_summary_rows


def test_build_stratified_summary_rows_contains_methodology_buckets_and_stats():
    rows = [
        {
            "scope_size": 2,
            "composite_size": 4,
            "success": 1.0,
            "befs_eta": 1.0,
            "greedy_eta": 2.0,
            "befs_rho": 0.0,
            "greedy_rho": 0.5,
            "h_initial": 10.0,
            "h_befs_final": 5.0,
            "h_greedy_final": 7.0,
            "rule_aware_resolved": 3.0,
            "no_rules_resolved": 2.0,
            "rule_aware_introduced": 0.0,
            "no_rules_introduced": 1.0,
        },
        {
            "scope_size": 4,
            "composite_size": 8,
            "success": 0.0,
            "befs_eta": 1.5,
            "greedy_eta": 2.5,
            "befs_rho": 0.25,
            "greedy_rho": 0.75,
            "h_initial": 8.0,
            "h_befs_final": 4.0,
            "h_greedy_final": 6.0,
            "rule_aware_resolved": 2.0,
            "no_rules_resolved": 1.0,
            "rule_aware_introduced": 1.0,
            "no_rules_introduced": 2.0,
        },
    ]

    out = _build_stratified_summary_rows(rows)
    assert "bucket_definitions" in out
    assert "by_scope" in out
    assert "by_composite" in out
    assert "overall" in out
    assert "2" in out["by_scope"]
    assert "small" in out["by_composite"]

    overall_metrics = out["overall"]["metrics"]
    assert overall_metrics["befs_relative_h_reduction"]["mean"] == 0.5
    assert overall_metrics["success"]["mean"] == 0.5
    assert overall_metrics["rule_gain_resolved"]["mean"] == 1.0
    assert overall_metrics["rule_gain_introduced"]["mean"] == 1.0
