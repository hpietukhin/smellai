import json

from workflows.planner_eval_batch_workflow import _load_batch_list, _aggregate_rows


def test_load_batch_list():
    data = {
        "cases": [
            {"project": "P1", "elements": ["a.A"]},
            {"project": "P2", "elements": ["b.B", "b.C"]},
        ]
    }
    
    import tempfile
    from pathlib import Path
    p = Path(tempfile.mkstemp(suffix='.json')[1])
    p.write_text(json.dumps(data))
    try:
        eps = _load_batch_list(str(p))
        assert len(eps) == 2
        assert eps[0]["project"] == "P1"
    finally:
        p.unlink()


def test_aggregate_rows_has_overall_and_buckets():
    rows = [
        {
            "scope_size": 2.0,
            "composite_size": 4.0,
            "befs_eta": 1.0,
            "greedy_eta": 2.0,
            "h_initial": 10.0,
            "h_befs_final": 5.0,
            "h_greedy_final": 7.0,
        },
        {
            "scope_size": 6.0,
            "composite_size": 12.0,
            "befs_eta": 1.5,
            "greedy_eta": 2.5,
            "h_initial": 8.0,
            "h_befs_final": 4.0,
            "h_greedy_final": 6.0,
        },
    ]
    out = _aggregate_rows(rows)
    assert "overall" in out
    assert "by_scope" in out
    assert "by_composite" in out
    assert "bucket_definitions" in out
    assert "befs_relative_h_reduction" in out["overall"]["metrics"]
