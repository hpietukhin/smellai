from workflows.planner_eval_workflow import _build_validity_metadata


def test_validity_metadata_contains_core_sections():
    data = _build_validity_metadata(
        project="JUnit4",
        elements={"org.example.Foo"},
        heuristic="range-based",
        locality="none",
    )
    assert data["construct_validity"]
    assert data["internal_validity"]
    assert data["external_validity"]
    assert data["filters"]["project"] == "JUnit4"
    assert data["filters"]["locality"] == "none"
