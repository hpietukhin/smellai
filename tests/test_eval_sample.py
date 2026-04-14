"""Tests for EvalSample — the unified evaluation-sample schema."""

import pytest
from pydantic import ValidationError

from smellai_datasets.schema import EvalSample


class TestEvalSampleSchema:
    """Structural and validation tests for EvalSample."""

    def test_valid_swe_sample(self):
        sample = EvalSample(
            source="swe",
            sample_id="swe:abc123",
            inputs={"project_name": "checkstyle", "commit_id": "abc123"},
            expectations={"class_after": "..."},
            tags={"is_pure": True},
        )
        assert sample.source == "swe"
        assert sample.sample_id == "swe:abc123"
        assert sample.inputs["project_name"] == "checkstyle"
        assert sample.expectations["class_after"] == "..."
        assert sample.tags["is_pure"] is True

    def test_valid_rminer_sample(self):
        sample = EvalSample(
            source="rminer",
            sample_id="rminer:pair_001",
            inputs={"pair_id": "pair_001"},
            expectations={},
            tags={},
        )
        assert sample.source == "rminer"

    def test_valid_tdd_sample(self):
        sample = EvalSample(
            source="tdd",
            sample_id="tdd:proj:sha:rule:comp:10",
            inputs={"rule": "java:S138"},
            expectations={"close_commit": ""},
            tags={"severity": "HIGH"},
        )
        assert sample.source == "tdd"

    def test_invalid_source_raises(self):
        with pytest.raises(ValidationError):
            EvalSample(source="unknown", sample_id="x:1", inputs={}, expectations={}, tags={})  # type: ignore

    def test_empty_sample_id_raises(self):
        with pytest.raises(ValidationError):
            EvalSample(source="swe", sample_id="", inputs={}, expectations={}, tags={})

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            EvalSample(
                source="swe",
                sample_id="swe:1",
                inputs={},
                expectations={},
                tags={},
                extra_field="not_allowed",  # type: ignore
            )

    def test_frozen_prevents_mutation(self):
        sample = EvalSample(
            source="swe", sample_id="swe:1", inputs={}, expectations={}, tags={}
        )
        with pytest.raises(Exception):  # ValidationError or TypeError
            sample.source = "rminer"  # type: ignore

    def test_model_dump_round_trip(self):
        original = EvalSample(
            source="swe",
            sample_id="swe:test",
            inputs={"key": "value", "num": 42},
            expectations={"result": [1, 2, 3]},
            tags={"flag": True},
        )
        data = original.model_dump()
        restored = EvalSample.model_validate(data)
        assert restored == original

    def test_model_dump_json(self):
        sample = EvalSample(
            source="rminer",
            sample_id="rminer:p1",
            inputs={"pair_id": "p1"},
            expectations={},
            tags={},
        )
        json_str = sample.model_dump_json()
        assert '"source":"rminer"' in json_str or '"source": "rminer"' in json_str

    def test_default_empty_dicts(self):
        sample = EvalSample(source="swe", sample_id="swe:x")
        assert sample.inputs == {}
        assert sample.expectations == {}
        assert sample.tags == {}
