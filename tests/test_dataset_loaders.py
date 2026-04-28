"""Tests for smellai_datasets loaders: raw DataFrames and EvalSample projections."""

import json
from pathlib import Path

import pytest

from smellai_datasets import load_eval_samples, load_eval_df, EvalSample
from smellai_datasets.loaders import load_swe_raw_df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def swe_json(tmp_path: Path) -> Path:
    """Minimal SWE-Refactor JSON with two records."""
    records = [
        {
            "uniqueId": "pair_001",
            "projectName": "checkstyle",
            "commitId": "abc123",
            "type": "Extract Method",
            "filePathBefore": "Foo.java",
            "filePathAfter": "Foo.java",
            "sourceCodeBeforeForWhole": "class Foo {}",
            "sourceCodeAfterForWhole": "class Foo { void bar() {} }",
            "sourceCodeBeforeRefactoring": "// before",
            "sourceCodeAfterRefactoring": "// after",
            "compileCommand": "mvn compile",
            "compileJDK": 11,
            "isPureRefactoring": True,
            "hasTestC": False,
        },
        {
            "uniqueId": "pair_002",
            "projectName": "guava",
            "commitId": "def456",
            "type": "Move Method+Extract Method",
            "filePathBefore": "Bar.java",
            "filePathAfter": "Bar.java",
            "sourceCodeBeforeForWhole": "class Bar {}",
            "sourceCodeAfterForWhole": "class Bar { void baz() {} }",
            "sourceCodeBeforeRefactoring": "// before2",
            "sourceCodeAfterRefactoring": "// after2",
            "compileCommand": "./gradlew build",
            "compileJDK": 1.8,  # should be coerced to 8
            "isPureRefactoring": False,
            "hasTestC": True,
        },
    ]
    path = tmp_path / "pure_refactoring_data.json"
    path.write_text(json.dumps(records))
    return path


# ---------------------------------------------------------------------------
# load_swe_raw_df
# ---------------------------------------------------------------------------

class TestLoadSweRawDf:
    def test_shape(self, swe_json):
        df = load_swe_raw_df(swe_json)
        assert len(df) == 2
        assert "pair_id" in df.columns
        assert "refactoring_type" in df.columns

    def test_jdk_coercion(self, swe_json):
        df = load_swe_raw_df(swe_json)
        row = df[df["pair_id"] == "pair_002"].iloc[0]
        assert row["jdk_version"] == 8  # 1.8 → 8

    def test_is_compound_flag(self, swe_json):
        df = load_swe_raw_df(swe_json)
        row1 = df[df["pair_id"] == "pair_001"].iloc[0]
        row2 = df[df["pair_id"] == "pair_002"].iloc[0]
        assert row1["is_compound"] == False  # noqa: E712
        assert row2["is_compound"] == True  # "+" in type  # noqa: E712

    def test_is_pure(self, swe_json):
        df = load_swe_raw_df(swe_json)
        assert df[df["pair_id"] == "pair_001"].iloc[0]["is_pure"] == True  # noqa: E712
        assert df[df["pair_id"] == "pair_002"].iloc[0]["is_pure"] == False  # noqa: E712


# ---------------------------------------------------------------------------
# load_eval_samples — SWE
# ---------------------------------------------------------------------------

class TestLoadEvalSamplesSwe:
    def test_returns_eval_samples(self, swe_json):
        samples = load_eval_samples(["swe"], swe_path=swe_json)
        assert len(samples) == 2
        assert all(isinstance(s, EvalSample) for s in samples)

    def test_source_is_swe(self, swe_json):
        samples = load_eval_samples(["swe"], swe_path=swe_json)
        assert all(s.source == "swe" for s in samples)

    def test_sample_id_format(self, swe_json):
        samples = load_eval_samples(["swe"], swe_path=swe_json)
        ids = {s.sample_id for s in samples}
        assert "swe:pair_001" in ids
        assert "swe:pair_002" in ids

    def test_inputs_keys(self, swe_json):
        samples = load_eval_samples(["swe"], swe_path=swe_json)
        s = next(s for s in samples if s.sample_id == "swe:pair_001")
        for key in (
            "project_name", "commit_id", "refactoring_type",
            "file_path_before", "file_path_after", "class_before",
            "source_before", "jdk_version", "compile_command",
        ):
            assert key in s.inputs, f"Missing input key: {key}"

    def test_expectations_keys(self, swe_json):
        samples = load_eval_samples(["swe"], swe_path=swe_json)
        s = next(s for s in samples if s.sample_id == "swe:pair_001")
        assert "class_after" in s.expectations
        assert "source_after" in s.expectations

    def test_tags_keys(self, swe_json):
        samples = load_eval_samples(["swe"], swe_path=swe_json)
        s = next(s for s in samples if s.sample_id == "swe:pair_001")
        assert "is_pure" in s.tags
        assert "is_compound" in s.tags
        assert "has_tests" in s.tags

    def test_limit(self, swe_json):
        samples = load_eval_samples(["swe"], swe_path=swe_json, limit=1)
        assert len(samples) == 1


# ---------------------------------------------------------------------------
# load_eval_df
# ---------------------------------------------------------------------------

class TestLoadEvalDf:
    def test_columns(self, swe_json):
        df = load_eval_df(["swe"], swe_path=swe_json)
        assert set(df.columns) >= {"source", "sample_id", "inputs", "expectations", "tags"}

    def test_source_column(self, swe_json):
        df = load_eval_df(["swe"], swe_path=swe_json)
        assert set(df["source"].unique()) == {"swe"}
