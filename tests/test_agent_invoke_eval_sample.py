"""Tests for agent invoke boundary — EvalSample → internal state."""

from unittest.mock import MagicMock

import pytest

from smellai_datasets.schema import EvalSample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SWE_SAMPLE = EvalSample(
    source="swe",
    sample_id="swe:abc123",
    inputs={
        "project_name": "checkstyle",
        "commit_id": "abc123",
        "refactoring_type": "Extract Method",
        "file_path_before": "src/Foo.java",
        "file_path_after": "src/Foo.java",
        "class_before": "class Foo { void longMethod() {} }",
        "source_before": "void longMethod() {}",
        "jdk_version": 11,
        "compile_command": "mvn compile",
    },
    expectations={
        "class_after": "class Foo { void extracted() {} void longMethod() {} }",
        "source_after": "void longMethod() { extracted(); }",
    },
    tags={"is_pure": True, "is_compound": False, "has_tests": False},
)

RMINER_SAMPLE = EvalSample(
    source="rminer",
    sample_id="rminer:pair_001",
    inputs={
        "pair_id": "pair_001",
        "before_code": "class Foo { void bigMethod() {} }",
        "file_path": "src/Foo.java",
        "refactoring_types": ["Extract Method"],
        "refactoring_descriptions": ["Extract Method foo() from bigMethod()"],
        "diff_hunks": [
            {
                "old_start": 1,
                "old_count": 3,
                "new_start": 1,
                "new_count": 5,
                "removed_lines": ["    void bigMethod() {}"],
                "added_lines": ["    void bigMethod() { foo(); }", "    void foo() {}"],
                "context_lines": [],
            }
        ],
        "sonar_issues": [],
    },
    expectations={},
    tags={},
)


# ---------------------------------------------------------------------------
# SWE agent boundary
# ---------------------------------------------------------------------------

class TestSweAgentBoundary:
    def test_sample_to_refactoring_record_fields(self):
        """_sample_to_refactoring_record maps EvalSample inputs correctly."""
        from agents.swe_eval.agent import _sample_to_refactoring_record

        record = _sample_to_refactoring_record(SWE_SAMPLE)

        assert record.projectName == "checkstyle"
        assert record.commitId == "abc123"
        assert record.type == "Extract Method"
        assert record.filePathBefore == "src/Foo.java"
        assert record.filePathAfter == "src/Foo.java"
        assert record.sourceCodeBeforeForWhole == "class Foo { void longMethod() {} }"
        assert record.sourceCodeAfterForWhole == ""  # agent generates this
        assert record.compileJDK == 11
        assert record.compileCommand == "mvn compile"
        assert record.isPureRefactoring is True
        assert record.hasTestC is False

    def test_sample_to_refactoring_record_defaults(self):
        """compile result flags default to True (SWE-Refactor guarantee)."""
        from agents.swe_eval.agent import _sample_to_refactoring_record

        record = _sample_to_refactoring_record(SWE_SAMPLE)

        assert record.compileResultBefore is True
        assert record.compileResultCurrent is True

    def test_invoke_agent_rejects_wrong_source(self, tmp_path):
        """invoke_agent raises ValueError for non-swe EvalSample."""
        from agents.swe_eval.agent import invoke_agent

        wrong_sample = EvalSample(
            source="rminer",
            sample_id="rminer:1",
            inputs={"pair_id": "1"},
            expectations={},
            tags={},
        )
        mock_agent = MagicMock()

        with pytest.raises(ValueError, match="source='swe'"):
            invoke_agent(mock_agent, wrong_sample, workspace_path=tmp_path)

    def test_invoke_agent_calls_agent_invoke(self, tmp_path):
        """invoke_agent calls agent.invoke with the built initial state."""
        from agents.swe_eval.agent import invoke_agent

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [],
            "refactored_code": "class Foo {}",
            "compile_success": True,
            "test_success": True,
            "error_message": None,
        }

        invoke_agent(mock_agent, SWE_SAMPLE, workspace_path=tmp_path)

        mock_agent.invoke.assert_called_once()
        call_kwargs = mock_agent.invoke.call_args[0][0]
        assert call_kwargs["record"].projectName == "checkstyle"
        assert call_kwargs["record"].commitId == "abc123"
        assert call_kwargs["smell_detector"] is not None

    def test_invoke_agent_accepts_injected_smell_detector(self, tmp_path):
        """invoke_agent forwards an injected smell detector into graph state."""
        from agents.swe_eval.agent import invoke_agent
        from domain.detector import StaticDetector

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [],
            "compile_success": True,
            "test_success": True,
            "error_message": None,
        }
        detector = StaticDetector()

        invoke_agent(
            mock_agent,
            SWE_SAMPLE,
            workspace_path=tmp_path,
            smell_detector=detector,
        )

        call_state = mock_agent.invoke.call_args[0][0]
        assert call_state["smell_detector"] is detector


# ---------------------------------------------------------------------------
# RMiner agent boundary
# ---------------------------------------------------------------------------

class TestRminerAgentBoundary:
    def test_invoke_agent_reads_pair_id(self):
        """invoke_agent reads pair_id from sample.inputs, not a manifest file."""
        from agents.rminer_eval.agent import invoke_agent

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"predictions": []}

        result = invoke_agent(mock_agent, RMINER_SAMPLE)

        assert result["pair_id"] == "pair_001"
        assert result["filename"] == "src/Foo.java"

    def test_invoke_agent_passes_inputs_to_graph(self):
        """invoke_agent passes all EvalSample inputs to the LangGraph state."""
        from agents.rminer_eval.agent import invoke_agent

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"predictions": [{"refactoring_index": 0}]}

        invoke_agent(mock_agent, RMINER_SAMPLE)

        call_state = mock_agent.invoke.call_args[0][0]
        assert call_state["before_code"] == "class Foo { void bigMethod() {} }"
        assert call_state["filename"] == "src/Foo.java"
        assert call_state["refactoring_types"] == ["Extract Method"]
        assert len(call_state["diff_hunks"]) == 1

    def test_invoke_agent_rejects_wrong_source(self):
        """invoke_agent raises ValueError for non-rminer EvalSample."""
        from agents.rminer_eval.agent import invoke_agent

        wrong_sample = EvalSample(
            source="swe",
            sample_id="swe:1",
            inputs={"commit_id": "1"},
            expectations={},
            tags={},
        )
        mock_agent = MagicMock()

        with pytest.raises(ValueError, match="source='rminer'"):
            invoke_agent(mock_agent, wrong_sample)

    def test_invoke_agent_returns_predictions(self):
        """invoke_agent returns predictions from agent result."""
        from agents.rminer_eval.agent import invoke_agent

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "predictions": [{"refactoring_index": 0, "hunk_index": 0}]
        }

        result = invoke_agent(mock_agent, RMINER_SAMPLE)

        assert result["predictions"] == [{"refactoring_index": 0, "hunk_index": 0}]
