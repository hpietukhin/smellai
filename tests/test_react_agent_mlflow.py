"""Tests for the react_agent_mlflow pipeline."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.models import DACOSSample
from src.pipelines import (
    _build_expectations,
    _build_inputs,
    _extract_agent_response,
    _resolve_judge_models,
    _sample_to_prompt,
    predict_refactoring,
)


@pytest.fixture
def sample_dacos():
    """Create a sample DACOS entry for testing."""
    return DACOSSample(
        id=1,
        project_name="test/project",
        path_to_file="src/main/Test.java",
        smell_name="Long Method",
        smell_description="Method is too long",
        sample_constraints="Keep interfaces stable",
        has_smell=True,
        is_class=False,
        iscm=True,  # Complex Method
        isim=False,
        islp=False,
        isma=False,
    )


def test_sample_to_prompt(sample_dacos):
    """Test prompt generation from DACOS sample."""
    prompt = _sample_to_prompt(sample_dacos)

    assert "DACOS sample 1" in prompt
    assert "test/project" in prompt
    assert "src/main/Test.java" in prompt
    assert "Long Method" in prompt
    assert "Method is too long" in prompt
    assert "Keep interfaces stable" in prompt


def test_build_expectations(sample_dacos):
    """Test expectations building."""
    expectations = _build_expectations(sample_dacos)

    assert expectations["sample_id"] == 1
    assert expectations["smell_name"] == "Long Method"
    assert expectations["smell_description"] == "Method is too long"


def test_build_inputs(sample_dacos):
    """Test inputs building."""
    inputs = _build_inputs(sample_dacos)

    assert "inputs" in inputs
    assert isinstance(inputs["inputs"], str)
    assert "DACOS sample 1" in inputs["inputs"]


def test_extract_agent_response():
    """Test extracting response from agent state."""
    state = {
        "messages": [
            HumanMessage(content="Test question"),
            AIMessage(content="Test answer"),
        ]
    }

    response = _extract_agent_response(state)
    assert response == "Test answer"


def test_extract_agent_response_no_messages():
    """Test error handling when no messages in state."""
    state = {"messages": []}

    with pytest.raises(ValueError, match="Agent returned no messages"):
        _extract_agent_response(state)


def test_resolve_judge_models_with_models():
    """Test resolving judge models when models list provided."""
    models = _resolve_judge_models(judge_models=["model1", "model2"])
    assert models == ["model1", "model2"]


def test_resolve_judge_models_with_single_model():
    """Test resolving judge models when single model provided."""
    models = _resolve_judge_models(judge_model="model1")
    assert models == ["model1"]


def test_resolve_judge_models_from_env():
    """Test resolving judge models from environment."""
    with patch.dict("os.environ", {"MLFLOW_JUDGE_MODEL": "custom/model"}):
        models = _resolve_judge_models()
        assert models == ["custom/model"]


@patch("src.pipelines.react_agent_mlflow._invoke_agent")
@patch("src.pipelines.react_agent_mlflow._get_agent_context")
def test_predict_refactoring_success(mock_context, mock_invoke):
    """Test successful prediction."""
    mock_invoke.return_value = {
        "messages": [
            HumanMessage(content="Test"),
            AIMessage(content="Refactoring suggestion"),
        ]
    }
    mock_context.return_value = Mock()

    result = predict_refactoring("Test prompt")
    assert result == "Refactoring suggestion"
    mock_invoke.assert_called_once()


@patch("src.pipelines.react_agent_mlflow._invoke_agent")
@patch("src.pipelines.react_agent_mlflow._get_agent_context")
def test_predict_refactoring_error(mock_context, mock_invoke):
    """Test error handling in prediction."""
    mock_invoke.side_effect = Exception("Test error")
    mock_context.return_value = Mock()

    result = predict_refactoring("Test prompt")
    assert "error" in result
    assert "Test error" in result


def test_smell_detection_f1_perfect_match():
    """Test F1 score with perfect detection."""
    from src.pipelines.react_agent_mlflow import smell_detection_f1

    outputs = "This code has a Complex Method smell that needs refactoring."
    expectations = {"smell_name": "Complex Method"}

    f1_score = smell_detection_f1(outputs, expectations)
    assert abs(f1_score - 1.0) < 0.001


def test_smell_detection_f1_no_detection():
    """Test F1 score when smell is not detected."""
    from src.pipelines.react_agent_mlflow import smell_detection_f1

    outputs = "The code looks good and follows best practices."
    expectations = {"smell_name": "Complex Method"}

    f1_score = smell_detection_f1(outputs, expectations)
    assert abs(f1_score - 0.0) < 0.001


def test_smell_detection_f1_false_positive():
    """Test F1 score with false positive detection."""
    from src.pipelines.react_agent_mlflow import smell_detection_f1

    outputs = "This code has a Long Method smell."
    expectations = {"smell_name": "Complex Method"}

    # F1 is 0 because we detected wrong smell (false positive) and missed correct smell (false negative)
    f1_score = smell_detection_f1(outputs, expectations)
    assert abs(f1_score - 0.0) < 0.001


def test_smell_detection_f1_partial_match():
    """Test F1 score with both correct and incorrect detections."""
    from src.pipelines.react_agent_mlflow import smell_detection_f1

    outputs = "This code has a Complex Method and Long Method smell."
    expectations = {"smell_name": "Complex Method"}

    # TP=1 (complex method), FP=1 (long method), FN=0
    # Precision = 1/2 = 0.5, Recall = 1/1 = 1.0
    # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 0.667
    f1_score = smell_detection_f1(outputs, expectations)
    assert abs(f1_score - 0.6666666666666666) < 0.001


def test_smell_detection_f1_no_expectation():
    """Test F1 score when no smell is expected."""
    from src.pipelines.react_agent_mlflow import smell_detection_f1

    outputs = "The code looks good."
    expectations = {}

    f1_score = smell_detection_f1(outputs, expectations)
    assert abs(f1_score - 0.0) < 0.001
