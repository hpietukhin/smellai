"""MLflow scorers for SWE-Refactor evaluation outputs."""

from __future__ import annotations

from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer


@scorer
def _boolean_scorer(outputs: dict, key: str, label: str) -> Feedback:
    """Score a boolean output key as 1.0/0.0 with a readable rationale."""
    value = outputs.get(key, False)
    return Feedback(
        value=1.0 if value else 0.0,
        rationale=f"{label}: {'yes' if value else 'no'}",
    )


def compile_scorer(outputs: dict) -> Feedback:
    """Score whether generated code compiled successfully."""
    return _boolean_scorer(outputs, "compile_success", "Compilation")


def test_scorer(outputs: dict) -> Feedback:
    """Score whether tests passed, treating compile failure as tests-not-run."""
    compile_ok = outputs.get("compile_success", False)
    if not compile_ok:
        return Feedback(value=0.0, rationale="Compilation failed, tests not run")
    return _boolean_scorer(outputs, "test_success", "Tests")


def overall_scorer(outputs: dict) -> Feedback:
    """Score overall success: compilation and tests both passed."""
    compile_ok = outputs.get("compile_success", False)
    test_ok = outputs.get("test_success", False)
    success = compile_ok and test_ok
    if success:
        rationale = "Both compilation and tests passed"
    elif not compile_ok:
        rationale = "Compilation failed"
    else:
        rationale = "Compilation passed but tests failed"
    return Feedback(value=1.0 if success else 0.0, rationale=rationale)


SWE_SCORERS = {
    "compile": compile_scorer,
    "test": test_scorer,
    "overall": overall_scorer,
}


def get_swe_scorers():
    """Return standard SWE compile/test/overall scorers."""
    return list(SWE_SCORERS.values())


__all__ = [
    "SWE_SCORERS",
    "compile_scorer",
    "get_swe_scorers",
    "overall_scorer",
    "test_scorer",
]
