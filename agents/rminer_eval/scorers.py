"""DEPRECATED: MLflow scorers for RMiner refactoring mapping evaluation.

Use Composite Refactorings 2020 planner metrics via workflows/planner_eval_workflow.py.
"""

from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer


def _calculate_line_overlap(start1: int, end1: int, start2: int, end2: int) -> int:
    """Calculate overlapping lines between two ranges."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0, overlap_end - overlap_start + 1)


@scorer
def mapping_accuracy(outputs: dict, expectations: dict) -> Feedback:
    """
    Compute accuracy: fraction of predictions that overlap with ground truth hunks.

    A prediction is correct if its line range overlaps with any ground truth hunk.
    """
    predictions = outputs.get("predictions", [])
    hunks = expectations.get("diff_hunks", [])

    if not predictions:
        return Feedback(value=0.0, rationale="No predictions made by agent")

    correct = 0
    details = []

    for pred in predictions:
        pred_start = pred.get("line_start", 0)
        pred_end = pred.get("line_end", 0)
        pred_hunk_idx = pred.get("predicted_hunk_index", -1)

        has_overlap = False
        for i, hunk in enumerate(hunks):
            hunk_start = hunk.get("old_start", 0)
            hunk_end = hunk_start + hunk.get("old_count", 1) - 1

            overlap = _calculate_line_overlap(
                pred_start, pred_end, hunk_start, hunk_end
            )
            if overlap > 0:
                has_overlap = True
                details.append(f"Pred {pred_hunk_idx}→Hunk {i}: ✓ (overlap={overlap})")
                break

        if has_overlap:
            correct += 1
        else:
            details.append(f"Pred {pred_hunk_idx}: ✗ (no overlap)")

    accuracy = correct / len(predictions)

    return Feedback(
        value=accuracy,
        rationale=f"{correct}/{len(predictions)} predictions correct. "
        + "; ".join(details[:5]),
    )


@scorer
def hunk_coverage(outputs: dict, expectations: dict) -> Feedback:
    """
    Compute coverage: fraction of ground truth hunks covered by at least one prediction.
    """
    predictions = outputs.get("predictions", [])
    hunks = expectations.get("diff_hunks", [])

    if not hunks:
        return Feedback(value=1.0, rationale="No hunks to cover")

    covered_hunks = set()

    for pred in predictions:
        pred_start = pred.get("line_start", 0)
        pred_end = pred.get("line_end", 0)

        for i, hunk in enumerate(hunks):
            hunk_start = hunk.get("old_start", 0)
            hunk_end = hunk_start + hunk.get("old_count", 1) - 1

            overlap = _calculate_line_overlap(
                pred_start, pred_end, hunk_start, hunk_end
            )
            if overlap > 0:
                covered_hunks.add(i)

    coverage = len(covered_hunks) / len(hunks)

    return Feedback(
        value=coverage,
        rationale=f"{len(covered_hunks)}/{len(hunks)} hunks covered by predictions",
    )


@scorer
def prediction_completeness(outputs: dict, expectations: dict) -> Feedback:
    """
    Check if agent made predictions for all refactorings.
    """
    predictions = outputs.get("predictions", [])
    expected_count = expectations.get("num_refactorings", 0)

    if expected_count == 0:
        return Feedback(value=1.0, rationale="No refactorings expected")

    actual_count = len(predictions)
    ratio = min(actual_count / expected_count, 1.0)

    if actual_count == expected_count:
        rationale = f"Agent made exactly {expected_count} predictions as expected"
    elif actual_count < expected_count:
        rationale = (
            f"Agent made {actual_count}/{expected_count} predictions (missing some)"
        )
    else:
        rationale = (
            f"Agent made {actual_count} predictions but only {expected_count} expected"
        )

    return Feedback(value=ratio, rationale=rationale)
