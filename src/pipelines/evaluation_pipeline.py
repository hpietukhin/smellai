"""
LangGraph evaluation pipeline for code smell detection.

This module assembles all pipeline nodes into a LangGraph StateGraph and provides
the entry point function for running evaluations.

**Architecture**: Linear pipeline with error handling
**Flow**: START → fetch_sample → clone_repo → detect_smells → judge_evaluation → END
"""

import json
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from src.pipelines.nodes import (
    EvaluationState,
    clone_repo_node,
    detect_smells_node,
    fetch_sample_node,
    judge_evaluation_node,
)

# Global retriever instance (initialized once, reused)
_retriever = None


def get_retriever_instance():
    """
    Get or create the global retriever instance.

    Lazy initialization to avoid loading vector DB on module import.
    The retriever is created once and reused across all evaluations.

    Returns:
        Configured LangChain retriever for smell knowledge base

    Raises:
        RuntimeError: If vector DB initialization fails
    """
    global _retriever

    if _retriever is None:
        print("Initializing vector database and retriever...")

        try:
            from src.data.vector_db import load_and_create_vector_db

            # Load vector DB and get retriever
            # Using in-memory for development; change to file path for production
            _, _retriever = load_and_create_vector_db(
                smells_pattern="smells/content/smells/**/*.md",
                dataset_path="mem://deeplake/smells",  # In-memory for speed
                k=20,
            )

            print("✓ Retriever initialized and cached")

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize retriever: {e}. "
                "Make sure 'smells' repository is cloned in the project root."
            ) from e

    return _retriever


def create_evaluation_graph():
    """
    Create and compile the LangGraph evaluation pipeline.

    Assembles nodes into a linear graph with the following flow:
    START → fetch_sample → clone_repo → detect_smells → judge_evaluation → END

    Each node updates the shared EvaluationState and handles errors gracefully.
    Errors in early nodes propagate through but allow later nodes to create
    appropriate error responses.

    **Task**: P4-PIPELINE-006 - Create LangGraph pipeline

    Returns:
        Compiled LangGraph instance ready for invocation

    Example:
        >>> graph = create_evaluation_graph()
        >>> result = graph.invoke({"sample_id": 12345})
        >>> print(result["evaluation_result"].f1_score)
    """
    # Create state graph
    workflow = StateGraph(EvaluationState)

    # Add nodes
    print("Building evaluation pipeline...")

    workflow.add_node("fetch_sample", fetch_sample_node)
    print("  ✓ Added fetch_sample node")

    workflow.add_node("clone_repo", clone_repo_node)
    print("  ✓ Added clone_repo node")

    workflow.add_node("detect_smells", detect_smells_node)
    print("  ✓ Added detect_smells node")

    workflow.add_node("judge_evaluation", judge_evaluation_node)
    print("  ✓ Added judge_evaluation node")

    # Add edges (linear flow)
    workflow.set_entry_point("fetch_sample")
    workflow.add_edge("fetch_sample", "clone_repo")
    workflow.add_edge("clone_repo", "detect_smells")
    workflow.add_edge("detect_smells", "judge_evaluation")
    workflow.add_edge("judge_evaluation", END)

    print("  ✓ Connected pipeline flow")

    # Compile graph
    compiled_graph = workflow.compile()

    print("✓ Pipeline compiled successfully")

    return compiled_graph


def run_evaluation(sample_id: int) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline for a single sample.

    This is the main entry point that Promptfoo or other tools will call.
    It orchestrates the entire pipeline:
    1. Initialize retriever (once, cached)
    2. Create/get compiled graph
    3. Invoke with sample_id
    4. Extract and serialize results

    **Task**: P4-PIPELINE-007 - Create pipeline entry point function

    Args:
        sample_id: DACOS sample ID to evaluate

    Returns:
        Dictionary with evaluation results (JSON-serializable)
        Structure:
        {
            "sample_id": int,
            "file_path": str,
            "overall_score": float,
            "precision": float,
            "recall": float,
            "f1_score": float,
            "summary": str,
            "timestamp": str,
            "git_sha": str,
            "detected_smells_count": int,
            "ground_truth_smells_count": int,
            "error": str or null
        }

    Example:
        >>> result = run_evaluation(12345)
        >>> print(f"F1 Score: {result['f1_score']}")
        >>> if result['error']:
        ...     print(f"Error: {result['error']}")

    Notes:
        - Retriever is initialized once and reused
        - Graph is created fresh each time (lightweight)
        - All errors are caught and included in result
        - Output is JSON-serializable for Promptfoo
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION PIPELINE - Sample {sample_id}")
    print(f"{'='*60}")

    try:
        # Ensure retriever is initialized
        get_retriever_instance()

        # Create graph (lightweight, no heavy initialization)
        graph = create_evaluation_graph()

        # Create initial state
        initial_state = {"sample_id": sample_id}

        # Invoke pipeline
        print("\n▶ Starting pipeline execution...")
        final_state = graph.invoke(initial_state)

        print("\n✓ Pipeline execution complete")

        # Extract evaluation result
        evaluation_result = final_state.get("evaluation_result")

        if evaluation_result is None:
            # Pipeline failed before evaluation
            return {
                "sample_id": sample_id,
                "file_path": final_state.get("file_path", "unknown"),
                "overall_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "summary": "Pipeline failed before evaluation",
                "timestamp": "",
                "git_sha": final_state.get("commit_sha", "unknown"),
                "detected_smells_count": 0,
                "ground_truth_smells_count": len(final_state.get("ground_truth", [])),
                "error": final_state.get("error", "Unknown error"),
            }

        # Convert to dictionary
        result_dict = {
            "sample_id": evaluation_result.sample_id,
            "file_path": evaluation_result.file_path,
            "overall_score": evaluation_result.overall_score,
            "precision": evaluation_result.precision,
            "recall": evaluation_result.recall,
            "f1_score": evaluation_result.f1_score,
            "summary": evaluation_result.summary,
            "timestamp": evaluation_result.timestamp,
            "git_sha": evaluation_result.git_sha,
            "detected_smells_count": len(final_state.get("llm_detections", [])),
            "ground_truth_smells_count": len(
                [gt for gt in final_state.get("ground_truth", []) if gt.is_present]
            ),
            "error": final_state.get("error"),
        }

        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Sample ID: {result_dict['sample_id']}")
        print(f"File: {result_dict['file_path']}")
        print(f"Overall Score: {result_dict['overall_score']:.2f}/5.0")
        print(f"Precision: {result_dict['precision']:.2f}")
        print(f"Recall: {result_dict['recall']:.2f}")
        print(f"F1 Score: {result_dict['f1_score']:.2f}")
        print(f"Detected: {result_dict['detected_smells_count']} smells")
        print(f"Ground Truth: {result_dict['ground_truth_smells_count']} smells")
        if result_dict["error"]:
            print(f"Error: {result_dict['error']}")
        print(f"{'='*60}\n")

        return result_dict

    except Exception as e:
        print(f"\n✗ Pipeline execution failed: {e}")

        # Return error result
        import traceback
        from datetime import datetime

        return {
            "sample_id": sample_id,
            "file_path": "unknown",
            "overall_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "summary": f"Pipeline execution failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "git_sha": "unknown",
            "detected_smells_count": 0,
            "ground_truth_smells_count": 0,
            "error": f"{str(e)}\n{traceback.format_exc()}",
        }


# CLI interface for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python evaluation_pipeline.py <sample_id>")
        print("Example: python evaluation_pipeline.py 12345")
        sys.exit(1)

    sample_id = int(sys.argv[1])
    result = run_evaluation(sample_id)

    # Pretty print result
    print("\nFinal Result (JSON):")
    print(json.dumps(result, indent=2))
