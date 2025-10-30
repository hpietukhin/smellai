gg"""
LangGraph nodes for code smell detection evaluation pipeline.

This module defines the state schema and node functions for the evaluation pipeline.
Each node performs a specific step in the process and updates the shared state.

**Architecture**: LangGraph StateGraph with TypedDict state
**Nodes**: fetch_sample → clone_repo → detect_smells → judge_evaluation
"""

from datetime import datetime
from typing import List, Optional, TypedDict

from src.agents.detector import detect_smells as detect_smells_agent
from src.agents.judge import evaluate_detections
from src.data.git_ops import (clone_and_read_file, derive_repo_url,
                              get_commit_before_date)
from src.data.mysql_connector import fetch_sample_by_id
from src.models.entities import (EvaluationResult, SmellAnnotation,
                                 SmellDetection)


class EvaluationState(TypedDict):
    """
    State schema for the evaluation pipeline.

    This TypedDict defines all fields that nodes can read from and write to
    during pipeline execution. LangGraph automatically manages state updates.

    **Task**: P4-PIPELINE-001 - Create LangGraph state definition
    """
    # Input
    sample_id: int

    # Sample metadata (from database)
    file_path: Optional[str]
    project_name: Optional[str]
    repo_url: Optional[str]
    commit_sha: Optional[str]
    ground_truth: Optional[List[SmellAnnotation]]

    # Code content (from git)
    file_content: Optional[str]

    # Detection results (from LLM)
    llm_detections: Optional[List[SmellDetection]]

    # Evaluation results (from judge)
    evaluation_result: Optional[EvaluationResult]

    # Error tracking
    error: Optional[str]


def fetch_sample_node(state: EvaluationState) -> EvaluationState:
    """
    Fetch sample data from DACOS database and derive repository information.

    This node:
    1. Fetches sample record from MySQL
    2. Derives GitHub repository URL from project name
    3. Gets commit SHA before cutoff date (2023-01-24)
    4. Extracts ground truth annotations

    **Task**: P4-PIPELINE-002 - Create fetch_sample node

    Args:
        state: Current pipeline state with sample_id

    Returns:
        Updated state with file_path, repo_url, commit_sha, ground_truth

    Example:
        >>> state = {"sample_id": 12345}
        >>> new_state = fetch_sample_node(state)
        >>> print(new_state["repo_url"])
        'https://github.com/alibaba/arthas'
    """
    print(f"\n=== Fetching Sample {state['sample_id']} ===")

    try:
        # Fetch sample from database
        sample = fetch_sample_by_id(state['sample_id'])

        if sample is None:
            return {
                **state,
                'error': f"Sample {state['sample_id']} not found in database"
            }

        print(f"✓ Sample found: {sample.project_name}")
        print(f"  Path: {sample.path_to_file}")
        print(f"  Has smell: {sample.has_smell}")

        # Derive repository URL
        repo_url = derive_repo_url(sample.project_name)
        print(f"✓ Derived repo URL: {repo_url}")

        # Get commit SHA before cutoff date
        commit_sha = get_commit_before_date(
            repo_url,
            before_date="2023-01-24"
        )

        if commit_sha is None:
            return {
                **state,
                'error': f"No commits found before 2023-01-24 for {repo_url}"
            }

        print(f"✓ Found commit: {commit_sha[:8]}")

        # Extract ground truth annotations
        ground_truth = sample.to_annotations()
        active_smells = [ann for ann in ground_truth if ann.is_present]

        print(f"✓ Ground truth: {len(active_smells)} active smells")
        for smell in active_smells:
            print(f"  - {smell.smell_type}")

        # Update state
        return {
            **state,
            'file_path': sample.path_to_file,
            'project_name': sample.project_name,
            'repo_url': repo_url,
            'commit_sha': commit_sha,
            'ground_truth': ground_truth,
            'error': None
        }

    except Exception as e:
        print(f"✗ Error fetching sample: {e}")
        return {
            **state,
            'error': f"Failed to fetch sample: {str(e)}"
        }


def clone_repo_node(state: EvaluationState) -> EvaluationState:
    """
    Clone repository and read target file content.

    This node:
    1. Uses sparse checkout to efficiently retrieve single file
    2. Checks out specific commit SHA
    3. Reads file content as string

    **Task**: P4-PIPELINE-003 - Create clone_repo node

    Args:
        state: Current pipeline state with repo_url, commit_sha, file_path

    Returns:
        Updated state with file_content

    Example:
        >>> state = {
        ...     "repo_url": "https://github.com/alibaba/arthas",
        ...     "commit_sha": "abc123",
        ...     "file_path": "src/main/java/Example.java"
        ... }
        >>> new_state = clone_repo_node(state)
        >>> print(len(new_state["file_content"]))
        1234
    """
    print(f"\n=== Cloning Repository ===")

    # Check for errors from previous nodes
    if state.get('error'):
        print(f"⊘ Skipping due to previous error: {state['error']}")
        return state

    try:
        # Clone and read file
        file_content = clone_and_read_file(
            repo_url=state['repo_url'],
            commit_sha=state['commit_sha'],
            file_path=state['file_path']
        )

        print(f"✓ File read: {len(file_content)} characters")
        print(f"  First 100 chars: {file_content[:100]}...")

        # Update state
        return {
            **state,
            'file_content': file_content,
            'error': None
        }

    except Exception as e:
        print(f"✗ Error cloning repository: {e}")
        return {
            **state,
            'error': f"Failed to clone repository: {str(e)}"
        }


def detect_smells_node(state: EvaluationState) -> EvaluationState:
    """
    Detect code smells using LLM with RAG.

    This node:
    1. Retrieves relevant smell documentation (RAG)
    2. Sends code + context to Cerebras LLM
    3. Parses structured output to SmellDetection objects

    **Task**: P4-PIPELINE-004 - Create detect_smells node
    **Note**: Requires retriever to be initialized externally and passed via closure

    Args:
        state: Current pipeline state with file_content

    Returns:
        Updated state with llm_detections

    Example:
        >>> state = {"file_content": "public class Example { ... }"}
        >>> new_state = detect_smells_node(state)
        >>> print(f"Detected {len(new_state['llm_detections'])} smells")
    """
    print(f"\n=== Detecting Code Smells ===")

    # Check for errors from previous nodes
    if state.get('error'):
        print(f"⊘ Skipping due to previous error: {state['error']}")
        return state

    try:
        # Import retriever (should be initialized globally or passed in)
        from src.pipelines.evaluation_pipeline import get_retriever_instance
        retriever = get_retriever_instance()

        # Detect smells
        detections = detect_smells_agent(
            code=state['file_content'],
            retriever=retriever
        )

        print(f"✓ Detected {len(detections)} smells")
        for detection in detections:
            print(f"  - {detection.smell_type} ({detection.severity}) at {detection.location}")

        # Update state
        return {
            **state,
            'llm_detections': detections,
            'error': None
        }

    except Exception as e:
        print(f"✗ Error detecting smells: {e}")
        return {
            **state,
            'llm_detections': [],
            'error': f"Failed to detect smells: {str(e)}"
        }


def judge_evaluation_node(state: EvaluationState) -> EvaluationState:
    """
    Evaluate detection quality using LLM-as-judge.

    This node:
    1. Compares LLM detections against ground truth
    2. Assigns scores using 5-level rubric
    3. Calculates precision, recall, F1 metrics
    4. Returns comprehensive EvaluationResult

    **Task**: P4-PIPELINE-005 - Create judge_evaluation node

    Args:
        state: Current pipeline state with ground_truth and llm_detections

    Returns:
        Updated state with evaluation_result

    Example:
        >>> state = {
        ...     "ground_truth": [...],
        ...     "llm_detections": [...],
        ...     "sample_id": 123
        ... }
        >>> new_state = judge_evaluation_node(state)
        >>> print(f"F1 Score: {new_state['evaluation_result'].f1_score}")
    """
    print(f"\n=== Evaluating Detection Quality ===")

    # Check for errors from previous nodes
    if state.get('error'):
        print(f"⊘ Skipping due to previous error: {state['error']}")

        # Create error evaluation result
        from datetime import datetime
        error_result = EvaluationResult(
            sample_id=state['sample_id'],
            file_path=state.get('file_path', 'unknown'),
            overall_score=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            evaluations=[],
            summary=f"Evaluation skipped: {state['error']}",
            timestamp=datetime.now().isoformat(),
            git_sha=state.get('commit_sha', 'unknown')
        )

        return {
            **state,
            'evaluation_result': error_result
        }

    try:
        # Evaluate detections
        evaluation_result = evaluate_detections(
            ground_truth=state['ground_truth'],
            detected_smells=state['llm_detections'],
            sample_id=state['sample_id'],
            file_path=state['file_path'],
            git_sha=state['commit_sha']
        )

        print(f"✓ Evaluation complete")
        print(f"  Overall Score: {evaluation_result.overall_score:.2f}/5.0")
        print(f"  Precision: {evaluation_result.precision:.2f}")
        print(f"  Recall: {evaluation_result.recall:.2f}")
        print(f"  F1: {evaluation_result.f1_score:.2f}")

        # Update state
        return {
            **state,
            'evaluation_result': evaluation_result,
            'error': None
        }

    except Exception as e:
        print(f"✗ Error in evaluation: {e}")

        # Create error evaluation result
        from datetime import datetime
        error_result = EvaluationResult(
            sample_id=state['sample_id'],
            file_path=state.get('file_path', 'unknown'),
            overall_score=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            evaluations=[],
            summary=f"Evaluation failed: {str(e)}",
            timestamp=datetime.now().isoformat(),
            git_sha=state.get('commit_sha', 'unknown')
        )

        return {
            **state,
            'evaluation_result': error_result,
            'error': f"Failed to evaluate: {str(e)}"
        }
