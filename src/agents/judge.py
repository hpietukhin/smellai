"""
LLM-as-judge evaluation agent for code smell detection quality.

This module implements the evaluation agent that assesses the quality of
smell detections against ground truth using an LLM as judge.

**Structured Output**: Uses Pydantic models (EvaluationResult) for type-safe evaluation responses
"""

import json
from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from src.agents.detector import get_llm_completion
from src.models.entities import EvaluationResult, SmellAnnotation, SmellDetection


def create_evaluation_prompt() -> tuple[PromptTemplate, PydanticOutputParser]:
    """
    Create prompt template for LLM-as-judge evaluation with rubric.

    Builds a comprehensive evaluation prompt that includes:
    - Expert evaluator role definition
    - 5-level rubric (EXCELLENT to INCORRECT)
    - Matching criteria for approximate locations
    - Format instructions for structured output

    **Task**: P3-JUDGE-001 - Create evaluation prompt template
    **Structured Output**: Uses PydanticOutputParser for EvaluationResult model

    Returns:
        Tuple of (PromptTemplate, PydanticOutputParser)

    Example:
        >>> prompt, parser = create_evaluation_prompt()
        >>> formatted = prompt.format(ground_truth="...", detected_smells="...")
        >>> response = get_llm_completion([{"role": "user", "content": formatted}])
        >>> result = parser.parse(response)
        >>> print(f"Overall score: {result.overall_score}/5.0")

    Notes:
        - Rubric provides clear scoring guidelines
        - Handles approximate location matching
        - Returns comprehensive metrics (precision, recall, F1)
    """
    # Create Pydantic output parser
    parser = PydanticOutputParser(pydantic_object=EvaluationResult)

    # Create evaluation prompt with rubric
    template = """You are an expert evaluator for code smell detection systems. Your task is to evaluate the quality of LLM-detected code smells against ground truth annotations.

**Evaluation Rubric** (5-point scale):

- **EXCELLENT (5 points)**: Smell type matches exactly, location is precise (same method/class), description is accurate and detailed
- **GOOD (4 points)**: Smell type matches, location is approximately correct (same class, close method), description is mostly accurate
- **ACCEPTABLE (3 points)**: Smell type matches but location is imprecise, or location matches but smell type is related/similar
- **POOR (2 points)**: Partial match - either smell type or general location is somewhat related but not accurate
- **INCORRECT (1 point)**: No meaningful match - wrong smell type and wrong location, or false positive

**Location Matching Criteria**:
- **Exact**: Same class and method name
- **Approximate**: Same class, similar method name or nearby methods
- **Class-level**: Same class but method unclear
- **File-level**: Same file but location unclear

**Ground Truth Annotations**:
```json
{ground_truth}
```

**Detected Smells**:
```json
{detected_smells}
```

**Evaluation Instructions**:
1. For each detected smell, find the best matching ground truth annotation
2. Assign a score (EXCELLENT, GOOD, ACCEPTABLE, POOR, INCORRECT)
3. Provide clear justification for each score
4. Calculate aggregate metrics:
   - **Precision**: (True Positives) / (True Positives + False Positives)
   - **Recall**: (True Positives) / (True Positives + False Negatives)
   - **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)
   - **Overall Score**: Average of all individual scores (0.0 to 5.0)
5. Provide a summary of detection quality

**Metrics Guidelines**:
- True Positive: EXCELLENT (5) or GOOD (4) scores
- Partial Match: ACCEPTABLE (3)
- False Positive: POOR (2) or INCORRECT (1) with no ground truth match
- False Negative: Ground truth smells not detected

{format_instructions}

Respond with valid JSON matching the schema above."""

    # Create prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["ground_truth", "detected_smells"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    return prompt, parser


def evaluate_detections(
    ground_truth: List[SmellAnnotation],
    detected_smells: List[SmellDetection],
    sample_id: int,
    file_path: str,
    git_sha: str,
    model: str = "cerebras/llama3.1-8b",
    temperature: float = 0.1,
) -> EvaluationResult:
    """
    Evaluate detection quality using LLM-as-judge.

    Implements comprehensive evaluation by:
    1. Formatting ground truth and detections as JSON
    2. Creating evaluation prompt with rubric
    3. Getting structured LLM evaluation
    4. Parsing and returning EvaluationResult

    **Task**: P3-JUDGE-002 - Create evaluation function
    **Structured Output**: Returns EvaluationResult Pydantic model

    Args:
        ground_truth: List of ground truth SmellAnnotation objects
        detected_smells: List of detected SmellDetection objects
        sample_id: DACOS sample ID
        file_path: Path to evaluated file
        git_sha: Git commit SHA
        model: Cerebras model to use
        temperature: LLM temperature (0.1 for slight variation)

    Returns:
        EvaluationResult object with scores and metrics

    Example:
        >>> ground_truth = [
        ...     SmellAnnotation(smell_type="Complex Method", is_present=True)
        ... ]
        >>> detected = [
        ...     SmellDetection(smell_type="Complex Method", location="UserService.process()", ...)
        ... ]
        >>> result = evaluate_detections(ground_truth, detected, 123, "path/to/file", "abc123")
        >>> print(f"Precision: {result.precision:.2f}")
        >>> print(f"Recall: {result.recall:.2f}")
        >>> print(f"F1: {result.f1_score:.2f}")

    Notes:
        - Handles cases with no detections (all false negatives)
        - Handles cases with no ground truth (all false positives)
        - Uses temperature=0.1 for consistent but not identical evaluations
        - Graceful error handling returns default low scores
    """
    try:
        # Convert inputs to JSON for LLM
        ground_truth_json = json.dumps(
            [gt.model_dump() for gt in ground_truth], indent=2
        )

        detected_smells_json = json.dumps(
            [ds.model_dump() for ds in detected_smells], indent=2
        )

        print("Creating evaluation prompt...")

        # Create prompt with ground truth and detections
        prompt_template, parser = create_evaluation_prompt()
        formatted_prompt = prompt_template.format(
            ground_truth=ground_truth_json, detected_smells=detected_smells_json
        )

        # Get LLM evaluation
        print(f"Evaluating with {model}...")
        messages = [
            {
                "role": "system",
                "content": "You are an expert evaluator. Always respond with valid JSON.",
            },
            {"role": "user", "content": formatted_prompt},
        ]

        response_text = get_llm_completion(
            messages, model=model, temperature=temperature, max_tokens=4096
        )

        print(f"✓ Received evaluation ({len(response_text)} characters)")

        # Parse structured output
        print("Parsing evaluation result...")
        result = parser.parse(response_text)

        # Add required fields
        result.sample_id = sample_id
        result.file_path = file_path
        result.git_sha = git_sha

        print("✓ Evaluation complete:")
        print(f"  Overall Score: {result.overall_score:.2f}/5.0")
        print(f"  Precision: {result.precision:.2f}")
        print(f"  Recall: {result.recall:.2f}")
        print(f"  F1 Score: {result.f1_score:.2f}")

        return result

    except Exception as e:
        print(f"⚠ Error in evaluation: {e}")
        print("  Returning default evaluation result")

        # Return fallback result on error
        from datetime import datetime

        return EvaluationResult(
            sample_id=sample_id,
            file_path=file_path,
            overall_score=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            evaluations=[],
            summary=f"Evaluation failed: {str(e)}",
            timestamp=datetime.now().isoformat(),
            git_sha=git_sha,
        )
