"""LangGraph agent for SWE-Refactor evaluation workflow.

Workflow (basic): A0 (setup) → A5 (generate) → A6 (verify)
Workflow (composite): A0 → A1 → A2 → A3 → [A4 → A5 → A6] (loop) → END
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, List, Literal, Optional, TypedDict

import networkx as nx
from langchain_core.messages import BaseMessage
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.dependency_analysis.agent import build_smell_graph_from_events
from swe_refactor.adapters import sample_to_refactoring_record
from swe_refactor.dataset import RefactoringRecord
from smellai_datasets.schema import EvalSample
from domain.models import SmellEvent
from swe_refactor.persistence.models import SmellEventRecord
from swe_refactor.runtime import setup_project_workspace, verify_refactoring
from agents.swe_eval.config import DEFAULT_CONFIG, SWEEvalAgentConfig
from agents.swe_eval.prompts import SYSTEM_PROMPT, get_refactoring_prompt
from domain.detector import SmellDetectionError, SmellDetector
from sonarqube.detector import SonarQubeDetector

if TYPE_CHECKING:
    from swe_refactor.persistence.database import AnalyticsDB

LOGGER = logging.getLogger(__name__)
EXTRACT_METHOD = "Extract Method"


def _refactoring_outcome(compile_success: bool, test_success: bool) -> str:
    if not compile_success:
        return "compile_failed"
    if not test_success:
        return "test_failed"
    return "success"


class SWEEvalState(TypedDict):
    """State for SWE evaluation agent.

    Supports both basic (single refactoring) and composite (iterative) modes.
    """

    # ===== EXISTING FIELDS (basic mode) =====
    messages: Annotated[list[BaseMessage], add_messages]
    record: RefactoringRecord
    workspace_path: Path
    project_path: Path
    refactored_code: str | None
    refactored_target_code: str | None
    compile_success: bool
    test_success: bool
    retry_count: int
    error_message: str | None

    # ===== NEW: Composite refactoring fields =====
    # Smell detection (A1)
    detected_smells: List[SmellEvent]  # Smells in current iteration

    # Prioritization (A2)
    smell_graph: Optional[nx.DiGraph]  # Dependency graph
    priority_queue: List[str]  # Sorted smell_ids (highest PZ first)

    # Selection (A3)
    current_smell: Optional[str]  # smell_id currently being refactored

    # Mapping (A4)
    refactoring_type: Optional[str]  # Mapped refactoring type for current smell

    # Loop control
    refactoring_iteration: int  # 0, 1, 2, ...
    max_refactorings: int  # N-action limit (default: 5)

    # Metrics
    initial_smells: List[SmellEvent]  # Snapshot after A1 (iteration 0)
    smells_resolved_count: int
    smells_created_count: int
    refactoring_history: List[dict]  # [{iteration, smell, type, outcome}, ...]

    # Persistence
    session_id: str  # thread_id from LangGraph
    analytics_db: Optional[AnalyticsDB]  # SQLModel database instance

    # Smell detection backend config
    smell_detector: SmellDetector
    sonar_url: str  # Default: "http://localhost:9000"
    sonar_cache_dir: Optional[str]  # Default: "./sonar_cache"

    # Token tracking
    total_tokens: int
    tokens_by_node: dict  # {node_name: token_count}


def create_swe_eval_agent(
    model_name: str | None = None, enable_composite: bool = False
) -> StateGraph:
    """Create LangGraph agent for SWE-Refactor evaluation.

    Args:
        model_name: LLM model to use. If None, uses default from config.
        enable_composite: If True, enables composite refactoring mode (A1-A6 loop).

    Returns:
        Compiled LangGraph StateGraph
    """
    model_name = model_name or DEFAULT_CONFIG[SWEEvalAgentConfig.MODEL_NAME]
    model = ChatLiteLLM(model=model_name)

    def get_smell_detector(state: SWEEvalState) -> SmellDetector:
        """Return the injected detector or a default SonarQube backend."""
        detector = state.get("smell_detector")
        if detector is not None:
            return detector
        return SonarQubeDetector(sonar_url=state.get("sonar_url", "http://localhost:9000"))

    def a0_setup(state: SWEEvalState) -> dict:
        """A0: Setup - Clone repo, checkout parent commit, switch JDK."""
        record = state["record"]
        workspace_path = state["workspace_path"]

        LOGGER.info(
            "A0: Setting up workspace for %s @ %s",
            record.projectName,
            record.commitId[:8],
        )

        setup = setup_project_workspace(record, workspace_path)
        return {
            "project_path": setup.project_path,
            "error_message": setup.error,
        }

    def a5_generate(state: SWEEvalState) -> dict:
        """A5: Generate - LLM generates refactored code."""
        record = state["record"]

        LOGGER.info("A5: Generating refactoring (%s)", record.type)

        target_code = None
        if record.filePathBefore != record.filePathAfter:
            target_file = state["project_path"] / record.filePathAfter
            if target_file.exists():
                try:
                    target_code = target_file.read_text(encoding="utf-8")
                except Exception as e:
                    LOGGER.warning("Failed to read target file %s: %s", target_file, e)

        prompt = get_refactoring_prompt(
            record.type,
            record.sourceCodeBeforeForWhole,
            target_code,
            record.filePathBefore,
            record.filePathAfter,
        )

        # Add smell context if in composite mode
        if current_smell := state.get("current_smell"):
            refactoring_type = state.get("refactoring_type", EXTRACT_METHOD)
            smell_context = f"\n\n## Target Code Smell\n{current_smell}\n\n## Refactoring Type\n{refactoring_type}\n\nPlease focus on resolving this specific smell while maintaining code correctness."
            prompt += smell_context

        if state.get("retry_count", 0) > 0:
            error_msg = state.get("error_message", "")
            prompt += f"\n\n## Previous Attempt Failed\n\nCompilation error:\n{error_msg}\n\nPlease fix and try again."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = model.invoke(messages)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Track token usage in composite mode
            if analytics_db := state.get("analytics_db"):
                from swe_refactor.persistence.models import TokenUsage

                if hasattr(response, "response_metadata"):
                    usage = response.response_metadata.get("token_usage", {})
                    if usage:
                        token_record = TokenUsage(
                            session_id=state.get("session_id", "unknown"),
                            iteration=state.get("refactoring_iteration", 0),
                            node_name="A5_generate",
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                            total_tokens=usage.get("total_tokens", 0),
                            model=model_name or "unknown",
                        )
                        analytics_db.log_token_usage(token_record)
        except Exception as e:
            LOGGER.error("LLM invocation failed: %s", e)
            return {
                "error_message": f"LLM error: {str(e)}",
                "refactored_code": None,
            }

        refactored_code, refactored_target = _extract_code_from_response(
            response_text,
            record.filePathBefore,
            record.filePathAfter,
        )

        if not refactored_code:
            return {
                "error_message": "Failed to extract code from LLM response",
                "refactored_code": None,
            }

        LOGGER.info("A5: Generated %d chars of refactored code", len(refactored_code))

        return {
            "refactored_code": refactored_code,
            "refactored_target_code": refactored_target,
            "messages": [response],
        }

    def a6_verify(state: SWEEvalState) -> dict:
        """A6: Verify - Compile and test refactored code."""
        record = state["record"]
        project_path = state["project_path"]
        refactored_code = state["refactored_code"]

        LOGGER.info("A6: Verifying refactored code")

        verification = verify_refactoring(
            record,
            project_path,
            refactored_code=refactored_code,
            refactored_target_code=state.get("refactored_target_code"),
        )

        if not verification.compile_success:
            return {
                "compile_success": False,
                "test_success": False,
                "error_message": verification.error,
            }

        test_success = verification.test_success

        # Re-scan for smells after successful compilation (composite mode)
        after_smells = []
        diff = {"resolved": [], "created": [], "persisted": []}

        if verification.compile_success and state.get("analytics_db"):
            detector = get_smell_detector(state)

            try:
                after_smells = detector.detect(Path(project_path))

                before_smells = state.get("detected_smells", [])
                diff = SmellDetector.compare(before_smells, after_smells)

                LOGGER.info(
                    "A6: Smell diff - resolved: %d, created: %d, persisted: %d",
                    len(diff["resolved"]),
                    len(diff["created"]),
                    len(diff["persisted"]),
                )
            except SmellDetectionError as e:
                LOGGER.warning("A6: Failed to re-scan smells: %s", e)

        # Log refactoring attempt (composite mode)
        if state.get("analytics_db"):
            from swe_refactor.persistence.models import RefactoringAttempt

            analytics_db = state["analytics_db"]

            # Capture git diff if compilation was successful
            code_diff = None
            if verification.compile_success:
                try:
                    from git import Repo

                    repo = Repo(project_path)
                    # Get diff between HEAD and working tree (uncommitted changes)
                    code_diff = repo.git.diff("HEAD")
                except Exception as e:
                    LOGGER.warning("A6: Failed to capture git diff: %s", e)

            attempt = RefactoringAttempt(
                session_id=state.get("session_id", "unknown"),
                iteration=state.get("refactoring_iteration", 0),
                smell_id=state.get("current_smell", "unknown"),
                refactoring_type=state.get("refactoring_type", EXTRACT_METHOD),
                outcome=_refactoring_outcome(verification.compile_success, test_success),
                retries=state.get("retry_count", 0),
                smells_resolved=len(diff["resolved"]),
                smells_created=len(diff["created"]),
                code_diff=code_diff,
            )
            analytics_db.log_refactoring_attempt(attempt)

        return {
            "compile_success": True,
            "test_success": test_success,
            "error_message": None,
            "detected_smells": after_smells
            if after_smells
            else state.get("detected_smells", []),
            "smells_resolved_count": state.get("smells_resolved_count", 0)
            + len(diff["resolved"]),
            "smells_created_count": state.get("smells_created_count", 0)
            + len(diff["created"]),
        }

    def should_retry(state: SWEEvalState) -> Literal["retry", "end"]:
        """Decide whether to retry generation after failure."""
        max_retries = DEFAULT_CONFIG[SWEEvalAgentConfig.MAX_RETRIES]

        if state.get("compile_success", False):
            return "end"

        if state.get("retry_count", 0) >= max_retries:
            LOGGER.warning("Max retries (%d) reached", max_retries)
            return "end"

        return "retry"

    # ===== NEW: Composite refactoring nodes =====

    def a1_detect_smells(state: SWEEvalState) -> dict:
        """A1: Detect code smells through the configured detector backend."""
        record = state["record"]
        project_path = state["project_path"]
        session_id = state.get("session_id", "unknown")
        iteration = state.get("refactoring_iteration", 0)
        detector = get_smell_detector(state)

        LOGGER.info("A1: Detecting smells (iteration %d)", iteration)
        LOGGER.debug("A1: Using smell detector %s for %s", detector.__class__.__name__, record.projectName)

        try:
            detected_smells = detector.detect(Path(project_path))
        except SmellDetectionError as e:
            LOGGER.error("A1: Smell detection failed: %s", e)
            return {
                "detected_smells": [],
                "initial_smells": state.get("initial_smells", []),
                "error_message": f"Smell detection failed: {e}",
            }

        LOGGER.info("A1: Detected %d smells", len(detected_smells))

        # Log to analytics DB if available
        if analytics_db := state.get("analytics_db"):
            for smell in detected_smells:
                record = SmellEventRecord.from_domain(smell, session_id=session_id, iteration=iteration)
                analytics_db.log_smell_event(record)

        # Save initial snapshot (iteration 0)
        initial_smells = (
            detected_smells if iteration == 0 else state.get("initial_smells", [])
        )

        return {
            "detected_smells": detected_smells,
            "initial_smells": initial_smells,
        }

    def a2_prioritize_smells(state: SWEEvalState) -> dict:
        """A2: Prioritize smells using the canonical domain smell graph."""
        detected_smells = state.get("detected_smells", [])

        if not detected_smells:
            LOGGER.warning("A2: No smells to prioritize")
            return {"priority_queue": [], "smell_graph": None}

        LOGGER.info("A2: Prioritizing %d smells", len(detected_smells))

        smell_graph = build_smell_graph_from_events(detected_smells)
        priority_sequence = smell_graph.calculate_priorities()

        LOGGER.info(
            "A2: Priority queue: %s",
            [
                f"{item['smell_type']}:{item['location']}"
                for item in priority_sequence[:3]
            ],
        )

        priority_ids = [item["smell_id"] for item in priority_sequence]

        return {
            "priority_queue": priority_ids,
            "smell_graph": smell_graph.graph,
        }

    def a3_select_next_smell(state: SWEEvalState) -> dict:
        """A3: Select next smell from priority queue."""
        priority_queue = state.get("priority_queue", [])
        iteration = state.get("refactoring_iteration", 0)
        max_refactorings = state.get("max_refactorings", 5)

        if iteration >= max_refactorings:
            LOGGER.info("A3: Max iterations (%d) reached", max_refactorings)
            return {"current_smell": None}

        if not priority_queue:
            LOGGER.info("A3: No more smells to refactor")
            return {"current_smell": None}

        current_smell = priority_queue[0]
        remaining_queue = priority_queue[1:]

        LOGGER.info("A3: Selected smell %s (iteration %d)", current_smell, iteration)

        return {
            "current_smell": current_smell,
            "priority_queue": remaining_queue,
        }

    def a4_map_smell_to_refactoring(state: SWEEvalState) -> dict:
        """A4: Map smell to refactoring type using LLM."""
        current_smell = state.get("current_smell")

        if not current_smell:
            return {"refactoring_type": None}

        # Parse smell_id: {type}:{file}:{line}
        parts = current_smell.split(":", 2)
        if len(parts) != 3:
            LOGGER.warning("A4: Invalid smell_id format: %s", current_smell)
            return {"refactoring_type": EXTRACT_METHOD}  # default fallback

        smell_type = parts[0]

        # Simple mapping (can be enhanced with LLM later)
        mapping = {
            "Long Method": EXTRACT_METHOD,
            "Complex Method": EXTRACT_METHOD,
            "God Class": "Extract Class",
            "Large Class": "Extract Class",
            "Long Parameter List": "Introduce Parameter Object",
            "Duplicated Conditions": EXTRACT_METHOD,
            "Conditional Complexity": "Replace Conditional with Polymorphism",
        }

        refactoring_type = mapping.get(smell_type, EXTRACT_METHOD)

        LOGGER.info("A4: Mapped %s → %s", smell_type, refactoring_type)

        return {"refactoring_type": refactoring_type}

    def increment_iteration(state: SWEEvalState) -> dict:
        """Increment refactoring iteration counter."""
        iteration = state.get("refactoring_iteration", 0)
        new_iteration = iteration + 1

        LOGGER.info("Incrementing iteration: %d → %d", iteration, new_iteration)

        return {
            "refactoring_iteration": new_iteration,
            "retry_count": 0,  # Reset retry count for new iteration
        }

    def should_continue_refactoring(state: SWEEvalState) -> Literal["continue", "end"]:
        """Decide whether to continue refactoring after A3."""
        if state.get("current_smell") is None:
            return "end"
        return "continue"

    def after_verify(state: SWEEvalState) -> Literal["retry", "next_iteration"]:
        """Decide what to do after A6 verification."""
        max_retries = DEFAULT_CONFIG[SWEEvalAgentConfig.MAX_RETRIES]

        # If compilation failed and we haven't exceeded retries, retry
        if not state.get("compile_success", False):
            if state.get("retry_count", 0) < max_retries:
                return "retry"

        # Otherwise move to next iteration
        return "next_iteration"

    workflow = StateGraph(SWEEvalState)

    # Add all nodes
    workflow.add_node("a0_setup", a0_setup)
    workflow.add_node("a5_generate", a5_generate)
    workflow.add_node("a6_verify", a6_verify)

    if enable_composite:
        # Add composite refactoring nodes
        workflow.add_node("a1_detect_smells", a1_detect_smells)
        workflow.add_node("a2_prioritize_smells", a2_prioritize_smells)
        workflow.add_node("a3_select_next_smell", a3_select_next_smell)
        workflow.add_node("a4_map_smell_to_refactoring", a4_map_smell_to_refactoring)
        workflow.add_node("increment_iteration", increment_iteration)

        # Composite mode workflow: A0 → A1 → A2 → A3 → [A4 → A5 → A6 → increment → A1] or END
        workflow.set_entry_point("a0_setup")
        workflow.add_edge("a0_setup", "a1_detect_smells")
        workflow.add_edge("a1_detect_smells", "a2_prioritize_smells")
        workflow.add_edge("a2_prioritize_smells", "a3_select_next_smell")

        # Conditional: continue to A4 or end
        workflow.add_conditional_edges(
            "a3_select_next_smell",
            should_continue_refactoring,
            {
                "continue": "a4_map_smell_to_refactoring",
                "end": END,
            },
        )

        workflow.add_edge("a4_map_smell_to_refactoring", "a5_generate")
        workflow.add_edge("a5_generate", "a6_verify")

        # Conditional: retry A5 or move to next iteration
        workflow.add_conditional_edges(
            "a6_verify",
            after_verify,
            {
                "retry": "a5_generate",
                "next_iteration": "increment_iteration",
            },
        )

        # Loop back to A1 for next smell
        workflow.add_edge("increment_iteration", "a1_detect_smells")
    else:
        # Basic mode workflow: A0 → A5 → A6 → [retry A5 or END]
        workflow.set_entry_point("a0_setup")
        workflow.add_edge("a0_setup", "a5_generate")
        workflow.add_edge("a5_generate", "a6_verify")

        workflow.add_conditional_edges(
            "a6_verify",
            should_retry,
            {
                "retry": "a5_generate",
                "end": END,
            },
        )

    return workflow.compile()


def _extract_code_from_response(
    response: str,
    source_file: str,
    target_file: str,
) -> tuple[str | None, str | None]:
    """Extract Java code from LLM response.

    Handles both single-file and multi-file responses.

    Args:
        response: LLM response text
        source_file: Expected source file path
        target_file: Expected target file path

    Returns:
        Tuple of (source_code, target_code). target_code is None for single-file refactorings.
    """
    if f"// FILE: {source_file}" in response and f"// FILE: {target_file}" in response:
        return _extract_multi_file(response, source_file, target_file)

    pattern = r"```java\s*(.*?)\s*```"
    matches = re.findall(pattern, response, re.DOTALL)

    if not matches:
        LOGGER.warning("No Java code block found in response")
        return None, None

    return matches[0].strip(), None


def _extract_multi_file(
    response: str,
    source_file: str,
    target_file: str,
) -> tuple[str | None, str | None]:
    """Extract source and target code from multi-file response."""
    source_pattern = rf"// FILE: {re.escape(source_file)}\s*```java\s*(.*?)\s*```"
    target_pattern = rf"// FILE: {re.escape(target_file)}\s*```java\s*(.*?)\s*```"

    source_match = re.search(source_pattern, response, re.DOTALL)
    target_match = re.search(target_pattern, response, re.DOTALL)

    source_code = source_match.group(1).strip() if source_match else None
    target_code = target_match.group(1).strip() if target_match else None

    return source_code, target_code


def invoke_agent(
    agent: StateGraph,
    sample: EvalSample,
    workspace_path: str | Path,
    analytics_db=None,
    max_refactorings: int = 5,
    sonar_url: str = "http://localhost:9000",
    sonar_cache_dir: str | None = None,
    smell_detector: SmellDetector | None = None,
) -> dict:
    """Invoke agent for a single SWE EvalSample.

    Args:
        agent: Compiled LangGraph agent
        sample: EvalSample with source="swe"
        workspace_path: Base workspace directory
        analytics_db: Optional AnalyticsDB instance for composite mode
        max_refactorings: Max refactoring iterations (N-action limit)
        sonar_url: SonarQube server URL
        sonar_cache_dir: SonarQube cache directory
        smell_detector: Optional injected detector backend

    Returns:
        Dictionary with evaluation results
    """
    if sample.source != "swe":
        raise ValueError(f"SWE agent expects source='swe', got {sample.source!r}")
    record = sample_to_refactoring_record(sample)
    workspace_path = Path(workspace_path)
    workspace_path.mkdir(parents=True, exist_ok=True)

    session_id = str(uuid.uuid4())
    smell_detector = smell_detector or SonarQubeDetector(sonar_url=sonar_url)

    result = agent.invoke(
        {
            "messages": [],
            "record": record,
            "workspace_path": workspace_path,
            "project_path": None,
            "refactored_code": None,
            "refactored_target_code": None,
            "compile_success": False,
            "test_success": False,
            "retry_count": 0,
            "error_message": None,
            # Composite mode fields
            "session_id": session_id,
            "analytics_db": analytics_db,
            "refactoring_iteration": 0,
            "max_refactorings": max_refactorings,
            "smell_detector": smell_detector,
            "sonar_url": sonar_url,
            "sonar_cache_dir": sonar_cache_dir,
            "detected_smells": [],
            "priority_queue": [],
            "initial_smells": [],
            "smells_resolved_count": 0,
            "smells_created_count": 0,
            "refactoring_history": [],
            "total_tokens": 0,
            "tokens_by_node": {},
            "current_smell": None,
            "refactoring_type": None,
            "smell_graph": None,
        }
    )

    output = {
        "project": record.projectName,
        "commit": record.commitId,
        "type": record.type,
        "compile_success": result.get("compile_success", False),
        "test_success": result.get("test_success", False),
        "error": result.get("error_message"),
    }

    # Add composite mode metrics
    if analytics_db:
        output["session_id"] = session_id
        output["smells_resolved"] = result.get("smells_resolved_count", 0)
        output["smells_created"] = result.get("smells_created_count", 0)
        output["total_tokens"] = result.get("total_tokens", 0)
        output["iterations"] = result.get("refactoring_iteration", 0)

        # Print summary
        summary = analytics_db.get_session_summary(session_id)
        print(f"\n=== Session {session_id[:8]} Summary ===")
        print(f"Iterations: {summary.get('total_iterations', 0)}")
        print(f"Successful refactorings: {summary.get('successful_refactorings', 0)}")
        print(f"Smells resolved: {summary.get('smells_resolved', 0)}")
        print(f"Smells created: {summary.get('smells_created', 0)}")
        print(f"Total tokens: {summary.get('total_tokens', 0)}")

    return output
