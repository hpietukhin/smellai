"""LangGraph agent for SWE-Refactor evaluation workflow.

Workflow: A0 (setup) → A5 (generate) → A6 (verify)
"""

import logging
import re
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from swe_refactor.dataset import RefactoringRecord
from swe_refactor.utils import (
    clone_repository,
    compile_project,
    force_checkout_commit,
    get_previous_commit,
    get_repo_url,
    replace_java_code,
    switch_java_version,
)
from agents.tools.java_test_tools import detect_build_system, run_tests
from agents.swe_eval.config import DEFAULT_CONFIG, SWEEvalAgentConfig
from agents.swe_eval.prompts import SYSTEM_PROMPT, get_refactoring_prompt

LOGGER = logging.getLogger(__name__)


class SWEEvalState(TypedDict):
    """State for SWE evaluation agent."""

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


def create_swe_eval_agent(model_name: str | None = None) -> StateGraph:
    """Create LangGraph agent for SWE-Refactor evaluation.

    Args:
        model_name: LLM model to use. If None, uses default from config.

    Returns:
        Compiled LangGraph StateGraph
    """
    if model_name is None:
        model_name = DEFAULT_CONFIG[SWEEvalAgentConfig.MODEL_NAME]

    model = ChatLiteLLM(model=model_name)

    def a0_setup(state: SWEEvalState) -> dict:
        """A0: Setup - Clone repo, checkout parent commit, switch JDK."""
        record = state["record"]
        workspace_path = state["workspace_path"]

        LOGGER.info(
            "A0: Setting up workspace for %s @ %s",
            record.projectName,
            record.commitId[:8],
        )

        try:
            repo_url = get_repo_url(record.projectName)
        except KeyError:
            error_msg = f"Unknown project: {record.projectName}"
            LOGGER.error(error_msg)
            return {
                "error_message": error_msg,
                "project_path": workspace_path / record.projectName,
            }

        project_path = workspace_path / record.projectName

        if not project_path.exists():
            success = clone_repository(repo_url, project_path)
            if not success:
                return {
                    "error_message": f"Failed to clone {repo_url}",
                    "project_path": project_path,
                }

        parent_commit = get_previous_commit(project_path, record.commitId)
        if not parent_commit:
            return {
                "error_message": f"Failed to get parent of {record.commitId}",
                "project_path": project_path,
            }

        success = force_checkout_commit(project_path, parent_commit)
        if not success:
            return {
                "error_message": f"Failed to checkout {parent_commit}",
                "project_path": project_path,
            }

        success = switch_java_version(record.compileJDK, project_path)
        if not success:
            LOGGER.warning("Failed to switch Java version to %d", record.compileJDK)

        LOGGER.info(
            "A0: Setup complete. Project at %s, commit %s",
            project_path,
            parent_commit[:8],
        )

        return {
            "project_path": project_path,
            "error_message": None,
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

        source_file = project_path / record.filePathBefore
        success = replace_java_code(source_file, refactored_code)

        if not success:
            return {
                "error_message": f"Failed to write {source_file}",
                "compile_success": False,
                "test_success": False,
            }

        if state.get("refactored_target_code"):
            target_file = project_path / record.filePathAfter
            replace_java_code(target_file, state["refactored_target_code"])

        compile_result = compile_project(project_path, record.compileCommand)

        if not compile_result.success:
            error_summary = "\n".join(
                compile_result.error_summary or ["Unknown compile error"]
            )
            LOGGER.warning("A6: Compilation failed:\n%s", error_summary)
            return {
                "compile_success": False,
                "test_success": False,
                "error_message": error_summary,
            }

        LOGGER.info("A6: Compilation succeeded")

        test_success = True
        if record.hasTestC:
            build_system = detect_build_system(str(project_path))
            if build_system:
                test_result = run_tests(str(project_path), build_system)
                test_success = test_result.success

                if not test_success:
                    LOGGER.warning("A6: Tests failed (%d failures)", test_result.failed)
                else:
                    LOGGER.info("A6: Tests passed (%d tests)", test_result.total)

        return {
            "compile_success": True,
            "test_success": test_success,
            "error_message": None,
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

    workflow = StateGraph(SWEEvalState)

    workflow.add_node("a0_setup", a0_setup)
    workflow.add_node("a5_generate", a5_generate)
    workflow.add_node("a6_verify", a6_verify)

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
    record: RefactoringRecord,
    workspace_path: str | Path,
) -> dict:
    """Invoke agent for single refactoring record.

    Args:
        agent: Compiled LangGraph agent
        record: Refactoring record to process
        workspace_path: Base workspace directory

    Returns:
        Dictionary with evaluation results
    """
    workspace_path = Path(workspace_path)
    workspace_path.mkdir(parents=True, exist_ok=True)

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
        }
    )

    return {
        "project": record.projectName,
        "commit": record.commitId,
        "type": record.type,
        "compile_success": result.get("compile_success", False),
        "test_success": result.get("test_success", False),
        "error": result.get("error_message"),
    }
