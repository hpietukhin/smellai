"""LangGraph agent for Java test analysis.

This module provides a LangGraph agent that can analyze Java projects,
run tests, and report on test failures. The agent uses tools to detect
build systems, execute tests, and parse results.

# TODO SPEC-001: Implement test generation capabilities for methods without test coverage.
# Agent 3 (Test Generation Agent) is currently a placeholder.
# Need to implement test generation for uncovered methods.
# MEDIUM priority.
# (See TECHNICAL_SPECIFICATION.md §3.2)

# TODO SPEC-002: Implement behavior preservation checks beyond test execution.
# Agent 6 (Verification Agent) currently reuses Agent 2's test execution.
# Need to implement additional behavior preservation checks beyond running tests.
# MEDIUM priority.
# (See TECHNICAL_SPECIFICATION.md §3.2)
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_litellm import ChatLiteLLM

from agents.tools.java_test_tools import get_java_test_tools
from agents.java_test.config import DEFAULT_CONFIG, JavaTestAgentConfig


class JavaTestState(TypedDict):
    """State for Java test analysis agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    project_path: str
    build_system: str | None


def create_java_test_agent() -> StateGraph:
    """Create a LangGraph agent for Java test analysis.

    The agent is configurable at runtime via RunnableConfig.
    Supported configuration keys:
    - model_name: Name of the LLM model to use (default: gpt-4o-mini)

    Returns:
        Compiled LangGraph StateGraph
    """
    # Initialize tools
    tools = get_java_test_tools()

    # Define agent node
    def agent(state: JavaTestState, config: RunnableConfig) -> dict:
        """Agent node that calls LLM with tools."""
        # Get configuration
        configurable = config.get("configurable", {})
        model_name = configurable.get(
            JavaTestAgentConfig.MODEL_NAME,
            DEFAULT_CONFIG[JavaTestAgentConfig.MODEL_NAME],
        )

        # Initialize LLM using LiteLLM
        llm = ChatLiteLLM(model=model_name)
        llm_with_tools = llm.bind_tools(tools)

        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Create graph
    graph_builder = StateGraph(JavaTestState)

    # Add nodes
    graph_builder.add_node("agent", agent)
    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)

    # Add edges
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
    )
    graph_builder.add_edge("tools", "agent")

    # Compile
    return graph_builder.compile()


def analyze_java_tests(
    project_path: str,
    *,
    model_name: str = DEFAULT_CONFIG[JavaTestAgentConfig.MODEL_NAME],
) -> dict:
    """Analyze Java tests in a project using the agent.

    Args:
        project_path: Path to the Java project
        model_name: LLM model to use

    Returns:
        Dictionary with analysis results
    """
    agent = create_java_test_agent()

    # Initial message to agent
    initial_message = {
        "role": "user",
        "content": f"""Analyze the Java tests in the project at: {project_path}

Please:
1. Detect the build system (Maven or Gradle)
2. Run the tests
3. Report on which tests passed and which failed
4. For any failed tests, provide details about the failures

Be concise but thorough in your analysis.""",
    }

    # Run agent with configuration
    result = agent.invoke(
        {
            "messages": [initial_message],
            "project_path": project_path,
            "build_system": None,
        },
        config={"configurable": {JavaTestAgentConfig.MODEL_NAME: model_name}},
    )

    # Extract final response
    final_message = result["messages"][-1]

    return {
        "project_path": project_path,
        "response": final_message.content,
        "messages": result["messages"],
    }
