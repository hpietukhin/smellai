#!/usr/bin/env python3
"""Example: Using the Java Test Analysis Agent.

This script demonstrates how to use the Java test agent to analyze
a Java project and report on test results.
"""

from agents.java_test.agent import analyze_java_tests, create_java_test_agent
from agents.java_test.config import JavaTestAgentConfig


def basic_example():
    """Basic usage example."""
    print("=== Basic Example ===\n")

    # Analyze a project (replace with actual path)
    project_path = "/path/to/your/java/project"

        result = analyze_java_tests(
        project_path,
        model_name="gpt-4o-mini",
    )

    print(f"Project: {result['project_path']}")
    print(f"\nAnalysis:\n{result['response']}")


def custom_agent_example():
    """Example using custom agent."""
    print("\n=== Custom Agent Example ===\n")

    # Create agent (configuration is passed at runtime)
    agent = create_java_test_agent()

    # Use with custom prompt and configuration
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": """Analyze the Java project at /path/to/project.
            
            Focus on:
            1. Identifying all failed tests
            2. Grouping failures by root cause
            3. Suggesting specific fixes
            """,
                }
            ],
            "project_path": "/path/to/project",
            "build_system": None,
        },
        config={"configurable": {JavaTestAgentConfig.MODEL_NAME: "gpt-4o-mini"}},
    )

    # Extract final response
    final_message = result["messages"][-1]
    print(f"Response: {final_message.content}")


def anthropic_example():
    """Example using Anthropic Claude via LiteLLM."""
    print("\n=== Anthropic Example ===\n")
    
    result = analyze_java_tests(
        "/path/to/project",
        model_name="claude-3-5-sonnet-20241022",
    )
    
    print(result["response"])
if __name__ == "__main__":
    # Run examples (uncomment the ones you want to try)

    # basic_example()
    # custom_agent_example()
    # anthropic_example()

    print("\nTo run examples, edit this file and uncomment the example you want.")
    print("Make sure to replace '/path/to/project' with an actual Java project path.")
