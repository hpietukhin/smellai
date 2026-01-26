# Java Test Analysis Agent

LangGraph-based agent for analyzing Java projects and running tests.

## Quick Start

```bash
# Run test analysis
uv run workflows/java_test_workflow.py --project /path/to/java/project

# With specific model
uv run workflows/java_test_workflow.py \
    --project /path/to/project \
    --model gpt-4o
```

## Visualization

Interactive web UI for analyzing multi-agent refactoring execution:

```bash
uv run python tools/visualize_smell_prioritization.py
# Open http://localhost:8080
```

See [../VISUALIZATION_USAGE.md](../VISUALIZATION_USAGE.md) for complete guide.

## Python API

```python
from agents.java_test_agent import analyze_java_tests

result = analyze_java_tests("/path/to/java/project")
print(result["response"])
```

## Documentation

See [docs/java_test_agent.md](../docs/java_test_agent.md) for full documentation.

## Components

- **tools/**: Test runner tools (Maven/Gradle)
- **java_test/agent.py**: LangGraph agent implementation
- **java_test/config.py**: Agent configuration
- **../workflows/java_test_workflow.py**: CLI workflow script

## Features

- ✅ Auto-detect Maven/Gradle build systems
- ✅ Run tests and parse XML reports
- ✅ LLM-powered failure analysis
- ✅ Support for OpenAI and Anthropic models
- ✅ Structured test result extraction
- ✅ JSON output for programmatic use
