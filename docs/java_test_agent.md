# Java Test Analysis Agent

A LangGraph-based agent for analyzing Java projects, running tests, and reporting on test results. This agent automatically detects Maven or Gradle build systems, executes tests, and provides detailed analysis of test failures.

## Architecture

The Java test analysis system consists of three main components:

### 1. Test Runner Tools (`agents/tools/java_test_tools.py`)

Low-level tools for interacting with Java build systems:

- **`detect_java_build_system`**: Detects Maven or Gradle in a project
- **`run_java_tests`**: Executes tests and returns detailed results
- **`get_test_output`**: Retrieves stdout/stderr from test runs

These tools handle:
- Build system detection (pom.xml for Maven, build.gradle/build.gradle.kts for Gradle)
- Test execution with configurable timeout
- XML parsing of Surefire (Maven) and Gradle test reports
- Structured test result extraction (pass/fail, error messages, stack traces)

### 2. LangGraph Agent (`agents/java_test/agent.py`)

A ReAct-style agent that uses the test tools to analyze projects:

- **State Management**: Tracks messages, project path, and build system
- **Tool Calling**: Automatically selects and invokes appropriate tools
- **Multi-Step Reasoning**: Can detect build system, run tests, and analyze failures
- **Configuration**: Uses `agents/java_test/config.py` for defaults

The agent uses a StateGraph with:
- `agent` node: LLM with bound tools for decision-making
- `tools` node: Executes selected tools
- Conditional edges: Routes between agent and tools based on tool calls

### 3. Workflow Script (`workflows/java_test_workflow.py`)

Command-line interface for running the agent:

```bash
uv run workflows/java_test_workflow.py --project /path/to/project
```

## Quick Start

### Basic Usage

```bash
# Analyze a Java project
uv run workflows/java_test_workflow.py --project /path/to/java/project

# Use a different model
uv run workflows/java_test_workflow.py \
    --project /path/to/project \
    --model gpt-4

# Use Anthropic provider
uv run workflows/java_test_workflow.py \
    --project /path/to/project \
    --provider anthropic \
    --model claude-3-5-sonnet-20241022

# JSON output for programmatic use
uv run workflows/java_test_workflow.py \
    --project /path/to/project \
    --json
```

### Python API

```python
from agents.java_test.agent import analyze_java_tests

# Analyze tests
result = analyze_java_tests(
    "/path/to/java/project",
    model_name="gpt-4o-mini",
)

print(result["response"])
```

### Custom Agent Integration

```python
from agents.java_test.agent import create_java_test_agent
from agents.java_test.config import JavaTestAgentConfig

# Create agent
agent = create_java_test_agent()

# Run with custom prompt and config
result = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "Run tests and identify the root cause of failures"
        }],
        "project_path": "/path/to/project",
        "build_system": None
    },
    config={"configurable": {JavaTestAgentConfig.MODEL_NAME: "gpt-4o-mini"}}
)
```

## How It Works

### 1. Build System Detection

The agent first identifies the build system:

- Checks for `pom.xml` → Maven
- Checks for `build.gradle` or `build.gradle.kts` → Gradle
- Returns error if neither found

### 2. Test Execution

Runs appropriate command:

```bash
# Maven
mvn clean test

# Gradle
gradle clean test
```

Captures:
- Exit code
- stdout/stderr
- Test reports (XML)

### 3. Result Parsing

Parses XML test reports:

**Maven**: `target/surefire-reports/TEST-*.xml`
**Gradle**: `build/test-results/test/TEST-*.xml`

Extracts for each test:
- Test name (fully qualified)
- Status (PASS, FAIL, ERROR, SKIPPED)
- Duration
- Error message and type
- Stack trace

### 4. Agent Analysis

The LangGraph agent:

1. Calls `detect_java_build_system` tool
2. Calls `run_java_tests` tool
3. Analyzes results using LLM reasoning
4. Provides summary with:
   - Total tests, pass/fail counts
   - Failed test details
   - Potential root causes
   - Recommendations

## Test Result Structure

```python
@dataclass
class TestResult:
    name: str                      # e.g., "com.example.UserServiceTest.testLogin"
    status: Literal["PASS", "FAIL", "ERROR", "SKIPPED"]
    duration: float                # in seconds
    error_message: Optional[str]   # assertion message
    error_type: Optional[str]      # exception class
    failure_trace: Optional[str]   # stack trace

@dataclass
class TestRunSummary:
    build_system: Literal["maven", "gradle"]
    total: int
    passed: int
    failed: int
    errors: int
    skipped: int
    duration: float
    exit_code: int
    tests: list[TestResult]
    stdout: str
    stderr: str
```

## Integration with RefactoringMiner Workflow

This agent can be integrated into the RefactoringMiner evaluation pipeline:

### Before/After Test Comparison

```python
from agents.java_test.agent import analyze_java_tests
from rminer.rminer_utils import checkout_commit

# Analyze tests before refactoring
checkout_commit(repo_path, before_commit)
before_results = analyze_java_tests(repo_path)

# Analyze tests after refactoring
checkout_commit(repo_path, after_commit)
after_results = analyze_java_tests(repo_path)

# Compare
if before_results["success"] and not after_results["success"]:
    print("Refactoring broke tests!")
```

### MLflow Integration

```python
import mlflow
from agents.java_test.agent import analyze_java_tests

with mlflow.start_run():
    result = analyze_java_tests(project_path)
    
    # Log metrics
    mlflow.log_metrics({
        "total_tests": result["total"],
        "passed_tests": result["passed"],
        "failed_tests": result["failed"],
    })
    
    # Log artifacts
    mlflow.log_param("project_path", project_path)
    mlflow.log_text(result["response"], "analysis.txt")
```

## Configuration

### Environment Variables

```bash
# OpenAI (default)
export OPENAI_API_KEY="your-key"

# Anthropic
export ANTHROPIC_API_KEY="your-key"
```

### Supported Models

**OpenAI**:
- `gpt-4o-mini` (default, fast and cost-effective)
- `gpt-4o`
- `gpt-4-turbo`
- `gpt-4`

**Anthropic**:
- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`

## Limitations

1. **Build System Support**: Only Maven and Gradle
2. **Test Report Format**: Requires standard Surefire/Gradle XML format
3. **Execution Environment**: Requires Maven/Gradle CLI tools installed
4. **Timeout**: Default 300s (configurable)
5. **No Static Analysis**: Only runs actual tests, no AST parsing

## Future Enhancements

- [ ] Coverage analysis (JaCoCo integration)
- [ ] Test discovery without execution
- [ ] Support for other build systems (Ant, Bazel)
- [ ] Parallel test execution
- [ ] Test flakiness detection
- [ ] Integration with CI/CD systems

## Examples

### Example 1: Simple Test Analysis

```bash
uv run workflows/java_test_workflow.py \
    --project ~/repos/spring-petclinic
```

Output:
```
================================================================================
Java Test Analysis Results
================================================================================

Project: /Users/user/repos/spring-petclinic
Model: gpt-4o-mini (openai)

--------------------------------------------------------------------------------

I've analyzed the Java tests in your Spring PetClinic project:

Build System: MAVEN detected

Test Results Summary:
- Total Tests: 42
- Passed: 40
- Failed: 2
- Errors: 0
- Skipped: 0
- Duration: 45.23 seconds

Failed Tests:

1. org.springframework.samples.petclinic.owner.OwnerControllerTests.testFindOwner
   - Error: Expected status <200> but was <404>
   - Type: AssertionError
   - Root Cause: The owner endpoint is returning 404, likely due to missing
     test data setup in the database

2. org.springframework.samples.petclinic.vet.VetControllerTests.testGetVet
   - Error: NullPointerException at VetController.java:45
   - Type: NullPointerException
   - Root Cause: Vet service is not properly mocked, returning null

Recommendations:
- Fix test data initialization in OwnerControllerTests
- Add proper mock setup for VetService in VetControllerTests
```

### Example 2: JSON Output

```bash
uv run workflows/java_test_workflow.py \
    --project ~/repos/my-app \
    --json > results.json
```

## Dependencies

Required Python packages (from `pyproject.toml`):

```toml
dependencies = [
    "langchain>=0.3.0",
    "langgraph>=0.2.0",
    "langchain-anthropic",
    "langchain-openai",
    "python-dotenv>=1.0.0",
]
```

System requirements:
- Java 8+ (for running tests)
- Maven 3.6+ or Gradle 6.0+
- Python 3.11+

## Troubleshooting

### "No Java build system detected"

Ensure your project has either:
- `pom.xml` (Maven)
- `build.gradle` or `build.gradle.kts` (Gradle)

### "Tests timed out"

Increase timeout:

```python
from agents.tools.java_test_tools import run_tests

result = run_tests(project_path, "maven", timeout=600)  # 10 minutes
```

### "Failed to parse test results"

Check that tests actually ran and generated XML reports:

```bash
# Maven
ls target/surefire-reports/

# Gradle
ls build/test-results/test/
```

## Related Documentation

- [RefactoringMiner README](README_RMINER.md)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Maven Surefire Reports](https://maven.apache.org/surefire/maven-surefire-plugin/)
- [Gradle Test Reports](https://docs.gradle.org/current/userguide/java_testing.html)
