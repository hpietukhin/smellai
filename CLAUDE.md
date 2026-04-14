# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

SmellAI is a master's thesis research system for evaluating LLM-based code smell detection and refactoring using multi-agent orchestration (LangGraph). The system evaluates how well AI agents can map refactorings to code changes while accounting for smell dependencies and cascading effects.

**Core technologies**: Python 3.11+, LangGraph, LiteLLM, MLflow, SonarQube, SQLModel, NetworkX

## Essential commands

### Dependency management
```bash
# Install dependencies (including dev)
uv sync --all-groups

# Install specific dependency group
uv sync --group dev

# Add new package
uv add package-name

# Run Python with project venv
uv run python script.py
```

### SonarQube (code smell detection)
```bash
# Start SonarQube container
docker compose -f sonarqube/docker-compose.yml up -d

# Scan repository for smells
uv run sonarqube/commit_scan.py --repo-url <url> --commit <sha>
```

### MLflow (experiment tracking)
```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Opens http://localhost:5000

# List datasets
uv run scripts/manage_datasets.py list

# Inspect dataset
uv run scripts/manage_datasets.py get --name <dataset-name> --show-records
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_file.py

# Run with coverage
uv run pytest --cov=agents --cov-report=html
```

### Evaluation workflows
```bash
# RefactoringMiner evaluation (refactoring mapping)
uv run workflows/rminer_eval_workflow.py \
    --dataset-name rminer-eval-dataset \
    --model gpt-4o-mini

# SWE-Refactor evaluation (behavior preservation)
uv run workflows/swe_eval_workflow.py \
    --dataset-name swe-refactor-dataset \
    --mode composite

# Java test analysis
uv run workflows/java_test_workflow.py --project /path/to/java/project
```

### Visualization
```bash
# Agent execution visualizer (port 8080)
uv run python tools/visualize_smell_prioritization.py

# Generate smell prioritization graph
uv run scripts/prioritize_smells.py --smells-file smells.json
```

## Architecture

### Multi-agent system (LangGraph-based)

The system implements 6+ specialized agents in a research workflow:

- **A0 (Test Coverage)**: `agents/java_test/agent.py` - Detects Maven/Gradle, runs tests, parses results
- **A1 (Smell Detection)**: `sonarqube/commit_scan.py` - Detects 8 smell types via SonarQube API
- **A2 (Developer Query)**: Planned - Query developer to select which smells to address
- **A3 (Prioritization)**: `agents/dependency_analysis/agent.py` - Analyzes positive/negative smell dependencies using NetworkX
- **A4 (Prompt Prep)**: Planned - Prepares refactoring prompts from smell prioritization
- **A5 (Refactoring Execution)**: `agents/rminer_eval/agent.py` + `agents/swe_eval/agent.py` - LLM-based refactoring mapping and code generation
- **A6 (Behavior Verification)**: Reuses A0 - Verifies refactorings preserve behavior via test execution
- **A7 (Test Generation)**: Planned - Generate tests for uncovered methods

### Agent workflow modes

**Basic mode** (single refactoring):
```
A0 (setup) → A5 (generate) → A6 (verify)
```

**Composite mode** (iterative multi-refactoring):
```
A0 (setup) → A1 (detect) → A2 (prioritize) → A3 (select) →
  [A4 (map) → A5 (generate) → A6 (verify)] loop → END
```

### State management pattern

Agents use TypedDict states with message reduction:
```python
class SWEEvalState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # Auto-reduces
    record: RefactoringRecord
    detected_smells: List[SmellEvent]
    # ... additional fields
```

Nodes are pure functions: `(state: State) -> dict` returning partial state updates.

### LLM integration (LiteLLM)

Provider-agnostic LLM access supporting OpenAI, Anthropic, Cerebras:
```python
from langchain_litellm import ChatLiteLLM

llm = ChatLiteLLM(model="gpt-4o-mini")  # or claude-sonnet-4-5-20250929
response = llm.invoke(messages)
```

**Default models**:
- RMiner agent: `gpt-4o-mini` (refactoring mapping)
- SWE-Eval agent: `claude-sonnet-4-5-20250929` (code generation)
- Java test agent: Tool-based (no LLM generation)

**Structured output pattern**: Try structured output first, fall back to JSON parsing if model lacks support.

### Dataset adapter pattern

All datasets implement `Dataset` interface (`datasets/base.py`):
```python
class Dataset(ABC):
    @abstractmethod
    def request(self) -> list[dict]: ...
    @abstractmethod
    def get_dataset_name(self) -> str: ...
    @abstractmethod
    def get_tags(self) -> dict: ...
```

**Current adapters**:
- `RMinerAdapter` (`datasets/rminer.py`) - RefactoringMiner 2.0 ground truth
- `SWERefactorAdapter` (`datasets/swe_refactor.py`) - SWE-Refactor behavior preservation

Output format is MLflow GenAI: `{"inputs": dict, "expectations": dict, "tags": dict}`

### Analytics persistence (separate from LangGraph checkpoints)

Uses SQLModel ORM for structured analytics separate from LangGraph state:
- `ToolCall` - Tool invocations with timing (node_name, tool_name, duration_ms)
- `SmellEvent` - Smell lifecycle (detected/resolved/created) with session/iteration tracking
- `RefactoringAttempt` - Refactoring outcome with smells_resolved/created counts
- `TokenUsage` - LLM token consumption by node
- `SmellDependency` - Positive/negative smell relationships

Database: `test_analytics.db` (SQLite)

Access via `AnalyticsDB` (`swe_refactor/persistence/database.py`)

### Smell prioritization (PZ formula)

Dependency-aware prioritization using NetworkX graph:
```
PZ_i = Severity_i + Σ(w_impact for each positive dependency)
```

Positive dependencies = refactoring helps resolve other smells (green edges)
Negative dependencies = refactoring may create new smells (red dashed edges)

Implementation: `agents/dependency_analysis/agent.py` + `scripts/prioritize_smells.py`

## Key integration points

### SonarQube
- **Setup**: Docker container on port 9000 (`sonarqube/docker-compose.yml`)
- **Auth**: `SONAR_TOKEN` env var (create in UI: My Account → Security)
- **Scanner**: `sonarqube/commit_scan.py` runs scanner locally → polls API → fetches issues
- **Rule mapping**: 8 smell types mapped to SonarQube rules (java:S1541=Complex Method, java:S138=Long Method, etc.)
- **Output**: `SmellEvent` objects with file path, line number, severity (HIGH/MEDIUM/LOW)

### MLflow
- **Tracking DB**: `sqlite:///mlflow.db` (default)
- **Dataset format**: MLflow GenAI (inputs/expectations/metadata)
- **Custom scorers**: Domain-specific metrics (mapping_accuracy, hunk_coverage, compile_success_rate, test_pass_rate)
- **Factories**: `RMinerDatasetFactory`, `SWERefactorDatasetFactory` convert domain data to MLflow format
- **Manager**: `mlflow_utils/datasets/manager.py` for dataset CRUD

### RefactoringMiner
- **Manifest**: JSON with before/after file pairs + refactoring metadata
- **Location**: `rminer_data/manifest.json`
- **Adapter**: `datasets/rminer.py` converts to MLflow GenAI format
- **Usage**: Ground truth for evaluating LLM refactoring mapping accuracy

## Common patterns

### Configuration (Enum-based)
```python
class AgentConfig(str, Enum):
    MODEL_NAME = "model_name"
    MAX_RETRIES = "max_retries"

DEFAULT_CONFIG = {
    AgentConfig.MODEL_NAME: "gpt-4o-mini",
    AgentConfig.MAX_RETRIES: 3,
}
```

### Error handling (Result objects)
```python
CompileResult(success: bool, error_summary: list[str] | None)
# Caller checks result.success and decides response
```

### Repository operations (GitPython utilities)
```python
from swe_refactor.utils import (
    clone_repository,
    force_checkout_commit,
    get_previous_commit,
    replace_java_code,
)
```

### Build system detection
```python
if (path / "pom.xml").exists():
    compile_command = "mvn clean compile"
elif (path / "build.gradle").exists():
    compile_command = "./gradlew clean build -x test"
```

### Java version switching (jenv)
```python
from swe_refactor.utils import switch_java_version

switch_java_version(jdk_version="17", project_path=path)
# Uses jenv to switch JDK before build
```

## Important notes

### Environment variables
**NEVER edit `.env` files** - only user may change them. Required vars:
- `OPENAI_API_KEY` - OpenAI API key
- `CEREBRAS_API_KEY` - Cerebras API key (optional)
- `SONAR_TOKEN` - SonarQube auth token
- `SONAR_URL` - SonarQube server (default: http://localhost:9000)
- `MLFLOW_TRACKING_URI` - MLflow DB (default: sqlite:///mlflow.db)

### Dependencies not in pyproject.toml
These packages are required by tools but not explicitly in dependencies:
- `langchain-litellm` - Used by java_test and swe_eval agents
- `sqlmodel` - Used by analytics persistence layer
- `nicegui` - Used by visualization tool

If missing after `uv sync`, install manually:
```bash
uv pip install langchain-litellm sqlmodel nicegui
```

### Running visualization tool
The visualizer runs on port 8080. If port is in use, kill existing process or change port in code (line 1074).

### Analytics database
`test_analytics.db` tracks agent execution separately from LangGraph checkpoints. Don't confuse with `mlflow.db` (MLflow tracking).

### Git operations
- Use `uv run` prefix when executing Python scripts to ensure correct venv
- Always use absolute paths for repository operations
- JDK switching requires jenv installed on system

## Documentation

- `TECHNICAL_SPECIFICATION.md` - Complete system architecture and design
- `README.md` - Quick start and usage guide
- `VISUALIZATION_USAGE.md` - Visualizer guide with examples
- `docs/README_RMINER.md` - RefactoringMiner workflow details
- `docs/java_test_agent.md` - Java test analysis agent docs
- `docs/sonarqube_smells.md` - SonarQube integration and smell types
- `docs/SYSTEM_DESIGN_SUMMARY.md` - Multi-agent architecture overview


General rules for code contributions and modifications:
## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

paper where the theory behind builded system is described in /Users/havriil.pietukhin/uni/masterThesis/conf_paper
/conf.tex. Use it as a spec, READ WHEN NOT COMPLETELY CLEAR when we are talking about processing logic or discussing data models, adhere to it
