# SmellAI: Technical requirements specification

**Version**: 1.1
**Date**: 2026-01-12
**Project**: Master's thesis - Multi-agent system for code smell detection, prioritization, and refactoring evaluation
**Author**: Based on codebase analysis

---

## Table of contents

1. [Overview](#1-overview)
2. [Goals and objectives](#2-goals-and-objectives)
3. [System architecture](#3-system-architecture)
4. [Feature specifications](#4-feature-specifications)
5. [Data contracts and interfaces](#5-data-contracts-and-interfaces)
6. [Integration points](#6-integration-points)
7. [Technical considerations](#7-technical-considerations)
8. [Non-functional requirements](#8-non-functional-requirements)

---

## 1. Overview

### 1.1 System purpose

SmellAI is a research-focused multi-agent system designed to evaluate code smell detection and refactoring approaches for master's thesis research. The system combines:

- **Static analysis** using SonarQube for automated code smell detection
- **LLM-powered agents** built with LangGraph for test analysis and refactoring evaluation
- **Dependency analysis** to model relationships between code smells and optimize refactoring sequences
- **MLflow integration** for comprehensive experiment tracking and evaluation
- **RefactoringMiner integration** for ground truth refactoring data

The core research question evaluates how well AI agents can map detected refactorings to specific code changes while accounting for code smell dependencies and their cascading effects.

### 1.2 Key capabilities

1. **Code smell detection**: Integration with SonarQube to identify 8 types of code smells with severity levels
2. **Java test analysis**: Automated detection of build systems, test execution, and failure analysis
3. **Refactoring mapping**: LLM-based mapping of refactorings to diff hunks in Git commits
4. **Dependency-aware prioritization**: Graph-based analysis of smell dependencies to determine optimal refactoring sequences
5. **Experiment tracking**: Complete evaluation pipeline with MLflow for reproducibility
6. **Visualization**: NetworkX-based graphs showing smell relationships and priority sequences

---

## 2. Goals and objectives

### 2.1 Research goals

1. **Evaluate LLM performance** in mapping refactorings to code changes using RefactoringMiner 2.0 dataset
2. **Implement dependency-aware prioritization** based on positive/negative smell relationships
3. **Measure impact** of refactoring sequences on code quality metrics
4. **Provide reproducible experiments** through comprehensive MLflow tracking

### 2.2 System objectives

1. **Multi-agent orchestration**: Implement LangGraph-based agents with clear separation of concerns
2. **Evaluation framework**: MLflow GenAI evaluation with custom scorers for refactoring accuracy
3. **Data integration**: Seamless integration with RefactoringMiner, SonarQube, and Git repositories
4. **Extensibility**: Modular architecture supporting future agent additions and alternative LLM providers

### 2.3 Non-goals

- Production-grade refactoring execution on live codebases (system applies refactorings for evaluation purposes only)
- Automated code generation or modification without validation
- Integration with IDEs or development environments
- Production deployment or scalability beyond research evaluation

**Note**: The system DOES apply refactorings during evaluation to measure their impact and verify behavior preservation, but this is done in isolated environments on research datasets, not on production code.

---

## 3. System architecture

### 3.1 High-level architecture layers

The system consists of four main layers:

1. **External Services Layer**: SonarQube (Docker), LLM providers (OpenAI/Anthropic/etc.), RefactoringMiner dataset
2. **Application Layer**: MLflow tracking, multi-agent orchestration (LangGraph), six specialized agents
3. **Data Layer**: Git repositories, MLflow database (SQLite), SonarQube API, RefactoringMiner JSON data
4. **Infrastructure Layer**: Docker compose, Python environment (uv), logging configuration

### 3.2 Agent architecture

The system implements six specialized agents in a research-oriented workflow.

**Planned agent workflow**:
1. **A0**: Check if tests exist for refactoring target. If not, call A7 (Test Generation Agent).
2. **A1**: Detect code smells via SonarQube
3. **A2**: Query developer (or use pre-configured priorities) to select which smells to address
4. **A3**: Prioritize selected smells based on dependency analysis
5. **A4**: Prepare refactoring prompts based on prioritized list (or retrieve from prompt database)
6. **A5**: Execute refactorings in a loop
7. **A6**: Verify refactoring correctness and behavior preservation via test execution. If tests not found, request A7 to generate tests.

**Architecture evolution note**: Current implementation uses separate agents. Future iterations may consolidate into a single agent with multiple capabilities, adding complexity gradually as needed.

**Agent descriptions**:

### Agent status table

| Agent | Name | Location | Status |
|-------|------|----------|--------|
| **A0** | Test coverage verification | `agents/java_test/agent.py` | ✅ Working |
| **A1** | Code smell detection | `sonarqube/commit_scan.py` | ✅ Working |
| **A2** | Developer query | TBD | 🔄 Planned |
| **A3** | Smell prioritization | `agents/dependency_analysis/agent.py`, `scripts/prioritize_smells.py` | ✅ Working |
| **A4** | Refactoring prompt preparation | TBD | 🔄 Planned |
| **A5** | Refactoring execution | `agents/rminer_eval/agent.py` | ✅ Working |
| **A6** | Behavior preservation | `agents/java_test/agent.py` (reuses A0) | ✅ Working |
| **A7** | Test generation | TBD | 🔄 Planned |

### Detailed agent specifications

**A0: Test coverage verification agent**
- Technology: LangGraph + LangChain tools
- Location: `agents/java_test/agent.py`
- Function: Detects Maven/Gradle build systems, runs tests, parses XML results
- Output: Test run summary with pass/fail/error/skipped status
- Called by: Workflow initialization and A6 (behavior preservation)
- Calls: A7 if test coverage is insufficient

**A1: Code smell detection agent**
- Technology: SonarQube API integration
- Location: `sonarqube/commit_scan.py`
- Function: Detects 8 code smell types with severity levels (Complex Method, Long Method, Large Class, etc.)
- Output: Normalized smell detection records with location and severity

**A2: Developer query agent (planned)**
- Status: Planned (currently auto-selects all detected smells)
- Function: Query developer to select which smells to address
- Method: Interactive prompt or pre-configured priorities
- TODO: Implement developer interaction interface

**A3: Smell prioritization agent**
- Technology: NetworkX + dependency rules
- Location: `agents/dependency_analysis/agent.py`, `scripts/prioritize_smells.py`
- Function: Analyzes positive/negative smell dependencies, calculates priority scores using PZ formula
- Formula: `PZ_i = Severity_i + Σ(w_impact for each positive dependency)`
- Output: Prioritized refactoring sequence + visualization graphs (NetworkX)

**A4: Refactoring prompt preparation agent (planned)**
- Status: Planned (currently prompts generated inline in A5)
- Function: Prepares refactoring prompts based on prioritized smell list
- Method: Retrieves from prompt database or generates based on smell type
- TODO: Implement prompt database and retrieval logic

**A5: Refactoring execution agent**
- Technology: LangGraph + LiteLLM
- Location: `agents/rminer_eval/agent.py`
- Function: Maps refactorings to diff hunks using LLM reasoning, applies refactorings in priority order
- Output: Refactoring-to-hunk mappings with reasoning
- Runs in: Loop over prioritized smell list from A3

**A6: Behavior preservation verification agent**
- Function: Verifies that applied refactorings preserve code behavior by running tests
- Implementation: Reuses A0's test execution capabilities
- Note: Tests are run on code after each refactoring is applied to verify correctness
- Action: Calls A7 if tests are not found
- TODO: Implement behavior preservation checks beyond test execution (e.g., semantic equivalence analysis)

**A7: Test generation agent (planned)**
- Status: Planned (will be implemented in future)
- Function: Generate tests for uncovered methods or classes
- Called by: A0 or A6 when test coverage is insufficient
- TODO: Implement test generation capabilities using LLM-based test synthesis

### 3.3 Technology stack

#### Core technology summary

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Language | Python | 3.11+ | Core implementation |
| Orchestration | LangGraph | Latest | Multi-agent workflows |
| LLM Framework | LangChain | Latest | Tool definitions, message handling |
| LLM Interface | LiteLLM | Latest | Provider-agnostic LLM access |
| Experiment Tracking | MLflow | 3.0+ | Run tracking, evaluation framework |
| Static Analysis | SonarQube | 10.6.0 | Code smell detection |
| Version Control | GitPython | Latest | Repository operations |
| Graph Analysis | NetworkX | Latest | Dependency modeling |
| Visualization | Matplotlib | Latest | Graph rendering |
| Data Validation | Pydantic | 2.0+ | Type-safe data models |
| Package Manager | uv | Latest | Dependency management |

#### Detailed technology breakdown

**1. LLM Orchestration**
- `langgraph` >=0.2.0 - Multi-agent workflow orchestration
- `langchain` >=0.3.0 - LLM framework and tool definitions
- `langchain-community` >=0.3.0 - Community-contributed integrations
- `langchain-openai` - OpenAI provider integration
- `langchain-anthropic` - Anthropic provider integration
- `litellm` >=1.0.0 - Unified interface for OpenAI, Anthropic, Cerebras, Google

**2. LLM Providers** (at least one required)
- **OpenAI**: GPT-4o, GPT-4o-mini (default for most agents)
  - Requires: `OPENAI_API_KEY` environment variable
- **Anthropic**: Claude 3.5 Sonnet, Claude Opus
  - Requires: `ANTHROPIC_API_KEY` environment variable (optional)
- **Cerebras**: Fast inference (experimental)
  - Requires: `CEREBRAS_API_KEY` environment variable (optional)

**3. Experiment tracking**
- `mlflow` >=3.0.0 - Experiment tracking, evaluation framework, model registry
  - Automatic LangGraph tracing via `mlflow.langchain.autolog()`
  - Default tracking: `sqlite:///mlflow.db`
  - Dataset management and versioning

**4. Static analysis**
- **SonarQube**: `sonarqube:10.6.0-community` (Docker container)
  - Code smell detection (8 smell types)
  - Requires: `SONAR_TOKEN` environment variable
  - Default URL: `http://localhost:9000`

**5. Development tools**

*Dependency management*:
- `uv` (recommended) - Fast dependency resolution and installation
- `pip` - Alternative package manager

*Code quality*:
- `ruff` >=0.5.0 - Linting and formatting
- `mypy` >=1.10.0 - Type checking
- `pre-commit` - Git hooks for code quality

*Testing*:
- `pytest` >=8.0.0 - Test framework
- `pytest-cov` >=4.1.0 - Coverage reporting
- `pytest-asyncio` >=0.23.0 - Async test support

*Environment management*:
- `python-dotenv` >=1.0.0 - .env file loading
- `pydantic-settings` >=2.0.0 - Settings management

**6. Data and model libraries**

*Data validation*:
- `pydantic` >=2.0.0 - Structured output from LLMs, entity models validation

*Data manipulation*:
- `pandas` >=2.2.0 - Data analysis and processing
- `datasets` >=4.2.0 (dev) - Dataset processing utilities

*HTTP client*:
- `requests` >=2.31.0 - SonarQube API integration

**7. Infrastructure**

*Containerization*:
- Docker Engine >=24.0
- Docker Compose >=2.20

*Version control*:
- Git >=2.30
- GitPython >=3.1.0 - Python Git integration

**8. System dependencies**

*Required*:
- Python 3.11+ (3.11 or 3.12)
- Docker Engine 24.0+ (for SonarQube)
- Docker Compose 2.20+
- Git 2.30+
- Maven 3.6+ or Gradle 6.0+ (for Java test analysis)

*Optional*:
- MySQL 8.0+ (only if using DACOS dataset adapter)
- uv package manager (recommended for faster dependency resolution)

**9. Planned technologies** (not yet implemented)

*Vector database* (for future LLM-based smell detection):
- DeepLake <4.0.0 - Vector storage for smell knowledge base
- Google Generative AI Embeddings - Semantic search

*Note*: Current implementation uses SonarQube for smell detection. LLM-based detection with RAG is planned future work.

### 3.4 Code organization and design patterns

**Module structure**:
- `agents/` - LangGraph agent implementations organized by capability
  - `rminer_eval/` - Refactoring mapping agent
  - `java_test/` - Test analysis agent
  - `dependency_analysis/` - Dependency analysis logic
  - `tools/` - LangChain tool definitions
- `workflows/` - High-level orchestration scripts and CLI entry points
- `models/` - Pydantic data models (RefactoringMiner structures)
- `mlflow_utils/` - MLflow integration (server management, dataset operations, evaluation)
- `sonarqube/` - SonarQube API integration and Docker deployment
- `rminer/` - RefactoringMiner data processing utilities
- `repo_utils/` - Git repository operations
- `scripts/` - Utility scripts (prioritization, dataset creation)
- `tests/` - Test suite with test data

**Design patterns implemented**:

1. **State pattern** (LangGraph): Each agent maintains typed state dictionary, immutable state transitions
   - **State persistence**: Currently in-memory only. Context is cleared after each complete refactoring operation (get before-state → run tests → apply refactoring → run tests → get after-state).
   - TODO: Add simple persistence mechanism for long-running workflows (low priority - not critical for current evaluation-based workflow)
2. **Tool pattern** (LangChain): Functions decorated with `@tool` for LLM function calling
3. **Factory pattern**: Dataset factories (`rminer_factory.py`) create MLflow GenAI records
4. **Builder pattern**: Agent construction via StateGraph builder API
5. **Repository pattern**: `DatasetManager` abstracts MLflow dataset operations
6. **Strategy pattern**: Pluggable scorers for MLflow evaluation
7. **Decorator pattern**: `@require_git_import` for graceful degradation when Git unavailable

**Key architectural decisions**:
- Pydantic models for all data contracts (type safety, validation)
- LiteLLM for provider-agnostic LLM access (supports OpenAI, Anthropic, Cerebras, Google)
- Structured output from LLMs via `with_structured_output()` when supported
- Fail-fast error handling (no silent failures, no fallback defaults)
- Centralized logging configuration

### 3.5 Logging infrastructure

**Configuration**: `logging_config.py` provides centralized setup

**Features**:
- Dual-handler system: console (stdout) and optional file
- Console logs at configured level (default: INFO)
- File logs at DEBUG level (captures everything)
- Timestamp format: `YYYY-MM-DD HH:MM:SS`
- Message format: `[LEVEL] module_name - message`

**Usage across modules**:

| Module | Usage | Purpose |
|--------|-------|---------|
| `workflows/*.py` | INFO level | Workflow progress, major steps |
| `sonarqube/commit_scan.py` | INFO/WARNING | Clone operations, scan progress, issues |
| `repo_utils/*.py` | INFO/ERROR | Git operations, failures |
| `rminer/*.py` | INFO | Data processing, parsing |
| `mlflow_utils/*.py` | INFO | Server status, dataset operations |

**MLflow tracing**: Automatic via LangGraph integration, captures:
- All agent node inputs/outputs
- Execution timeline and latency
- LLM calls with prompts and responses
- Tool invocations

### 3.6 Known gaps and limitations

**Missing components**:
1. **Dataset adapter for input data**: No generic adapter interface for different input formats. Current implementation hardcodes RefactoringMiner JSON parsing. Future work should abstract this behind a common interface.

2. **Test generation agent (A7)**: Placeholder only, not implemented as it's not the primary research focus.

3. **Batch processing optimization**: Sequential evaluation only; no parallel processing support.

4. **Configuration management**: Relies on environment variables and function parameters. No centralized configuration module or YAML/TOML config files.

---

## 4. Feature specifications

### 4.1 Code smell detection

**Feature**: Automated detection of code smells using SonarQube

**Inputs**:
- Git repository URL
- Commit SHA or branch name
- SonarQube server credentials

**Processing**:
1. Clone repository at specified commit
2. Configure SonarQube scanner
3. Run analysis via Docker container
4. Poll for completion
5. Fetch issues via REST API
6. Normalize to internal format

**Outputs**:
- List of detected smells with:
  - Rule ID (e.g., `java:S1541`)
  - Smell type (e.g., "Complex Method")
  - Location (file, line, column)
  - Severity (HIGH/MEDIUM/LOW)
  - Message and description

**Important notes**:
- **Rule filtering**: Only rules in `RULE_NAME_MAP` (see section 5.1) are processed. All other SonarQube findings are ignored.
- **Snapshot approach**: Each commit is analyzed as an independent code snapshot. SonarQube's stateful features (issue tracking, quality gate history) are not used.
- **Project keys**: Each analysis represents a point-in-time state, not a historical project progression.

**Supported smell types** (8 total):
1. Complex Method (`java:S1541`)
2. Long Method (`java:S138`)
3. Long Parameter List (`java:S107`)
4. Conditional Complexity (`java:S1067`)
5. God Class (`java:S1200`)
6. Large Class (`java:S110`)
7. Duplicated Conditions (`java:S1871`)
8. Print Statements (`java:S106`)

**Data contract**: See section 5.1

---

### 4.2 Java test analysis

**Feature**: Automated build system detection and test execution with failure analysis

**Inputs**:
- Project directory path
- Optional: specific test classes or methods
- Optional: LLM model for failure analysis

**Processing**:
1. **Build system detection**:
   - Check for `pom.xml` (Maven)
   - Check for `build.gradle` or `build.gradle.kts` (Gradle)
   - Return detected system or None

2. **Test execution**:
   - Run `mvn clean test` or `gradle clean test`
   - Capture stdout/stderr
   - Record exit code

3. **Result parsing**:
   - Parse XML reports (Surefire or Gradle format)
   - Extract test names, status, duration, error messages
   - Aggregate statistics

4. **LLM analysis** (optional):
   - Provide failure details to LLM
   - Request root cause analysis and recommendations
   - Return structured analysis

**Outputs**:
- `TestRunSummary`:
  - Build system detected
  - Total/passed/failed/error/skipped counts
  - Duration
  - Exit code
  - List of individual test results
  - stdout/stderr output

**Data contract**: See section 5.2

---

### 4.3 Refactoring mapping

**Feature**: LLM-powered mapping of refactorings to diff hunks in Git commits

**Inputs**:
- Before/after code snippets
- List of refactorings (type, description)
- Diff hunks from Git commit
- Optional: SonarQube issues (TODO: Document when included vs excluded)
- Optional: Dependency analysis (TODO: Document when included vs excluded)

**Processing**:
1. Parse refactoring metadata (type, description)
2. Parse diff hunks (old/new line ranges, content). Each hunk is annotated with start-line/end-line for LLM context.
3. Construct LLM prompt with:
   - File context
   - Refactoring descriptions
   - Hunk summaries (with explicit line numbers to aid LLM comprehension)
   - Optional smell context
   - See datamodels in code for exact prompt structure (TODO: Add reference link)
4. Invoke LLM with structured output
5. Parse and validate response:
   - Check that hunk line ranges are within valid bounds
   - If validation fails (e.g., out-of-bounds lines), re-prompt LLM
   - Handle N-to-N relationship between before-state and after-state hunks

**Context management**:
- All context is cleared after each complete refactoring operation cycle
- Refactoring operation = get before-state → run tests → apply refactoring → run tests → get after-state
- TODO: Implement token counting and truncation strategy for large files (e.g., "God Classes" exceeding context window)

**Outputs**:
- `RefactoringMappingOutput`:
  - Overall analysis
  - List of mappings (refactoring_index → hunk line ranges, reasoning)
  - **Note on reasoning field**: Currently used only for human debugging and qualitative analysis, not programmatically evaluated. Format is plain text, unstructured.

**Evaluation metrics** (MLflow scorers):
- **F1 score**: Balances precision and recall to avoid "shotgunning" (predicting everything). Replaces simple overlap-based mapping accuracy.
- **Hunk coverage**: Fraction of actual hunks covered by predictions
- **Prediction completeness**: Whether expected number of predictions made

**Ground truth definition**:
- Source: Manually annotated datasets containing before/after code fragments
- Derivation: Ground truth hunks are all hunks from the after-refactoring state
- Adapter: TODO: Implement adapter for new dataset format (specification to be provided)

**Data contract**: See section 5.3

---

### 4.4 Dependency-aware prioritization

**Feature**: Graph-based analysis of code smell dependencies to optimize refactoring order

**Dependency types explained**:

- **Positive dependency**: Removal of one code smell leads to removal of another code smell. Example: Refactoring a Long Method (extracting smaller methods) often removes Duplicated Code, Feature Envy, and Switch Statements that were embedded in the long method.

- **Negative dependency**: Removal of one code smell introduces a new code smell. Example: Refactoring a Large Class by extracting new classes may introduce new Data Classes (classes with only data, no behavior) or Inappropriate Intimacy (excessive coupling between the extracted classes).

**Inputs**:
- List of detected smells (from SonarQube)
- Optional: code metrics (WMC, CBO, RFC, LCOM)

**Processing**:
1. **Dependency rule application**:
   - Map each smell to dependency rules (see section 5.4 for complete rules)
   - Identify positive dependencies (smells that can be solved by refactoring this smell)
   - Identify negative dependencies (smells that may be introduced by refactoring this smell)
   - Rules citation: Based on Markovič & Polášek. TODO: Create comprehensive map of dependency rules with detailed citations.

2. **Priority calculation** (Best-first search approach):
   - Formula: `PZ_i = Severity_i + Σ(w_impact for positive deps) - penalty(negative deps)`
   - Severity: HIGH=3, MEDIUM=2, LOW=1
   - Impact weight: 2 per positive dependency (reward for cascading benefits)
   - Negative dependencies: Reduce priority by considering side effects
   - **Strategy**: Find refactoring with maximum positive dependencies AND minimum negative dependencies
   - **Future consideration**: May evolve to count whole sum of dependencies on graph path rather than best-first

3. **Greedy sequence generation with cycle detection**:
   - Select smell with max(PZ) that minimizes negative side effects
   - Apply refactoring (see note below on execution vs simulation)
   - Update dependency graph (remove solved, add created)
   - Recalculate PZ scores
   - Repeat until no smells remain
   - **Cycle detection**: If refactoring A creates smell B, and refactoring B creates smell A, mark as outlier
   - TODO: Implement cycle detection mechanism and max-step limit to prevent infinite loops
   - TODO: Investigate Airflow capabilities for handling problematic cyclic dependency situations

4. **Visualization**:
   - Create directed graph with NetworkX
   - Green edges: positive dependencies (refactoring A solves B)
   - Red edges: negative dependencies (refactoring A introduces B)
   - Node size: proportional to PZ score

**Outputs**:
- Prioritized sequence of refactorings
- Dependency graph visualization (PNG)
- Per-smell dependency analysis

**Note on refactoring execution**: Despite the term "simulation" in some documentation, the system DOES apply refactorings to code. The workflow is:
1. Analyze before-state code
2. Run tests on before-state
3. Apply refactoring transformation to code
4. Run tests on after-state to verify behavior preservation
5. Compare states and measure impact

**Dependency rules**: See section 5.4

---

### 4.5 MLflow evaluation framework

**Feature**: Comprehensive experiment tracking and evaluation

**Components**:

1. **Dataset management**:
   - Create GenAI datasets from RefactoringMiner data
   - Store pair_id (input), ground truth hunks (expectations), metadata (tags)
   - Support listing, searching, and versioning

2. **Evaluation runner**:
   - Load dataset by ID or name
   - Invoke agent on each record
   - Apply custom scorers
   - Aggregate results

3. **Auto server management**:
   - Start MLflow server automatically if not running
   - Configure tracking URI
   - Create/set experiment

4. **Custom scorers**:
   - `mapping_accuracy`: Correctness of refactoring-to-hunk assignments
   - `hunk_coverage`: Percentage of actual changes identified
   - `prediction_completeness`: Expected vs actual prediction count

**Data contract**: See section 5.5

---

### 4.6 LangChain tool specifications

The system provides LangChain tools for agent function calling. Tools are Python functions decorated with `@tool` that agents can invoke.

#### 4.6.1 Java test analysis tools

**Location**: `agents/tools/java_test_tools.py`

**Tool 1: `detect_java_build_system`**
- **Purpose**: Identify Maven or Gradle in project
- **Input**: `project_path` (str) - Path to Java project directory
- **Output**: String describing detected build system or error message
- **Behavior**: Checks for `pom.xml` (Maven) or `build.gradle`/`build.gradle.kts` (Gradle)
- **Used by**: Java test agent (A0/A6)

**Tool 2: `run_java_tests`**
- **Purpose**: Execute tests and return summary
- **Inputs**:
  - `project_path` (str) - Path to Java project directory
  - `clean` (bool, default=True) - Whether to run clean before tests
- **Output**: Dictionary with test results
  - `success` (bool)
  - `total`, `passed`, `failed`, `errors`, `skipped` (int counts)
  - `duration` (float, seconds)
  - `failed_tests` (list of failed test details)
  - `error` (str, if build system not detected)
- **Behavior**: Runs `mvn clean test` or `gradle clean test`, parses XML reports (Surefire/Gradle format)
- **Timeout**: 300 seconds (5 minutes)
- **Used by**: Java test agent (A0/A6)

**Tool 3: `get_test_output`**
- **Purpose**: Retrieve recent test execution logs
- **Input**: `project_path` (str) - Path to Java project directory
- **Output**: String containing stdout and stderr (last 2000 characters)
- **Behavior**: Re-runs tests without clean (fast), captures console output
- **Timeout**: 60 seconds (1 minute)
- **Used by**: Java test agent (A0/A6) for detailed failure analysis

**Tool export**: `get_java_test_tools()` returns list of all three tools

#### 4.6.2 SonarQube scanning tool

**Location**: `sonarqube/tool.py`

**Tool: `scan_commit_smells`**
- **Purpose**: Scan Git commit for code smells using SonarQube
- **Inputs**:
  - `repo_url` (str, required) - Git repository URL
  - `commit_sha` (str, required) - Commit SHA to analyze
  - `file_paths` (List[str], optional) - Specific files to scan (None = entire commit)
  - `sonar_url` (str, default="http://localhost:9000") - SonarQube server URL
  - `cache_dir` (str, optional) - Directory for caching scan results
- **Output**: Dictionary with scan results
  - `commit_sha` (str)
  - `files_scanned` (int)
  - `total_smells` (int)
  - `smells_by_file` (dict mapping file paths to smell lists)
  - `error` (str, if scan failed)
- **Behavior**:
  1. Checks `SONAR_TOKEN` environment variable
  2. Verifies SonarQube is running via `/api/system/status`
  3. Auto-starts SonarQube via Docker Compose if not running
  4. Clones repository at specified commit
  5. Runs SonarQube scanner
  6. Fetches issues via REST API
  7. Normalizes to internal format (maps rule IDs to smell names)
- **Smell types detected**: 8 types (Complex Method, Long Method, Long Parameter List, Conditional Complexity, God Class, Large Class, Duplicated Conditions, Print Statements)
- **Error handling**: Returns error dictionary if SONAR_TOKEN missing, SonarQube fails to start, or scan fails
- **Used by**: Potential integration with refactoring mapping agent (currently not in main workflow)

**Auto-start mechanism**: If SonarQube not accessible, tool calls `_start_sonarqube()` which:
1. Locates `sonarqube/docker-compose.yml`
2. Runs `docker compose up -d`
3. Polls `/api/system/status` for 60 iterations (2-second intervals)
4. Raises RuntimeError if startup timeout exceeded

---

## 5. Data contracts and interfaces

### 5.1 Code smell detection contract

**Input**: SonarQube API response

**Output**: Normalized smell record

```python
{
    "rule": "java:S1541",           # SonarQube rule ID
    "message": "Method complexity...",
    "severity": "MAJOR",            # BLOCKER/CRITICAL/MAJOR/MINOR/INFO
    "component": "src/.../MyClass.java",
    "line": 45,
    "textRange": {
        "startLine": 45,
        "endLine": 78,
        "startOffset": 4,
        "endOffset": 5
    }
}
```

**Internal format** after normalization:

```python
class SmellDetection(BaseModel):
    smell_type: str                  # "Complex Method"
    location: str                    # "MyClass.java:45-78"
    description: str
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    refactoring_suggestion: str
    confidence: Optional[float]
```

**Rule mapping**:

```python
RULE_NAME_MAP = {
    "java:S1541": "Complex Method",
    "java:S138": "Long Method",
    "java:S107": "Long Parameter List",
    "java:S1067": "Conditional Complexity",
    "java:S1200": "God Class",
    "java:S110": "Large Class",
    "java:S1871": "Duplicated Conditions",
    "java:S106": "Print Statements",
}
```

**Severity mapping** (SonarQube → Internal format):

```python
SEVERITY_MAP = {
    "BLOCKER": "HIGH",
    "CRITICAL": "HIGH",
    "MAJOR": "MEDIUM",
    "MINOR": "LOW",
    "INFO": "LOW"
}
```

Note: TODO: Verify this mapping table exists in codebase and document exact location.

---

### 5.2 Java test analysis contract

**Input**: Project directory path

**Output**: Test run summary

```python
@dataclass
class TestResult:
    """Individual test result"""
    name: str
    status: Literal["PASS", "FAIL", "ERROR", "SKIPPED"]
    duration: float = 0.0
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    failure_trace: Optional[str] = None

@dataclass
class TestRunSummary:
    """Summary of test run"""
    build_system: Literal["maven", "gradle"]
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration: float = 0.0
    exit_code: int = 0
    tests: list[TestResult] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and self.failed == 0 and self.errors == 0

# Note: Skipped tests are not checked in success condition
# Policy: Tests should not be skipped during evaluation
```

**LangChain tools**:

```python
@tool
def detect_java_build_system(project_path: str) -> str:
    """Detect Java build system (Maven or Gradle)"""
    pass

@tool
def run_java_tests(project_path: str, build_system: str) -> dict:
    """Run tests using detected build system"""
    pass

@tool
def get_test_output(project_path: str) -> str:
    """Retrieve test logs and output"""
    pass
```

---

### 5.3 Refactoring mapping contract

**Input**: LangGraph state

```python
class RMinerEvalState(dict):
    """State for RMiner evaluation agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    before_code: str
    filename: str
    refactoring_types: List[str]
    refactoring_descriptions: List[str]
    diff_hunks: List[dict]
    sonar_issues: List[dict]
    dependency_analysis: List[DependencyAnalysis]
    predictions: List[dict]
```

**Diff hunk format**:

```python
{
    "old_start": 45,
    "old_count": 15,
    "new_start": 45,
    "new_count": 8,
    "lines": ["- removed line", "+ added line", "  context"],
    "header": "@@ -45,15 +45,8 @@ class MyClass {"
}
```

**Output**: Structured LLM response

```python
class RefactoringMapping(BaseModel):
    """A mapping between a refactoring and diff hunk line ranges"""
    refactoring_index: int       # 0-based index
    hunk_start_line: int         # Start line of mapped hunk
    hunk_end_line: int           # End line of mapped hunk
    reasoning: str               # Plain text explanation (for human debugging only)

    # Validation: Custom validator checks that line ranges are within bounds
    # If LLM returns invalid ranges, re-prompt for correction

class RefactoringMappingOutput(BaseModel):
    """Complete output from the refactoring mapping agent"""
    analysis: str                # Overall analysis of the refactoring
    mappings: List[RefactoringMapping]  # N-to-N relationship supported
```

**Note**: Previous version used `hunk_index` (integer), but this is being replaced with explicit line ranges (`hunk_start_line`, `hunk_end_line`) to better handle N-to-N mappings between before-state and after-state hunks.

---

### 5.4 Dependency analysis contract

**Dependency semantics**:

- **Positive dependencies**: List of code smells that will likely be removed when this smell is refactored. These are cascading benefits. Example: Refactoring "Long Method" typically removes "Duplicated Code", "Feature Envy", and "Switch Statement" that were part of the long method.

- **Negative dependencies**: List of code smells that may be introduced when this smell is refactored. These are potential side effects. Example: Refactoring "Large Class" may introduce "Data Class" (new classes with only getters/setters) or "Inappropriate Intimacy" (new classes tightly coupled).

**Input**: List of SonarQube issues

```python
[
    {
        "rule": "java:S1541",
        "message": "Method complexity is 15...",
        "severity": "MAJOR",
        ...
    },
    ...
]
```

**Output**: Dependency analysis

```python
class DependencyAnalysis(BaseModel):
    """Analysis of dependencies for a specific code smell"""
    smell_type: str
    rule_id: str
    positive_dependencies: List[str]  # Smells that will be removed (cascading benefits)
    negative_dependencies: List[str]  # Smells that may be introduced (side effects)

# Example:
{
    "smell_type": "Long Method",
    "rule_id": "java:S138",
    "positive_dependencies": [
        "Switch Statement",
        "Feature Envy",
        "Duplicated Code",
        "Divergent Change",
        "Comments",
        "Long Parameter List"
    ],
    "negative_dependencies": [
        "Long Method",
        "Long Parameter List"
    ]
}
```

**Dependency rules** (hardcoded):

```python
DEPENDENCY_RULES = {
    "Long Method": {
        "positive": ["Switch Statement", "Feature Envy", "Duplicated Code", ...],
        "negative": ["Long Method", "Long Parameter List"]
    },
    "Large Class": {
        "positive": ["Data Clumps", "Feature Envy", "Bad Class Content"],
        "negative": ["Long Method", "Data Class", "Inappropriate Intimacy", ...]
    },
    ...
}
```

---

### 5.5 MLflow GenAI contract

**Dataset record format**:

```python
{
    "pair_id": "commit_sha:file_path",     # Unique identifier
    "before_code": "...",                  # Code before refactoring
    "filename": "MyClass.java",
    "refactoring_types": ["Extract Method"],
    "refactoring_descriptions": ["Extract Method private validate(user User) : boolean..."],
    "diff_hunks": [...],                   # Parsed diff hunks
    "ground_truth": [0, 2],                # Expected hunk line ranges (from after-state)
    "metadata": {
        "repository": "https://github.com/...",
        "commit_sha": "abc123...",
        "refactoring_count": 1             # Can be >1 for commits with multiple refactorings
    }
}
```

**Important notes**:
- **Multiple refactorings**: `refactoring_count` can be any positive integer. The mapping agent handles all refactorings in a commit simultaneously.
- **Multi-file refactorings**: Current `pair_id` format is `commit_sha:file_path`. TODO: Verify how new dataset handles refactorings spanning multiple files (e.g., Move Class, Pull Up Method).
- **Ground truth source**: Manually annotated datasets with before/after code fragments. Ground truth hunks are derived from the after-refactoring state.
- **Dataset evolution**: New dataset format specification to be provided. TODO: Implement adapter for new format.

**Evaluation result**:

```python
{
    "run_id": "mlflow_run_id",
    "predictions": [
        {
            "refactoring_index": 0,
            "hunk_start_line": 45,
            "hunk_end_line": 78,
            "reasoning": "..."
        }
    ],
    "scores": {
        "f1_score": 0.85,                  # Replaces mapping_accuracy
        "precision": 0.90,                 # Avoid rewarding "shotgunning"
        "recall": 0.80,
        "hunk_coverage": 0.75,
        "prediction_completeness": 1.0
    },
    "metadata": {
        "model": "gpt-4o-mini",
        "temperature": 0.0,                # All agents use temperature 0 for determinism
        "timestamp": "2026-01-12T10:30:00Z"
    }
}
```

**Metric changes**:
- **F1 score replaces mapping_accuracy**: The old `len(overlap) / len(expected)` metric only measured recall, allowing models to "shotgun" by predicting everything. F1 score balances precision and recall using the formula: `F1 = 2 * (precision * recall) / (precision + recall)`

---

### 5.6 RefactoringMiner data models

**Refactoring operation**:

```python
class Refactoring(BaseModel):
    """Individual refactoring operation detected by RefactoringMiner"""
    type: str                           # "Extract Method", "Rename Class", ...
    description: str                    # Human-readable description
    validation: Optional[str]           # "TP", "FP", or None
    comment: Optional[str]              # Validator comment
    detection_tools: Optional[str]      # Comma-separated tool names
    validators: Optional[str]           # Validator names
    left_side_locations: List[RefactoringLocation]
    right_side_locations: List[RefactoringLocation]
```

**Location information**:

```python
class RefactoringLocation(BaseModel):
    """Location information for code elements"""
    file_path: str
    start_line: int
    end_line: int
    start_column: Optional[int]
    end_column: Optional[int]
    code_element: Optional[str]         # Method/class/variable name
```

**Commit data**:

```python
class RMinerCommit(BaseModel):
    """Git commit with refactoring information"""
    id: int
    repository: str
    sha1: str
    url: str
    author: str
    time: str
    refactorings: List[Refactoring]
    ref_diff_execution_time: Optional[int]
```

**Statistical summary**:

```python
class RefactoringStats(BaseModel):
    """Statistical summary of refactoring analysis"""
    total_commits: int
    total_repositories: int
    total_refactorings: int
    refactoring_type_counts: Dict[str, int]
    validation_counts: Dict[str, int]
    top_repositories: List[Dict[str, Any]]
    clusters_found: int
    clusters_detail: List[Dict[str, Any]]
```

---

## 6. Integration points

### 6.1 SonarQube integration

**Deployment**: Docker Compose with PostgreSQL backend

**Location**: `sonarqube/docker-compose.yml`

**Services**:
1. **SonarQube server**:
   - Image: `sonarqube:10.6.0-community`
   - Port: 9000
   - Database: PostgreSQL (not embedded)
   - Volumes: data, extensions, logs (persistent)
   - Memory: 512MB-1GB JVM heap
   - Network: `smellai-network` (isolated)

2. **PostgreSQL database**:
   - Image: `postgres:13`
   - Database name: `sonar`
   - Health checks: pg_isready with 10 retries
   - Dependency: SonarQube waits for PostgreSQL health

**Starting SonarQube**:
- Manual: `./sonarqube/sonarqube_server.sh start` or `docker compose -f sonarqube/docker-compose.yml up -d`
- Automatic: `sonarqube/tool.py` provides `_start_sonarqube()` with health checks

**Connection**:
- Protocol: HTTP REST API
- Authentication: Token-based (from `.env`)
- Base URL: `http://localhost:9000` (default)

**Key endpoints**:
- `/api/issues/search`: Fetch detected issues
- `/api/ce/task`: Poll for analysis completion
- `/api/system/status`: Health check
- `/api/qualitygates/project_status`: Quality gate status (informational only, does not block analysis)

**Configuration** (`.env`): `SONAR_URL` and `SONAR_TOKEN` required

**Auto-start behavior**: The `scan_commit_smells` tool checks if SonarQube is accessible and attempts to start it via Docker Compose if not running.

**Network connectivity**:
- SonarQube container runs on `smellai-network` (isolated Docker network)
- Exposes port 9000 to host machine
- Python agents on host access via `http://localhost:9000`
- Scanner CLI (also in container) can reach server via internal network
- See `sonarqube/commit_scan.py` for implementation details

**Quality gate policy**: Quality gate status is retrieved for informational purposes only. Analysis does NOT fail if quality gate fails. This is a research system focused on smell detection, not enforcement of quality standards.

---

### 6.2 LLM provider integration

**Supported providers** (via LiteLLM):
- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic (Claude 3.5 Sonnet)
- Google (Gemini 2.5)
- Cerebras (Llama 3.1-8b)

**Configuration**:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**Model selection**: Configurable via agent config or CLI (`--model` flag). Examples: `gpt-4o-mini` (default), `claude-3-5-sonnet`, `cerebras/llama3.1-8b`

**Temperature settings**:
- **Value**: 0.0 (deterministic) for ALL agents
- **Scope**: Centralized configuration
- **Rationale**: Refactoring is a deterministic process. Temperature 0 ensures reproducible outputs across evaluation runs.
- **Per-agent configuration**: Not currently supported. All agents use the same temperature setting.

---

### 6.3 MLflow integration

**Tracking URI**:
- Default: `sqlite:///mlflow.db`
- Configurable via `--tracking-uri`
- Storage: Local SQLite database in project root (typically `mlruns/` folder)

**Server management**: `ensure_mlflow_server()` context manager auto-starts server if needed

**Dataset creation**: `DatasetManager.create_dataset()` creates GenAI datasets from manifest paths with experiment association

**Data persistence**:
- **MLflow runs**: SQLite database in `mlruns/` folder
- **Datasets**: Parsed from JSON to pandas DataFrame (not stored in database)
- **SonarQube data**: Separate PostgreSQL database (managed by SonarQube, do not touch)

**Concurrency considerations**:
- **Current limitation**: Sequential evaluation only. Do not run multiple agents on the same data in parallel.
- **Potential conflicts**: Multiple evaluation scripts calling `ensure_mlflow_server()` may conflict over SQLite locks or port 5000
- TODO: Investigate parallel evaluation by breaking datasets into chunks for different agent instances
- TODO: Implement proper concurrency handling for MLflow server management

---

### 6.4 Git repository integration

**Operations**:
- Clone repositories (full checkout)
- Checkout specific commits
- Parse unified diffs
- Extract file content

**Implementation**: GitPython. See `repo_utils/` for helpers.

**Important**: **Sparse checkout has been removed** from the design. Initial design considered sparse checkout for efficiency, but this breaks SonarQube's ability to perform cross-file static analysis (e.g., detecting coupling, inheritance issues). Full checkout is required for accurate smell detection.

**Diff parsing**: `parse_diff_hunks()` from `rminer.create_rminer_dataset` parses unified diffs into structured hunk dictionaries.

**Assumptions**:
- Datasets contain only source code files (no binary files, no massive auto-generated files)
- Project root detection: Auxiliary bash script locates Java project root (searches for `pom.xml` or `build.gradle`). Agent tools require explicit `project_path` parameter.

---

### 6.5 RefactoringMiner data integration

**Mode**: Pre-computed JSON files only. The system does NOT run RefactoringMiner at runtime.

**Manifest format** (JSON):
```json
{
    "commitSHA1": "abc123...",
    "repository": "https://github.com/...",
    "filePairs": [
        {
            "leftFile": "src/Before.java",
            "rightFile": "src/After.java",
            "refactorings": [...]
        }
    ]
}
```

**Dataset creation flow**:
1. Read pre-computed manifest JSON (generated by RefactoringMiner 2.0)
2. For each commit:
   - Extract file pairs
   - Parse refactorings
   - Fetch Git diffs
   - Parse diff hunks
3. Create MLflow GenAI records
4. Register dataset

**Important notes**:
- **No runtime execution**: RefactoringMiner JAR is not executed as part of this pipeline
- **Dataset source**: Using existing datasets that were created with RefactoringMiner results
- **Schema verification**: TODO: Verify that manifest format matches raw RefactoringMiner 2.0 output or if intermediate processing is needed

---

## 7. Technical considerations

### 7.1 Error handling

**Strategy**: Fail fast with structured logging

**Principles**:
- No silent failures
- No fallbacks that mask missing data
- No chained defaults in business logic (`a or b or c`)
- No hidden retries unless explicitly requested
- Catch only expected exceptions
- Log with context, then re-raise

**Rationale**: Chained defaults (`a or b or c`) mask missing data and create silent failures. Instead, fail fast with clear error messages.

---

### 7.2 Asynchronous operations

**Policy**: No try-catch on async requests

**Rationale**: Let failures propagate for visibility. No try-catch blocks around async requests unless handling specific, expected exceptions.

---

### 7.3 Logging and observability

**Configuration**: `logging_config.py`

**Levels**:
- INFO: Normal operations
- WARNING: Recoverable issues
- ERROR: Failures requiring attention

**Structured logging**: Use `extra` dict for context (commit SHAs, file paths, counts)

**MLflow tracing**:
- Automatic via LangGraph integration
- Captures all agent inputs/outputs
- Records execution timeline
- Enables debugging and comparison

---

### 7.4 Testing strategy

**Test types**:
1. **Unit tests**: Core logic in isolation
2. **Integration tests**: Agent workflows end-to-end
3. **Evaluation tests**: MLflow scorers

**Test data**:
- Location: `tests/test_data/`
- RefactoringMiner samples
- SonarQube mock responses

**Coverage targets**:
- Data models: 100%
- Core agents: >80%
- Integration workflows: >70%

---

### 7.5 Dependency management

**Tool**: uv (fast Python package installer)

**Key commands**: `uv pip install .` (install), `uv add <package>` (add dependency), `uv pip sync uv.lock` (reproducible install)

**Configuration**: `pyproject.toml`

---

### 7.6 Version control policies

**Branch strategy**:
- `master`: Stable releases
- `revised`: Current development
- Feature branches: Short-lived, merged via PR

**Commit conventions**:
- Atomic commits (one logical change)
- Descriptive messages
- No WIP commits in main branches

**Git policies** (from CLAUDE.md):
- Never run destructive operations without explicit approval
- Keep commits atomic, list paths explicitly
- Quote paths with brackets/parentheses
- No amending unless explicitly approved
- Coordinate before reverting other agents' work

---

## 8. Non-functional requirements

### 8.1 Reproducibility

**Requirements**:
- Log Git SHA for every analyzed commit
- Log LLM model, temperature, provider
- Store full prompts in MLflow
- Record experiment configuration
- Version agent graph definitions

**MLflow metadata**: Captures git_sha, timestamp, model, temperature, agent_version in run metadata

---

### 8.2 Performance

**Targets**:
- Evaluate 100 samples in <30 minutes (sequential)
- Single refactoring mapping in <10 seconds
- MLflow dataset creation: <5 minutes for 100 commits

**Dataset size configuration**:
- Sample count (e.g., "100 samples") is configurable via .sh scripts or Python script parameters
- Not a system limitation - adjust based on research requirements
- See evaluation scripts for limit configuration options

**Optimization strategies**:
- Full Git clones (sparse checkout removed - see section 6.4)
- Shallow clones where possible (depth=1 for single commit analysis)
- Batch processing for datasets

**Parallel evaluation**:
- **Current status**: Sequential evaluation only. Parallel processing NOT currently supported.
- **Experience**: Tested with 1 agent only
- **Future potential**: Break dataset into chunks and feed to separate agent instances
- TODO: Investigate parallel evaluation capabilities and test concurrency handling

---

### 8.3 Extensibility

**Design principles**:
- Modular agent architecture
- Provider-agnostic LLM interface
- Pluggable scorers
- Configurable dependency rules

**Extension points**:
1. New agents: Implement LangGraph StateGraph
2. New LLM providers: Configure via LiteLLM
3. New scorers: Implement MLflow scorer interface
4. New dependency rules: Update `DEPENDENCY_RULES` dict

---

### 8.4 Security

**Sensitive data**:
- API keys in `.env` (git-ignored)
- No secrets in code or commits
- Read-only database access

**Required environment variables** (`.env`): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `SONAR_URL`, `SONAR_TOKEN`

---

### 8.5 Documentation

**Standards**:
- Docstrings for all public functions (Google style)
- Type hints throughout
- Architecture diagrams in docs/
- LaTeX thesis chapter in docs/methodology_system_design.tex

**Generated docs**:
- Presentation: `slides.md` (Slidev)
- System design summary: `docs/SYSTEM_DESIGN_SUMMARY.md`
- Architecture: `docs/architecture.md`

---

## Appendices

### A. Dependency rules reference

Complete dependency rules from `agents/dependency_analysis/agent.py`:

```python
DEPENDENCY_RULES = {
    "Long Method": {
        "positive": [
            "Switch Statement",
            "Feature Envy",
            "Duplicated Code",
            "Divergent Change",
            "Comments",
            "Long Parameter List",
        ],
        "negative": ["Long Method", "Long Parameter List"],
    },
    "Complex Method": {
        "positive": [
            "Switch Statement",
            "Feature Envy",
            "Duplicated Code",
            "Divergent Change",
            "Comments",
            "Long Parameter List",
        ],
        "negative": ["Long Method", "Long Parameter List"],
    },
    "Conditional Complexity": {
        "positive": [
            "Switch Statement",
            "Feature Envy",
            "Duplicated Code",
            "Divergent Change",
            "Comments",
            "Long Parameter List",
        ],
        "negative": ["Long Method", "Long Parameter List"],
    },
    "Long Parameter List": {
        "positive": ["Long Parameter List", "Data Clumps"],
        "negative": ["Data Class"],
    },
    "Large Class": {
        "positive": ["Data Clumps", "Feature Envy", "Bad Class Content"],
        "negative": [
            "Long Method",
            "Data Class",
            "Inappropriate Intimacy",
            "Message Chains",
        ],
    },
    "God Class": {
        "positive": ["Data Clumps", "Feature Envy", "Bad Class Content"],
        "negative": [
            "Long Method",
            "Data Class",
            "Inappropriate Intimacy",
            "Message Chains",
        ],
    },
    "Duplicated Conditions": {
        "positive": ["Divergent Change", "Shotgun Surgery"],
        "negative": ["Large Class", "Bad Inheritance"],
    },
    "Print Statements": {
        "positive": ["Needless Part"],
        "negative": ["Data Class", "Lazy Class"],
    },
}
```

### B. Prioritization algorithm

**Formula**: `PZ_i = Severity_i + Σ(w_impact for positive deps) - penalty(negative deps)`

**Parameters**:
- Severity weights: HIGH=3, MEDIUM=2, LOW=1
- Impact weight: w_impact = 2 (per positive dependency)
- Negative dependency penalty: Reduces priority based on potential side effects

**Best-first search algorithm**:
1. Calculate PZ for all detected smells
   - Higher PZ = more cascading benefits AND fewer side effects
   - Goal: Maximize positive dependencies, minimize negative dependencies
2. Select smell with max(PZ) considering both benefits and costs
3. Apply refactoring (actual code transformation, not simulation)
4. Update graph:
   - Remove solved smells based on positive dependencies (cascading effect)
   - Add created smells based on negative dependencies (side effects)
5. Recalculate PZ scores with updated graph
6. Check for cycles:
   - If refactoring A creates smell B, and refactoring B creates smell A, mark as outlier
   - Implement max-step limit to prevent infinite loops
7. Repeat until no smells remain or cycle detected

**Rationale**: Prioritize refactorings that solve multiple smells (high positive dependency count) while minimizing introduction of new smells (low negative dependency count). The algorithm balances severity with cascading impact and side effects.

**Future evolution**: May evolve to count whole sum of dependencies along graph paths rather than greedy best-first selection.

### C. MLflow scorer implementation

**Updated F1 Score Scorer** (replaces mapping_accuracy):

```python
from mlflow.metrics import make_genai_metric

def f1_score_scorer(eval_result, ground_truth):
    """
    Calculate F1 score for refactoring-to-hunk mapping predictions.

    F1 = 2 * (precision * recall) / (precision + recall)

    This replaces the old mapping_accuracy scorer which only measured recall
    and rewarded "shotgunning" (predicting everything).
    """
    predictions = set(eval_result["predicted_line_ranges"])
    expected = set(ground_truth["ground_truth_line_ranges"])

    if not expected and not predictions:
        return 1.0  # Perfect score if nothing to predict and nothing predicted

    if not expected or not predictions:
        return 0.0  # No match if one is empty

    # Calculate overlap
    overlap = predictions.intersection(expected)

    # Precision: How many predictions were correct?
    precision = len(overlap) / len(predictions) if predictions else 0.0

    # Recall: How many ground truth items were found?
    recall = len(overlap) / len(expected) if expected else 0.0

    # F1 score
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

f1_score = make_genai_metric(
    name="f1_score",
    eval_fn=f1_score_scorer,
    greater_is_better=True,
    aggregations=["mean", "variance", "p90", "p50"]
)
```

**Why F1 instead of mapping_accuracy?**

The old `mapping_accuracy = len(overlap) / len(expected)` had a critical flaw:

- **Example**: Ground truth = `[1]`, Prediction = `[1, 2, 3, 4, 5]`
- **Old score**: `1/1 = 1.0` (perfect!)
- **Problem**: Model "shotgunned" by guessing everything and still got perfect score

F1 score penalizes both false positives (extra predictions) and false negatives (missed ground truth items).

---

## Change log

| Version | Date | Changes |
|---------|------|---------|
| 1.1 | 2026-01-12 | Addressed 34 ambiguities identified in specification review. Major updates: clarified agent workflow, removed sparse checkout, added F1 score metric, documented temperature settings, clarified refactoring execution approach, added TODOs for future work. |
| 1.0 | 2026-01-12 | Initial specification based on codebase analysis |

---

## References

1. RefactoringMiner 2.0: Tsantalis, N., Ketkar, A., Dig, D. IEEE TSE 2022
2. Stack Overflow: "A practical guide to writing technical specs" (guide source)
3. Dependency analysis theory: Markovič & Polášek (cited in system design docs)
4. DACOS dataset: Nandani, H., Saad, M., & Sharma, T. MSR 2023 (referenced in architecture.md)
