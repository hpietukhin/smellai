SmellAI - Multi-Agent Refactoring System

> **📋 Documentation**: [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) is the authoritative source for system architecture and design. This README covers quick start and usage.

## What is this?

A multi-agent system for code smell detection, dependency-aware prioritization, and behavior-preserving refactoring. The system implements 8 specialized agents (A0-A7) that orchestrate a complete refactoring workflow: detecting code smells via SonarQube (A1), prioritizing them using dependency graph analysis (A3), executing refactorings in optimal order (A5), and verifying behavior preservation through test execution (A0/A6). Built for master's thesis research on LLM-based refactoring evaluation using ground truth datasets (RefactoringMiner 2.0, SWE-Refactor). Uses LangGraph for multi-agent orchestration, MLflow for experiment tracking, and SonarQube for automated smell detection.

**Key capabilities**:
- 🔍 Automated code smell detection (8 smell types via SonarQube)
- 📊 Dependency-aware prioritization (NetworkX graph analysis)
- 🤖 LLM-based refactoring mapping (OpenAI, Anthropic, Cerebras)
- ✅ Behavior preservation verification (Maven/Gradle test execution)
- 📈 Experiment tracking and analysis (MLflow)
- 🔄 Multiple dataset support via adapter pattern

**Agent workflow**: A0 (Test Coverage) → A1 (SonarQube Scan) → A2 (Developer Query) → A3 (Prioritization) → A4 (Prompt Prep) → A5 (Refactoring Loop) → A6 (Behavior Verification) → A7 (Test Generation if needed)

## Quick start

```bash
# 1. Install dependencies
uv pip install .

# 2. Configure environment variables
cp .env.example .env
# Edit .env with your API keys (OPENAI_API_KEY, SONAR_TOKEN)

# 3. Start SonarQube (for smell detection)
docker compose -f sonarqube/docker-compose.yml up -d

# 4. Create evaluation dataset (RefactoringMiner)
uv run scripts/create_rminer_dataset.py \
    --manifest rminer_data/manifest.json \
    --limit 5 \
    --experiment rminer-evaluation

# 5. Run evaluation
uv run workflows/rminer_eval_workflow.py \
    --dataset-name rminer-eval-dataset \
    --experiment rminer-evaluation

# 6. View results
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

See [Working Example](#working-example) for complete end-to-end tutorial.

## Installation

**Prerequisites:**
- Python 3.11+ (installed automatically by uv)
- Docker (for SonarQube)
- Maven 3.6+ or Gradle 6.0+ (for Java test analysis)

**Install dependencies:**
```bash
# uv handles Python version and dependencies automatically
uv pip install .
```

**Install uv** (if not already installed):
- macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- See https://docs.astral.sh/uv/ for other platforms

### Environment variables

Copy `.env.example` to `.env` and configure with your API keys:
```bash
cp .env.example .env
# Edit .env with your actual values
```

**Required variables:**
- `OPENAI_API_KEY`: OpenAI API key for GPT models (get from https://platform.openai.com/api-keys)
- `CEREBRAS_API_KEY`: Cerebras API key for fast inference (alternative LLM provider)
- `SONAR_TOKEN`: SonarQube authentication token (create in SonarQube UI: My Account → Security)
- `SONAR_URL`: SonarQube server URL (default: `http://localhost:9000`)

**Optional variables:**
- `MLFLOW_TRACKING_URI`: MLflow tracking database (default: `sqlite:///mlflow.db`)
- `ANTHROPIC_API_KEY`: Anthropic API key for Claude models (optional alternative provider)

## Datasets

The system supports multiple evaluation datasets via adapter pattern (`datasets/` directory):

### RefactoringMiner 2.0 (Primary)

Ground truth for refactoring mapping evaluation. Contains real refactorings from 188 open-source projects.

**Create dataset:**
```bash
uv run scripts/create_rminer_dataset.py \
    --manifest rminer_data/manifest.json \
    --limit 20 \
    --experiment rminer-evaluation
```

**List datasets:**
```bash
uv run scripts/manage_datasets.py list
```

**Inspect dataset:**
```bash
uv run scripts/manage_datasets.py get --name rminer-eval-dataset --show-records
```

**Run evaluation:**
```bash
uv run workflows/rminer_eval_workflow.py \
    --dataset-name rminer-eval-dataset \
    --model gpt-4o-mini
```

**View results:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

See [docs/README_RMINER.md](docs/README_RMINER.md) for detailed RefactoringMiner workflow documentation.

### SWE-Refactor (Alternative)

Alternative refactoring dataset with before/after code examples. Located in `swe_refactor/` directory.

### DACOS (Alternative)

MySQL-based dataset for code smell detection evaluation. Requires MySQL 8.0+ and dataset import.

**Note**: Datasets are interchangeable via adapter interface in `datasets/base.py`. All evaluation workflows work with any supported dataset.

## Agent workflows

### A0/A6: Java test analysis (Test Coverage & Behavior Verification)

Auto-detects Maven/Gradle, runs tests, provides LLM-powered failure analysis.

```bash
uv run workflows/java_test_workflow.py --project /path/to/java/project
```

See [docs/java_test_agent.md](docs/java_test_agent.md) for detailed documentation.

### A1: SonarQube smell detection

Start SonarQube container and scan repository for code smells.

```bash
# Start SonarQube
docker compose -f sonarqube/docker-compose.yml up -d

# Scan repository
uv run sonarqube/commit_scan.py --repo-url https://github.com/user/repo --commit abc123
```

See [docs/sonarqube_smells.md](docs/sonarqube_smells.md) for smell types and configuration.

### A3: Dependency-aware prioritization

Analyzes smell dependencies and calculates optimal refactoring order.

```bash
uv run scripts/prioritize_smells.py --smells-file smells.json
```

Generates prioritization graphs visualizing positive/negative dependencies.

### Agent Execution Visualizer

Interactive web UI for analyzing agent execution and understanding refactoring decisions:

```bash
uv run python tools/visualize_smell_prioritization.py
# Open http://localhost:8080
```

**Features:**
- 📊 Agent execution timeline (node invocations, durations)
- 🕸️ Smell dependency graph with PZ prioritization
- 📝 Iteration details (outcomes, retries, smells resolved/created)
- 🔧 Tool call logs for debugging
- 📄 Code diff viewer showing actual changes
- 💡 Decision rationale (PZ scores, dependencies)
- 📚 Real-world composite refactoring examples

See [VISUALIZATION_USAGE.md](VISUALIZATION_USAGE.md) for complete guide.

### A5: Refactoring execution

LLM-based refactoring mapping and execution (see RefactoringMiner workflow above).

## Working example

For a complete end-to-end tutorial with all agents, see [docs/react_agent_mlflow.md](docs/react_agent_mlflow.md).

This guide covers:
- Setting up SonarQube and MLflow
- Running complete evaluation workflow
- Analyzing results and metrics
- Troubleshooting common issues

## Advanced usage

### Dependency management with uv

- Lock dependencies: `uv pip compile pyproject.toml -o uv.lock`
- Reproducible install: `uv pip sync uv.lock`
- Add package: `uv add package-name`

### Jupyter notebook integration

See detailed Jupyter setup in [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) section 3.5.

## Documentation

- **[TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md)** - Complete system architecture and design
- **[docs/SYSTEM_DESIGN_SUMMARY.md](docs/SYSTEM_DESIGN_SUMMARY.md)** - Multi-agent architecture overview
- **[docs/architecture.md](docs/architecture.md)** - Detailed component descriptions
- **[docs/java_test_agent.md](docs/java_test_agent.md)** - Java test analysis agent
- **[docs/README_RMINER.md](docs/README_RMINER.md)** - RefactoringMiner workflow
- **[docs/sonarqube_smells.md](docs/sonarqube_smells.md)** - SonarQube integration and smell types