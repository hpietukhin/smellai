# Tech Stack Document: LLM-Based Code Smell Detection

**Version**: 1.0  
**Date**: 2025-10-18  
**Status**: Draft  
**Project**: Master's Thesis - Code Smell Detection Using LLMs

## 1. Core Technology Stack

### 1.1 Programming Language
- **Python**: `>=3.11,<3.13` (uv/pip version constraint)

### 1.2 LLM Orchestration
- **langgraph**: `>=0.2.0`
- **langchain**: `>=0.3.0`
- **langchain-community**: `>=0.3.0`
- **langchain-google-genai**: `>=2.0.0`
- **litellm**: `>=1.0.0`

### 1.3 LLM Provider
- **LiteLLM with Cerebras**: `cerebras/llama3.1-8b` (detection & judging)
  - Uses LiteLLM's unified API to access Cerebras models
  - High-speed inference via Cerebras accelerators
  - Requires CEREBRAS_API_KEY environment variable
- **Google Generative AI Embeddings**: `models/text-embedding-004` (768 dimensions)
  - Used for semantic search in vector database
  - Requires GOOGLE_API_KEY environment variable

### 1.4 Experiment Tracking & Tracing
- **mlflow**: `>=3.0.0` (tracing and experiment tracking)
  - Automatic LangGraph tracing via `mlflow.langchain.autolog()`
  - Local tracking server at `./mlruns` (default)

### 1.5 Vector Database
- **deeplake**: `<4.0.0` (v3.x)

### 1.6 Database
- **MySQL**: `>=8.0`
- **mysql-connector-python**: `>=9.0.0`
  - Connection pooling for DACOS dataset access
  - Configuration via environment variables (MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD)

### 1.7 Version Control Integration
- **GitPython**: `>=3.1.0`
  - Sparse checkout for efficient file retrieval
  - Commit selection before cutoff dates

### 1.8 Baseline Tool
- **SonarQube**: `sonarqube:10.6.0-community` (Docker)
- **sonar-scanner-cli**: `5.0` (Docker)

## 2. Development Tools

### 2.1 Dependency Management
- **uv**: Latest (recommended package manager)
  - Fast dependency resolution and installation
  - Lockfile: `uv.lock`
- **pip**: Alternative if uv not available

### 2.2 Code Quality
- **ruff**: `>=0.5.0` (linting & formatting)
- **mypy**: `>=1.10.0` (type checking)
- **autoflake**: `>=2.3.1` (remove unused imports, dev dependency)
- **autopep8**: `>=2.3.2` (PEP 8 formatting, dev dependency)
- **isort**: `>=7.0.0` (import sorting, dev dependency)

### 2.3 Testing
- **pytest**: `>=8.0.0`
- **pytest-cov**: `>=4.1.0`
- **pytest-asyncio**: `>=0.23.0`

### 2.4 Environment Management
- **python-dotenv**: `>=1.0.0`
- **pydantic-settings**: `>=2.0.0`

## 3. Data & Model Libraries

### 3.1 Data Validation
- **pydantic**: `>=2.0.0`
  - Structured output from LLMs
  - Entity models validation

### 3.2 Data Manipulation
- **pandas**: `>=2.2.0`
  - Optional for data analysis
- **datasets**: `>=4.2.0` (dev dependency, for data processing)

### 3.3 HTTP Client
- **requests**: `>=2.31.0`
  - SonarQube API integration

### 3.4 Additional Development Tools
- **openai**: `>=2.3.0` (dev dependency, for experimentation)
- **weave**: `>=0.52.9` (dev dependency, for experiment tracking)
- **set-env-colab-kaggle-dotenv**: `>=0.1.4` (dev dependency, environment setup)

## 4. Infrastructure & Deployment

### 4.1 Containerization
- **Docker Engine**: `>=24.0`
- **Docker Compose**: `>=2.20`

### 4.2 Shell Scripting
- **Bash**: `>=4.0`

## 5. Documentation Tools

### 5.1 Notebook Environment
- **jupyter**: `>=1.0.0`
- **ipython**: `>=8.20.0`

## 6. System Dependencies

### 6.1 Required
- Python 3.11+
- MySQL 8.0+
- Docker Engine 24.0+
- Docker Compose 2.20+
- Git 2.30+
- Bash 4.0+
- MLflow 3.0+ (for tracing and tracking)

### 6.2 Optional
- uv (latest, recommended package manager for faster dependency resolution)

## 7. Repository Structure

```
smellai/
├── .claude/                # Claude AI configuration
├── .git/                   # Git repository
├── .github/                # GitHub workflows (if any)
├── .idea/                  # IDE configuration (git-ignored)
├── .pytest_cache/          # Pytest cache (git-ignored)
├── .ruff_cache/            # Ruff cache (git-ignored)
├── .venv/                  # Virtual environment (git-ignored)
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── detector.py             # LLM smell detector with RAG
│   │   └── judge.py                # LLM-as-judge evaluator
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── evaluation_pipeline.py  # Main LangGraph pipeline
│   │   └── nodes.py                # Pipeline nodes (fetch, clone, detect, judge)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mysql_connector.py      # DACOS database access
│   │   ├── git_ops.py              # Git operations (sparse checkout)
│   │   └── vector_db.py            # DeepLake vector database setup
│   ├── models/
│   │   ├── __init__.py
│   │   └── entities.py             # Pydantic models (SmellDetection, EvaluationResult, etc.)
│   └── smellai.egg-info/           # Package metadata (auto-generated)
├── tests/                           # Test directory (minimal)
├── infra/
│   └── sonarqube/
│       └── baseline_scan.py        # Python script for SonarQube baseline
├── pipeline_reference/              # Reference implementation
├── docs/
│   ├── architecture.md             # System architecture documentation
│   ├── tech_stack.md               # This file - technology stack
│   ├── tasks.md                    # Task tracking
│   └── sonarqube_smells.md         # SonarQube smell mapping
├── eval_results/                   # Evaluation outputs (git-ignored)
├── .env                            # Environment variables (git-ignored)
├── .env.example                    # Template for environment variables
├── .gitignore
├── .pre-commit-config.yaml         # Pre-commit hooks configuration
├── .python-version                 # Python version specification
├── pyproject.toml                  # Project dependencies (uv/pip)
├── uv.lock                         # Locked dependencies
├── CLAUDE.md                       # Instructions for Claude AI agents
├── LICENSE                         # MIT License
└── README.md                       # Project overview
```

**Key Directories**:
- `src/` - Main source code (agents, pipelines, data access, models)
- `infra/` - Infrastructure scripts (SonarQube baseline)
- `docs/` - Project documentation
- `tests/` - Test suite (unit and integration tests)
- `pipeline_reference/` - Reference implementation from prototype
- `eval_results/` - Generated evaluation results (git-ignored)
- `.venv/` - Python virtual environment (git-ignored)