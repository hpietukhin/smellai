# Tech Stack Document: LLM-Based Code Smell Detection

**Version**: 1.0  
**Date**: 2025-10-18  
**Status**: Draft  
**Project**: Master's Thesis - Code Smell Detection Using LLMs

## 1. Core Technology Stack

### 1.1 Programming Language
- **Python**: `>=3.11,<3.13` (uv/pip version constraint)

### 1.2 LLM Orchestration
- **langgraph**: `^0.2.0`
- **langchain**: `^0.3.0`
- **langchain-community**: `^0.3.0`
- **litellm**: `^1.0.0`

### 1.3 LLM Provider
- **ChatLiteLLM with Cerebras**: `cerebras/llama3.1-8b` (detection & judging)
  - Uses LiteLLM's unified API to access Cerebras models
  - High-speed inference via Cerebras accelerators
- **Google Embeddings**: `text-embedding-004`

### 1.4 Evaluation Framework
- **promptfoo**: `^0.80.0` (npm global install)

### 1.5 Vector Database
- **deeplake**: `<4.0.0` (v3.x)

### 1.6 Database
- **MySQL**: `>=8.0`
- **mysql-connector-python**: `^9.0.0`

### 1.7 Version Control Integration
- **GitPython**: `^3.1.0`

### 1.8 Baseline Tool
- **SonarQube**: `sonarqube:10.6.0-community` (Docker)
- **sonar-scanner-cli**: `5.0` (Docker)

## 2. Development Tools

### 2.1 Dependency Management
- **uv**: `>=0.5.0` (recommended package manager)
- **pip**: Alternative if uv not available

### 2.2 Code Quality
- **ruff**: `^0.5.0` (linting & formatting)
- **mypy**: `^1.10.0` (type checking)

### 2.3 Testing
- **pytest**: `^8.0.0`
- **pytest-cov**: `^4.1.0`
- **pytest-asyncio**: `^0.23.0`

### 2.4 Environment Management
- **python-dotenv**: `^1.0.0`
- **pydantic-settings**: `^2.0.0` (for configuration management)

## 3. Data & Model Libraries

### 3.1 Data Validation
- **pydantic**: `^2.0.0`

### 3.2 Data Manipulation
- **pandas**: `^2.2.0`

### 3.3 HTTP Client
- **requests**: `^2.31.0`

## 4. Infrastructure & Deployment

### 4.1 Containerization
- **Docker Engine**: `>=24.0`
- **Docker Compose**: `>=2.20`

### 4.2 Shell Scripting
- **Bash**: `>=4.0`

## 5. Documentation Tools

### 5.1 Notebook Environment
- **jupyter**: `^1.0.0`
- **ipython**: `^8.20.0`

### 5.2 Markdown Rendering
- **python-markdown**: `^3.6.0`

## 6. System Dependencies

### 6.1 Required
- Python 3.11+
- MySQL 8.0+
- Docker Engine 24.0+
- Docker Compose 2.20+
- Node.js 18+ (for Promptfoo)
- Git 2.30+
- Bash 4.0+

### 6.2 Optional
- uv 0.5+ (recommended package manager)

## 7. Repository Structure

```
project/
├── .github/
│   └── workflows/
├── src/
│   ├── __init__.py
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── evaluation_pipeline.py
│   │   └── nodes.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── judge.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mysql_connector.py
│   │   ├── git_ops.py
│   │   └── vector_db.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── entities.py
│   └── config/
│       ├── __init__.py
│       ├── settings.py
│       └── defaults.py
├── experiments/
│   └── notebooks/
│       └── prototype_eval.ipynb
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── conftest.py
├── infra/
│   ├── sonarqube/
│   │   ├── docker-compose.yml
│   │   ├── analyze_baseline.sh
│   │   └── README.md
│   └── mysql/
│       ├── import_dacos.sh
│       └── README.md
├── scripts/
│   ├── verify_env.py
│   ├── clone_repos.py
│   └── export_test_cases.py
├── eval_results/           # Git-ignored
├── data/                   # Git-ignored
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
├── promptfoo.config.yaml
├── README.md
├── CLAUDE.md
├── CONSTITUTION.md
└── docs/
    ├── 01-architecture.md
    ├── 02-tech-stack.md
    ├── 03-setup-instructions.md
    ├── 04-key-concepts.md
    └── 05-development-conventions.md
```