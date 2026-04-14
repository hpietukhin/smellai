---
title: SmellAI — Onboarding
author: Master's Thesis Codebase Tour
theme:
  name: dark
---

# SmellAI

**What is this project?**

A research system for evaluating **LLM-based code smell detection and refactoring** using multi-agent orchestration.

<!-- pause -->

**Core idea:**
1. Detect code smells in Java repos (via SonarQube)
2. Prioritize them by dependency graph
3. Ask an LLM agent to refactor
4. Verify behavior is preserved (tests)
5. Track everything in MLflow

<!-- pause -->

**Stack:** Python 3.11 · LangGraph · LiteLLM · MLflow · SonarQube · SQLModel · NetworkX

<!-- end_slide -->

# Repository Layout

```
smellai/
├── agents/            ← LangGraph agents (A0–A7)
├── smellai_datasets/  ← Raw → DataFrame pipeline
├── datasets/          ← RMiner adapter & utilities
├── swe_refactor/      ← SWE-Refactor domain objects & tools
├── sonarqube/         ← SonarQube scanner integration
├── mlflow_utils/      ← MLflow dataset management
├── workflows/         ← End-to-end evaluation workflows
├── scripts/           ← CLI tools and dataset scripts
├── tools/             ← Interactive visualizer (NiceGUI)
├── evals/             ← Custom MLflow scorers
├── repo_utils/        ← Git/build helpers shared across agents
├── models/            ← Shared Pydantic domain models
├── tests/             ← pytest test suite
└── prompts/           ← Prompt templates
```

<!-- end_slide -->

# `agents/` — Multi-Agent System

The core of the project. Each subfolder is one LangGraph agent.

<!-- pause -->

**Agent pipeline (A0 → A6):**

```
A0 (setup)  →  A1 (detect)  →  A3 (prioritize)
   →  A5 (refactor)  →  A6 (verify)
```

<!-- pause -->

| Subfolder | Role |
| --- | --- |
| `java_test/` | **A0/A6** — Run Maven/Gradle tests, parse results |
| `dependency_analysis/` | **A3** — Build smell dependency graph, compute PZ score |
| `rminer_eval/` | **A5** — Map refactorings from RMiner ground truth |
| `swe_eval/` | **A5** — Generate refactored code, verify compilation |
| `baseline/` | **Baseline** — Simple non-agentic reference implementation |
| `tools/` | Shared LangChain tool definitions used by multiple agents |

<!-- end_slide -->

# `agents/java_test/`

**Purpose:** Run Java project tests and report pass/fail — used both for setup (A0) and behavior verification (A6).

```
agent.py    ← LangGraph graph: detect build system → run tests → parse results
__init__.py ← Package export
```

<!-- pause -->

**Key logic in `agent.py`:**
- Detects `pom.xml` → Maven, `build.gradle` → Gradle
- Runs compile + test, captures stdout/stderr
- Returns `CompileResult(success, error_summary)`

<!-- end_slide -->

# `agents/dependency_analysis/`

**Purpose:** Analyze positive/negative smell dependencies, produce a prioritized list via the PZ formula.

```
agent.py    ← NetworkX graph builder + PZ scorer
__init__.py ← Package export
```

<!-- pause -->

**PZ formula:**
```
PZ_i = Severity_i + Σ(w for each positive dependency)
```

- **Green edges** = refactoring smell i also helps smell j (positive)
- **Red dashed edges** = refactoring smell i may create smell j (negative)

<!-- end_slide -->

# `agents/rminer_eval/`

**Purpose:** Given a RMiner manifest (before/after files + refactoring type), ask the LLM to map which hunks implement which refactoring.

```
agent.py    ← LangGraph graph with structured output + JSON fallback
config.py   ← AgentConfig enum, DEFAULT_CONFIG (model, retries)
scorers.py  ← mapping_accuracy, hunk_coverage custom MLflow scorers
__init__.py ← Package export
```

<!-- pause -->

Default model: **`gpt-4o-mini`** (cheap, fast for mapping task)

<!-- end_slide -->

# `agents/swe_eval/`

**Purpose:** Given a smell + context, generate refactored Java code and verify it compiles.

```
agent.py    ← LangGraph graph: load → generate → compile → retry loop
config.py   ← same pattern as rminer_eval
prompts.py  ← System/user prompt templates for code generation
__init__.py ← Package export
```

<!-- pause -->

Default model: **`claude-sonnet-4-5-20250929`**

Uses `swe_refactor/utils/` for git ops and jenv switching.

<!-- end_slide -->

# `agents/baseline/` and `agents/tools/`

**`baseline/`** — Non-agentic reference: direct LLM call, no graph, used to compare against the full multi-agent approach.

```
agent.py    ← Single-shot LLM call baseline
__init__.py ← same
```

<!-- pause -->

**`tools/`** — Shared LangChain `@tool` functions imported by multiple agents.

```
java_test_tools.py  ← Tools: run_maven_tests, run_gradle_tests, parse_test_output
__init__.py         ← same
```

<!-- end_slide -->

# `smellai_datasets/` — Dataset Pipeline

**Purpose:** Convert raw data sources → canonical pandas DataFrame → Parquet → MLflow.

```
models.py        ← DiffHunk Pydantic model (canonical data class)
converter.py     ← Raw → DataFrame: rminer_to_df, swe_refactor_to_df, tdd_to_df
preprocessor.py  ← Dedup, filter, train/test split, save/load Parquet
mlflow_bridge.py ← DataFrame → MLflow GenAI format (hf_to_genai_records, load_for_evaluation)
config.py        ← DATASET_CONFIGS (dedup keys, filters) + MLFLOW_COLUMN_MAP
__init__.py      ← Package export
```

<!-- pause -->

**Data flow:**
```
Raw JSON/DB  →  converter  →  DataFrame  →  preprocessor  →  Parquet
                                                 ↓
                                          mlflow_bridge  →  MLflow experiment
```

<!-- end_slide -->

# `datasets/` — RMiner Adapter

**Purpose:** Parse RefactoringMiner output and turn it into the canonical format used by agents.

```
rminer_utils.py          ← Parse RMiner JSON: extract file pairs, hunks, refactoring type
create_rminer_dataset.py ← CLI: build MLflow dataset from RMiner manifest
extract_rminer_data.py   ← Extract raw RMiner JSON from a git repo
explore_rminer.py        ← REPL-style exploration of RMiner data
__init__.py              ← Package export
```

<!-- pause -->

**RMiner manifest** (`rminer_data/manifest.json`):
```json
{ "before": "Foo.java", "after": "Foo.java",
  "refactoring": "Extract Method", "hunks": [...] }
```

<!-- end_slide -->

# `swe_refactor/` — SWE Domain Layer

**Purpose:** Domain objects and utilities for the SWE-Refactor evaluation (the full code generation + test pipeline).

```
dataset.py        ← RefactoringRecord (the main domain object passed between agents)
dataset_card.md   ← Human-readable description of the SWE-Refactor dataset
```

<!-- pause -->

**Subfolders:**

| Subfolder | Purpose |
| --- | --- |
| `analytics/` | `__init__.py` only — namespace for analytics models |
| `persistence/` | SQLModel ORM: `ToolCall`, `SmellEvent`, `RefactoringAttempt`, `TokenUsage` |
| `smell_detection/` | `__init__.py` only — namespace for detection helpers |
| `utils/` | `build_util.py` (Maven/Gradle), `jenv_util.py` (JDK switching), `project_util.py` (path helpers) |

<!-- end_slide -->

# `swe_refactor/persistence/`

**Purpose:** Track agent execution analytics in SQLite, separate from LangGraph checkpoints.

```
database.py  ← AnalyticsDB class: session management, CRUD for all event types
models.py    ← SQLModel tables: ToolCall, SmellEvent, RefactoringAttempt, TokenUsage, SmellDependency
__init__.py  ← Package export
```

<!-- pause -->

Database file: **`test_analytics.db`** (do not confuse with `mlflow.db`)

<!-- end_slide -->

# `sonarqube/` — Smell Detection

**Purpose:** Run SonarQube scanner against a Java repo at a specific commit and collect code smell issues.

```
commit_scan.py       ← Main scanner: clone repo, checkout commit, run sonar-scanner, poll API, emit SmellEvents
tool.py              ← LangChain @tool wrapper around commit_scan for agent use
docker-compose.yml   ← SonarQube 10.x container on port 9000
sonarqube_server.sh  ← Helper to start/stop server
__init__.py          ← Package export
```

<!-- pause -->

**8 smell types mapped:**
`Long Method`, `Complex Method`, `Complex Class`, `Long Parameter List`,  
`Duplicated Code`, `Feature Envy`, `Deep Hierarchy`, `Brain Class`

<!-- end_slide -->

# `mlflow_utils/` — Experiment Management

**Purpose:** Manage MLflow datasets and run evaluations programmatically.

```
auto_server.py   ← Auto-start MLflow server if not running
cli.py           ← CLI entry point: list, get, delete datasets
runner.py        ← Run MLflow evaluate() with custom scorers
server.py        ← Server lifecycle (start/stop/health-check)
__init__.py      ← Package export
```

**`datasets/` subfolder:**
```
manager.py   ← DatasetManager: create_dataset_from_records, list, get, delete
__init__.py  ← same
```

<!-- end_slide -->

# `workflows/` — End-to-End Pipelines

**Purpose:** Orchestrate agents into complete evaluation runs, from loading a dataset to logging results in MLflow.

```
eval_workflow.py              ← Unified entry point (replaces old rminer + swe workflows)
rminer_eval_workflow.py       ← Legacy: RMiner mapping evaluation
swe_eval_workflow.py          ← Legacy: SWE-Refactor generation evaluation
composite_analysis_workflow.py← Multi-smell iterative refactoring loop
java_test_workflow.py         ← Standalone: just run Java tests on a project
smell_cooccurrence_workflow.py← Analyze which smells appear together
baseline_eval_workflow.py     ← Baseline (non-agentic) evaluation run
common.py                     ← Shared helpers: load dataset, init MLflow run, log results
utils.py                      ← Shared utilities across workflows
```

<!-- end_slide -->

# `scripts/` — Developer CLI Tools

**Purpose:** One-off and recurring scripts for dataset management, visualization, and demo runs.

```
prioritize_smells.py           ← CLI: compute PZ scores for a smells JSON file
extract_compound_refactorings.py ← Extract multi-step refactoring chains from RMiner
dashboard.py                   ← Streamlit analytics dashboard
run_composite_demo.sh          ← Shell script to run composite demo end-to-end
run_demo_eval.py               ← same
run_visualizer.py / .sh        ← Launch the NiceGUI visualizer
visualize_analytics.py         ← Plot analytics from test_analytics.db
```

**`datasets/` subfolder:**
```
analyze.py        ← Analyze dataset statistics (coverage, balance)
preprocess.py     ← Preprocess raw data into Parquet via smellai_datasets pipeline
inspect_rminer.py ← Interactive inspection of RMiner JSON
__init__.py       ← same
```

<!-- end_slide -->

# `tools/` — Interactive Visualizer

**Purpose:** NiceGUI web app (port 8080) for interactively exploring smell prioritization and agent execution.

```
visualize_smell_prioritization.py  ← Full NiceGUI app: graph view, PZ table, dependency explorer
example_manifests/                  ← Sample smell JSON files for demo/testing
```

<!-- pause -->

Launch:
```bash
uv run python tools/visualize_smell_prioritization.py
# → http://localhost:8080
```

<!-- end_slide -->

# `evals/` — Custom Scorers

**Purpose:** MLflow-compatible scorer functions that measure domain-specific quality of agent outputs.

*(Currently empty — scorers live in `agents/rminer_eval/scorers.py` and `agents/swe_eval/config.py`)*

Planned scorers:
- `mapping_accuracy` — fraction of hunks correctly attributed
- `compile_success_rate` — fraction of generated code that compiles
- `test_pass_rate` — fraction of test suites that pass after refactoring

<!-- end_slide -->

# `repo_utils/` — Git & Build Helpers

**Purpose:** Shared utilities for cloning, checking out, and manipulating Java repositories.

```
operations.py    ← clone_repository, force_checkout_commit, get_previous_commit
test_execution.py← run_tests_in_repo (wraps Maven/Gradle, returns structured result)
errors.py        ← Custom exception types for repo operations
__init__.py      ← Package export
```

<!-- end_slide -->

# `models/` — Shared Domain Models

**Purpose:** Pydantic models shared across multiple packages (not agent-specific).

```
refactoring.py  ← RefactoringType enum + RefactoringLocation model
__init__.py     ← Package export
```

Note: `DiffHunk` (the other key model) lives in `smellai_datasets/models.py`.

<!-- end_slide -->

# `tests/` — Test Suite

**Purpose:** pytest tests covering agents, utilities, and data contracts.

```
conftest.py                  ← Shared fixtures (temp repos, fake MLflow, sample records)
test_commit_scan.py          ← SonarQube scanner integration tests
test_data_contracts.py       ← Assert dataset schema stays stable across pipeline changes
test_java_test_agent.py      ← Agent A0/A6 unit tests
test_mlflow_server.py        ← MLflow server lifecycle tests
test_repo_utils.py           ← Git operation tests
test_rminer_utils.py         ← RMiner JSON parsing tests
test_smell_cooccurrence.py   ← Cooccurrence workflow tests
test_smell_prioritization.py ← PZ scorer and graph tests
test_workflows_common.py     ← Shared workflow helper tests
test_workflows_utils.py      ← same
test_data/                   ← Fixtures: sample Java files, RMiner JSON, smell JSONs
```

Run all: `uv run pytest`

<!-- end_slide -->

# `prompts/` — Prompt Templates

**Purpose:** Standalone prompt files used during dataset search and agent prompt construction.

```
dataset_search_prompt.md  ← Prompt for searching/selecting relevant dataset records
```

Agent-specific prompts live in `agents/swe_eval/prompts.py`.

<!-- end_slide -->

# Key Concepts to Remember

**1. State flows as TypedDict** — each agent node returns `dict` (partial state update), never mutates.

**2. LiteLLM = model switcher** — change `model=` string to swap OpenAI ↔ Anthropic ↔ Cerebras.

**3. Two databases, two purposes:**
- `mlflow.db` — experiment runs, datasets, metrics (MLflow)
- `test_analytics.db` — agent execution events (SQLModel/SQLite)

**4. Smell lifecycle:**
```
detected → (refactoring attempt) → resolved | created (new smell)
```

**5. Dataset pipeline order:**
```
Raw → converter.py → DataFrame → preprocessor.py → Parquet → mlflow_bridge.py → MLflow
```

<!-- end_slide -->

# Quick Start

```bash
# 1. Install deps
uv sync --all-groups

# 2. Start SonarQube
docker compose -f sonarqube/docker-compose.yml up -d

# 3. Start MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 4. Run unified eval
uv run workflows/eval_workflow.py \
    --dataset-name rminer-eval-dataset \
    --model gpt-4o-mini

# 5. Visualize
uv run python tools/visualize_smell_prioritization.py
```

<!-- pause -->

**Environment variables required:** `OPENAI_API_KEY`, `SONAR_TOKEN`, `SONAR_URL`, `MLFLOW_TRACKING_URI`

<!-- end_slide -->

# That's It

Questions? Check:
- `CLAUDE.md` — full architecture reference
- `TECHNICAL_SPECIFICATION.md` — design decisions
- `docs/` — per-component deep dives
- `mlflow ui` — experiment history

<!-- pause -->

```
Good luck. Leave the codebase better than you found it.
```
