# React Agent MLflow Evaluation

Use this guide to run the MLflow-based evaluation pipeline for the LangGraph ReAct agent that produces refactoring suggestions from the DACOS dataset.

## Prerequisites

- Python 3.11 (matches the checked-in virtual environment).
- DACOS MySQL instance reachable from your machine.
- API keys for the LLMs you plan to use:
  - `OPENAI_API_KEY` for the default agent model (gpt-4.1-mini) and judge (openai:/gpt-4.1-mini).
  - `ANTHROPIC_API_KEY` when using Anthropic models (e.g., `MODEL=anthropic/claude-sonnet-4-5-20250929`).
  - Other provider-specific keys as needed (see Model Provider Format below).
- All project dependencies installed in the active environment:
  ```bash
  uv pip install -e .
  uv pip list | grep -E "(mlflow|langgraph|mysql-connector)"
  ```
  The output should include `mlflow 3.5.x`, `langgraph 1.0.x`, and `mysql-connector-python`.

## Required Environment Variables

Create a `.env` file or export the variables before running the pipeline.

| Variable | Purpose |
| --- | --- |
| `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DATABASE`, `MYSQL_USER`, `MYSQL_PASSWORD` | DACOS MySQL connectivity (required). |
| `MODEL` | Optional override for the LangGraph agent model (default: `gpt-4.1-mini`). Supports provider-prefixed format (see below). |
| `MLFLOW_TRACKING_URI` | MLflow tracking backend URI. Use `sqlite:///path/to/mlflow.db` for SQLite or `http://localhost:5000` for tracking server (default: `http://localhost:5000`). **Note:** Dataset features require a SQL backend (SQLite, PostgreSQL, or MySQL). |
| `MLFLOW_EXPERIMENT_NAME` | Optional experiment name (default: `react-agent-mlflow`). |
| `MLFLOW_JUDGE_MODEL` | Optional override for the LLM-as-judge model (default: `openai:/gpt-4.1-mini`). Must use provider-prefixed format. |

Ensure the provider-specific API keys are also exported, for example:
```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
echo "OPENAI_API_KEY=sk-openai-..." >> .env
```

### Model Provider Format

The `MODEL` environment variable supports multiple provider formats, parsed by the agent's model initialization logic (`src/agents/react_agent/graph.py`):

**Supported Formats:**
- `provider:/model` - e.g., `openai:/gpt-4.1-mini`, `anthropic:/claude-sonnet-4-5-20250929`
- `provider://model` - e.g., `openai://gpt-4.1-mini`
- `provider/model` - e.g., `anthropic/claude-sonnet-4-5-20250929`
- `provider:model` - e.g., `openai:gpt-4.1-mini`
- `model` - e.g., `gpt-4.1-mini` (defaults to `openai` provider)

**Provider Detection:**
The parser splits on separators (`:/`, `://`, `/`, `:`) and validates that the provider component is alphabetic. If no provider is detected, it defaults to `openai`.

**Examples:**
```bash
# Use Anthropic's Claude
export MODEL="anthropic/claude-sonnet-4-5-20250929"

# Use OpenAI GPT-4 (explicit provider)
export MODEL="openai:/gpt-4.1-mini"

# Use default OpenAI provider (implicit)
export MODEL="gpt-4.1-mini"

# Use with protocol-style separator
export MODEL="anthropic://claude-3-opus-20240229"
```

**Judge Model Format:**
The `MLFLOW_JUDGE_MODEL` must always include a provider prefix and uses the same parsing logic. The default is `openai:/gpt-4.1-mini`.

## Running the Evaluation

1. Activate the project environment:
   ```bash
   source .venv/bin/activate
   ```
2. Execute the evaluation pipeline. Specify sample IDs explicitly or let the script sample from the DACOS catalogue:
   ```bash
   # Evaluate specific samples
  uv run python -m src.pipelines.react_agent_mlflow --sample-ids 101 205

   # Sample up to three random DACOS entries
  uv run python -m src.pipelines.react_agent_mlflow --limit 3

   # Use a specific judge model
   uv run python -m src.pipelines.react_agent_mlflow --limit 5 --judge-model "anthropic:/claude-sonnet-4-5-20250929"

   # Use a predefined sample preset (e.g., test_5 loads 5 samples per smell type)
   uv run python -m src.pipelines.react_agent_mlflow --sample-preset test_5
   ```
3. The script prints aggregated metrics to the console and logs full traces, prompts, and judge scores to MLflow.

### CLI Options

The evaluation pipeline supports the following command-line arguments:

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--sample-ids` | `int` (multiple) | None | Explicit DACOS sample IDs to evaluate. When provided, `--limit` is ignored. |
| `--limit` | `int` | 5 | Maximum number of random samples to fetch when `--sample-ids` is not specified. |
| `--sample-preset` | `str` | None | Predefined dataset selector. Currently supports `test_5` (loads 5 samples per smell type). Overrides both `--sample-ids` and `--limit`. |
| `--judge-model` | `str` | `$MLFLOW_JUDGE_MODEL` or `openai:/gpt-4.1-mini` | Override for the LLM-as-judge scorer. Must use provider-prefixed format. |
| `--log-level` | `str` | `$PIPELINE_LOG_LEVEL` or `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR). |

**Note:** All CLI arguments override their corresponding environment variables.

## Evaluation Methodology

The pipeline implements a two-stage evaluation approach for the ReAct agent's refactoring suggestions:

### Stage 1: Agent Execution
The LangGraph ReAct agent (`src/agents/react_agent/graph.py`) receives a prompt constructed from DACOS sample metadata (sample ID, repository, file path, annotated smell, and description). The agent:
1. Uses available DACOS tools (`load_dacos_samples`, `fetch_dacos_sample`) to retrieve code context
2. Applies reasoning-action cycles to formulate refactoring advice
3. Returns a final response addressing the identified smell

### Stage 2: Scoring
The agent's output is evaluated using:
1. **LLM-as-judge scorer** (`_quality_judge`): An LLM evaluates refactoring quality using the rubric: excellent, good, acceptable, poor. The judge considers whether the response explicitly cites the annotated smell, proposes actionable steps, and remains grounded in the dataset.
2. **Heuristic scorers**:
   - `mentions_smell`: Binary check for whether the annotated smell name appears in the response
   - `smell_detection_f1`: F1 score measuring precision and recall of smell detection against ground truth

### Traceability
Each evaluation example includes:
- **inputs**: The prompt sent to the agent (constructed from DACOS sample metadata)
- **expectations**: Ground-truth smell name and description from DACOS annotations
- **tags**: Smell categorization for aggregating metrics by smell type

All traces, expectations, and scores are logged to MLflow for reproducibility and analysis.

## Inspecting MLflow Results

Launch the MLflow UI against the configured tracking URI to inspect traces and scores:

### Using SQLite Backend (Recommended)

If you have a `mlflow.db` SQLite database (required for dataset features) use the project environment to avoid migration mismatches:
```bash
# Using absolute path
uv run mlflow ui --backend-store-uri sqlite:////Users/havriil.pietukhin/PycharmProjects/smellai/mlflow.db --port 5000

# Or use the provided script
./start_mlflow_server.sh
```

### Using HTTP Tracking Server

If using a remote MLflow tracking server:
```bash
# Server should already be running with SQL backend
# Start the UI with the same environment
uv run mlflow ui --backend-store-uri http://remote-tracking-server --port 5000
# Or just access the existing UI if it is already hosted
open http://localhost:5000
```

### Using FileStore (Limited Features)

If using local file-based tracking (./mlruns) - **Note:** Dataset features are not supported:
Use the same command pattern, replacing the backend URI as needed.

Navigate to the `react-agent-mlflow` experiment, open a run, and review the GenAI evaluation tables for judge outputs and heuristic scores.

**Important:** To use MLflow dataset features (like `create_dataset`), you must use a SQL backend. Configure your `.env` with:
```bash
MLFLOW_TRACKING_URI=sqlite:///$(pwd)/mlflow.db
```

## Troubleshooting

- **Missing dependencies**: re-run `uv pip install -e .` inside the active environment and confirm with `uv pip list`.
- **MySQL connection errors**: verify the DACOS instance is reachable and credentials are correct; run `uv run python -m src.data.mysql_connector` if you have a connectivity check implemented.
- **LLM authentication issues**: confirm the relevant API key environment variables are exported in the same shell session.
- **Judge cost control**: set `MLFLOW_JUDGE_MODEL` to a cheaper provider/model if needed.
- **"create_dataset is not supported with FileStore" error**: This means you're using MLflow's FileStore backend (`./mlruns` directory) which doesn't support dataset features. You must configure a SQL backend:
  1. Add to `.env`: `MLFLOW_TRACKING_URI=sqlite:///$(pwd)/mlflow.db`
  2. Restart your scripts to pick up the new tracking URI
  3. Launch MLflow UI with: `mlflow ui --backend-store-uri sqlite:///path/to/mlflow.db --port 5000`
- **"Each record must have an 'inputs' field" error**: You are likely using a dataset created with an older version of `create_dacos_mlflow_dataset.py`. Regenerate the dataset so that every row includes a top-level `inputs` object (the updated script writes both `inputs` and nested `inputs.inputs` columns) or add the column manually before calling `dataset.merge_records`.
- **"Can't locate revision identified by 'bf29a5ff90ea'" error**: The CLI that launched MLflow UI is older than the project dependency set and lacks the corresponding Alembic migration. Run the UI with `uv run mlflow ui ...` (or via `./start_mlflow_server.sh`, which now falls back to `.venv`) so the bundled MLflow version is used, or upgrade your global/pipx `mlflow` installation to match the version declared in `pyproject.toml`.
