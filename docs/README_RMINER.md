# RefactoringMiner Evaluation Workflow

This project includes MLflow-based evaluation pipelines for refactoring mapping agents.

## Quick Start

### 1. Create an MLflow Dataset
```bash
# Create dataset from RefactoringMiner manifest
# NOTE: --experiment must match what you use in evaluation
uv run scripts/create_rminer_dataset.py \
    --manifest rminer_data/manifest.json \
    --limit 20 \
    --experiment rminer-evaluation \
    --tracking-uri sqlite:///mlflow.db

# Note the dataset_id printed at the end (e.g., "d-abc123def456")
```

### 2. List Available Datasets
```bash
# List all datasets
uv run scripts/manage_datasets.py list --tracking-uri sqlite:///mlflow.db

# List datasets with JSON output
uv run scripts/manage_datasets.py list --json
```

### 3. Inspect a Dataset
```bash
# Get dataset by name
uv run scripts/manage_datasets.py get --name rminer-eval-dataset --show-records

# Get dataset by ID
uv run scripts/manage_datasets.py get --id d-abc123def456 --show-records
```

### 4. Run Evaluation
```bash
# Run evaluation pipeline
# NOTE: --dataset-id is preferred if you have it
uv run agent_workflows/rminer_eval.py \
    --dataset-name rminer-eval-dataset \
    --experiment rminer-evaluation \
    --tracking-uri sqlite:///mlflow.db \
    --model gpt-4o-mini
```

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--tracking-uri` | `sqlite:///mlflow.db` | MLflow tracking server |
| `--experiment` | `rminer-evaluation` | Experiment name (must be consistent) |
| `--model` | `gpt-4o-mini` | LLM model for agent |

## View Results
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```
