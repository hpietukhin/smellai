# mlflow_utils

MLflow infrastructure: server lifecycle management, dataset CRUD, and evaluation orchestration.

## Key Files

- **server.py** - `MLflowServer` class for start/stop/status management of the MLflow UI process
- **auto_server.py** - `ensure_mlflow_server` context manager and `setup_mlflow_tracking` for automatic server startup
- **runner.py** - `EvaluationRunner` orchestrating server + dataset + evaluation pipeline
- **cli.py** - CLI entry point (`server`, `datasets`, `evaluate` subcommands)
- **datasets/manager.py** - `DatasetManager` for MLflow GenAI dataset CRUD operations

## Usage

```bash
# CLI: manage server
uv run -m mlflow_utils.cli server start --port 5000
uv run -m mlflow_utils.cli server status

# CLI: manage datasets
uv run -m mlflow_utils.cli datasets list --experiment rminer-evaluation
uv run -m mlflow_utils.cli datasets create --manifest rminer_data/manifest.json --limit 20

# CLI: run evaluation
uv run -m mlflow_utils.cli evaluate --model gpt-4o-mini --limit 5
```

```python
from mlflow_utils import setup_mlflow_tracking, DatasetManager

# Auto-start server and configure tracking
setup_mlflow_tracking(
    tracking_uri="http://localhost:5000",
    experiment_name="my-experiment",
)

# Create dataset from records
manager = DatasetManager(tracking_uri="http://localhost:5000")
manager.create_dataset_from_records(records, name="my-dataset", experiment="my-experiment")
```
