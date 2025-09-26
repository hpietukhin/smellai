# smellai
master thesis
SmellAI - Thesis Experiments

W&B Quickstart

1. Install: `pip install wandb`
2. Login: `wandb login` (or set `WANDB_API_KEY`)
3. Project: defaults to `mt` (override via env)
4. Minimal usage:
```python
import os, wandb
os.environ.setdefault("WANDB_PROJECT", "mt")
run = wandb.init(project=os.getenv("WANDB_PROJECT"))
wandb.config.update({"experiment": "demo"})
wandb.log({"metric": 1})
run.finish()
```

Project setup

Prerequisites
- Python 3.11+
- pip

Create virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```

Install minimal dependencies
```bash
pip install -U pandas wandb
```

Environment variables
- `WANDB_API_KEY`: your W&B API key (or run `wandb login`)
- `WANDB_PROJECT`: defaults to `mt`
- `CLASSES_CSV_PATH`: path to pre-edited classes CSV
- `REFACTORINGS_CSV_PATH`: path to pre-edited refactorings CSV

Example `.env` content (optional)
```bash
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
WANDB_PROJECT=mt
CLASSES_CSV_PATH=/absolute/path/to/classes.csv
REFACTORINGS_CSV_PATH=/absolute/path/to/refactorings.csv
```

Run a minimal experiment
```python
from src.pipelines.experiment_pipeline import load_dataset, run_experiment

# Optional: pass paths explicitly instead of env
config = {
    "classes_csv": "/absolute/path/to/classes.csv",
    "refactorings_csv": "/absolute/path/to/refactorings.csv",
}

df_classes, df_refactorings = load_dataset(config)
run_experiment(
    df_classes,
    df_refactorings,
    dataset_name="demo-dataset",
    dataset_version="v0",
    connector_name="mysql",
)
```