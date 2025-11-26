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
- uv (installs Python automatically if needed)

Install dependencies
```bash
# uv handles Python version and dependencies automatically
uv pip install .
```

Environment variables
Copy `.env.example` to `.env` and fill in your values:
```bash
cp .env.example .env
# Edit .env with your actual values
```

Required variables:
- `WANDB_API_KEY`: your W&B API key (or run `uvx wandb login`)
- `WANDB_PROJECT`: defaults to `mt`
- `CLASSES_CSV_PATH`: path to pre-edited classes CSV
- `REFACTORINGS_CSV_PATH`: path to pre-edited refactorings CSV

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

## Dependency Management with uv

### Basic Setup
- Install uv: see `https://docs.astral.sh/uv/` (macOS/Linux/Homebrew supported)
- Install deps: `uv pip install .` (from pyproject.toml)
- CLI tools via uvx: `uvx wandb login`
- Lock dependencies: `uv pip compile pyproject.toml -o uv.lock`
- Reproducible install: `uv pip sync uv.lock`

### Jupyter Notebook Integration

#### Setup for Notebooks
```bash
# 1. Add ipykernel and uv as dev dependencies
uv add --dev ipykernel uv

# 2. Create a dedicated Jupyter kernel for this project
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=smellai

# 3. Optional: Seed environment with pip for %pip magic support
uv venv --seed
```

#### Managing Dependencies in Notebooks

**Method 1: Using uv commands (Recommended)**
```python
# Add dependencies permanently to pyproject.toml
!uv add pydantic numpy matplotlib

# Install dependencies temporarily (session only)
!uv pip install requests beautifulsoup4

# Add development dependencies
!uv add --dev pytest black ruff
```

**Method 2: Using %pip magic (if seeded)**
```python
# Only works if environment was created with --seed
%pip install package-name
```

#### Best Practices for Notebook Dependencies

1. **Persistent Dependencies**: Use `!uv add package-name` to add packages to your project permanently
2. **Temporary Testing**: Use `!uv pip install package-name` for quick experiments
3. **Development Tools**: Use `!uv add --dev package-name` for testing/linting tools
4. **Project Isolation**: Always use the dedicated kernel (`smellai`) to ensure proper environment isolation

#### Example Notebook Cell Structure
```python
# Cell 1: Install dependencies
!uv add autopep8 autoflake weave isort openai datasets --dev

# Cell 2: Handle compatibility issues
!uv pip install "httpx<0.28"  # Temporary fix for OpenAI compatibility

# Cell 3: Import and use
import weave
import openai
# ... rest of your code
```

#### Troubleshooting
- **Kernel not found**: Restart Jupyter and select the `smellai` kernel
- **Package not found**: Ensure you're using the correct kernel and run `!uv pip list` to verify installation
- **Import errors**: Restart kernel after installing new packages

## RefactoringMiner Evaluation Workflow

1. Generate extraction pairs (optional if you already have `rminer_data`):
     ```bash
     uv run src/data/rminer_extract.py --data /path/to/oracle/data.json --output ./rminer_data --max-commits 50
     ```
     This populates `rminer_data/manifest.json` with Java before/after pairs collected from RefactoringMiner.

2. Build the MLflow dataset from the manifest. Baseline run without SonarQube:
     ```bash
     uv run infra/mlflow/rminer_dataset.py --manifest rminer_data/manifest.json --limit 20 --dry-run
     ```

     Enable SonarQube enrichment with the new `--enable-sonar` flag to map smells onto refactored hunks:
     ```bash
     uv run infra/mlflow/rminer_dataset.py --manifest rminer_data/manifest.json --limit 20 --enable-sonar --dry-run
     ```

     Required environment variables when Sonar is enabled:
     - `SONAR_URL` (defaults to `http://localhost:9000` if unset)
     - `SONAR_TOKEN` (must be provided or passed with `--sonar-token`)

     Optional flags:
     - `--sonar-cache-dir` to reuse previous scan results (default `.sonar_cache`)
     - `--local-scanner` to use a local `sonar-scanner` binary instead of Docker

### How smell mapping works

- Each pair is scanned at the parent commit (`parent_sha`) so smells represent the pre-refactoring state.
- SonarQube issues are filtered to the file under analysis and only kept when their line number falls inside the diff hunk span (`old_start` to `old_start + old_count`).
- Mapped smells are stored in `expectations.code_smells` with hunk metadata so downstream tooling can reason about affected regions.

Example (truncated) expectation payload with smell annotations:

```json
{
    "inputs": {"pair_id": "4af8b1c2d3"},
    "expectations": {
        "file_path": "src/main/java/com/example/Foo.java",
        "diff_hunks": [
            {"old_start": 42, "old_count": 3, "new_start": 42, "new_count": 5}
        ],
        "code_smells": [
            {
                "smell_type": "Long Method",
                "line": 43,
                "severity": "MEDIUM",
                "message": "Refactor this method to reduce its Cognitive Complexity.",
                "rule": "java:S138",
                "hunk_index": 0
            }
        ],
        "num_smells": 1
    }
}
```

To inspect or export created datasets, use the CLI helper:

```bash
uv run infra/rminer_dataset_cli.py list
uv run infra/rminer_dataset_cli.py get --name rminer-eval-dataset --show-records
```