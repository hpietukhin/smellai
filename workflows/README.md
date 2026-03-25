# workflows

CLI workflow scripts for running evaluations and analyses. Each script is a standalone entry point.

## Workflows

| Workflow | Description | Command |
|----------|-------------|---------|
| **eval_workflow.py** | Unified MLflow GenAI evaluation (preferred) | `uv run workflows/eval_workflow.py --source swe --hf-dataset-path data/processed/swe` |
| **swe_eval_workflow.py** | SWE-Refactor agent evaluation (compile + test) | `uv run workflows/swe_eval_workflow.py --dataset <path> --limit 10` |
| **rminer_eval_workflow.py** | RMiner mapping accuracy evaluation | `uv run workflows/rminer_eval_workflow.py --manifest rminer_data/manifest.json --limit 5` |
| **java_test_workflow.py** | Java test analysis (build detection + test run) | `uv run workflows/java_test_workflow.py --project /path/to/project` |
| **composite_analysis_workflow.py** | Composite refactoring analysis from RMiner manifest | `uv run workflows/composite_analysis_workflow.py --manifest rminer_data/manifest.json` |
| **smell_cooccurrence_workflow.py** | Smell co-occurrence graph visualization | `uv run workflows/smell_cooccurrence_workflow.py --manifest <manifest>` |

## Shared Modules

- **common.py** - MLflow setup helpers (`setup_workflow_mlflow`, `save_agent_graph`, `print_eval_results`)
- **utils.py** - Logging configuration, manifest loading, matplotlib helpers

## Usage

```bash
# Unified evaluation (recommended)
uv run workflows/eval_workflow.py --source swe --hf-dataset-path data/processed/swe --limit 5

# Draw agent graph without running evaluation
uv run workflows/eval_workflow.py --source swe --draw-graph

# RMiner evaluation with specific model
uv run workflows/rminer_eval_workflow.py --manifest rminer_data/manifest.json --model claude-sonnet-4-5-20250929
```
