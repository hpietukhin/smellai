# Compound Refactoring Extraction Guide

## Overview

Script: `scripts/extract_compound_refactorings.py`

Extracts **compound refactorings** from SWE-Refactor dataset. Compound refactorings combine multiple operations in one refactoring action.

## Compound Refactoring Types

Total: **177 records (16.1%)** out of 1,099

1. **Extract And Move Method** — 142 records (12.9%)
   - Extract a method AND move it to another class
   - Common pattern: extract helper → move to utility class

2. **Move And Rename Method** — 21 records (1.9%)
   - Move method to another class AND rename simultaneously
   - Indicator: filePathBefore ≠ filePathAfter

3. **Move And Inline Method** — 14 records (1.3%)
   - Move method to another class AND inline it
   - Complex refactoring requiring careful analysis

## Quick Start

```bash
# Extract all compound refactorings (177 records)
uv run python scripts/extract_compound_refactorings.py

# Just show statistics (no output file)
uv run python scripts/extract_compound_refactorings.py --stats-only

# Extract specific type
uv run python scripts/extract_compound_refactorings.py --type "Extract And Move Method"

# Filter by project + limit size
uv run python scripts/extract_compound_refactorings.py --project checkstyle --limit 20

# Filter by JDK version
uv run python scripts/extract_compound_refactorings.py --jdk 17

# Custom output path
uv run python scripts/extract_compound_refactorings.py --output /tmp/my_dataset.json

# Create MLflow dataset (for experiment tracking)
uv run python scripts/extract_compound_refactorings.py \
  --create-mlflow-dataset \
  --mlflow-dataset-name "compound-eval-subset"
```

## Example Workflows

### 1. Small Test Run (Java 17 only, 10 records)
```bash
uv run python scripts/extract_compound_refactorings.py \
  --jdk 17 \
  --limit 10 \
  --output compound_test.json

# Run evaluation
uv run workflows/swe_eval_workflow.py \
  --dataset compound_test.json \
  --enable-composite \
  --analytics-db compound_test.db
```

### 2. Single Project Deep Dive (Checkstyle)
```bash
uv run python scripts/extract_compound_refactorings.py \
  --project checkstyle \
  --output checkstyle_compound.json

uv run workflows/swe_eval_workflow.py \
  --dataset checkstyle_compound.json \
  --enable-composite
```

### 3. Extract And Move Method Focus
```bash
uv run python scripts/extract_compound_refactorings.py \
  --type "Extract And Move Method" \
  --limit 50 \
  --output extract_move_50.json
```

### 4. MLflow Dataset for Tracking
```bash
uv run python scripts/extract_compound_refactorings.py \
  --jdk 17 \
  --limit 30 \
  --create-mlflow-dataset \
  --mlflow-dataset-name "compound-jdk17-subset"

# Use in MLflow evaluation
# (integrate with workflows/swe_eval_workflow.py --dataset-name)
```

## Dataset Statistics Example

```
DATASET STATISTICS
============================================================

Refactoring Type Distribution:
  Extract And Move Method: 142 (80.2%) [COMPOUND]
  Move And Rename Method: 21 (11.9%) [COMPOUND]
  Move And Inline Method: 14 (7.9%) [COMPOUND]

Project Distribution (top 10):
  checkstyle: 35 (19.8%)
  guava: 28 (15.8%)
  pmd: 22 (12.4%)
  ...

JDK Version Distribution:
  Java 8: 15 (8.5%)
  Java 11: 78 (44.1%)
  Java 17: 54 (30.5%)
  Java 21: 30 (16.9%)

Refactoring Complexity:
  Compound: 177 (100.0%)
  Atomic: 0 (0.0%)

Compilation Success Rates:
  Before: 177/177 (100.0%)
  After: 176/177 (99.4%)
============================================================
```

## Output Format

### JSON Output
Standard SWE-Refactor record format (same as original dataset):
```json
[
  {
    "projectName": "checkstyle",
    "commitId": "a1b2c3d4...",
    "type": "Extract And Move Method",
    "sourceCodeBeforeForWhole": "...",
    "sourceCodeAfterForWhole": "...",
    "compileJDK": 17,
    ...
  }
]
```

### MLflow Dataset
Converted to MLflow GenAI format with:
- **inputs**: project, commit, type, source code before, file paths
- **outputs**: source code after
- **metadata**: JDK version, compile command, test coverage

## Integration with Evaluation

```bash
# 1. Extract compound subset
uv run python scripts/extract_compound_refactorings.py \
  --jdk 17 \
  --limit 20 \
  --output compound_subset.json

# 2. Run composite evaluation
uv run workflows/swe_eval_workflow.py \
  --dataset compound_subset.json \
  --enable-composite \
  --max-refactorings 5 \
  --analytics-db compound_eval.db \
  --model claude-sonnet-4-5-20250929

# 3. Visualize results
./scripts/run_visualizer.sh
# In UI: Load compound_eval.db → Select session
```

## Filter Combinations

All filters can be combined:

```bash
# Java 17 + guava + Extract And Move + limit 15
uv run python scripts/extract_compound_refactorings.py \
  --type "Extract And Move Method" \
  --project guava \
  --jdk 17 \
  --limit 15 \
  --output guava_extract_move_j17.json
```

## Notes

- **Compound detection**: Any refactoring type containing "And" in the name
- **Dataset source**: Auto-detects `swe_refactor/SWE-Refactor.zip` or `/tmp/SWE-Refactor/pure_refactoring_data.json`
- **Compilation guarantee**: 100% compile before, 99.4% compile after
- **Test coverage**: Very limited (only 1/1099 records in full dataset have test coverage marked)

## Recommended Subsets for Evaluation

| Subset | Size | Description | Command |
|--------|------|-------------|---------|
| Quick test | 10 | Mixed types, JDK 17 | `--jdk 17 --limit 10` |
| Extract+Move focus | 50 | Most common compound type | `--type "Extract And Move Method" --limit 50` |
| Single project | ~20-35 | Deep dive one project | `--project checkstyle` |
| Modern Java | ~84 | JDK 17+ only | `--jdk 17` or `--jdk 21` |
| Full compound | 177 | All compound refactorings | (no filters) |

## Troubleshooting

**Error: Dataset not found**
```bash
# Extract the ZIP first:
unzip swe_refactor/SWE-Refactor.zip -d /tmp/SWE-Refactor/
# Or use --dataset-path to specify location
```

**No records match filter**
- Check available projects: `--stats-only` first to see distribution
- Remove filters one by one to find valid combinations

**MLflow dataset creation fails**
- Ensure MLflow is installed: `uv pip install mlflow`
- Check MLflow tracking URI is configured
