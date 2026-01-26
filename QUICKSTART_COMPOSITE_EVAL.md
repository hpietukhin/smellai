# Quick Start: Composite Evaluation with Compound Refactorings

## What You Have Now

✅ **Extraction Script**: `scripts/extract_compound_refactorings.py`
✅ **Sample Dataset**: `compound_sample_j17.json` (15 compound refactorings, Java 17)
✅ **Visualization Tool**: `scripts/run_visualizer.sh`
✅ **Documentation**: `scripts/README_COMPOUND_EXTRACTION.md`

## Step-by-Step Workflow

### 1. Extract Compound Refactorings

```bash
# View statistics of all compound refactorings (177 total)
uv run python scripts/extract_compound_refactorings.py --stats-only

# Extract specific subset (already created for you)
uv run python scripts/extract_compound_refactorings.py \
  --type "Extract And Move Method" \
  --jdk 17 \
  --limit 15 \
  --output compound_sample_j17.json
```

**Available Filters**:
- `--type`: "Extract And Move Method", "Move And Rename Method", "Move And Inline Method"
- `--project`: checkstyle, pmd, hibernate-search, junit5, etc.
- `--jdk`: 8, 11, 17, 21
- `--limit`: Number of records to extract

### 2. Run Composite Evaluation

**Basic Mode** (single refactoring, behavior preservation only):
```bash
uv run workflows/swe_eval_workflow.py \
  --dataset compound_sample_j17.json \
  --model claude-sonnet-4-5-20250929
```

**Composite Mode** (multi-agent: smell detection → prioritization → iterative refactoring):
```bash
uv run workflows/swe_eval_workflow.py \
  --dataset compound_sample_j17.json \
  --enable-composite \
  --max-refactorings 5 \
  --analytics-db compound_eval.db \
  --model claude-sonnet-4-5-20250929
```

### 3. Visualize Results

```bash
# Start NiceGUI visualizer
./scripts/run_visualizer.sh
# or
uv run python scripts/run_visualizer.py
```

Then in the browser (http://localhost:8080):
1. Click "Load Database"
2. Select `compound_eval.db`
3. Choose session to visualize
4. Explore:
   - Smell dependency graph (with PZ scores)
   - Agent execution timeline
   - Iteration-by-iteration refactoring outcomes
   - Tool call logs

### 4. View MLflow Results

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Open http://localhost:5000
# Navigate to "swe-refactor-evaluation" experiment
```

## Understanding Composite Mode

**Agent Workflow** (A0 → A1 → A3 → [A4 → A5 → A6] loop):

1. **A0 (Setup)**: Clone repo, detect build system, run initial tests
2. **A1 (Smell Detection)**: SonarQube scan for 8 smell types
3. **A3 (Prioritization)**: Dependency analysis, compute PZ scores
4. **A4 (Prompt Prep)**: Select next smell to refactor (planned)
5. **A5 (Refactoring)**: LLM generates code changes
6. **A6 (Verification)**: Re-run tests, check compilation
7. **Repeat** steps 4-6 until N refactorings or no smells remain

**PZ Score** (Prioritization with Dependencies):
```
PZ_i = Severity_i + Σ(positive_dependency_weights)
```
- **Positive dependencies**: Refactoring helps resolve other smells (prioritize higher)
- **Negative dependencies**: Refactoring may create new smells (deprioritize)

## Sample Datasets by Use Case

### Quick Test (5 minutes)
```bash
uv run python scripts/extract_compound_refactorings.py \
  --jdk 17 \
  --limit 5 \
  --output quick_test.json
```

### Extract+Move Focus (most common compound type)
```bash
uv run python scripts/extract_compound_refactorings.py \
  --type "Extract And Move Method" \
  --jdk 11 \
  --limit 30 \
  --output extract_move_j11.json
```

### Single Project Deep Dive
```bash
uv run python scripts/extract_compound_refactorings.py \
  --project checkstyle \
  --output checkstyle_all_compound.json
# Result: 43 compound refactorings from checkstyle
```

### Modern Java Only
```bash
uv run python scripts/extract_compound_refactorings.py \
  --jdk 21 \
  --output modern_java.json
# Result: 38 Java 21 compound refactorings
```

## Key Metrics to Track

**From MLflow**:
- `compile_success_rate`: Fraction of generated code that compiles
- `test_pass_rate`: Fraction of compilable code that passes tests
- `overall_success_rate`: Both compile + tests pass

**From Analytics DB** (visualizer):
- Smells detected per iteration
- Smells resolved vs created per refactoring
- PZ scores and dependency relationships
- Token usage by agent node
- Tool call timing (performance profiling)

## Compound Refactoring Statistics

**Total in SWE-Refactor**: 177 out of 1,099 (16.1%)

**By Type**:
- Extract And Move Method: 142 (80.2%)
- Move And Rename Method: 21 (11.9%)
- Move And Inline Method: 14 (7.9%)

**By Project** (top 5):
- checkstyle: 43 (24.3%)
- pmd: 36 (20.3%)
- hibernate-search: 31 (17.5%)
- junit5: 20 (11.3%)
- hibernate-orm: 10 (5.6%)

**By JDK**:
- Java 8: 27 (15.3%)
- Java 11: 93 (52.5%)
- Java 17: 19 (10.7%)
- Java 21: 38 (21.5%)

**Quality**:
- Compilation before: 100.0% (177/177)
- Compilation after: 100.0% (177/177)
- Pure refactorings: 100.0% (no feature changes)

## Troubleshooting

**"Dataset not found"**
```bash
# Extract the ZIP if needed
unzip swe_refactor/SWE-Refactor.zip -d /tmp/SWE-Refactor/
```

**"Port 8080 already in use"** (visualizer)
```bash
# Find and kill process
lsof -ti:8080 | xargs kill -9
# Or edit port in tools/visualize_smell_prioritization.py:1074
```

**"No records match filter"**
- Run `--stats-only` first to see available combinations
- Remove filters one by one to find valid subsets

**SonarQube not running** (needed for composite mode)
```bash
docker compose -f sonarqube/docker-compose.yml up -d
# Wait 30 seconds for startup
# Create token: http://localhost:9000 → My Account → Security
# Set in .env: SONAR_TOKEN=your_token
```

## Next Steps

1. **Small test run**: Use `compound_sample_j17.json` (15 records)
2. **Review results**: Check MLflow metrics and visualizer
3. **Scale up**: Extract larger subset (50-100 records)
4. **Iterate**: Adjust agent configuration based on findings
5. **Document**: Record insights for thesis

## Research Questions to Explore

- How well do LLMs map compound refactorings compared to atomic ones?
- Do positive smell dependencies accurately predict refactoring success?
- Which smell types cascade most (create new smells)?
- What's the optimal iteration limit (max-refactorings)?
- How does model choice (GPT-4o vs Claude Sonnet) affect outcomes?

## Files Reference

- **Extract script**: `scripts/extract_compound_refactorings.py`
- **Evaluation workflow**: `workflows/swe_eval_workflow.py`
- **Visualizer**: `scripts/run_visualizer.sh` → `tools/visualize_smell_prioritization.py`
- **Dataset model**: `swe_refactor/dataset.py`
- **Analytics DB**: `swe_refactor/persistence/database.py`
- **Documentation**: `scripts/README_COMPOUND_EXTRACTION.md`, `VISUALIZATION_USAGE.md`
