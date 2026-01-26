# Complete Guide: Composite Evaluation with Visualization

## TL;DR - Quick Start

```bash
# 1. Start SonarQube (if not running)
docker compose -f sonarqube/docker-compose.yml up -d

# 2. Run evaluation (1 record demo, ~10 min)
uv run python scripts/run_demo_eval.py

# 3. Visualize results
./scripts/run_visualizer.sh
# Browser: http://localhost:8080 → Load "demo_composite.db"

# 4. View MLflow metrics
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Browser: http://localhost:5000 → "swe-refactor-evaluation"
```

---

## What You Have

### 📊 Datasets Created

| File | Records | Type | JDK | Description |
|------|---------|------|-----|-------------|
| `compound_demo.json` | 2 | Extract+Move | 17 | Quick demo (commons-io) |
| `compound_sample_j17.json` | 15 | Extract+Move | 17 | Sample evaluation set |
| Full dataset (in ZIP) | 177 | All compound | 8-21 | Complete compound refactorings |

### 🛠️ Scripts Created

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/extract_compound_refactorings.py` | Extract compound refactorings | Filter by type/project/JDK |
| `scripts/run_demo_eval.py` | Automated evaluation runner | Checks SonarQube, runs eval |
| `scripts/run_visualizer.sh` | Start NiceGUI visualizer | Port 8080 interactive UI |
| `scripts/run_composite_demo.sh` | Full demo workflow | Bash version |

### 📚 Documentation

| File | Content |
|------|---------|
| `scripts/README_COMPOUND_EXTRACTION.md` | Dataset extraction guide |
| `QUICKSTART_COMPOSITE_EVAL.md` | Workflow overview |
| `RUN_COMPOSITE_DEMO.md` | Step-by-step instructions |
| This file | Complete reference |

---

## System Architecture

### Multi-Agent Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    COMPOSITE MODE WORKFLOW                    │
└──────────────────────────────────────────────────────────────┘

[INPUT: Compound Refactoring Record]
   ├─ projectName: "commons-io"
   ├─ commitId: "abc123..."
   ├─ type: "Extract And Move Method"
   └─ sourceCode: before/after

         ↓

┌──────────────────────────────────────────────────────────────┐
│ A0: Test Coverage Agent (Java Test)                          │
│ - Clone repository to workspace                              │
│ - Detect build system (Maven/Gradle)                         │
│ - Run tests, capture coverage                                │
│ → OUTPUT: CompileResult, TestResult                          │
└──────────────────────────────────────────────────────────────┘

         ↓

┌──────────────────────────────────────────────────────────────┐
│ A1: Smell Detection Agent (SonarQube)                        │
│ - Run SonarQube scanner on workspace                         │
│ - Fetch issues from SonarQube API                            │
│ - Map to 8 smell types (Complex Method, Long Method, etc.)   │
│ → OUTPUT: List[SmellEvent(action="detected")]                │
└──────────────────────────────────────────────────────────────┘

         ↓

┌──────────────────────────────────────────────────────────────┐
│ A3: Dependency Analysis Agent (Prioritization)               │
│ - Build smell dependency graph (NetworkX)                    │
│ - Identify positive/negative dependencies                    │
│ - Compute PZ scores: PZ_i = Severity_i + Σ(deps)            │
│ → OUTPUT: Prioritized smell list                             │
└──────────────────────────────────────────────────────────────┘

         ↓

      ┌────────────────────────────────────┐
      │  ITERATION LOOP (max N times)      │
      └────────────────────────────────────┘

         ↓

┌──────────────────────────────────────────────────────────────┐
│ A4: Prompt Preparation (TODO)                                 │
│ - Select highest PZ smell                                     │
│ - Prepare refactoring context                                 │
│ → OUTPUT: Refactoring prompt                                  │
└──────────────────────────────────────────────────────────────┘

         ↓

┌──────────────────────────────────────────────────────────────┐
│ A5: Refactoring Execution Agent (LLM)                        │
│ - Generate code changes via LLM                               │
│ - Apply changes to workspace files                            │
│ - Record RefactoringAttempt                                   │
│ → OUTPUT: Modified source code                                │
└──────────────────────────────────────────────────────────────┘

         ↓

┌──────────────────────────────────────────────────────────────┐
│ A6: Behavior Verification Agent (Reuse A0)                   │
│ - Compile modified code                                       │
│ - Run tests                                                    │
│ - Update RefactoringAttempt.outcome                          │
│ → OUTPUT: CompileResult, TestResult                          │
└──────────────────────────────────────────────────────────────┘

         ↓

    ┌──────────────────┐
    │ Tests pass?      │
    └──────────────────┘
       ↓         ↓
     YES        NO
       ↓         ↓
  Re-scan    Rollback
  (A1)       changes
       ↓         ↓
  Update     Record
  smells     failure
       ↓
   ┌──────────────────┐
   │ Continue loop?   │
   │ - Smells remain  │
   │ - Under max iter │
   └──────────────────┘
      ↓         ↓
    YES        NO
      ↓         ↓
   REPEAT     END

         ↓

[OUTPUT: Evaluation Results]
   ├─ compile_success_rate
   ├─ test_pass_rate
   ├─ overall_success_rate
   ├─ Analytics DB (SmellEvent, RefactoringAttempt, etc.)
   └─ MLflow run (metrics, artifacts)
```

### Data Flow

```
SWE-Refactor Dataset
  └─ compound_demo.json (filtered)
       ↓
  Evaluation Workflow
       ↓
  ┌──────────────────────────────────────┐
  │ Multi-Agent Execution (LangGraph)    │
  └──────────────────────────────────────┘
       ↓                    ↓
  Analytics DB         MLflow DB
  (demo_composite.db)  (mlflow.db)
       ↓                    ↓
  Visualizer           MLflow UI
  (NiceGUI)           (Web Interface)
```

---

## Detailed Walkthrough

### 1. Dataset Preparation

**View compound refactorings statistics**:
```bash
uv run python scripts/extract_compound_refactorings.py --stats-only
```

Output:
```
Found 177 compound refactorings (16.1%)

Refactoring Type Distribution:
  Extract And Move Method: 142 (80.2%) [COMPOUND]
  Move And Rename Method: 21 (11.9%) [COMPOUND]
  Move And Inline Method: 14 (7.9%) [COMPOUND]

Project Distribution:
  checkstyle: 43 (24.3%)
  pmd: 36 (20.3%)
  hibernate-search: 31 (17.5%)
  ...

JDK Version Distribution:
  Java 8: 27 (15.3%)
  Java 11: 93 (52.5%)
  Java 17: 19 (10.7%)
  Java 21: 38 (21.5%)
```

**Create custom subset**:
```bash
# Example: All compound refactorings from checkstyle (Java 11)
uv run python scripts/extract_compound_refactorings.py \
  --project checkstyle \
  --jdk 11 \
  --output checkstyle_j11.json

# Example: Move+Rename focus (rare type)
uv run python scripts/extract_compound_refactorings.py \
  --type "Move And Rename Method" \
  --output move_rename.json

# Example: Modern Java subset
uv run python scripts/extract_compound_refactorings.py \
  --jdk 21 \
  --limit 20 \
  --output modern_java.json
```

### 2. Environment Setup

**Start SonarQube** (required for composite mode):
```bash
docker compose -f sonarqube/docker-compose.yml up -d

# Wait 30-60 seconds for startup
# Verify: http://localhost:9000 shows login page
```

**Configure SonarQube token** (first time only):
1. Open http://localhost:9000
2. Login: `admin` / `admin` (will prompt to change)
3. My Account → Security → Generate Token
4. Add to `.env`:
   ```bash
   echo "SONAR_TOKEN=your_token_here" >> .env
   ```

**Verify environment**:
```bash
# Check Docker containers
docker ps | grep sonarqube

# Check Python environment
uv sync

# Check dataset
ls -lh compound_demo.json
```

### 3. Run Evaluation

#### Option A: Automated Script (Recommended)

```bash
# Quick demo (1 record, 3 iterations, ~10 min)
uv run python scripts/run_demo_eval.py

# Custom configuration
uv run python scripts/run_demo_eval.py \
  --records 5 \
  --iterations 5 \
  --dataset compound_sample_j17.json \
  --db my_eval.db \
  --model gpt-4o-mini
```

#### Option B: Manual Workflow Command

```bash
uv run workflows/swe_eval_workflow.py \
  --dataset compound_demo.json \
  --enable-composite \
  --max-refactorings 3 \
  --analytics-db demo_composite.db \
  --model claude-sonnet-4-5-20250929 \
  --limit 1 \
  --workspace /tmp/swe-eval-workspace \
  --sonar-url http://localhost:9000
```

**Parameters explained**:
- `--dataset`: Input JSON with refactoring records
- `--enable-composite`: Enable multi-agent mode (A0→A1→A3→A5→A6)
- `--max-refactorings`: Max iterations per record (N-action limit)
- `--analytics-db`: SQLite DB for agent execution tracking
- `--model`: LLM model (claude-sonnet-4-5-20250929, gpt-4o-mini, etc.)
- `--limit`: Number of records to evaluate
- `--workspace`: Directory for cloned repositories

**Expected terminal output**:
```
Creating agent...
Model: claude-sonnet-4-5-20250929
Mode: Composite
Records: 1

[Record 1/1] commons-io @ abc123
[A0] Cloning repository...
[A0] Running tests... PASSED (145/145)
[A1] SonarQube scan... Found 12 smells
[A3] Prioritization... Top: Complex Method (PZ=8.5)

[Iteration 1]
  [A5] Generating refactoring...
  [A6] Compiling... SUCCESS
  [A6] Tests... PASSED (145/145)
  [A1] Re-scan... 10 smells remain
  → SUCCESS: 2 resolved, 0 created

[Iteration 2]
  [A5] Generating refactoring...
  [A6] Compiling... SUCCESS
  [A6] Tests... PASSED (144/145) - 1 FAILURE
  → PARTIAL: 1 resolved, 1 created

[Iteration 3]
  [A5] Generating refactoring...
  [A6] Compiling... FAILED
  → FAILURE: Rolled back

EVALUATION RESULTS
──────────────────────────────────
compile_success_rate: 0.6667
test_pass_rate: 0.5000
overall_success_rate: 0.3333
──────────────────────────────────
```

### 4. Visualization

**Start NiceGUI visualizer**:
```bash
./scripts/run_visualizer.sh
# Opens http://localhost:8080
```

**UI walkthrough**:

1. **Load Database** (top left)
   - Enter path: `demo_composite.db`
   - Click "Load Database"
   - Select session from dropdown

2. **Smell Dependency Graph** (main center panel)
   - **Nodes**: Smells (size = PZ score, color = severity)
   - **Green edges**: Positive dependencies (helps resolve)
   - **Red dashed edges**: Negative dependencies (may create)
   - **Click node**: Show details in sidebar

3. **Agent Timeline** (bottom panel)
   - **Horizontal bars**: Node executions (A0, A1, A3, A5, A6)
   - **Bar length**: Duration
   - **Click bar**: Show tool calls

4. **Iteration Selector** (left sidebar)
   - **Slider**: Navigate through iterations
   - Shows before/after smell counts
   - Refactoring outcomes (SUCCESS/FAILURE/PARTIAL)

5. **Smell Details** (left sidebar)
   - Current iteration smells
   - PZ scores
   - Dependencies
   - File locations

6. **Tool Call Logs** (left sidebar)
   - Debugging information
   - Tool invocations
   - Timing data

7. **Code Diff** (left sidebar)
   - Changes made per iteration
   - Syntax-highlighted diff

**Example visualization session**:
```
Iteration 0 (Initial):
  Graph: 12 nodes (smells detected by SonarQube)
  Edges: 8 dependencies identified
  Top PZ: Complex Method (score 8.5)

Iteration 1 (After first refactoring):
  Graph: 10 nodes (2 smells resolved)
  New edges: 1 smell created (negative dep)
  Outcome: SUCCESS

Iteration 2:
  Graph: 9 nodes
  Outcome: PARTIAL (test failure)

Iteration 3:
  Graph: 9 nodes (no change)
  Outcome: FAILURE (compile error)
```

### 5. MLflow Analysis

**Start MLflow UI**:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Opens http://localhost:5000
```

**Navigate to experiment**:
1. Click **"swe-refactor-evaluation"**
2. View runs sorted by date
3. Click run to see details

**Key metrics**:
- `compile_success_rate`: Fraction of iterations that compiled
- `test_pass_rate`: Fraction of compilable iterations with passing tests
- `overall_success_rate`: Both compile AND tests pass

**Artifacts**:
- Input/output records
- Agent configuration
- Error logs (if any)

**Compare runs**:
- Select multiple runs
- Click "Compare"
- View metric differences (e.g., different models)

---

## Research Use Cases

### Use Case 1: Model Comparison

**Question**: Does Claude Sonnet outperform GPT-4o on compound refactorings?

```bash
# Run with Claude
uv run python scripts/run_demo_eval.py \
  --records 10 \
  --model claude-sonnet-4-5-20250929 \
  --db eval_claude.db

# Run with GPT-4o
uv run python scripts/run_demo_eval.py \
  --records 10 \
  --model gpt-4o \
  --db eval_gpt4o.db

# Compare in MLflow UI
mlflow ui
# Select both runs → Compare
```

### Use Case 2: Dependency Analysis

**Question**: Do positive dependencies predict refactoring success?

```bash
# Run evaluation
uv run python scripts/run_demo_eval.py \
  --records 20 \
  --iterations 5

# Visualize
./scripts/run_visualizer.sh
# Load demo_composite.db
# For each iteration:
#   - Note top PZ smell
#   - Check dependencies (green edges)
#   - Observe outcome (SUCCESS/FAILURE)
#   - Correlate positive deps with success
```

### Use Case 3: Cascade Effects

**Question**: Which refactorings create new smells (negative dependencies)?

```bash
# Extract Move+Inline (complex type)
uv run python scripts/extract_compound_refactorings.py \
  --type "Move And Inline Method" \
  --output move_inline.json

# Evaluate
uv run python scripts/run_demo_eval.py \
  --dataset move_inline.json \
  --records 14 \
  --db cascade_study.db

# Analyze in visualizer:
#   - Track smell creation (red nodes appearing)
#   - Identify negative dependencies (red dashed edges)
#   - Document patterns
```

### Use Case 4: Iteration Efficiency

**Question**: What's the optimal max-refactorings limit?

```bash
# Test different limits
for N in 2 5 10; do
  uv run python scripts/run_demo_eval.py \
    --records 10 \
    --iterations $N \
    --db eval_iter${N}.db
done

# Compare success rates in MLflow
# Plot: X-axis=iterations, Y-axis=overall_success_rate
```

---

## Advanced Configuration

### Custom Smell Types

Edit `agents/swe_eval/agent.py` to add smell types:
```python
SMELL_TYPES = [
    "Complex Method",
    "Long Method",
    "Long Parameter List",
    # Add custom rules here
]
```

### Adjust Prioritization Weights

Edit `scripts/prioritize_smells.py`:
```python
PZ_WEIGHTS = {
    "positive_dependency": 2.0,  # Increase weight
    "negative_dependency": -1.5,
    "severity_multiplier": 1.0,
}
```

### Change Iteration Logic

Edit `agents/swe_eval/agent.py` to customize loop conditions:
```python
def should_continue(state):
    # Custom logic
    if state["iteration"] >= max_refactorings:
        return False
    if len(state["detected_smells"]) == 0:
        return False
    # Add custom conditions
    return True
```

---

## Troubleshooting

### Common Issues

#### 1. SonarQube Connection Errors

**Symptom**: `Connection refused: localhost:9000`

**Fix**:
```bash
# Check if running
docker ps | grep sonarqube

# Restart
docker compose -f sonarqube/docker-compose.yml restart

# Check logs
docker logs smellai-sonarqube | tail -50

# Wait for "SonarQube is operational"
```

#### 2. Token Authentication Failed

**Symptom**: `401 Unauthorized: SonarQube API`

**Fix**:
```bash
# Generate new token at http://localhost:9000
# My Account → Security → Generate Token

# Update .env
echo "SONAR_TOKEN=sqa_abc123..." >> .env

# Verify
grep SONAR_TOKEN .env
```

#### 3. Compilation Errors in Workspace

**Symptom**: `CompileResult(success=False)`

**Fix**:
```bash
# Check JDK version
jenv versions

# Install required JDK
# Java 8: brew install --cask adoptopenjdk8
# Java 11: brew install --cask adoptopenjdk11
# Java 17: brew install openjdk@17
# Java 21: brew install openjdk@21

# Add to jenv
jenv add /path/to/jdk

# Verify
jenv versions
```

#### 4. Visualizer Shows Empty Graph

**Symptom**: No data in visualizer UI

**Fix**:
```bash
# Verify database exists
ls -lh demo_composite.db

# Check records
sqlite3 demo_composite.db "SELECT COUNT(*) FROM smell_events;"

# Ensure evaluation completed successfully
# Re-run evaluation if needed
```

#### 5. Port Already in Use

**Symptom**: `Address already in use: 8080`

**Fix**:
```bash
# Kill existing process
lsof -ti:8080 | xargs kill -9

# Or change port in tools/visualize_smell_prioritization.py:1074
# ui.run(title="...", port=8081)
```

---

## Performance Tips

### Speed up evaluation:

1. **Use faster model**: `--model gpt-4o-mini`
2. **Reduce iterations**: `--max-refactorings 2`
3. **Limit records**: `--limit 1`
4. **Filter by JDK**: Use Java 11 (most records, good tooling)
5. **Cache SonarQube**: Reuse `--sonar-cache-dir`

### Optimize resource usage:

```bash
# Limit Docker memory
docker compose -f sonarqube/docker-compose.yml up -d \
  --scale postgres=1 \
  --memory="2g"

# Clean up workspace
rm -rf /tmp/swe-eval-workspace/*

# Vacuum databases
sqlite3 demo_composite.db "VACUUM;"
sqlite3 mlflow.db "VACUUM;"
```

---

## File Organization

```
smellai/
├── compound_demo.json          # 2 record demo dataset
├── compound_sample_j17.json    # 15 record sample
├── demo_composite.db           # Analytics database
├── mlflow.db                   # MLflow tracking database
├── scripts/
│   ├── extract_compound_refactorings.py  # Dataset extractor
│   ├── run_demo_eval.py                  # Automated eval runner
│   ├── run_visualizer.sh                 # Visualizer launcher
│   └── README_COMPOUND_EXTRACTION.md     # Extraction guide
├── workflows/
│   └── swe_eval_workflow.py              # Main evaluation workflow
├── agents/
│   ├── java_test/                        # A0, A6
│   ├── swe_eval/                         # Main composite agent
│   └── dependency_analysis/              # A3
├── tools/
│   └── visualize_smell_prioritization.py # NiceGUI visualizer
├── swe_refactor/
│   ├── dataset.py                        # Dataset loader
│   ├── persistence/                      # Analytics DB models
│   └── SWE-Refactor.zip                  # Original dataset
└── Documentation (this file and others)
```

---

## Next Steps

1. ✅ **Run demo**: `uv run python scripts/run_demo_eval.py`
2. ✅ **Visualize**: `./scripts/run_visualizer.sh` → Load `demo_composite.db`
3. 📊 **Analyze**: Check MLflow metrics, explore visualizations
4. 🔬 **Experiment**: Try different models, datasets, configurations
5. 📝 **Document**: Record findings for thesis

---

## Contact & Support

- GitHub Issues: [smellai/issues](https://github.com/user/smellai/issues)
- Documentation: See `scripts/README_*.md` files
- Dataset source: [SWE-Refactor benchmark](https://arxiv.org/abs/2410.13782)
