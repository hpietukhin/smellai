# Run Composite Evaluation Demo

## Current Status

✅ **SonarQube**: Running (docker containers started)
✅ **Dataset**: `compound_demo.json` (2 compound refactorings from commons-io, Java 17)
✅ **Visualizer**: Ready (`scripts/run_visualizer.sh`)

## Full Workflow (Step-by-Step)

### Step 1: Wait for SonarQube (30-60 seconds)

```bash
# Check status (should show "UP")
docker logs smellai-sonarqube | grep "SonarQube is operational"

# Or check via browser
open http://localhost:9000
```

**Wait until**: You see the SonarQube login page at http://localhost:9000

### Step 2: Run Composite Evaluation

**Quick demo (1 record, ~5-10 minutes)**:
```bash
uv run workflows/swe_eval_workflow.py \
  --dataset compound_demo.json \
  --enable-composite \
  --max-refactorings 3 \
  --analytics-db demo_composite.db \
  --model claude-sonnet-4-5-20250929 \
  --limit 1
```

**Full sample (15 records, ~30-60 minutes)**:
```bash
uv run workflows/swe_eval_workflow.py \
  --dataset compound_sample_j17.json \
  --enable-composite \
  --max-refactorings 5 \
  --analytics-db composite_eval.db \
  --model claude-sonnet-4-5-20250929
```

### Step 3: View Results in Visualizer

**Start visualizer**:
```bash
./scripts/run_visualizer.sh
# Opens http://localhost:8080
```

**In browser**:
1. Click **"Load Database"**
2. Enter path: `demo_composite.db` (or `composite_eval.db`)
3. Select session from dropdown
4. Explore:
   - **Smell dependency graph** (center): Nodes = smells, edges = dependencies, colors = PZ scores
   - **Agent timeline** (bottom): Node invocations with timing
   - **Iteration selector** (sidebar): Step through refactoring iterations
   - **Smell details** (sidebar): Before/after counts, outcomes
   - **Tool logs** (sidebar): Debugging info

### Step 4: View MLflow Results

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Opens http://localhost:5000
```

Navigate to **"swe-refactor-evaluation"** experiment to see:
- `compile_success_rate`
- `test_pass_rate`
- `overall_success_rate`
- Per-record outputs

## What Happens in Composite Mode

### Agent Execution Flow

```
┌─────────────────────────────────────────────────────────┐
│ A0: Setup                                               │
│ - Clone repo to workspace                              │
│ - Detect build system (Maven/Gradle)                   │
│ - Run initial tests                                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ A1: Smell Detection                                     │
│ - Run SonarQube scanner                                 │
│ - Fetch smell issues (8 types)                         │
│ - Store as SmellEvent(action="detected")               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ A3: Prioritization                                      │
│ - Build dependency graph (NetworkX)                     │
│ - Compute PZ scores: PZ_i = Severity + Σ(dependencies) │
│ - Rank smells by priority                              │
└─────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────┐
        │     ITERATION LOOP (N times)    │
        │  Max iterations = max-refactorings │
        └─────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ A4: Prompt Prep (TODO)                                  │
│ - Select highest PZ score smell                         │
│ - Prepare refactoring prompt                            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ A5: Refactoring Generation                              │
│ - LLM generates code changes                            │
│ - Apply changes to workspace                            │
│ - Record RefactoringAttempt                             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ A6: Behavior Verification                               │
│ - Compile code (Maven/Gradle)                           │
│ - Run tests                                              │
│ - Update RefactoringAttempt.outcome                     │
└─────────────────────────────────────────────────────────┘
                          ↓
                ┌─────────────────┐
                │ Tests pass?     │
                └─────────────────┘
                   ↓         ↓
                 YES        NO
                   ↓         ↓
        Re-scan smells   Rollback
        Record resolved  Record failure
                   ↓
        ┌─────────────────┐
        │ More smells?    │
        │ Under max iter? │
        └─────────────────┘
           ↓         ↓
         YES        NO
           ↓         ↓
      LOOP BACK    END
```

### Data Persistence

**Analytics DB** (`demo_composite.db`):
- `SmellEvent`: detected/resolved/created per iteration
- `RefactoringAttempt`: outcome, smells_resolved, smells_created
- `ToolCall`: node invocations with timing
- `TokenUsage`: LLM costs per node

**MLflow DB** (`mlflow.db`):
- Experiment runs with metrics
- Input/output records
- Model parameters

## Visualizer Features

### Smell Dependency Graph

**Nodes**:
- Size = PZ score (larger = higher priority)
- Color = Severity (red = high, yellow = medium, green = low)
- Label = Smell type + location

**Edges**:
- Green solid = Positive dependency (refactoring helps resolve target)
- Red dashed = Negative dependency (refactoring may create target)

**Interactions**:
- Click node → Show smell details in sidebar
- Use iteration selector → See how graph evolves

### Agent Timeline

**Horizontal bars**:
- Each bar = One node execution
- Length = Duration
- Color = Node type (A0/A1/A3/A5/A6)

**Click bar** → See tool calls and logs

### Iteration Selector

**Slider** → Navigate through refactoring iterations
- Iteration 0: Initial state (after A1 detection)
- Iteration 1+: After each refactoring attempt

**Per iteration**:
- Smells before
- Smells after
- Refactoring outcome
- Code diff (if available)

## Expected Output

### Terminal Output (during evaluation)
```
Creating agent...
Model: claude-sonnet-4-5-20250929
Mode: Composite
Records: 1
Workspace: /tmp/swe-eval-workspace

Running evaluation on 1 records...
[Agent A0] Cloning commons-io...
[Agent A0] Detected Maven project
[Agent A0] Running tests... PASSED (145/145)
[Agent A1] Running SonarQube scan...
[Agent A1] Detected 12 smells
[Agent A3] Computing PZ scores...
[Agent A3] Top priority: Complex Method (PZ=8.5)
[Iteration 1]
  [Agent A5] Generating refactoring...
  [Agent A6] Compiling... SUCCESS
  [Agent A6] Running tests... PASSED (145/145)
  [Agent A1] Re-scanning... 10 smells remain
  [Result] SUCCESS: 2 smells resolved, 0 created
[Iteration 2]
  ...

============================================================
EVALUATION RESULTS
============================================================
compile_success_rate: 1.0000
test_pass_rate: 1.0000
overall_success_rate: 1.0000
============================================================

Run ID: abc123...
```

### Visualizer UI

**Main graph** (center):
- 12 smell nodes initially
- Edges showing dependencies
- Animated transitions as you step through iterations

**Sidebar** (left):
- Iteration 0: 12 smells detected
- Iteration 1: SUCCESS (10 smells remain)
- Iteration 2: SUCCESS (7 smells remain)
- ...

**Timeline** (bottom):
- A0: 45s (setup + tests)
- A1: 120s (SonarQube scan)
- A3: 2s (prioritization)
- A5: 30s (LLM generation)
- A6: 35s (compile + tests)

## Troubleshooting

### "Connection refused: SonarQube"
```bash
# Check if running
docker ps | grep sonarqube

# Check logs
docker logs smellai-sonarqube

# Restart if needed
docker compose -f sonarqube/docker-compose.yml restart

# Wait for "SonarQube is operational" message
```

### "No token for SonarQube"
```bash
# Generate token at: http://localhost:9000
# Login: admin / admin (first time, will prompt to change)
# My Account → Security → Generate Token

# Add to .env
echo "SONAR_TOKEN=your_token_here" >> .env
```

### "Workspace permission denied"
```bash
# Ensure workspace directory is writable
chmod 755 /tmp/swe-eval-workspace
```

### "Port 8080 already in use"
```bash
# Kill existing process
lsof -ti:8080 | xargs kill -9

# Or edit port in tools/visualize_smell_prioritization.py:1074
```

### Evaluation takes too long
- Use `--limit 1` to test with single record first
- Reduce `--max-refactorings` to 2-3
- Use faster model: `--model gpt-4o-mini`

### Visualizer shows no data
- Verify database path: `ls -lh demo_composite.db`
- Check database has data: `sqlite3 demo_composite.db "SELECT COUNT(*) FROM smell_events;"`
- Ensure evaluation completed successfully

## Next Steps

1. **Run quick demo**: 1 record, 3 iterations (~10 min)
2. **Visualize results**: Load `demo_composite.db`
3. **Analyze patterns**: Which smells cascade? Success rate?
4. **Scale up**: Run on full `compound_sample_j17.json` (15 records)
5. **Experiment**: Try different models, projects, iteration limits

## Research Questions

Use the visualizer to investigate:

1. **Dependency accuracy**: Do positive dependencies actually help?
2. **Cascade effects**: Which refactorings create new smells?
3. **Prioritization impact**: Does PZ ordering improve success rate?
4. **Iteration efficiency**: Optimal max-refactorings value?
5. **Model comparison**: GPT-4o vs Claude Sonnet performance?

## Files Created

- `compound_demo.json` — 2 record demo dataset
- `compound_sample_j17.json` — 15 record sample dataset
- `scripts/extract_compound_refactorings.py` — Dataset extractor
- `scripts/run_visualizer.sh` — Visualizer launcher
- `scripts/run_composite_demo.sh` — Automated demo runner
- `scripts/README_COMPOUND_EXTRACTION.md` — Extraction guide
- `QUICKSTART_COMPOSITE_EVAL.md` — Full workflow guide
- This file — Step-by-step instructions
