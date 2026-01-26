# Composite Refactoring System - Comprehensive Test Plan

## Overview
Test plan for validating the composite refactoring system implementation, including workflow execution, analytics persistence, MLflow integration, and visualization.

---

## Phase 1: Unit Testing (Code-Level Verification)

### 1.1 Persistence Layer Tests

**Test: Analytics Database Schema**
```bash
# Verify database tables creation
sqlite3 test_analytics.db ".schema"
```
Expected tables:
- `tool_calls`
- `smell_events`
- `smell_dependencies`
- `refactoring_attempts`
- `token_usage`

**Test: CRUD Operations**
```python
from swe_refactor.persistence.database import AnalyticsDB
from swe_refactor.persistence.models import SmellEvent

db = AnalyticsDB("test_analytics.db")

# Test smell event logging
event = SmellEvent(
    session_id="test-123",
    iteration=0,
    id="Long Method:src/Test.java:42",
    smell_type="Long Method",
    location="src/Test.java:42",
    severity="HIGH",
    status="detected"
)
db.log_smell_event(event)

# Verify retrieval
events = db.get_session_summary("test-123")
assert len(events["smells"]) == 1
```

### 1.2 Smell Detection Utilities Tests

**Test: Local Project Scan**
```python
from swe_refactor.smell_detection.utils import scan_local_project

smells = scan_local_project(
    project_path="./test_project",
    project_key="test_scan",
    sonar_url="http://localhost:9000",
    session_id="test-456",
    iteration=0
)
assert isinstance(smells, list)
assert all(isinstance(s, SmellEvent) for s in smells)
```

**Test: Smell Comparison**
```python
from swe_refactor.smell_detection.utils import compare_smell_sets

before = [...]  # SmellEvent list
after = [...]   # SmellEvent list

diff = compare_smell_sets(before, after)
assert "resolved" in diff
assert "created" in diff
assert "persistent" in diff
```

### 1.3 Agent Node Tests

**Test: A1 Detect Smells**
- Mock SonarQube response
- Verify SmellEvent objects created
- Check analytics DB logging

**Test: A2 Prioritize Smells**
- Provide sample smells with dependencies
- Verify priority queue ordering (PZ scoring)
- Check smell graph structure (NetworkX DiGraph)

**Test: A3 Select Next Smell**
- Test iteration limit enforcement
- Verify queue popping behavior
- Test empty queue handling

**Test: A4 Map Smell to Refactoring**
- Test smell type → refactoring type mapping
- Verify default fallback

**Test: A5 Generate (Modified)**
- Verify smell context added to prompt when `current_smell` exists
- Mock LLM response with `response_metadata`
- Verify token usage logged to analytics DB

**Test: A6 Verify (Modified)**
- Mock successful compilation
- Mock SonarQube re-scan
- Verify smell diff calculation
- Check RefactoringAttempt logged

---

## Phase 2: Integration Testing (Workflow Execution)

### 2.1 Basic Mode (Backward Compatibility)

**Test: Basic Workflow Execution**
```bash
# Prerequisites
# - SWE-Refactor dataset available
# - No SonarQube required
# - MLflow server NOT required for this test

uv run workflows/swe_eval_workflow.py \
  --dataset /tmp/SWE-Refactor/pure_refactoring_data.json \
  --limit 1 \
  --model claude-sonnet-4-5-20250929 \
  --workspace /tmp/test-basic-workspace
```

**Expected Behavior:**
- Agent follows: A0 → A5 → A6
- No smell detection occurs
- No analytics DB created
- Compilation and test results returned
- No composite state fields used

**Verification:**
```bash
# Check workspace
ls /tmp/test-basic-workspace

# Verify no analytics DB
! [ -f analytics.db ]
```

### 2.2 Composite Mode (New Functionality)

**Prerequisites:**
1. Start SonarQube:
```bash
docker run -d --name sonarqube -p 9000:9000 sonarqube:latest
# Wait for startup, create token
```

2. Set environment variable:
```bash
export SONAR_TOKEN="your_sonarqube_token"
```

**Test: Composite Workflow Execution**
```bash
uv run workflows/swe_eval_workflow.py \
  --dataset /tmp/SWE-Refactor/pure_refactoring_data.json \
  --limit 1 \
  --model claude-sonnet-4-5-20250929 \
  --workspace /tmp/test-composite-workspace \
  --enable-composite \
  --max-refactorings 3 \
  --analytics-db ./test_composite_analytics.db \
  --sonar-url http://localhost:9000 \
  --sonar-cache-dir ./test_sonar_cache
```

**Expected Behavior:**
- Agent follows: A0 → A1 → A2 → A3 → [A4 → A5 → A6 → increment → A1] (loop up to 3 times)
- SonarQube scans project at each iteration
- Smells detected, prioritized, selected
- Analytics DB populated with:
  - SmellEvents (before/after each refactoring)
  - RefactoringAttempts (success/failure tracking)
  - TokenUsage (per A5 invocation)
- Session summary printed at end

**Verification:**
```bash
# Check analytics DB created
ls -lh test_composite_analytics.db

# Query analytics data
sqlite3 test_composite_analytics.db <<EOF
SELECT COUNT(*) FROM smell_events;
SELECT COUNT(*) FROM refactoring_attempts;
SELECT COUNT(*) FROM token_usage;
SELECT 
  iteration, 
  outcome, 
  smells_resolved, 
  smells_created 
FROM refactoring_attempts 
ORDER BY iteration;
EOF
```

**Test: Iteration Limit Enforcement**
```bash
# Run with max_refactorings=2
uv run workflows/swe_eval_workflow.py \
  --commit <commit_hash> \
  --project <project_name> \
  --enable-composite \
  --max-refactorings 2 \
  --analytics-db ./test_limit.db
```

Expected: Agent stops after 2 refactoring iterations

**Test: Empty Priority Queue**
```bash
# Use project with no smells (or all resolved)
# Should terminate early via A3 → END
```

### 2.3 Error Handling Tests

**Test: Compilation Failure with Retry**
- Mock LLM to produce invalid code
- Verify retry logic: A6 → A5 (up to MAX_RETRIES)
- Check retry count in RefactoringAttempt

**Test: SonarQube Unavailable**
```bash
# Stop SonarQube container
docker stop sonarqube

# Run composite mode
# Should log warning and continue (smell detection fails gracefully)
```

**Test: Analytics DB Write Failure**
- Use read-only filesystem for DB
- Verify workflow continues (analytics logging is non-blocking)

---

## Phase 3: MLflow Integration Testing

### 3.1 MLflow Server Setup

**Start MLflow Server:**
```bash
# Automated via workflow (auto_start_server=True)
# Or manually:
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

**Verify Server Running:**
```bash
curl http://localhost:5000/health
```

### 3.2 Experiment Tracking Tests

**Test: Run Logging (Basic Mode)**
```bash
uv run workflows/swe_eval_workflow.py \
  --dataset /tmp/SWE-Refactor/pure_refactoring_data.json \
  --limit 3 \
  --experiment test-basic-experiment \
  --tracking-uri http://localhost:5000
```

**Expected MLflow Metrics:**
- `compile_success_rate`: e.g., 0.6667
- `test_pass_rate`: e.g., 0.5000
- `overall_success_rate`: e.g., 0.3333

**Test: Run Logging (Composite Mode)**
```bash
uv run workflows/swe_eval_workflow.py \
  --dataset /tmp/SWE-Refactor/pure_refactoring_data.json \
  --limit 3 \
  --experiment test-composite-experiment \
  --tracking-uri http://localhost:5000 \
  --enable-composite \
  --max-refactorings 5 \
  --analytics-db ./mlflow_test_analytics.db
```

**Expected Additional Metrics (per record output):**
- `session_id`: UUID
- `smells_resolved`: integer
- `smells_created`: integer
- `total_tokens`: integer
- `iterations`: integer

**Test: Parameter Logging**
Verify MLflow logs parameters:
- `model`: "claude-sonnet-4-5-20250929"
- `enable_composite`: True/False
- `max_refactorings`: 5
- `sonar_url`: "http://localhost:9000"

**Test: Artifact Logging**
Check if code diffs, error logs, or graphs are logged as artifacts.

---

## Phase 4: MLflow UI Testing with agent-browser

### 4.1 UI Navigation Tests

**Test: Access Experiments List**
```bash
npx agent-browser \
  --url http://localhost:5000 \
  --action screenshot \
  --output mlflow_experiments.png
```

**Manual Verification:**
- Experiments page loads
- "test-basic-experiment" and "test-composite-experiment" visible
- Run counts displayed

**Test: Navigate to Experiment Runs**
```bash
npx agent-browser \
  --url "http://localhost:5000/#/experiments/<experiment_id>" \
  --action screenshot \
  --output mlflow_runs_list.png
```

**Verification:**
- Runs table displays with columns: Name, Status, Start Time, Duration, Metrics
- Metrics columns show: `compile_success_rate`, `test_pass_rate`, `overall_success_rate`

**Test: Open Individual Run Details**
```bash
npx agent-browser \
  --url "http://localhost:5000/#/experiments/<exp_id>/runs/<run_id>" \
  --action screenshot \
  --output mlflow_run_details.png
```

**Verification:**
- Run metadata visible
- Parameters section shows model, composite mode settings
- Metrics section shows success rates
- Artifacts section (if any) accessible

### 4.2 Metrics Visualization Tests

**Test: Compare Multiple Runs**
```bash
# Run workflow multiple times with different parameters
uv run workflows/swe_eval_workflow.py --limit 3 --model gpt-4o
uv run workflows/swe_eval_workflow.py --limit 3 --model claude-sonnet-4-5-20250929
uv run workflows/swe_eval_workflow.py --limit 3 --enable-composite --max-refactorings 3

# Navigate to compare view
npx agent-browser \
  --url "http://localhost:5000/#/compare-runs?runs=<run1>,<run2>,<run3>" \
  --action screenshot \
  --output mlflow_compare_runs.png
```

**Verification:**
- Side-by-side comparison table
- Metrics differences highlighted
- Parameter differences visible (model, composite mode)

**Test: Metrics Charts**
```bash
# Access chart view
npx agent-browser \
  --url "http://localhost:5000/#/experiments/<exp_id>" \
  --action click \
  --selector "button[data-test-id='chart-view-button']" \
  --then screenshot \
  --output mlflow_metrics_chart.png
```

**Verification:**
- Line/bar charts for metrics over runs
- X-axis: Run ID or timestamp
- Y-axis: Metric values (0.0 to 1.0 for success rates)

### 4.3 Composite Mode Metrics Verification

**Test: Custom Metrics Display**

Since composite mode returns additional fields (`smells_resolved`, `smells_created`, `total_tokens`, `iterations`) in outputs, we need to verify if MLflow captures these.

**Note:** MLflow GenAI `evaluate()` may only log aggregated metrics from scorers, not individual output fields.

**Workaround:** Create custom scorers for composite metrics:
```python
# Add to swe_eval_workflow.py

def smells_resolved_scorer(outputs: dict, inputs: dict) -> float:
    """Average smells resolved per record."""
    return float(outputs.get("smells_resolved", 0))

def smells_created_scorer(outputs: dict, inputs: dict) -> float:
    """Average smells created per record."""
    return float(outputs.get("smells_created", 0))

def total_tokens_scorer(outputs: dict, inputs: dict) -> float:
    """Total tokens used."""
    return float(outputs.get("total_tokens", 0))

# Add to scorers list when enable_composite
if args.enable_composite:
    scorers.extend([
        smells_resolved_scorer,
        smells_created_scorer,
        total_tokens_scorer,
    ])
```

**Test with Enhanced Scorers:**
```bash
# After adding scorers, re-run composite workflow
uv run workflows/swe_eval_workflow.py \
  --limit 3 \
  --enable-composite \
  --max-refactorings 5

# Check MLflow UI for new metrics
npx agent-browser \
  --url "http://localhost:5000/#/experiments/<exp_id>/runs/<run_id>" \
  --action screenshot \
  --output mlflow_composite_metrics.png
```

**Verification:**
- Metrics section shows:
  - `smells_resolved_scorer/mean`
  - `smells_created_scorer/mean`
  - `total_tokens_scorer/mean`
- Values match session summaries printed to console

---

## Phase 5: Analytics Database Visualization

### 5.1 Direct SQL Queries

**Test: Session Summary Query**
```sql
-- Get overview of all sessions
SELECT 
  session_id,
  COUNT(DISTINCT iteration) as iterations,
  SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as successful_refactorings,
  SUM(smells_resolved) as total_resolved,
  SUM(smells_created) as total_created
FROM refactoring_attempts
GROUP BY session_id;
```

**Test: Token Usage by Node**
```sql
SELECT 
  node_name,
  SUM(total_tokens) as total_tokens,
  AVG(total_tokens) as avg_tokens_per_call,
  COUNT(*) as invocations
FROM token_usage
GROUP BY node_name;
```

**Test: Smell Resolution Timeline**
```sql
SELECT 
  iteration,
  COUNT(*) as smells_detected,
  SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved,
  SUM(CASE WHEN status = 'created' THEN 1 ELSE 0 END) as created
FROM smell_events
WHERE session_id = '<session_id>'
GROUP BY iteration
ORDER BY iteration;
```

### 5.2 Python-Based Visualization

**Create Visualization Script: `scripts/visualize_analytics.py`**
```python
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_smell_evolution(db_path: str, session_id: str):
    """Plot smell counts over iterations."""
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
      iteration,
      SUM(CASE WHEN status = 'detected' THEN 1 ELSE 0 END) as detected,
      SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved,
      SUM(CASE WHEN status = 'created' THEN 1 ELSE 0 END) as created
    FROM smell_events
    WHERE session_id = ?
    GROUP BY iteration
    ORDER BY iteration
    """
    
    df = pd.read_sql_query(query, conn, params=(session_id,))
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['iteration'], df['detected'], marker='o', label='Detected')
    plt.plot(df['iteration'], df['resolved'], marker='s', label='Resolved')
    plt.plot(df['iteration'], df['created'], marker='^', label='Created')
    plt.xlabel('Iteration')
    plt.ylabel('Smell Count')
    plt.title(f'Smell Evolution - Session {session_id[:8]}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'smell_evolution_{session_id[:8]}.png')
    plt.close()
    
    conn.close()

def plot_refactoring_success_rate(db_path: str):
    """Plot success rate across all sessions."""
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
      session_id,
      SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
    FROM refactoring_attempts
    GROUP BY session_id
    """
    
    df = pd.read_sql_query(query, conn)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(df)), df['success_rate'])
    plt.xlabel('Session')
    plt.ylabel('Success Rate (%)')
    plt.title('Refactoring Success Rate by Session')
    plt.axhline(y=50, color='r', linestyle='--', label='50% Baseline')
    plt.legend()
    plt.xticks(range(len(df)), [sid[:8] for sid in df['session_id']], rotation=45)
    plt.tight_layout()
    plt.savefig('success_rate_by_session.png')
    plt.close()
    
    conn.close()

def plot_token_usage(db_path: str):
    """Plot token usage by node."""
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
      node_name,
      SUM(total_tokens) as total_tokens
    FROM token_usage
    GROUP BY node_name
    """
    
    df = pd.read_sql_query(query, conn)
    
    plt.figure(figsize=(8, 6))
    plt.pie(df['total_tokens'], labels=df['node_name'], autopct='%1.1f%%')
    plt.title('Token Usage Distribution by Node')
    plt.savefig('token_usage_distribution.png')
    plt.close()
    
    conn.close()

if __name__ == "__main__":
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "analytics.db"
    
    # Get first session for evolution plot
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT DISTINCT session_id FROM smell_events LIMIT 1")
    session_id = cursor.fetchone()
    conn.close()
    
    if session_id:
        plot_smell_evolution(db_path, session_id[0])
    
    plot_refactoring_success_rate(db_path)
    plot_token_usage(db_path)
    
    print("Visualizations saved:")
    print("- smell_evolution_<session>.png")
    print("- success_rate_by_session.png")
    print("- token_usage_distribution.png")
```

**Test: Generate Visualizations**
```bash
# After running composite workflow
uv run python scripts/visualize_analytics.py test_composite_analytics.db

# View generated images
open smell_evolution_*.png
open success_rate_by_session.png
open token_usage_distribution.png
```

### 5.3 Interactive Dashboard (Optional Enhancement)

**Create Streamlit Dashboard: `scripts/dashboard.py`**
```python
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

st.title("Composite Refactoring Analytics Dashboard")

db_path = st.sidebar.text_input("Database Path", "analytics.db")

if st.sidebar.button("Load Data"):
    conn = sqlite3.connect(db_path)
    
    # Session overview
    st.header("Session Overview")
    sessions_df = pd.read_sql_query("""
        SELECT 
          session_id,
          COUNT(DISTINCT iteration) as iterations,
          SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as successful_refactorings,
          SUM(smells_resolved) as total_resolved,
          SUM(smells_created) as total_created
        FROM refactoring_attempts
        GROUP BY session_id
    """, conn)
    st.dataframe(sessions_df)
    
    # Session selector
    session_id = st.selectbox("Select Session", sessions_df['session_id'].tolist())
    
    # Smell evolution chart
    st.header(f"Smell Evolution - {session_id[:8]}")
    smell_df = pd.read_sql_query("""
        SELECT 
          iteration,
          smell_type,
          status
        FROM smell_events
        WHERE session_id = ?
    """, conn, params=(session_id,))
    
    if not smell_df.empty:
        smell_counts = smell_df.groupby(['iteration', 'status']).size().reset_index(name='count')
        fig = px.line(smell_counts, x='iteration', y='count', color='status', markers=True)
        st.plotly_chart(fig)
    
    # Token usage
    st.header("Token Usage")
    token_df = pd.read_sql_query("""
        SELECT 
          node_name,
          SUM(total_tokens) as total_tokens,
          AVG(prompt_tokens) as avg_prompt,
          AVG(completion_tokens) as avg_completion
        FROM token_usage
        WHERE session_id = ?
        GROUP BY node_name
    """, conn, params=(session_id,))
    
    if not token_df.empty:
        fig = px.bar(token_df, x='node_name', y='total_tokens', title='Total Tokens by Node')
        st.plotly_chart(fig)
    
    conn.close()
```

**Test: Run Dashboard**
```bash
uv run streamlit run scripts/dashboard.py

# Use agent-browser to test
npx agent-browser \
  --url http://localhost:8501 \
  --action screenshot \
  --output streamlit_dashboard.png
```

---

## Phase 6: Graph Visualization Testing

### 6.1 LangGraph Agent Graph

**Test: Generate Basic Mode Graph**
```bash
uv run workflows/swe_eval_workflow.py --draw-graph

# Verify output
open swe_eval_agent_graph.png
```

**Expected Graph:**
```
START → a0_setup → a5_generate → a6_verify → [retry → a5_generate] or [end → END]
```

**Test: Generate Composite Mode Graph**
```bash
uv run workflows/swe_eval_workflow.py --draw-graph --enable-composite

# Verify output
open swe_eval_agent_graph.png
```

**Expected Graph:**
```
START → a0_setup → a1_detect_smells → a2_prioritize_smells → a3_select_next_smell
                                                                ↓
                                                              [continue]
                                                                ↓
                                                    a4_map_smell_to_refactoring
                                                                ↓
                                                           a5_generate
                                                                ↓
                                                            a6_verify
                                                                ↓
                                                        [retry or next_iteration]
                                                                ↓
                                                         increment_iteration
                                                                ↓
                                                          (loop to A1)
                                                                
                                         a3_select_next_smell → [end] → END
```

### 6.2 Smell Dependency Graph

**Test: Export Smell Graph from State**

Modify `a2_prioritize_smells()` to save graph:
```python
# In a2_prioritize_smells node
import pickle

if state.get("analytics_db"):
    graph_path = f"smell_graph_{state['session_id'][:8]}.pkl"
    with open(graph_path, 'wb') as f:
        pickle.dump(prioritizer.graph, f)
```

**Visualize NetworkX Graph:**
```python
import pickle
import networkx as nx
import matplotlib.pyplot as plt

with open('smell_graph_<session>.pkl', 'rb') as f:
    G = nx.read_gpickle(f)

plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=0.5, iterations=50)
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)

# Color code edges by dependency type
positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'positive']
negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'negative']

nx.draw_networkx_edges(G, pos, edgelist=positive_edges, edge_color='green', arrows=True, width=2)
nx.draw_networkx_edges(G, pos, edgelist=negative_edges, edge_color='red', arrows=True, width=2, style='dashed')

plt.title('Smell Dependency Graph')
plt.axis('off')
plt.tight_layout()
plt.savefig('smell_dependency_graph.png', dpi=300)
plt.show()
```

---

## Phase 7: End-to-End Validation

### 7.1 Complete Workflow Test

**Test Scenario: Multi-Record Composite Evaluation**
```bash
# Prerequisites: SonarQube running, dataset available

uv run workflows/swe_eval_workflow.py \
  --dataset /tmp/SWE-Refactor/pure_refactoring_data.json \
  --limit 5 \
  --model claude-sonnet-4-5-20250929 \
  --experiment e2e-composite-test \
  --tracking-uri http://localhost:5000 \
  --enable-composite \
  --max-refactorings 3 \
  --analytics-db ./e2e_test_analytics.db \
  --sonar-url http://localhost:9000 \
  --sonar-cache-dir ./e2e_sonar_cache
```

**Success Criteria:**
1. ✅ Workflow completes without crashes
2. ✅ All 5 records processed
3. ✅ MLflow run created with metrics
4. ✅ Analytics DB contains data for 5 sessions
5. ✅ Console shows session summaries for each record
6. ✅ SonarQube cache directory populated

**Verification Steps:**

1. **Check MLflow UI:**
```bash
npx agent-browser \
  --url "http://localhost:5000/#/experiments/<exp_id>" \
  --action screenshot \
  --output e2e_mlflow_results.png
```
Expected: Run with 5 evaluated records, aggregate metrics displayed

2. **Check Analytics DB:**
```sql
sqlite3 e2e_test_analytics.db <<EOF
-- Should have 5 unique sessions
SELECT COUNT(DISTINCT session_id) FROM refactoring_attempts;

-- Check total attempts
SELECT COUNT(*) FROM refactoring_attempts;

-- Check smell events
SELECT COUNT(*) FROM smell_events;

-- Check token usage
SELECT SUM(total_tokens) FROM token_usage;
EOF
```

3. **Generate Visualizations:**
```bash
uv run python scripts/visualize_analytics.py e2e_test_analytics.db

# Verify images created
ls -lh smell_evolution_*.png success_rate_by_session.png token_usage_distribution.png
```

4. **Compare Basic vs Composite Mode:**
```bash
# Run same records in basic mode
uv run workflows/swe_eval_workflow.py \
  --dataset /tmp/SWE-Refactor/pure_refactoring_data.json \
  --limit 5 \
  --experiment e2e-basic-test \
  --tracking-uri http://localhost:5000

# Compare in MLflow UI
npx agent-browser \
  --url "http://localhost:5000/#/compare-experiments?experiments=<exp1>,<exp2>" \
  --action screenshot \
  --output basic_vs_composite_comparison.png
```

Expected differences:
- Basic: Single refactoring per record
- Composite: Multiple refactorings per record (up to max_refactorings)
- Composite: Additional metrics (smells_resolved, tokens, iterations)

---

## Phase 8: Performance & Stress Testing

### 8.1 Large Dataset Test
```bash
uv run workflows/swe_eval_workflow.py \
  --dataset /tmp/SWE-Refactor/pure_refactoring_data.json \
  --limit 50 \
  --enable-composite \
  --max-refactorings 5 \
  --analytics-db ./stress_test_analytics.db
```

**Monitor:**
- Execution time
- Database size growth
- Memory usage
- SonarQube cache size

### 8.2 High Iteration Limit Test
```bash
uv run workflows/swe_eval_workflow.py \
  --commit <commit_with_many_smells> \
  --project <project> \
  --enable-composite \
  --max-refactorings 20
```

**Verify:**
- Agent terminates after 20 iterations or when priority queue empty
- No infinite loops

---

## Test Automation Scripts

### `tests/test_composite_workflow.sh`
```bash
#!/bin/bash
set -e

echo "Starting Composite Workflow Tests..."

# Test 1: Basic mode
echo "Test 1: Basic Mode"
uv run workflows/swe_eval_workflow.py \
  --dataset /tmp/SWE-Refactor/pure_refactoring_data.json \
  --limit 1 \
  --workspace /tmp/test-basic

# Test 2: Composite mode
echo "Test 2: Composite Mode"
uv run workflows/swe_eval_workflow.py \
  --dataset /tmp/SWE-Refactor/pure_refactoring_data.json \
  --limit 1 \
  --workspace /tmp/test-composite \
  --enable-composite \
  --max-refactorings 3 \
  --analytics-db ./test_composite.db

# Test 3: Analytics DB validation
echo "Test 3: Analytics DB Validation"
COUNT=$(sqlite3 test_composite.db "SELECT COUNT(*) FROM refactoring_attempts;")
if [ "$COUNT" -gt 0 ]; then
  echo "✅ Analytics DB populated"
else
  echo "❌ Analytics DB empty"
  exit 1
fi

# Test 4: Graph generation
echo "Test 4: Graph Generation"
uv run workflows/swe_eval_workflow.py --draw-graph --enable-composite
if [ -f "swe_eval_agent_graph.png" ]; then
  echo "✅ Graph generated"
else
  echo "❌ Graph generation failed"
  exit 1
fi

echo "All tests passed!"
```

---

## Summary Checklist

### Code-Level Tests
- [ ] Persistence layer CRUD operations
- [ ] Smell detection utils (scan, compare)
- [ ] Agent node unit tests (A1-A6)

### Integration Tests
- [ ] Basic mode execution (backward compatibility)
- [ ] Composite mode execution (new functionality)
- [ ] Iteration limit enforcement
- [ ] Error handling (compilation failure, SonarQube unavailable)

### MLflow Tests
- [ ] Experiment creation
- [ ] Run logging (basic metrics)
- [ ] Run logging (composite metrics)
- [ ] Parameter logging

### MLflow UI Tests (agent-browser)
- [ ] Experiments list page
- [ ] Run details page
- [ ] Compare runs page
- [ ] Metrics charts

### Analytics Visualization
- [ ] SQL queries work correctly
- [ ] Python visualization script generates charts
- [ ] Streamlit dashboard loads and displays data

### Graph Visualization
- [ ] LangGraph agent graph (basic mode)
- [ ] LangGraph agent graph (composite mode)
- [ ] Smell dependency graph (NetworkX)

### End-to-End
- [ ] Multi-record evaluation completes
- [ ] All systems integrated (workflow → SonarQube → analytics DB → MLflow)
- [ ] Basic vs composite comparison in MLflow
- [ ] Performance acceptable for large datasets

---

## Next Steps

Would you like me to:
1. **Load the agent-browser skill** and start executing UI tests?
2. **Create the visualization scripts** (`visualize_analytics.py`, `dashboard.py`)?
3. **Create the test automation script** (`test_composite_workflow.sh`)?
4. **Run a specific test** from the plan above?

Let me know where you'd like to start!
