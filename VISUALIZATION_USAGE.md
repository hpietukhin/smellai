# AI Agent Execution Visualizer - Usage Guide

## Overview

Enhanced visualization tool showing complete AI agent execution context:
- **Agent execution timeline** - when each node ran, how long it took
- **Smell dependency graph** - relationships between code smells with PZ prioritization
- **Iteration details** - outcomes, smells resolved/created, retries
- **Tool call logs** - all tool invocations per iteration
- **Code diff viewer** - git diffs showing actual code changes
- **Decision rationale** - why smells were prioritized (PZ scores, dependencies)

## Quick Start

```bash
# Start the visualization server
cd /Users/havriil.pietukhin/PycharmProjects/smellai3/smellai
uv run python tools/visualize_smell_prioritization.py

# Open in browser
open http://localhost:8080
```

## Loading Agent Execution Data

### Option 1: Load from Analytics Database (Recommended)

1. **Enter database path**: `test_analytics.db`
2. **Click "Load Database"**
3. **Select session** from dropdown in header
4. **Use iteration slider** to scrub through agent execution

**Available when database loaded:**
- ✅ Iteration timeline slider (enabled)
- ✅ Iteration prev/next buttons (enabled)
- ✅ Agent execution timeline
- ✅ Iteration details, tool logs, code diffs
- ✅ Smell dependency graph

### Option 2: Load Example Manifests (New!)

Load real composite refactoring examples from open-source projects:

1. **Select example** from "Examples" dropdown in left drawer
2. **View smell dependencies** in main graph
3. **See refactoring sequence** in manifest info panel
4. **Use smell priority slider** to step through prioritization

**Available examples:**
- **Simple Example** - Educational 4-smell scenario with dependencies
- **Composite Sequence 1** - Guava: 13 refactorings (Move Method + Move And Rename)
- **Composite Sequence 2** - Checkstyle: 34 Extract And Move Method refactorings
- **Composite Sequence 3** - Commons-Lang: 22 Extract Method refactorings

**Note:** Iteration controls disabled for examples (no agent execution history).

### Option 3: Load Sample Data

Click "Load Sample Data" to see a demo with synthetic smell data (no agent execution context).

## UI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ Header: AI Agent Execution Visualizer | [Session Selector]     │
├────────────┬───────────────────────────────────┬────────────────┤
│ Left       │ Main Content Area                 │ Right          │
│ Drawer     │                                   │ Drawer         │
│            │  ┌─────────────────────────────┐  │                │
│ ┌─────┐    │  │ Smell Dependency Graph     │  │ Iteration      │
│ │Load │    │  │ (with PZ prioritization)   │  │ Details        │
│ │ DB  │    │  └─────────────────────────────┘  │                │
│ └─────┘    │                                   │ Tool Call      │
│            │  ┌─────────────────────────────┐  │ Logs           │
│ Examples   │  │ Agent Execution Timeline    │  │                │
│ Dropdown   │  │ (or Refactoring Timeline)   │  │ Code Diff      │
│            │  └─────────────────────────────┘  │ Viewer         │
│ Iteration  │                                   │                │
│ Timeline   │                                   │                │
│ Slider*    │                                   │                │
│            │                                   │                │
│ Priority   │                                   │                │
│ Sequence   │                                   │                │
│            │                                   │                │
│ Smell      │                                   │                │
│ Details    │                                   │                │
└────────────┴───────────────────────────────────┴────────────────┘
* Iteration controls disabled for examples/sample data
```

## Features

### 1. Agent Execution Timeline (Bottom Panel)

**What it shows:**
- **Database mode**: Gantt-style chart with each agent node (A0-A6) execution
  - Color-coded by node type
  - Shows duration in milliseconds
  - Displays iteration number
- **Example/sample mode**: Refactoring attempts timeline
  - Green bars = successful refactorings
  - Red bars = failed refactorings
  - X-axis = iteration number

**Color Legend:**
- 🔵 A1 (detect_smells) - Blue
- 🟢 A2 (prioritize_smells) - Green
- 🟠 A3 (select_next_smell) - Orange
- 🟣 A4 (map_smell_to_refactoring) - Purple
- 🔴 A5 (generate) - Red
- 🔵 A6 (verify) - Cyan

**How to use:**
- Hover over bars to see node name and duration
- X-axis shows time in seconds from start

### 2. Smell Dependency Graph (Top Panel)

**What it shows:**
- Nodes: Code smells with PZ scores
- Edges: Dependencies between smells
  - Green solid = Positive dependency (resolving A helps resolve B)
  - Red dashed = Negative dependency (resolving A may create B)
- Node colors:
  - Red = High severity (≥3)
  - Orange = Medium severity (2)
  - Green = Low severity (<2)
  - Gray = Resolved smell

**How to use:**
- Click nodes to see detailed rationale in left drawer
- Drag to pan, scroll to zoom
- Blue border = current smell being refactored

### 3. Iteration Details (Right Drawer - Top)

**Shows for current iteration:**
- Target smell ID
- Refactoring type (Extract Method, Extract Class, etc.)
- Outcome: ✅ Success or ❌ Failure reason
- Retry count
- Smells detected/resolved/created
- Net impact (resolved - created)

### 4. Tool Call Logs (Right Drawer - Middle)

**Shows:**
- All tool invocations for current iteration
- Node name (A1-A6)
- Tool name (e.g., sonar.scan, llm.invoke)
- Duration in milliseconds
- Total time and call count

**Use case:**
- Debug agent behavior
- Identify performance bottlenecks
- Understand what tools were used

### 5. Code Diff Viewer (Right Drawer - Bottom)

**Shows:**
- Git diff of code changes made in current iteration
- Unified diff format with syntax highlighting
- Lines added (+) and removed (-)

**Diff format:**
```diff
diff --git a/src/File.java b/src/File.java
--- a/src/File.java
+++ b/src/File.java
@@ -10,5 +10,3 @@
-    old line
+    new line
```

**Note:** Diffs are captured only if refactoring compilation succeeds.

### 6. Iteration Playback Controls (Left Drawer)

**Iteration Timeline Slider:**
- ⚠️ **Only enabled in database mode**
- Scrub through agent iterations (0 to N)
- Updates all panels automatically
- Prev/Next buttons for step-by-step navigation
- **Disabled for examples/sample data** (no iteration history)

**Smell Priority Slider:**
- ✅ **Always enabled**
- Step through smell resolution sequence
- Shows priority order based on PZ scores
- Highlights current smell in graph

**Example Manifest Selector:**
- Load real composite refactoring examples
- Shows project info and refactoring sequence
- Displays smell dependencies for learning

## Database Schema

The visualization reads from these tables:

```sql
-- Agent execution sessions
CREATE TABLE smell_events (
    session_id TEXT,
    iteration INT,
    smell_id TEXT,
    smell_type TEXT,
    severity TEXT,
    file_path TEXT,
    line_number INT,
    action TEXT  -- "detected" | "resolved" | "created"
);

-- Refactoring outcomes
CREATE TABLE refactoring_attempts (
    session_id TEXT,
    iteration INT,
    smell_id TEXT,
    refactoring_type TEXT,
    outcome TEXT,  -- "success" | "test_failed" | "compile_failed"
    retries INT,
    smells_resolved INT,
    smells_created INT,
    code_diff TEXT  -- Git diff string
);

-- Tool invocations (for debugging)
CREATE TABLE tool_calls (
    session_id TEXT,
    iteration INT,
    node_name TEXT,
    tool_name TEXT,
    arguments TEXT,  -- JSON
    result TEXT,     -- JSON
    duration_ms FLOAT
);

-- Token usage tracking
CREATE TABLE token_usage (
    session_id TEXT,
    iteration INT,
    node_name TEXT,
    prompt_tokens INT,
    completion_tokens INT,
    total_tokens INT,
    model TEXT
);
```

## Workflow: Understanding an Agent Run

### For Database Sessions (Full History)
1. **Load database** → Select session → See overview
2. **Check iteration details** → Understand what agent tried to do
3. **View smell graph** → See what smells existed and dependencies
4. **Check code diff** → See actual changes made
5. **Review tool logs** → Debug if something went wrong
6. **Check timeline** → Understand performance
7. **Step through iterations** → Watch agent progression

### For Example Manifests (Learning Mode)
1. **Select example** → Load real-world refactoring scenario
2. **View smell graph** → Understand smell dependencies
3. **Read manifest info** → See refactoring sequence
4. **Use priority slider** → Step through prioritization logic
5. **Click smell nodes** → Examine PZ score calculations

## Example Questions This Answers

### "Why did the agent select this smell?"
→ Click smell node, see PZ score calculation and dependencies in left drawer

### "What code changes did the agent make?"
→ Check code diff in right drawer for current iteration

### "Why did this refactoring fail?"
→ Check iteration details (outcome), tool logs (errors), and diff (what was attempted)

### "How long did smell detection take?"
→ Check timeline chart, find A1_detect_smells bar duration

### "Did resolving smell A create new smells?"
→ Check iteration details "Smells Created" count, see red dashed edges in graph

### "Where is the agent currently in its execution?"
→ Check iteration slider position, see highlighted smell in graph

## Testing

```bash
# 1. Start server
uv run python tools/visualize_smell_prioritization.py

# 2. Open browser
open http://localhost:8080

# 3. Load test database
# Enter: test_analytics.db
# Click: Load Database

# 4. Verify:
# - Session selector shows 2 sessions
# - Iteration slider goes from 0 to 2
# - Graph shows 3 smells
# - Timeline shows node executions
# - Code diff shows changes (iteration 0)
```

## Troubleshooting

### "Session selector is empty"
- Check database path is correct
- Verify database has `smell_events` table with data
- Check browser console for errors

### "Code diff not showing"
- Diffs are only captured for successful compilations
- Run workflow with `--enable-composite` to capture diffs
- Check `refactoring_attempts.code_diff` column exists

### "Timeline chart is empty"
- **Database mode**: No tool calls logged (tool call logging not yet implemented in agent)
- **Example mode**: Shows refactoring attempts timeline instead (green/red bars)
- Timeline adapts based on available data

### "Iteration controls (prev/next) don't work"
- ⚠️ Iteration controls only work with database sessions
- They are disabled (grayed out) for examples and sample data
- Use "Smell Priority" slider instead for examples

### "Graph shows no smells"
- Database might not have smell events for selected iteration
- Try different session or iteration

## Files Modified

- `tools/visualize_smell_prioritization.py` - Main visualization (enhanced)
- `tools/example_manifests/*.json` - Real composite refactoring examples (4 files)
- `swe_refactor/persistence/models.py` - Added `code_diff` field
- `agents/swe_eval/agent.py` - Capture git diffs in A6 node
- `VISUALIZATION_USAGE.md` - This file

## Recent Updates

### 2026-01-26: UI Improvements
- ✅ Reduced text sizes throughout interface (9-12px fonts)
- ✅ Added example manifest loading (4 real-world refactoring examples)
- ✅ Iteration controls now disabled for examples/sample data
- ✅ Timeline shows refactoring attempts when tool calls unavailable
- ✅ Manifest info panel displays project details and refactoring sequences

## Future Enhancements

- [ ] Real-time updates as agent runs
- [ ] Side-by-side code comparison view
- [ ] Export reports (PDF, HTML)
- [ ] Filter by smell type
- [ ] Search through diffs
- [ ] Agent position animation
- [ ] Performance metrics dashboard
