# Approximate code duplication analysis

This note looks for **roughly duplicated Python functions**: not byte-for-byte copies, but functions whose source is highly similar.

## Method

The scan was run with `uv run python` and a TF-IDF + cosine-similarity script similar to the one requested, but with one important change:

- it uses `git ls-files --cached --others --exclude-standard` to respect `.gitignore`
- it scans only **existing** `.py` files from that list
- it extracts function and method source with `ast`
- it computes similarity on function source text

Why the `existing` filter matters: after deleting `swe_refactor/`, Git still knows about some tracked paths that no longer exist in the working tree. Those missing paths were skipped.

### Scan script used

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast, subprocess
from pathlib import Path

class FuncCollector(ast.NodeVisitor):
    def __init__(self, source: str):
        self.source = source
        self.stack = []
        self.items = []

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self._handle(node)

    def visit_AsyncFunctionDef(self, node):
        self._handle(node)

    def _handle(self, node):
        self.stack.append(node.name)
        segment = ast.get_source_segment(self.source, node)
        if segment:
            self.items.append((".".join(self.stack), segment))
        self.generic_visit(node)
        self.stack.pop()

raw = subprocess.check_output([
    "git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"
])
paths = [Path(p) for p in raw.decode().split("\0") if p.endswith(".py")]

funcs = []
for p in paths:
    if not p.exists():
        continue
    src = p.read_text(encoding="utf-8")
    tree = ast.parse(src)
    c = FuncCollector(src)
    c.visit(tree)
    for name, seg in c.items:
        funcs.append((str(p), name, seg))

names = [f"{p}::{n}" for p, n, _ in funcs]
sources = [s for _, _, s in funcs]
vec = TfidfVectorizer().fit_transform(sources)
sim = cosine_similarity(vec)
```

## Scope and headline numbers

- Unignored existing Python files scanned: **94**
- Functions/methods extracted: **504**
- Cross-file pairs at similarity **>= 0.75**: **13**
- Pairs at similarity **>= 0.80** including same-file matches: **19**

## Caveats

This is an **approximate** scan, so a few matches are expected to be noisy:

- tiny wrapper functions can look similar even when intentional
- generated Marimo notebook cells show up as functions named `_`
- nested functions and factory-style closures often look duplicated because they repeat boilerplate

So the useful signal is not every pair, but the **clusters**.

---

## Main duplication hotspots

### 1. Workflow wrapper duplication in `workflows/`

This is the strongest real duplication in production code.

Top matches:

- `workflows/baseline_eval_workflow.py::main.predict_fn`
  ↔ `workflows/eval_workflow.py::_make_rminer_predict_fn.predict_fn` (**0.92**)
- `workflows/common.py::_get_rminer_scorers`
  ↔ `workflows/eval_workflow.py::_get_rminer_scorers` (**0.90**)
- `workflows/common.py::make_rminer_eval_sample`
  ↔ `workflows/eval_workflow.py::_make_rminer_predict_fn.predict_fn` (**0.86**)
- `workflows/common.py::make_swe_eval_sample`
  ↔ `workflows/eval_workflow.py::_make_mini_swe_predict_fn.predict_fn` (**0.84**)
- `workflows/common.py::make_swe_eval_sample`
  ↔ `workflows/eval_workflow.py::_make_swe_predict_fn.predict_fn` (**0.84**)
- `workflows/composite_analysis_workflow.py::visualize_dependencies`
  ↔ `workflows/smell_cooccurrence_workflow.py::visualize_file_dependencies` (**0.80**)

### Why this matters

The workflow layer repeats the same patterns:

1. reconstruct an `EvalSample`
2. call the appropriate agent
3. wire the same scorer set
4. expose a slightly different CLI shell around the same core operation

This is not accidental duplication — it is repeated orchestration logic.

### What looks duplicated semantically

#### A. RMiner predict function construction

There are three near-overlapping places:

- `workflows/common.py::make_rminer_eval_sample`
- `smellai_datasets/schema.py::rminer_sample`
- `workflows/eval_workflow.py::_make_rminer_predict_fn.predict_fn`
- `workflows/baseline_eval_workflow.py::main.predict_fn`

This likely means the code does not have one clear owner for:

- RMiner sample construction
- RMiner predict function wiring

#### B. SWE predict function construction

Similar duplication exists for SWE:

- `workflows/common.py::make_swe_eval_sample`
- `workflows/eval_workflow.py::_make_swe_predict_fn.predict_fn`
- `workflows/eval_workflow.py::_make_mini_swe_predict_fn.predict_fn`

### Recommendation

Create one reusable workflow helper layer, for example:

- one canonical function for `EvalSample` construction per source
- one canonical helper for creating `predict_fn` wrappers
- one canonical scorer registry

Concretely, duplication would drop if:

- `workflows/eval_workflow.py` called shared factories only
- `workflows/baseline_eval_workflow.py` reused the same RMiner wrapper
- `_get_rminer_scorers()` existed in exactly one place

---

### 2. SWE setup duplication between main and ablation agents

Top match:

- `agents/swe_eval/agent.py::create_swe_eval_agent.a0_setup`
  ↔ `evals/ablation/mini_swe_agent/agent.py::_a0_setup` (**0.84**)

### Why this matters

The repo has two agents that both do the same setup pipeline:

1. resolve repo URL
2. clone if needed
3. find parent commit
4. checkout parent commit
5. switch JDK

That is real operational duplication, not just similar shape.

### Recommendation

Extract a shared helper such as:

- `repo_utils.prepare_swe_workspace(...)`
- or `agents/swe_eval/setup.py` reused by both agents

The helper should return structured setup data rather than duplicating the error-handling path in both places.

---

### 3. Graph-visualization duplication

Top match:

- `workflows/composite_analysis_workflow.py::visualize_dependencies`
  ↔ `workflows/smell_cooccurrence_workflow.py::visualize_file_dependencies` (**0.80**)

### Why this matters

Both functions:

- create a `networkx.DiGraph`
- add nodes with colors/types
- add dependency edges
- run a spring layout
- draw nodes, labels, and edges
- save the graph

They differ in input shape and semantics, but the drawing pipeline is clearly repeated.

### Recommendation

Extract a shared plotting helper, e.g.:

- `tools/graph_viz.py`
- `workflows/graph_rendering.py`

Keep the graph-building logic separate from the rendering logic.

---

### 4. Repeated SQLite loader boilerplate in `scripts/viz/dashboard.py`

Same-file high-similarity matches:

- `load_smell_events` ↔ `load_token_usage` (**0.83**)
- `load_smell_events` ↔ `load_refactoring_attempts` (**0.83**)
- `load_refactoring_attempts` ↔ `load_token_usage` (**0.80**)

### Why this matters

These are three variations of the same pattern:

1. open SQLite connection
2. define query
3. run `pd.read_sql_query(...)`
4. close connection
5. return DataFrame

### Recommendation

Introduce one private helper such as:

```python
def _read_session_query(db_path: str, query: str, params: tuple = ()) -> pd.DataFrame:
    ...
```

Then each loader only supplies the SQL text and params.

This is a small cleanup, but it is a clean, low-risk deduplication target.

---

### 5. Prompt-template duplication in `agents/swe_eval/prompts.py`

Same-file high-similarity match:

- `_move_method_prompt` ↔ `_extract_and_move_prompt` (**0.90**)

### Why this matters

These prompts share nearly the same structure:

- same target-file section
- same two-file output format
- same source/target path interpolation
- mostly different instruction wording

### Recommendation

Factor a shared prompt builder for “two-file move-style refactorings”, with only the instruction headline/body varying.

This is not urgent, but it is real duplication.

---

### 6. SonarQube API pagination duplication

Same-file high-similarity match:

- `sonarqube/commit_scan.py::fetch_issues_for_file`
  ↔ `sonarqube/commit_scan.py::fetch_all_project_issues` (**0.85**)

### Why this matters

Both functions duplicate:

- `requests.Session()` setup
- pagination loop
- `/api/issues/search` calls
- `rule_list` construction
- batch accumulation logic

The main difference is only query params:

- file-specific query
- project-wide query

### Recommendation

Create one internal helper like:

```python
def _fetch_issues_paginated(sonar_url, sonar_token, **params) -> list[dict]:
    ...
```

Then build the specific variants on top.

---

## Noisy / lower-value matches

These showed up in the scan but are less important structurally:

### Notebook cells

Top notebook matches:

- `notebooks/swe_refactor_dataset_explainer.py::_`
  ↔ `notebooks/swe_to_evalsample.py::_` (**0.83**)
- `notebooks/data_preparation.py::_`
  ↔ `notebooks/swe_to_evalsample.py::_` (**0.82**)

These come from generated Marimo cell functions named `_`.
They do indicate repeated notebook content, but this is lower priority than production-code duplication.

### Tiny wrappers / nested functions

Examples:

- nested functions inside `repo_utils/__init__.py`
- tiny scorer/wrapper closures

These are often artifacts of the similarity method rather than meaningful duplication.

---

## Most important consolidation targets

If the goal is to reduce duplicated logic with the biggest payoff, the order should be:

1. **Workflow wrappers in `workflows/`**
   - biggest real cluster
   - repeated sample reconstruction + scorer wiring
2. **Shared SWE setup for agents**
   - `agents/swe_eval` vs `evals/ablation/mini_swe_agent`
3. **Shared graph rendering helper**
   - two workflow visualizers repeat the same NetworkX/Matplotlib pipeline
4. **Shared SQL query helper in dashboard**
   - easy win, low risk
5. **Shared prompt builder for move-style prompts**
   - nice cleanup, lower priority
6. **Shared Sonar pagination helper**
   - small internal refactor, good hygiene

---

## Suggested concrete moves

### A. Simplify RMiner workflow construction

Potential target:

- keep `smellai_datasets/schema.py::rminer_sample` as the canonical RMiner sample builder
- remove wrapper duplication from `workflows/common.py`
- make both evaluation workflows call the same helper to build `predict_fn`

### B. Consolidate SWE setup

Potential target:

- create one setup helper in `repo_utils/` or a small shared module under `agents/swe_eval/`
- both `agents/swe_eval/agent.py` and `evals/ablation/mini_swe_agent/agent.py` call it

### C. Extract graph renderer

Potential target:

- one helper that accepts `graph`, `title`, `output_path`, and visual options
- each workflow builds its own graph, then delegates rendering

### D. Extract dashboard query helper

Potential target:

- one helper around `sqlite3.connect` + `pd.read_sql_query`

---

## Bottom line

The repo does have approximate duplication, but it is not evenly spread.

The scan shows three meaningful concentrations:

1. **workflow wrapper duplication**
2. **duplicated SWE setup path across main + ablation agents**
3. **repeated visualization / query boilerplate**

That means the highest-value cleanup is **not** a broad “deduplicate everything” pass.
It is a targeted pass over:

- `workflows/`
- `agents/swe_eval` + `evals/ablation/mini_swe_agent`
- `scripts/viz/dashboard.py`
- `sonarqube/commit_scan.py`

These are the places where similar code is repeated enough to justify consolidation.
