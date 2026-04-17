# TODO: repository restructuring for coherence

## Basis for this note
- Repo scan covered all tracked `*.py` files via `rg --files -g '*.py' .` and `tree --gitignore`.
- Structural expectations are taken from:
  - `docs/conf_Pietukhin_10_3_rev2-2.pdf`
  - `docs/langgraph_best_practices.md`
- Paper model to preserve during restructuring:
  - pipeline stages A–J
  - six specialised agent roles
  - dependency-aware planning over code smells
  - clear separation between detection, planning, execution, and verification

---

## 1. Parts of the codebase, mapped to the thesis system

### A. Workflow / orchestration layer
Purpose in paper: orchestrate stages A–J and evaluation runs.

Current files:
- `workflows/eval_workflow.py`
- `workflows/rminer_eval_workflow.py`
- `workflows/swe_eval_workflow.py`
- `workflows/baseline_eval_workflow.py`
- `workflows/java_test_workflow.py`
- `workflows/composite_analysis_workflow.py`
- `workflows/smell_cooccurrence_workflow.py`
- `workflows/common.py`
- `workflows/utils.py`

Observation:
- `workflows/` mixes reusable library helpers (`common.py`, `utils.py`) with CLI entrypoints (`argparse` scripts).
- There is both a unified eval workflow and dedicated eval workflows, which overlaps conceptually.

### B. Agent layer
Purpose in paper: specialised agents for test/build verification, dependency analysis, refactoring, and baselines/ablations.

Current files:
- `agents/java_test/*`
- `agents/dependency_analysis/*`
- `agents/rminer_eval/*`
- `agents/swe_eval/*`
- `agents/baseline/*`
- `evals/ablation/mini_swe_agent/*`
- `agents/tools/java_test_tools.py`

Observation:
- Agent responsibilities are not consistently packaged.
- `agents/swe_eval/agent.py` is a large mixed-responsibility file (~801 lines) containing nested node functions for A0/A1/A2/A3/A4/A5/A6 plus conversion/helpers.
- The ablation agent mirrors parts of the main SWE agent instead of sharing a narrower reusable execution core.

### C. Dataset layer
Purpose in paper: load benchmark data and project it into evaluation-ready records.

Current files:
- `smellai_datasets/*`
- `swe_refactor/dataset.py`
- `rminer/create_rminer_dataset.py`
- `mlflow_utils/datasets/*`
- `scripts/datasets/*`

Observation:
- Dataset logic is spread across multiple top-level areas.
- `smellai_datasets` looks like the intended unified dataset layer, but `swe_refactor/dataset.py` and parts of `rminer/` still act as parallel dataset modules.
- `mlflow_utils/datasets` manages MLflow datasets, which is an infra concern, not raw dataset modeling.

### D. Domain models / analytics persistence
Purpose in paper: represent smells, refactorings, dependencies, attempts, and execution analytics.

Current files:
- `models/refactoring.py`
- `swe_refactor/persistence/models.py`
- `swe_refactor/persistence/database.py`

Observation:
- Domain and persistence concepts are split across `models/` and `swe_refactor/persistence/`.
- `SmellEvent` is doing double duty: both in-memory domain object and DB row model.

### E. Infrastructure / adapters
Purpose in paper: external systems and execution environment.

Current files:
- `sonarqube/*`
- `mlflow_utils/*`
- `repo_utils/*`
- `swe_refactor/utils/*`
- `logging_config.py`

Observation:
- This is the strongest separation-of-concerns problem in the repo.
- Git/repo/build/test utilities are split between `repo_utils` and `swe_refactor/utils`.
- Some infra helpers live in agents (`agents/tools/java_test_tools.py`) instead of in a shared infrastructure layer.

### F. Analysis / visualization / tooling
Purpose in paper: inspect plans, visualize dependencies, study analytics.

Current files:
- `scripts/prioritize_smells.py`
- `tools/visualize_smell_prioritization.py`
- `scripts/viz/*`
- `notebooks/*`

Observation:
- Reusable planning logic lives in a script file instead of a reusable package.
- There are multiple analytics UIs / reports over the same conceptual data (`NiceGUI`, `Streamlit`, `matplotlib`).

---

## 2. Main duplication and structural incoherence found

## 2.1 Repo/git/project utilities are duplicated
Duplicated concepts:
- `clone_repository`
- `get_previous_commit`
- checkout/project mutation helpers

Current locations:
- `repo_utils/operations.py`
- `repo_utils/__init__.py`
- `swe_refactor/utils/project_util.py`
- `swe_refactor/utils/repos.py`

Why this is a problem:
- same concern exists in two top-level modules
- inconsistent semantics (`gitpython`-based in one place, `subprocess`-based in another)
- SWE agent and general repo utilities evolve separately

Recommendation:
- pick one home for repository/project operations
- keep one implementation per concern
- expose a thin stable API used by agents/workflows

## 2.2 Java build/test execution is split across layers
Current locations:
- `agents/tools/java_test_tools.py`
- `repo_utils/test_execution.py`
- `agents/java_test/agent.py`
- `swe_refactor/utils/build_util.py`

Why this is a problem:
- execution primitives live partly under agents and partly under utilities
- compile/test/build-system detection are not clearly separated from agent orchestration
- persistence models in `swe_refactor/persistence/models.py` explicitly mirror test result dataclasses, which is a sign the boundary is weak

Recommendation:
- move build/test execution into one infrastructure package
- keep agents as orchestration only

## 2.3 Workflow entrypoints are duplicated
Current overlap:
- `workflows/eval_workflow.py` overlaps with:
  - `workflows/rminer_eval_workflow.py`
  - `workflows/swe_eval_workflow.py`
  - `workflows/baseline_eval_workflow.py`
- scorer factory `_get_rminer_scorers` exists in both:
  - `workflows/common.py`
  - `workflows/eval_workflow.py`

Why this is a problem:
- more than one “main” way to run evaluation
- predict-function scaffolding is repeated
- CLI concerns and library concerns are mixed together

Recommendation:
- keep either:
  1. one generic evaluation workflow with source-specific adapters, or
  2. dedicated workflows only
- not both, unless one is clearly marked experimental

## 2.4 Dependency planning logic is scattered
Current locations:
- dependency rules: `agents/dependency_analysis/agent.py`
- scoring: `agents/dependency_analysis/scorer.py`
- prioritizer/graph logic: `scripts/prioritize_smells.py`
- composite analyses: `workflows/composite_analysis_workflow.py`, `workflows/smell_cooccurrence_workflow.py`
- composite execution path: `agents/swe_eval/agent.py`

Why this is a problem:
- core paper logic (dependency-aware planning) is partly reusable package code and partly script-local code
- same domain appears in agent layer, workflow layer, and scripts layer
- makes the planner hard to reuse and hard to test as a clean unit

Recommendation:
- extract smell planning to a reusable package/module
- let scripts and workflows only call that package

## 2.5 Dataset layer is half-unified, half-legacy
Current signs:
- `smellai_datasets/__init__.py` presents a unified API
- but scripts still import removed/legacy modules:
  - `smellai_datasets.converter`
  - `smellai_datasets.config`
  - `smellai_datasets.preprocessor`
- affected files include:
  - `scripts/datasets/analyze.py`
  - `scripts/datasets/inspect_rminer.py`
  - `scripts/datasets/preprocess.py`
  - `scripts/viz/extract_compound_refactorings.py`

Why this is a problem:
- the migration to a unified dataset package is incomplete
- some tools are already on the new API, others are still wired to the old API

Recommendation:
- finish the migration decisively
- remove or rewrite all callers of removed modules

## 2.6 Package/import structure contains stale paths
Concrete examples:
- `sonarqube/__init__.py` imports `smellai.sonarqube.tool`
- `rminer/create_rminer_dataset.py` tries `smellai.sonarqube.commit_scan`
- `scripts/rminer/explore_rminer.py` imports `smellai.rminer.rminer_utils`
- `scripts/rminer/explore_rminer.py` imports `smellai.models.refactoring`
- `agents/swe_eval/agent.py` imports `swe_refactor.smell_detection.utils`, but only `swe_refactor/smell_detection/__init__.py` exists
- `rminer/__init__.py` advertises lazy imports for missing modules like `.mysql_connector` and `.git_ops`

Why this is a problem:
- import graph does not reflect the real folder structure
- some modules are legacy shells from earlier layouts
- these stale references will keep breaking refactors and packaging

Recommendation:
- make the import graph truthful before moving code around further

## 2.7 Packaging metadata is inconsistent with the actual repo
From `pyproject.toml`:
- wheel packages include `code_analysis`, but that directory does not exist
- `evals` exists but is not in the build target
- `swe_refactor` is used as an importable namespace but is not declared in the wheel target

Why this is a problem:
- installable package layout is different from source-tree layout
- import success depends too much on “running from repo root”

Recommendation:
- fix packaging as part of restructuring, not after

## 2.8 Visualization tooling is fragmented
Current locations:
- `tools/visualize_smell_prioritization.py` (NiceGUI)
- `scripts/viz/dashboard.py` (Streamlit)
- `scripts/viz/visualize_analytics.py` (matplotlib/seaborn)

Why this is a problem:
- multiple overlapping UI/report surfaces
- difficult to know which is canonical
- hard to keep analytics schema changes synchronized

Recommendation:
- choose one primary analytics UI path
- keep the others only if clearly marked as exports/experiments

## 2.9 Hard-coded local paths are embedded in the dataset layer
Examples:
- `smellai_datasets/loaders.py`
- `swe_refactor/dataset.py`
- `scripts/datasets/inspect_rminer.py`
- notebooks in `notebooks/`

Why this is a problem:
- current structure depends on one machine layout
- this makes future package cleanup harder because behavior is hidden in path fallbacks

Recommendation:
- centralize path resolution/configuration
- remove user-machine absolute defaults from library code

## 2.10 Some files are too large and mix too many concerns
Large files worth splitting first:
- `agents/swe_eval/agent.py` (~801 lines)
- `scripts/prioritize_smells.py` (~467 lines)
- `smellai_datasets/loaders.py` (~391 lines)
- `repo_utils/__init__.py` (~286 lines)
- `workflows/eval_workflow.py` (~260 lines)

Recommendation:
- split by responsibility, not by arbitrary size
- follow LangGraph best-practice layout where possible: `state.py`, `nodes.py`, `graph.py`, `prompts.py`, `config.py`

---

## 3. Recommended target structure

This is the simplest coherent target I would recommend.

```text
agents/
  java_test/
    graph.py
    nodes.py
    state.py
  dependency_analysis/
    rules.py
    planner.py
    scoring.py
  rminer_eval/
    graph.py
    nodes.py
    state.py
    prompts.py
    config.py
  swe_eval/
    graph.py
    nodes.py
    state.py
    prompts.py
    config.py

ablations/
  mini_swe/

datasets/
  common/
    schema.py
    mlflow_bridge.py
  swe/
    loader.py
    models.py
  rminer/
    loader.py
    diff.py
    models.py

infrastructure/
  mlflow/
  sonarqube/
  repository/
  java/
  logging/

analytics/
  persistence/
  visualization/

workflows/
  rminer_eval.py
  swe_eval.py
  baseline_eval.py
  composite_refactoring.py

cli/
  eval.py
  run_java_tests.py
  visualize_analytics.py
  prioritize_smells.py
```

Notes:
- `workflows/` should contain orchestration code, not CLI parsing.
- `cli/` or `scripts/` should contain executable wrappers only.
- `datasets/` should become the single home for dataset loading/projection.
- `infrastructure/` should become the single home for SonarQube, MLflow, Git, build/test execution.
- `analytics/` should own persistence + dashboards.

---

## 4. Transition / operations needed to get there

## Phase 0 — lock the target boundaries first
1. Write a short architecture decision note defining the target folders.
2. Decide one rule: library code vs CLI code must live in separate folders.
3. Decide one rule: one concern gets one home only.

Success criterion:
- every folder has a single sentence describing its responsibility.

## Phase 1 — finish the dataset migration first
1. Create one canonical dataset package (`datasets/` or rename `smellai_datasets/` into that).
2. Move/merge:
   - `smellai_datasets/schema.py`
   - `smellai_datasets/loaders.py`
   - `smellai_datasets/mlflow_bridge.py`
   - `swe_refactor/dataset.py`
   - RMiner dataset loading bits from `rminer/`
3. Rewrite all legacy imports that still reference removed modules.
4. Move MLflow dataset registration helpers out of the dataset modeling layer and into MLflow infrastructure.

Success criterion:
- all dataset-related imports come from exactly one top-level package.

## Phase 2 — unify infrastructure
1. Merge `repo_utils` and `swe_refactor/utils` into one repository/build infra package.
2. Move Java build/test primitives out of `agents/tools/java_test_tools.py` into that infra package.
3. Keep SonarQube access entirely under one adapter package.
4. Keep MLflow access entirely under one adapter package.

Success criterion:
- no agent imports low-level git/build helpers from more than one place.

## Phase 3 — split the SWE agent by responsibility
1. Break `agents/swe_eval/agent.py` into:
   - `state.py`
   - `nodes.py`
   - `graph.py`
   - `prompts.py`
   - `mappers.py` or `converters.py` for `EvalSample -> RefactoringRecord`
2. Move smell detection / prioritization / mapping helpers into reusable modules instead of nested closures.
3. Keep the graph assembly file thin.

Success criterion:
- each node is individually testable without reading the whole file.

## Phase 4 — centralize planning/domain logic
1. Move dependency rules out of the agent module into a domain/planner module.
2. Move `SmellPrioritizer` out of `scripts/prioritize_smells.py` into reusable package code.
3. Make workflows/scripts import the same planner implementation.
4. Keep the script as a thin CLI wrapper only.

Success criterion:
- there is exactly one implementation of smell dependency planning.

## Phase 5 — simplify workflow entrypoints
1. Choose one evaluation entrypoint strategy:
   - either keep `workflows/eval_workflow.py` as canonical and downgrade the dedicated workflow files to wrappers,
   - or keep dedicated workflows and remove the generic one.
2. Move `argparse` code out of reusable workflow modules into `cli/` or `scripts/`.
3. Keep `workflows/common.py` only for actual reusable orchestration helpers; remove duplicated scorer factories.

Success criterion:
- each workflow has one obvious way to run it.

## Phase 6 — clean packaging/import truth
1. Fix `pyproject.toml` package list.
2. Remove non-existent package entries such as `code_analysis`.
3. Ensure every importable package is intentionally included.
4. Remove all stale `smellai.*` import references unless you really want a top-level `smellai/` package.
5. Decide whether namespace packages are intentional; if not, add explicit `__init__.py` files where needed.

Success criterion:
- package install layout matches source-tree layout.

## Phase 7 — consolidate analytics and visualization
1. Choose the canonical analytics UI:
   - NiceGUI, or
   - Streamlit, or
   - static plots only.
2. Move all analytics UI/report code under one folder.
3. Keep extra tools only if clearly labeled as experimental.

Success criterion:
- analytics consumers know exactly where to go for the main UI.

## Phase 8 — quarantine or delete legacy artifacts
1. Audit these first:
   - `scripts/datasets/*`
   - `scripts/rminer/explore_rminer.py`
   - `scripts/viz/extract_compound_refactorings.py`
   - `rminer/__init__.py`
   - `sonarqube/__init__.py`
   - notebooks with old import paths
2. Either:
   - rewrite them to the new structure, or
   - move them under `legacy/` or `experiments/`, or
   - delete them if superseded

Success criterion:
- no active code path imports removed modules.

---

## 5. Suggested order of execution

Recommended order:
1. dataset unification
2. infra unification
3. split `agents/swe_eval`
4. centralize planner/domain logic
5. simplify workflows/CLI
6. fix packaging/imports
7. consolidate analytics tooling
8. delete/quarantine legacy code

Reason:
- datasets and infra are the foundation
- once those are stable, agent/workflow cleanup becomes much simpler
- packaging should be fixed after the import graph is cleaned, not before

---

## 6. My recommended answer to the high-level design question

If the goal is **“one folder for datasets, one folder for workflows, clear LangGraph separation of concerns”**, my recommendation is:

- **Yes**: make datasets a single top-level package.
- **Yes**: keep workflows as orchestration only.
- **Yes**: move CLI code out of workflows.
- **Yes**: collapse repo/build/test/SonarQube/MLflow into a single infrastructure layer.
- **Yes**: split large LangGraph agents into `state/nodes/graph/prompts/config`.
- **No**: do not keep parallel old/new dataset APIs alive.
- **No**: do not keep duplicated repo utility layers.
- **No**: do not leave core planner logic in `scripts/`.

That structure fits both the thesis architecture and LangGraph best practices much better than the current mixed layout.
