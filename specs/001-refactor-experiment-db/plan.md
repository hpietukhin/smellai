# Implementation Plan: Refactor Experiment DB Pipeline into Importable, Extensible Module

**Branch**: `001-refactor-experiment-db` | **Date**: 2025-09-26 | **Spec**: /Users/havriil.pietukhin/PycharmProjects/smellai/specs/001-refactor-experiment-db/spec.md
**Input**: Feature specification from `/specs/001-refactor-experiment-db/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
2. Fill Technical Context
3. Fill the Constitution Check section
4. Evaluate Constitution Check
5. Execute Phase 0 → research.md (if unknowns remain)
6. Execute Phase 1 → quickstart.md (tracking), contracts N/A
7. Re-evaluate Constitution Check
8. Plan Phase 2 → Task generation approach
9. STOP - Ready for /tasks command
```

## Summary
Refactor `may25/experiment1.ipynb` to import a DB pipeline from `src/` and set up
Weights & Biases (W&B) tracking for reproducibility. Skip SonarQube fetching; use
pre-edited data as in the notebook. Default W&B project is `mt` per spec.

## Technical Context
**Language/Version**: Python 3.11 (target)  
**Primary Dependencies**: LangGraph (pipelines), W&B (tracking), LiteLLM (LLM provider)  
**Storage**: MySQL (via connector)  
**Testing**: pytest  
**Target Platform**: local macOS, Linux  
**Project Type**: single  
**Performance Goals**: N/A  
**Constraints**: No secrets in VCS; determinism of notebooks; artifacts/logging via W&B  
**Scale/Scope**: Small datasets; streaming optional

## Constitution Check
- Reproducible Experimentation: 
  - Importable pipelines from `src/`; W&B run logs include git SHA, dataset ID/version, seeds.
- Baseline Parity and Comparisons: 
  - Baseline runs (SonarQube) out of scope for this plan; no violation.
- Correctness and Safety: 
  - No code changes to target projects; ensure notebook runs, lints pass.
- Transparent LLM Configuration and Prompting: 
  - Log model name via LiteLLM, temperature/top_p from notebook defaults.
- Project Structure and Data Governance: 
  - Notebooks under `experiments/`; connectors in `src/`; W&B artifacts for datasets.

Gate: PASS with note (baseline parity deferred to feature 002).

## Project Structure
```
src/
├── tracking/
│   ├── __init__.py
│   └── wandb_tracking.py
├── connectors/
│   └── mysql_connector.py              # [planned]
└── pipelines/
    └── experiment_pipeline.py         # [planned]

experiments/
└── 001_demo.ipynb                     # example import usage
```

**Structure Decision**: Single project structure; import from `src/`.

## Phase 0: Outline & Research
- Confirm W&B best practices: login, init, config logging, artifacts (Context7 docs used).

## Phase 1: Design & Contracts
- Define `src/pipelines/experiment_pipeline.py` with functions:
  - `load_dataset(config) -> pd.DataFrame, pd.DataFrame`
  - `run_experiment(df_classes, df_refactorings, tracking)`
- Define `src/connectors/mysql_connector.py` (skeleton) implementing `schema()` and `fetch()`.
- Tracking: use `src/tracking` helpers to init, log, and finish runs.
- quickstart.md: minimal example for notebooks.

## Phase 2: Task Planning Approach
- Setup tasks: create `src/connectors`, `src/pipelines`, example notebook cell.
- Tests: minimal unit test for `schema()` returns expected columns.
- Implementation tasks: pipeline wiring + W&B logging.
- Polish: README updates.

## Estimated Output
- 12–15 tasks across setup, implementation, and docs.

## Complexity Tracking
N/A.

## Progress Tracking
- [ ] Phase 0: Research complete (/plan command)
- [ ] Phase 1: Design complete (/plan command)
- [ ] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

## Gate Status
- [ ] Initial Constitution Check: PASS
- [ ] Post-Design Constitution Check: PASS
- [ ] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented

*Based on Constitution v1.1.0 - See `.specify/memory/constitution.md`*
