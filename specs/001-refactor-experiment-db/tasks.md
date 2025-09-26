# Tasks: Refactor Experiment DB Pipeline into Importable, Extensible Module

**Input**: Design documents from `/specs/001-refactor-experiment-db/`
**Prerequisites**: plan.md (required)

## Phase 3.1: Setup
- [X] T001 Create directories `src/connectors/` and `src/pipelines/`
- [X] T002 [P] Add `src/connectors/mysql_connector.py` skeleton with `schema()` and `fetch()` signatures
- [X] T003 [P] Add `src/pipelines/experiment_pipeline.py` with `load_dataset(config)` and `run_experiment(...)` stubs
- [X] T004 [P] Ensure W&B installed and login instructions in README (WANDB_PROJECT default `mt`)

## Phase 3.2: Tests First (TDD)
- [ ] T005 [P] Unit test for MySQL connector `schema()` returns expected column names (tests/unit/test_mysql_schema.py)
- [ ] T006 [P] Unit test pipeline returns Pandas DataFrames (classes/refactorings) (tests/unit/test_pipeline_outputs.py)
- [ ] T007 [P] Integration test: notebook-like flow that inits W&B run and logs config/metrics using `src/tracking` (tests/integration/test_tracking_flow.py)

## Phase 3.3: Core Implementation
- [X] T008 Implement MySQL connector `schema()` and mock `fetch()` to read pre-edited data files (no DB access)
- [X] T009 Implement `load_dataset(config)` to assemble DataFrames from pre-edited sources; attach dataset metadata
- [X] T010 Implement `run_experiment(...)` to:
      - init run via `src/tracking.init_run(project='mt')`
      - log config (git SHA, dataset id/version, connector)
      - log simple metrics and finish run
- [ ] T011 Add W&B artifact logging for dataset snapshot `datasets/{dataset_name}:{version}` (metadata contains connector)

## Phase 3.4: Integration
- [ ] T012 Example notebook cell: import pipeline from `src/`, run load + experiment, log to W&B
- [ ] T013 Wire LiteLLM defaults from notebook into run config (temperature/top_p)

## Phase 3.5: Polish
- [ ] T014 [P] Update README with example import usage and W&B artifacts note
- [ ] T015 [P] Add `.env.example` under `docs/` with WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY

## Dependencies
- Setup (T001-T004) before Tests and Core
- Tests (T005-T007) before Core (T008-T011)
- Core before Integration (T012-T013)
- All before Polish (T014-T015)

## Parallel Example
```
# Run setup in parallel where possible:
Task: "T002 Add mysql_connector.py skeleton"
Task: "T003 Add experiment_pipeline.py stubs"
Task: "T004 Ensure W&B installed and README login"

# Run tests in parallel:
Task: "T005 Unit test schema()"
Task: "T006 Unit test pipeline outputs"
Task: "T007 Integration test tracking flow"
```

## Validation Checklist
- [ ] W&B run shows git SHA, dataset id/version, connector
- [ ] DataFrames returned per spec; stable column names documented
- [ ] Dataset artifact logged with naming `datasets/{dataset_name}:{version}` and connector in metadata
- [ ] No SonarQube fetching in this feature (pre-edited data only)
