# MLflow / Evaluation Onboarding Questions

This file is a structured interview log to reach a shared understanding of the cleanup plan around MLflow usage, evaluation workflows, onboarding, and stale code.

We will resolve decisions one by one, starting from root decisions that affect the rest of the design tree.

---

## Question 1: What is the single canonical evaluation path we want a new contributor to succeed with first?

### Why this is the root decision
The repository currently exposes at least two evaluation stories:

- **RMiner evaluation**
  - `workflows/rminer_eval_workflow.py`
  - requires a manifest
  - current README path appears stale

- **SWE evaluation**
  - `workflows/swe_eval_workflow.py`
  - appears more self-contained
  - repo already contains `swe_refactor/SWE-Refactor.zip`

If we do not choose one primary onboarding path, we cannot cleanly decide:

- what the README should show first
- which dataset path to standardize
- which helper scripts are supported vs legacy
- what “a junior can run evals immediately” actually means

### Recommended answer
**Make SWE evaluation the first canonical onboarding path, and treat RMiner as the second path.**

### Why this is the recommendation
Based on the current codebase:

- `smellai_datasets/loaders.py` already supports SWE input from `.zip`, `.json`, or a directory
- `workflows/swe_eval_workflow.py` is a current workflow with explicit CLI args
- the repo already contains `swe_refactor/SWE-Refactor.zip`

RMiner currently has more ambiguity:

- README references `rminer_data/manifest.json`, but that directory is not present in the repo root
- dataset provenance for the RMiner manifest is unclear
- some RMiner-related scripts/docs are stale or mismatched

### Candidate answers
- [ ] **Option 1:** SWE first, RMiner second **(recommended)**
- [ ] **Option 2:** RMiner first, SWE second
- [ ] **Option 3:** Both are equal first-class onboarding paths
- [ ] **Option 4:** Neither; define a new minimal demo eval path first
- [ ] **Option 5:** Other: _______________________________

### Your answer

<!-- Fill in here -->

---

## Next questions

To be filled one at a time after Question 1 is resolved.
