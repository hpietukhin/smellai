# TODO

## SWE-Refactor readiness for PDF-spec experiments

### High priority
- [ ] Add repository URL mappings for all 18 SWE-Refactor projects in `swe_refactor/utils/repos.py`
- [ ] Revisit SWE test metadata handling in `smellai_datasets/loaders.py`
  - [ ] Stop relying only on `hasTestC`
  - [ ] Infer `has_tests` from `testResult` / `coverageInfo` where appropriate
- [ ] Document clearly that `pure_refactoring_data.json` is the canonical SWE benchmark artifact used by this repo
- [ ] Add an explicit note about dataset-card vs file-level mismatches
  - [ ] `hasTestC` is almost always missing in the concrete file
  - [ ] `compileResultCurrent` has 2 false rows
  - [ ] `testResult` is missing for 1 row

### Needed for PDF stages F / E / H
- [ ] Add mandatory SonarQube enrichment/preprocessing for SWE samples
- [ ] Reconstruct pre-refactoring smell set `S0` from the target commit
- [ ] Map SonarQube findings to the 8 smell types from Table I in `docs/conf_Pietukhin_10_3_rev2-2.pdf`
- [ ] Add severity labels needed by the planner
- [ ] Build dependency graph edges from the PDF/thesis dependency rules
- [ ] Store planner-ready smell instances separately from raw SWE records

### Needed for PDF stages A / D / J
- [ ] Verify checkout/build/test flow works for all SWE projects
- [ ] Validate `compileCommand` execution per project/JDK combination
- [ ] Validate baseline test execution on parent commit, not only dataset metadata
- [ ] Define fallback behavior for rows where runtime verification disagrees with dataset metadata

### Needed for planning experiments
- [ ] Group SWE records by `commitId`
- [ ] Define how to derive the developer’s “first committed operation” from commits with multiple refactoring records
- [ ] Build evaluation protocol for greedy vs BeFS first-step comparison
- [ ] Separate single-refactoring evaluation from dependency-aware planning evaluation in documentation and code

### Nice to have
- [ ] Add a notebook/report that explicitly maps SWE fields to PDF stages A–J
- [ ] Add dataset sanity checks as tests
- [ ] Add per-project readiness report: repo URL, build works, tests work, Sonar works
