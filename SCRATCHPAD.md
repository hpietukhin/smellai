# Scratchpad: Planner Dataset Pipeline Validation

## Goal
Create evaluation datasets using the planner pipeline, validating every stage end-to-end.

## Pipeline Stages
1. `rminer_planner_to_df()` — oracle → per-commit DataFrame
2. `select_pilot_commits()` — diverse subset selection
3. Clone repos & resolve parent SHAs
4. SonarQube scan parent commits → S₀ smell sets
5. Merge into DataFrame, save as Parquet
6. Convert to MLflow GenAI records

## Current Status
- [ ] Stage 1: Validate converter output deeply (column types, content sanity)
- [ ] Stage 2: Validate pilot selection logic (diversity, capping)
- [ ] Stage 3: Validate git operations (clone, parent SHA resolution)
- [ ] Stage 4: Validate SonarQube scanning (need SonarQube running)
- [ ] Stage 5: Validate full Parquet output
- [ ] Stage 6: Validate MLflow bridge conversion

## Notes
- Stage 1: PASS. 494 commits, 0 nulls, 0 dupes. Move Class has empty first_refactoring_class — that's fine, LLM uses full description.

- Stage 2: PASS. 50 commits, 43 repos, max_per_repo respected.
- Stage 3: PASS. Clone + parent SHA resolution works for 3 test repos.

- Stage 4: PASS after 2 fixes:
  1. Upgraded SonarQube 10.6 → 26.3 (community) for scanner 8.x compat
  2. Pass token via `-Dsonar.token=` flag (scanner 8.x ignores properties file for initial auth)
- Stage 5: PASS. Parquet output has all 12 columns including parent_sha, smell_set_s0, smell_count_s0
- Stage 6: PASS. MLflow bridge converts correctly: inputs/expectations/tags all populated

- Small pilot dataset: DONE
  - 5 repos, 11 commits, 261s total scan (avg 23.7s/commit)
  - Saved to data/processed/planner_small
  - Registered in MLflow as "planner-small-5repos" (dataset ID: d-6b29572744ad4aea987451c34482b75f)
  - Baseline evaluation run: first_action_match=0.18, smell_coverage=0.37
  - MLflow run: http://localhost:5000/#/experiments/4/evaluation-runs

## Currently Working On
→ DONE. Pipeline validated end-to-end.

### Bottleneck Analysis (robovm example — worst case 10m09s)
| Phase | Time | % |
|-------|------|---|
| Checkout | 9s | 1% |
| Preprocess 14,309 files (6 langs) | 24s | 4% |
| **JavaSensor parse 14k files** | **290s** | **47%** |
| JS/TS/CSS/XML/Ruby sensors | 17s | 3% |
| Text/secrets sensor | 4s | 1% |
| SCM Publisher | 30s | 5% |
| Report upload | 5s | 1% |
| **Server-side processing** | **127s** | **21%** |
| Other | ~100s | 16% |

### Optimization options
1. **`sonar.exclusions`** — skip test dirs, resources, generated code
   - `**/src/test/**,**/test/**,**/node_modules/**,**/*.js,**/*.css,**/*.xml,**/*.rb`
   - Expected: ~50-70% fewer files to parse
2. **`sonar.sources=.` → `sonar.sources=src/main/java`** — only Java main sources
   - Problem: many repos have non-standard layouts
3. **`sonar.language=java`** — restrict to Java only (skips JS/CSS/XML sensors)
   - But: property removed in SQ 7.4+, use `sonar.inclusions=**/*.java` instead
4. **Sparse checkout** — only materialize `.java` files on disk
   - Saves checkout time but main bottleneck is scanner, not checkout
