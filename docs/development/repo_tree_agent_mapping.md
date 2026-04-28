# Repository tree → paper-stage mapping

Based on `tree -L 2 --gitignore` and the stage definitions in `docs/conf_Pietukhin_10_3_rev2-2.pdf`.

## Stage legend from the paper

- **A–D** — source loading, build-system detection, test coverage check, test generation, baseline test execution
- **E** — smell dependency graph construction
- **F** — SonarQube smell scan and normalization
- **G** — developer smell selection
- **H** — planning / Best-First Search ordering
- **I** — LLM refactoring execution
- **J** — post-refactoring verification, rollback, before/after comparison
- **meta** — infrastructure, docs, experiments, analysis, or project support; not a direct paper-stage agent

## Tree annotation table

| Path | Paper stage | Role | Kind | Duplication risk | Why / notes |
|---|---|---|---|---|---|
| `.` | meta | repository root | mixed | N/A | Contains both implementation and research artifacts. |
| `1` | meta | orphan placeholder | data | low | Empty/stray root artifact, not obviously duplicated. |
| `CLAUDE.md` | meta | repo operating spec | data-for-code | low | Canonical repo instructions. |
| `README.md` | meta | project overview and usage | data-for-code | low | Main entry documentation. |
| `SCRATCHPAD.md` | meta | working notes | data-for-code | medium | Overlaps with other notes/todo files. |
| `agents` | A–D, E, H, I, J | core agent implementations | code | low | Main orchestration logic. |
| `agents/__init__.py` | meta | package marker | code | low | Standard package file. |
| `agents/baseline` | H, I | baseline comparator agent | code | medium | Eval role overlaps other evaluation agents. |
| `agents/dependency_analysis` | E, H | dependency graph and prioritization | code | low | Clear owner for dependency logic. |
| `agents/java_test` | A–D, J | build/test detection and verification | code | low | Direct match to prep/verify stages. |
| `agents/rminer_eval` | I | RMiner-oriented refactoring/eval agent | code | medium | Similar purpose to `agents/swe_eval`. |
| `agents/swe_eval` | I, J | SWE-Refactor execution and verification | code | medium | Similar execution flow to `agents/rminer_eval`. |
| `agents/tools` | A–D, J | shared agent helpers | code | medium | Utility role overlaps infra packages. |
| `bfs_vs_greedy.md` | H | planner comparison notes | data-for-code | low | Directly tied to planning strategy analysis. |
| `compound_demo.json` | E, H, I | composite refactoring demo input | data | low | Example/demo artifact. |
| `compound_sample_j17.json` | E, H, I | composite sample input | data | low | Example/demo artifact. |
| `docs` | meta | project documentation | data-for-code | medium | Some overlap with root notes/slides. |
| `docs/conf_Pietukhin_10_3_rev2-2.pdf` | meta | thesis paper / main spec | data-for-code | low | Primary reference spec. |
| `docs/data_models.html` | meta | rendered data model docs | data-for-code | medium | Mirrors model/schema code. |
| `docs/development` | meta | development docs | data-for-code | low | Dedicated docs area. |
| `docs/langgraph_best_practices.md` | meta | framework guidance | data-for-code | low | Supporting engineering guidance. |
| `docs/mlflow_usage_presentation.html` | meta | MLflow usage presentation | data-for-code | medium | Topic overlaps `mlflow_utils` and `mlflow_questions.md`. |
| `docs/mt_dec` | meta | thesis/decision docs | data-for-code | low | Research decision support. |
| `docs/plans` | meta | planning docs | data-for-code | medium | Overlaps todo/planning files elsewhere. |
| `docs/repo_python_analysis_presentation.html` | meta | repo structure analysis presentation | data-for-code | medium | Overlaps README and restructure notes. |
| `docs/todo.md` | meta | documentation todo list | data-for-code | medium | Same category as root todo files. |
| `etxt.txt` | meta | scratch text notes | data | low | Stray note file, not duplicated structurally. |
| `evals` | meta, I | evaluation and ablation code | code | medium | Overlaps with `workflows` and `agents`. |
| `evals/__init__.py` | meta | package marker | code | low | Standard package file. |
| `evals/ablation` | I | alternative evaluation experiments | code | high | Duplicates main-agent evaluation behavior. |
| `experiments_jrnl` | meta | experiment journal | data-for-code | low | Research log area. |
| `experiments_jrnl/2026-03-18.md` | meta | experiment note | data-for-code | low | Journal artifact. |
| `experiments_jrnl/dataset_unification_prompt.md` | meta | dataset refactor note | data-for-code | low | Design/history note. |
| `experiments_jrnl/design-by-contract-tips.md` | meta | design tips note | data-for-code | low | Research/engineering note. |
| `logging_config.py` | meta | logging infrastructure | code | low | Central infra helper. |
| `mlflow.db.bak` | meta | MLflow backup database | data | low | Backup artifact. |
| `mlflow_questions.md` | meta | MLflow notes/questions | data-for-code | medium | Topic overlap with docs/presentation and `mlflow_utils`. |
| `mlflow_utils` | meta | MLflow integration and tracking infra | code | low | Canonical MLflow adapter layer. |
| `mlflow_utils/__init__.py` | meta | package marker | code | low | Standard package file. |
| `mlflow_utils/auto_server.py` | meta | MLflow server helper | code | medium | Similar concern to `server.py`. |
| `mlflow_utils/cli.py` | meta | MLflow CLI entrypoint | code | low | Clear entrypoint role. |
| `mlflow_utils/datasets` | meta | MLflow dataset management | code | medium | Dataset handling overlaps `smellai_datasets` and `scripts/datasets`. |
| `mlflow_utils/runner.py` | meta | MLflow run orchestration helper | code | low | Distinct infra role. |
| `mlflow_utils/server.py` | meta | MLflow server control | code | medium | Similar concern to `auto_server.py`. |
| `models` | meta, I | shared domain models | code | medium | Some conceptual overlap with persistence models. |
| `models/__init__.py` | meta | package marker | code | low | Standard package file. |
| `models/refactoring.py` | I | refactoring domain model | code | medium | Overlaps with dataset/persistence representations. |
| `notebooks` | meta | analysis and data-prep notebooks | code+data | low | Exploratory area, distinct from runtime code. |
| `notebooks/__marimo__` | meta | notebook runtime state | data | low | Tool-generated session state. |
| `notebooks/data_preparation.py` | meta | data prep notebook/script | code | medium | Similar work exists in `scripts/datasets`. |
| `notebooks/swe_refactor_dataset_explainer.py` | meta | SWE dataset analysis notebook | code | medium | Overlaps dataset docs/scripts. |
| `notebooks/swe_to_evalsample.py` | meta | SWE → EvalSample conversion notebook | code | high | Conversion logic likely overlaps dataset loaders. |
| `package.json` | meta | JS tooling/package config | data-for-code | low | Tooling config, not domain logic. |
| `prioritization.json` | E, H | prioritization output/example | data | low | Output artifact of dependency/planning logic. |
| `prompts` | A4, I | prompt-prep placeholder area | code/data-for-code | high | Prompt logic already lives in agent modules. |
| `pyproject.toml` | meta | Python project config | data-for-code | low | Canonical package config. |
| `repo_utils` | A–D, J | repo/build/test/git utilities | code | high | Strong overlap with `swe_refactor/utils`. |
| `repo_utils/__init__.py` | A–D, J | public repo utility API | code | high | Same domain as `swe_refactor/utils`. |
| `repo_utils/errors.py` | meta | repo utility error types | code | low | Support code, not duplicated. |
| `repo_utils/operations.py` | A, F, J | clone/checkout/repo operations | code | high | Similar concern to `swe_refactor/utils/repos.py`. |
| `repo_utils/test_execution.py` | D, J | test execution primitives | code | high | Similar concern to Java-test helpers and `swe_refactor/utils`. |
| `rminer` | meta, I | RefactoringMiner dataset/tooling layer | code | medium | Partly overlaps unified dataset package. |
| `rminer/__init__.py` | meta | package marker | code | low | Standard package file. |
| `rminer/create_rminer_dataset.py` | meta | RMiner dataset builder | code | high | Similar dataset-building role exists in loaders/bridges. |
| `rminer/diff_hunk.py` | I | diff hunk model/helper | code | low | Specialized type/helper. |
| `rminer/rminer_utils.py` | meta, I | RMiner parsing helpers | code | medium | Some loader overlap possible. |
| `scripts` | meta, E, H, I, J | operational scripts and demos | code | medium | Some scripts mirror workflow/tool functionality. |
| `scripts/datasets` | meta | dataset CLI scripts | code | high | Overlaps `smellai_datasets` and `mlflow_utils/datasets`. |
| `scripts/prioritize_smells.py` | E, H | prioritization CLI | code | medium | Logic overlaps `agents/dependency_analysis`. |
| `scripts/rminer` | meta | RMiner helper scripts | code | medium | Overlaps `rminer` package. |
| `scripts/run_composite_demo.sh` | A–J | end-to-end composite demo runner | code | low | Demo entrypoint, distinct role. |
| `scripts/run_demo_eval.py` | I, J | demo evaluation runner | code | medium | Similar concern to workflow entrypoints. |
| `scripts/run_visualizer.sh` | meta | visualizer launcher | code | medium | Overlaps Python visualizer launchers. |
| `scripts/viz` | meta | visualization scripts | code | medium | Overlaps `tools/visualize_smell_prioritization.py`. |
| `skills-lock.json` | meta | tooling/skills lockfile | data-for-code | low | Tooling metadata. |
| `slides.md` | meta | presentation source | data-for-code | medium | Overlaps slide planning files. |
| `slides_planning.md` | meta | presentation planning notes | data-for-code | high | Same content family as `slides.md` and PDF export. |
| `slides_planning.pdf` | meta | rendered presentation/planning PDF | data | high | Export of planning/source material. |
| `smell_dependencies.html` | E, H | rendered dependency docs/visualization | data-for-code | medium | Mirrors dependency logic and docs. |
| `smell_sequence_examples.txt` | H | refactoring sequence examples | data-for-code | low | Planning examples, no clear duplicate. |
| `smell_states.html` | E, H, J | rendered smell-state docs | data-for-code | medium | Similar information exists in models/code/docs. |
| `smellai_datasets` | meta, I | canonical dataset schema/loaders/bridges | code | high | Overlaps `swe_refactor/dataset.py` and parts of `rminer`. |
| `smellai_datasets/__init__.py` | meta | dataset public API | code | low | Canonical package surface. |
| `smellai_datasets/enrich_sonar.py` | F | enrich datasets with Sonar data | code | low | Clear smell-scan integration role. |
| `smellai_datasets/loaders.py` | meta, I | dataset loading/conversion hub | code | high | Central loader that overlaps notebooks and old dataset modules. |
| `smellai_datasets/mlflow_bridge.py` | meta | dataset → MLflow bridge | code | medium | Some concern overlap with `mlflow_utils/datasets`. |
| `smellai_datasets/models.py` | meta | shared dataset models | code | medium | Overlaps `schema.py` and domain models. |
| `smellai_datasets/schema.py` | meta, I | core `EvalSample` schema | code | low | Looks like the canonical shared schema. |
| `sonarqube` | F | SonarQube smell-detection adapter | code+config | low | Clear ownership of scan logic. |
| `sonarqube/__init__.py` | meta | package marker | code | low | Standard package file. |
| `sonarqube/commit_scan.py` | F | low-level SonarQube scanning pipeline | code | low | Infrastructure adapter used by `store/detector.py`. |
| `sonarqube/constants.py` | F | rule/severity mappings | code | low | Dedicated mapping module. |
| `sonarqube/docker-compose.yml` | F | SonarQube service config | data-for-code | low | Infra config for scanning. |
| `sonarqube/sonarqube_server.sh` | F | SonarQube server helper | code | medium | Some overlap with docker-compose-based startup. |
| `sonarqube/tool.py` | F | SonarQube tool wrapper | code | low | Support wrapper around scan system. |
| `store` | E, F, H | smell graph/store + detector abstractions | code | medium | Central architectural boundary for graph persistence and detector backends. |
| `store/detector.py` | F | canonical smell-detection abstraction + default SonarQube backend | code | low | High-level code should depend on this, not on `sonarqube.*` directly. |
| `swe_refactor` | meta, I, J | SWE-Refactor dataset, execution infra, analytics | code+data | high | Several subareas overlap newer unified packages. |
| `swe_refactor/analytics` | J, meta | analytics package | code | low | Dedicated analytics area. |
| `swe_refactor/dataset.py` | meta, I | SWE dataset adapter | code | high | Same problem space as `smellai_datasets/loaders.py`. |
| `swe_refactor/dataset_card.md` | meta | dataset documentation | data-for-code | low | Canonical dataset doc. |
| `swe_refactor/persistence` | meta, J | analytics DB models/storage | code | medium | Conceptual overlap with `models/`. |
| `swe_refactor/smell_detection` | F | smell-detection package | code | medium | Smell detection is primarily owned by `sonarqube/`. |
| `swe_refactor/utils` | A–D, J | repo/build/java helpers | code | high | Strong overlap with `repo_utils` and agent helper code. |
| `tests` | meta | repository test suite | code+data | low | Clear support role. |
| `tests/conftest.py` | meta | shared pytest fixtures | code | low | Standard test support. |
| `tests/test_agent_invoke_eval_sample.py` | I | agent invocation tests | code | low | Directly tests execution path. |
| `tests/test_commit_scan.py` | F | SonarQube scan tests | code | low | Directly tests stage F logic. |
| `tests/test_data` | meta | fixture data | data | low | Standard test data area. |
| `tests/test_data_contracts.py` | meta | data contract tests | code | low | Shared correctness checks. |
| `tests/test_dataset_loaders.py` | meta | dataset loader tests | code | low | Tests canonical loader layer. |
| `tests/test_eval_sample.py` | meta, I | `EvalSample` schema tests | code | low | Tests shared schema. |
| `tests/test_java_test_agent.py` | A–D, J | Java test-agent tests | code | low | Directly tied to prep/verify stages. |
| `tests/test_mlflow_server.py` | meta | MLflow infra tests | code | low | Infra test. |
| `tests/test_repo_utils.py` | A–D, J | repo utility tests | code | low | Tests infra utility layer. |
| `tests/test_rminer_utils.py` | meta, I | RMiner utility tests | code | low | Narrow utility tests. |
| `tests/test_smell_cooccurrence.py` | E, H | smell interaction analysis tests | code | low | Related to dependency reasoning. |
| `tests/test_smell_prioritization.py` | E, H | prioritization tests | code | low | Directly tests planning inputs. |
| `tests/test_workflows_common.py` | meta | workflow helper tests | code | low | Shared orchestration support. |
| `tests/test_workflows_utils.py` | meta | workflow utility tests | code | low | Shared orchestration support. |
| `todo_7_3.md` | meta | personal todo notes | data-for-code | medium | Same category as other todo files. |
| `todo_for_now_16_4.md` | meta | personal todo notes | data-for-code | medium | Same category as other todo files. |
| `todo_restructure.md` | meta | repo restructuring analysis | data-for-code | low | Important analysis doc; also evidence for duplication hotspots. |
| `tools` | meta, E, H, J | visualization tools and manifests | code+data | medium | Visualization concern overlaps `scripts/viz`. |
| `tools/example_manifests` | meta, I | demo manifest data | data | low | Example inputs for demos/tools. |
| `tools/visualize_smell_prioritization.py` | E, H, J | dependency/analytics visualizer | code | medium | Similar concern to `scripts/viz`. |
| `tree` | meta | custom tree wrapper script | code | medium | Wrapper around existing shell tooling. |
| `uv.lock` | meta | dependency lockfile | data-for-code | low | Canonical dependency lock. |
| `workflows` | A–J | top-level workflow orchestration | code | low | Canonical workflow entrypoint layer. |
| `workflows/baseline_eval_workflow.py` | H, I, J | baseline evaluation flow | code | medium | Similar evaluation pattern to other workflow files. |
| `workflows/common.py` | meta | shared workflow helpers/types | code | low | Common support layer. |
| `workflows/composite_analysis_workflow.py` | E, H, I, J | full composite workflow | code | low | Central composite orchestration. |
| `workflows/eval_workflow.py` | meta, I, J | generic evaluation workflow | code | medium | Generic concern overlaps dataset-specific workflows. |
| `workflows/java_test_workflow.py` | A–D, J | test/build verification workflow | code | low | Direct match to prep/verify stages. |
| `workflows/rminer_eval_workflow.py` | I, J | RMiner evaluation workflow | code | medium | Similar concern to generic eval workflow. |
| `workflows/smell_cooccurrence_workflow.py` | E, H | smell interaction analysis workflow | code | low | Distinct analysis workflow. |
| `workflows/swe_eval_workflow.py` | I, J | SWE-Refactor evaluation workflow | code | medium | Similar concern to generic eval workflow. |
| `workflows/utils.py` | meta | workflow utility helpers | code | low | Shared support layer. |

## Main duplication hotspots

1. **`repo_utils/` ↔ `swe_refactor/utils/`** — repo/build/test helpers are split across two places.
2. **`smellai_datasets/` ↔ `swe_refactor/dataset.py` ↔ parts of `rminer/`** — dataset loading/conversion is not fully centralized.
3. **`scripts/viz/` ↔ `tools/visualize_smell_prioritization.py`** — visualization logic appears in more than one area.
4. **`agents/swe_eval/` ↔ `agents/rminer_eval/` ↔ `evals/ablation/mini_swe_agent/`** — execution/evaluation logic has parallel variants.
5. **Root notes/todos/slides ↔ `docs/`** — documentation and planning material is spread across several locations.
