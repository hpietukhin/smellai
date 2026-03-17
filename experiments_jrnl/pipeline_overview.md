# Unified Dataset Pipeline — Architecture & Experiment Design

> Этот документ описывает целевую архитектуру пайплайна данных, связывает её с кодовой базой и показывает, как каждый этап соотносится с экспериментом из conf.tex.

## 1. Общий поток данных

```
                      ┌──────────────────┐
                      │  Raw Sources     │
                      │  JSON / ZIP / DB │
                      └────────┬─────────┘
                               │
              hf_datasets/converter.py
              (rminer_to_hf, swe_refactor_to_hf, tdd_to_hf)
                               │
                      ┌────────▼─────────┐
                      │  HF Dataset      │
                      │  (Arrow/Parquet) │ ← canonical storage
                      └────────┬─────────┘
                               │
              hf_datasets/preprocessor.py   (dedup, filter)
              hf_datasets/config.py         (DATASET_CONFIGS)
                               │
                      ┌────────▼─────────┐
                      │  Preprocessed HF │
                      │  on disk         │
                      └────────┬─────────┘
                               │
              hf_datasets/mlflow_bridge.py
              (hf_to_genai_records, load_for_evaluation)
                               │
                      ┌────────▼─────────┐
                      │  MLflow GenAI    │
                      │  records         │  {inputs, expectations, tags}
                      └────────┬─────────┘
                               │
              workflows/eval_workflow.py    (unified)
                               │
                  ┌────────────┴────────────┐
                  │                         │
          predict_fn (rminer)       predict_fn (swe)
          → agents/rminer_eval/     → agents/swe_eval/
                  │                         │
          mlflow.genai.evaluate()   mlflow.genai.evaluate()
          + scorers                 + scorers
```

## 2. Слои пайплайна

### 2.1 Raw → HF Dataset (converter)

| Источник | Функция | Формат на входе | Ключевые колонки на выходе |
|----------|---------|-----------------|---------------------------|
| RMiner 2.0 oracle | `rminer_to_hf()` | `data.json` (list of commits) | `commit_sha`, `repository`, `refactoring_type`, `description`, `validation` |
| SWE-Refactor | `swe_refactor_to_hf()` | `.zip` / `.json` / directory | `pair_id`, `project_name`, `commit_id`, `source_before`, `source_after`, `class_before`, `class_after`, `jdk_version`, `compile_command` |
| TDD | `tdd_to_hf()` | SQLite DB | `project`, `commit_sha`, `parent_sha`, `smell_type`, `severity`, `file_path` |

**Файл:** `hf_datasets/converter.py`

SWE конвертер инлайнит загрузку из ZIP/JSON/directory напрямую (функции `_load_swe_raw_jsons()` + `_flatten_swe_record()`), без промежуточных адаптеров.

### 2.2 Preprocessing (preprocessor + config)

**Конфигурация** (`hf_datasets/config.py:DATASET_CONFIGS`):
```python
"rminer": {"dedup_keys": ["commit_sha", "refactoring_type", "description"], ...}
"swe":    {"dedup_keys": ["pair_id"], ...}
"tdd":    {"dedup_keys": ["commit_sha", "smell_type", "file_path"], ...}
```

**Утилиты** (`hf_datasets/preprocessor.py`):
- `deduplicate(ds, key_cols)` — убирает дубликаты
- `filter_by(ds, **kwargs)` — фильтрация по значениям колонок
- `split(ds, train, val, test)` — деление (зарезервировано, для conf.tex не нужно)
- `save(ds, path)` / `load(path)` — Arrow persistence

**CLI:** `scripts/datasets/preprocess.py` — запуск препроцессинга из командной строки.

### 2.3 HF → MLflow Bridge

**Файл:** `hf_datasets/mlflow_bridge.py`

Центральный мост. Использует маппинг колонок из `hf_datasets/config.py:MLFLOW_COLUMN_MAP`:

```python
"swe": {
    "input_cols":       ["project_name", "commit_id", "refactoring_type",
                         "source_before", "class_before", "file_path_before",
                         "file_path_after", "jdk_version", "compile_command"],
    "expectation_cols": ["source_after", "class_after"],
    "tag_cols":         ["has_tests", "is_compound", "is_pure"],
}
```

Две функции:
- `hf_to_genai_records(ds, source)` — Dataset → list of `{inputs, expectations, tags}`
- `load_for_evaluation(path, source)` — load from disk + convert

### 2.4 Unified Workflow

**Файл:** `workflows/eval_workflow.py`

Заменяет два отдельных workflow (`rminer_eval_workflow.py`, `swe_eval_workflow.py`).

```
eval_workflow.py --source rminer|swe
                 --hf-dataset-path <path>    (preferred)
                 --raw-data-path <path>      (fallback)
```

Для каждого `source` автоматически выбираются:
- **agent**: `agents/rminer_eval/` или `agents/swe_eval/`
- **scorers**: `mapping_accuracy, hunk_coverage` (rminer) или `compile_success, test_pass, overall_success` (swe)
- **predict_fn**: реконструирует `RefactoringRecord` из flat HF row (для swe)

### 2.5 Agent Layer (не меняется)

| Agent | Файл | Роль (conf.tex) |
|-------|------|-----------------|
| SWE Eval Agent | `agents/swe_eval/agent.py` | Stages A–J: setup, detect, prioritize, refactor, verify |
| RMiner Eval Agent | `agents/rminer_eval/agent.py` | Mapping refactorings to diff hunks |
| Dependency Analysis | `agents/dependency_analysis/agent.py` | Stage E: dependency graph construction |

`RefactoringRecord` (`swe_refactor/dataset.py`) — входная модель для SWE agent, остается на месте.

## 3. Связь с экспериментом (conf.tex)

### 3.1 Stages A–J из conf.tex → код

| Stage | conf.tex | Реализация | Файл |
|-------|----------|------------|------|
| **A** | Load source, detect build system | `detect_build_system()` | `agents/tools/java_test_tools.py` |
| **B** | Check test coverage | `run_tests()` | `agents/tools/java_test_tools.py` |
| **C** | Generate missing tests (LLM) | Planned (в `SWEEvalState.test_generation_needed`) | `agents/swe_eval/agent.py` |
| **D** | Run test suite (baseline) | `run_tests()` node в LangGraph | `agents/swe_eval/agent.py` |
| **E** | Build dependency graph | `DEPENDENCY_RULES` + NetworkX graph | `agents/dependency_analysis/agent.py` |
| **F** | SonarQube scan → $S_0$ | `scan_commit()` | `sonarqube/commit_scan.py` |
| **G** | Developer selects smells | `SWEEvalState.selected_smells` | `agents/swe_eval/agent.py` |
| **H** | BeFS / Greedy planner → plan $\pi$ | PZ formula + prioritization | `scripts/prioritize_smells.py` |
| **I** | LLM refactoring (chain-of-thought) | `ChatLiteLLM` + Pydantic schema | `agents/swe_eval/agent.py` + `prompts.py` |
| **J** | Test verification, rollback | compile + test nodes, retry logic | `agents/swe_eval/agent.py` |

### 3.2 Реконструкция $S_0$ (pre-refactoring smell set)

> conf.tex §IV: "For each evaluated commit we: (1) reconstruct the pre-refactoring smell set $S_0$ from SonarQube"

**Как это работает в коде:**

1. Загружаем запись из HF dataset через bridge: `load_for_evaluation(path, "swe")`
2. `predict_fn` в `eval_workflow.py` реконструирует `RefactoringRecord` из flat row
3. Agent (stage F) использует `sonarqube/commit_scan.py:scan_commit()` на **parent commit**:
   - `get_previous_commit(commit_id)` → parent SHA
   - `force_checkout_commit(parent_sha)` → checkout
   - SonarQube REST API → issues → normalized `SmellEvent` objects
4. Результат: `SWEEvalState.detected_smells` = $S_0$

**Ключевые файлы:**
- `sonarqube/commit_scan.py` — scanner + issue normalization (8 smell types → Table 1 conf.tex)
- `swe_refactor/utils.py` — `get_previous_commit()`, `force_checkout_commit()`
- `agents/dependency_analysis/agent.py:RULE_NAME_MAP` — маппинг SonarQube rules → smell types

### 3.3 PZ-формула приоритизации (Eq. 2 conf.tex)

> $P^{conc}_i = f_i \cdot (w_{sev} \cdot sev(s_i) + \sum pos\_out^{conc}(s_i) - w_{neg} \cdot \sum neg\_out^{abs}(s_i))$

**Реализация:** `scripts/prioritize_smells.py`

- `SmellInstance` (dataclass) — node в графе, `.severity_score` маппит HIGH=3, MEDIUM=2, LOW=1
- `DEPENDENCY_RULES` из `agents/dependency_analysis/agent.py` — каталог зависимостей (Table 2 conf.tex)
- NetworkX directed graph: positive edges (green) и negative edges (red dashed)
- Функция вычисляет PZ для каждого smell → сортирует по убыванию → выдает plan $\pi$

**Планировщики (conf.tex Algorithms 1–2):**

- **Greedy (Alg. 1):** На каждом шаге выбирает $\arg\max P_i$, пересчитывает $S$
- **BeFS (Alg. 2):** State-space search с $h(S) = \sum sev(s)$, OPEN/CLOSED sets

> **Статус:** В `scripts/prioritize_smells.py` реализован greedy. BeFS-планировщик пока не реализован как отдельный модуль — это TODO для полного эксперимента.

### 3.4 Сравнение с developer's first action

> conf.tex §IV: "(3) compare each candidate against the refactoring type the developer applied first"

**Как это будет работать:**

1. HF dataset содержит `refactoring_type` для каждой записи — это ground truth (что сделал разработчик)
2. Планировщик получает $S_0$ и выдает ordered plan $\pi$
3. Первый элемент $\pi[0]$ = predicted first refactoring type
4. MLflow scorer сравнивает $\pi[0]$ с `expectations.refactoring_type`

**Метрики (conf.tex §V):**
- `plan_efficiency` $\eta = \text{steps} / \text{smells resolved}$
- `negative_dependency_rate` $\rho = \text{new smells} / \text{refactorings executed}$
- `compile_and_test_pass_rate` — через scorers в `workflows/swe_eval_workflow.py`

### 3.5 Dependency model (conf.tex §III-B)

| Smell | Positive (resolves) | Negative (introduces) | Source |
|-------|--------------------|-----------------------|--------|
| Long Method / Complex | Feature Envy, Dup. Code, Long Param List | Long Method, Long Param List | `DEPENDENCY_RULES` |
| Long Param List | Data Clumps | Data Class | `DEPENDENCY_RULES` |
| God Class / Large Class | Feature Envy, Data Clumps | Long Method, Data Class, Inapp. Intimacy | `DEPENDENCY_RULES` |
| Dup. Conditions | Divergent Change | Large Class | `DEPENDENCY_RULES` |

**Реализация:** `agents/dependency_analysis/agent.py:DEPENDENCY_RULES` (dict mapping smell_type → positive/negative lists)

8 smell types маппятся на SonarQube rules (`RULE_NAME_MAP` в том же файле, Table 1 conf.tex).

### 3.6 Commit chains для multi-smell evaluation

**Файл:** `hf_datasets/chain_builder.py`

`build_commit_chains(ds, source)` группирует записи HF dataset в упорядоченные цепочки:
- **rminer:** по repository + time → хронологический порядок коммитов
- **swe:** по project_name + commit_id → co-located refactorings в одном коммите
- **tdd:** по parent_sha linkage → smell lifecycle chains (introduced → resolved)

Это нужно для conf.tex: developer-committed refactoring sequences — цепочки показывают, какие рефакторинги разработчик применил последовательно.

## 4. Persistence layer

### 4.1 Analytics DB (отдельно от MLflow)

**Файлы:** `swe_refactor/persistence/database.py` + `models.py`

SQLModel ORM таблицы:
- `SmellEvent` — lifecycle smell (detected/resolved/created/persisted), `session_id + iteration`
- `RefactoringAttempt` — результат рефакторинга, counters smells_resolved/created
- `ToolCall` — tool invocations с timing
- `TokenUsage` — LLM token consumption per node
- `SmellDependency` — positive/negative relationships

Это **не** LangGraph checkpoints. Это аналитика для визуализации и экспорта в MLflow.

### 4.2 MLflow tracking

**Файл:** `mlflow_utils/datasets/manager.py:DatasetManager`

- `create_dataset_from_records(records, name, experiment, tags)` — регистрирует MLflow GenAI dataset
- `list_datasets()` / `get_dataset()` — CRUD

MLflow хранит evaluation runs, metrics, artifacts. Dashboard: `http://localhost:5000`.

## 5. Полный experimental workflow (conf.tex §IV)

```
1. Preprocess:
   scripts/datasets/preprocess.py --source swe --dataset-path SWE-Refactor.zip --dedup --output data/processed/swe

2. Run evaluation:
   workflows/eval_workflow.py --source swe --hf-dataset-path data/processed/swe --model claude-sonnet-4-5-20250929

   Internally per record:
   a) load_for_evaluation() → MLflow GenAI record
   b) predict_fn → RefactoringRecord
   c) Agent A0: clone repo, checkout parent commit, detect build system
   d) Agent F:  sonarqube/commit_scan.py → S₀ (pre-refactoring smells)
   e) Agent E:  dependency graph from S₀ + DEPENDENCY_RULES
   f) Agent H:  prioritize_smells.py → plan π (Greedy or BeFS)
   g) Agent I:  LLM refactoring (ChatLiteLLM + chain-of-thought prompt)
   h) Agent J:  compile + test → pass/fail
   i) Scorers:  compare π[0] vs developer's first action

3. View results:
   mlflow ui → http://localhost:5000
```

## 6. Что ещё не реализовано (TODO)

| Item | Описание | Связь с conf.tex |
|------|----------|------------------|
| **BeFS planner** | Algorithm 2 как отдельный модуль (сейчас только greedy) | §III-C, Algorithm 2 |
| **First-action scorer** | MLflow scorer: π[0] vs developer's refactoring_type | §IV, metric "match" |
| **Plan metrics** | η (plan efficiency), ρ (negative dep rate) | §V |
| **RMiner raw format** | Конвертер для raw RM output (не oracle) | Отложен |
| **Stage C** | LLM test generation для uncovered methods | §III-A, Stage C |
| **Weight tuning** | $w_{sev}$, $w_{neg}$ per project | §V |

## 7. File index

| Файл | Роль |
|------|------|
| `hf_datasets/converter.py` | Raw → HF Dataset |
| `hf_datasets/preprocessor.py` | Dedup, filter, split, save/load |
| `hf_datasets/config.py` | DATASET_CONFIGS + MLFLOW_COLUMN_MAP |
| `hf_datasets/mlflow_bridge.py` | HF → MLflow GenAI records |
| `hf_datasets/chain_builder.py` | Commit chain reconstruction |
| `hf_datasets/models.py` | DiffHunk model |
| `workflows/eval_workflow.py` | Unified evaluation entry point |
| `workflows/swe_eval_workflow.py` | SWE-specific scorers (compile, test) |
| `workflows/rminer_eval_workflow.py` | RMiner-specific scorers (mapping, hunk coverage) |
| `agents/swe_eval/agent.py` | LangGraph agent (stages A–J) |
| `agents/rminer_eval/agent.py` | RMiner mapping agent |
| `agents/dependency_analysis/agent.py` | DEPENDENCY_RULES + DependencyAnalysis model |
| `scripts/prioritize_smells.py` | PZ formula, greedy planner, NetworkX graph |
| `sonarqube/commit_scan.py` | SonarQube REST scan → SmellEvent |
| `swe_refactor/dataset.py` | RefactoringRecord model |
| `swe_refactor/persistence/models.py` | SmellEvent, ToolCall, etc. (analytics ORM) |
| `mlflow_utils/datasets/manager.py` | MLflow dataset CRUD |
