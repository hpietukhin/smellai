# Prompts for researching the SmellAI codebase

These prompts are written for an LLM/coding agent that will inspect this repository and produce a precise map of agents, entry points, ownership boundaries, and ambiguities.

---

## Prompt 1 — Repository inventory and runtime surface

Ты исследуешь кодовую базу `smellai` как новый инженер-исследователь.

Сделай следующее:
1. Просмотри все верхнеуровневые папки и кратко классифицируй их как: runtime code / experiments / docs / tooling / tests / notebooks / legacy / ambiguous.
2. Для каждой папки укажи, участвует ли она в реальном исполнении multi-agent системы или только поддерживает исследования.
3. Отдельно найди все CLI / entry points:
   - `if __name__ == "__main__"`
   - Fire/argparse-based `main(...)`
   - `project.scripts` в `pyproject.toml`
4. Построй таблицу: `path | role | invoked by | invokes | runtime critical?`.
5. Если какой-то каталог упоминается в импортах/доках, но отсутствует физически в репозитории, пометь это как critical ambiguity.

Начни с файлов:
- `README.md`
- `pyproject.toml`
- `workflows/*.py`
- `agents/**/*.py`
- `smellai_datasets/*.py`
- `docs/development/repo_tree_agent_mapping.md`

Ожидаемый результат:
- короткое summary,
- список реальных entry points,
- список каталогов, которые можно считать каноническими.

---

## Prompt 2 — Agent map vs thesis/spec map

Сопоставь фактическую реализацию агентов с заявленной схемой A0–A7 из thesis/spec.

Сделай следующее:
1. Используй `docs/conf_Pietukhin_10_3_rev2-2.pdf` как главную спецификацию, а `README.md` и `docs/drafts/TECHNICAL_SPECIFICATION.md` — как вторичные источники.
2. Для каждого агента A0–A7 определи:
   - реализован ли он,
   - где лежит код,
   - какая функция/класс является главным entry point,
   - кто его вызывает,
   - что он возвращает,
   - есть ли тесты.
3. Отдельно зафиксируй расхождения между:
   - paper/spec,
   - README,
   - текущим кодом.
4. Если один и тот же номер агента по-разному интерпретируется в разных местах (например A2/A3/A4), выпиши exact mismatch.
5. Укажи, какие агенты реально являются отдельными агентами, а какие — просто узлы внутри одного LangGraph workflow.

Сфокусируйся на:
- `agents/swe_eval/agent.py`
- `agents/rminer_eval/agent.py`
- `agents/java_test/agent.py`
- `agents/dependency_analysis/agent.py`
- `agents/baseline/agent.py`
- `workflows/eval_workflow.py`

Ожидаемый результат:
- таблица `Agent ID | intended role | actual implementation | status | ambiguity`.

---

## Prompt 3 — End-to-end execution flow and ownership boundaries

Проследи фактический путь выполнения для основных сценариев.

Нужно разобрать минимум 3 сценария:
1. RMiner evaluation
2. SWE evaluation (basic)
3. SWE evaluation (composite)

Для каждого сценария определи:
- внешний entry point,
- workflow file,
- factory function создания агента,
- invoke function,
- dataset loader,
- внешние зависимости (SonarQube, MLflow, swe_refactor, Java toolchain, git repos),
- где происходит orchestration,
- где происходит LLM call,
- где происходит verification.

Построй call chain вида:
`CLI -> workflow.main -> loader -> create_agent -> invoke_agent -> node/tool/util`.

Отдельно определи ownership boundaries:
- что считается каноническим слоем orchestration,
- что считается каноническим слоем dataset abstraction,
- что считается каноническим слоем infra/utilities,
- где есть дублирование.

Если находишь 2+ возможных “источника истины”, помечай это как ambiguity и объясняй, какой из них сейчас фактически используется.

---

## Prompt 4 — Ambiguity and missing-dependency audit

Проведи аудит неоднозначностей так, как если бы завтра другой агент должен был автономно модифицировать эту кодовую базу.

Проверь:
1. Все импорты на отсутствующие модули/пакеты.
2. Все ссылки в README/docs на отсутствующие директории или устаревшие пути.
3. Несоответствие между package surface (`pyproject.toml`, `__init__.py`) и реальной структурой файлов.
4. Пустые или неиспользуемые каталоги, особенно если их название намекает на runtime ownership (например `prompts/`).
5. Параллельные реализации одной и той же роли (например dataset loading, repo utils, visualization, evaluation wrappers).
6. Все planned/TBD агенты и то, как код сейчас их эмулирует.

Формат результата:
- `critical` — ломает импорт/запуск или делает ownership неясным,
- `major` — мешает навигации и безопасным изменениям,
- `minor` — косметическая неоднозначность.

Для каждой проблемы дай:
- evidence,
- impact,
- minimal fix,
- ideal fix.

---

## Prompt 5 — Disambiguation plan for future agents

Сформируй план, как сделать кодовую базу понятной для будущих агентов и инженеров.

Нужно предложить:
1. Каноническую карту каталогов (`agents/`, `workflows/`, `smellai_datasets/`, `sonarqube/`, `evals/`, `repo_utils/`, etc.).
2. Явную матрицу ownership: какой каталог за что отвечает.
3. Как переименовать или реструктурировать неоднозначные зоны.
4. Какие README/ARCHITECTURE/AGENTS docs добавить.
5. Какие import-level smoke tests нужны, чтобы ловить missing packages вроде `swe_refactor`.
6. Как обозначить legacy / planned / experimental код, чтобы агент не принимал его за production runtime.

Итог оформи как roadmap в 3 слоя:
- quick fixes,
- structural fixes,
- documentation fixes.

---

## Short one-shot prompt

Исследуй кодовую базу `smellai` и дай мне карту системы с фокусом на multi-agent runtime. Я хочу понять:
1. какие здесь есть агенты и где лежат их части,
2. какие файлы являются реальными точками входа,
3. какие каталоги канонические, а какие legacy/experimental,
4. есть ли неоднозначности в ownership или mapping agent->code,
5. какие отсутствующие зависимости/директории ломают однозначную навигацию,
6. как минимально устранить эти неоднозначности.

Обязательно:
- проверь `README.md`, `pyproject.toml`, `workflows/`, `agents/`, `smellai_datasets/`, `evals/`, `sonarqube/`, `docs/development/repo_tree_agent_mapping.md`;
- выпиши несовпадения между документацией и кодом;
- отдельно пометь critical ambiguity, если модуль упоминается в импортах, но отсутствует в репозитории.
