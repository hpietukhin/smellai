# Design by Contract для Dataset Pipeline (Pydantic)

> Контракты на стыках слоёв пайплайна. Проверяются при каждом реальном запуске — заменяют интеграционные тесты.

## Принцип

```
converter output  →  SWERow / RMinerRow    (postcondition converter = precondition preprocessor)
                            ↓
preprocessor      →  тот же тип, без изменений
                            ↓
bridge output     →  GenAIRecord            (postcondition bridge = precondition workflow)
                            ↓
predict_fn input  →  RefactoringRecord      (уже есть в swe_refactor/dataset.py)
```

Каждая модель — контракт на границе слоя. Если converter выдал мусор, падает на `SWERow(**row)` с понятной ошибкой, а не тихо прокидывает пустоту до LLM agent.

## Подход: `Annotated` + `Field()` (Pydantic v2 standard)

Pydantic v2 рекомендует `Annotated` pattern для reusable constrained types.
Два источника constraints — оба работают внутри `Annotated`:

| Источник | Пример | Когда использовать |
|----------|--------|--------------------|
| `pydantic.Field()` | `Annotated[int, Field(gt=0)]` | Стандарт, уже используется в проекте |
| `annotated_types` | `Annotated[int, Gt(0)]` | Framework-agnostic, но лишний import |

**Выбор для проекта: `Field()`** — consistency с `RefactoringRecord`, `DiffHunk` и остальными моделями.

### Reusable type aliases (определяем один раз, используем везде)

```python
from typing import Annotated, Literal
from pydantic import Field

# Shared type aliases — smellai_datasets/contracts.py
NonEmptyStr   = Annotated[str, Field(min_length=1)]
CommitSha     = Annotated[str, Field(min_length=6, max_length=40)]
JDKVersion    = Annotated[int, Field(gt=0, le=25)]
JavaFilePath  = Annotated[str, Field(pattern=r".*\.java$")]

SWERefactoringType = Literal[
    "Extract Method", "Move Method", "Inline Method",
    "Extract And Move Method", "Move And Rename Method",
    "Move And Inline Method",
]

RMinerValidation = Literal["TP", "FP", ""]
```

Aliases переиспользуются в `SWERow`, `RMinerRow`, `GenAIRecord` — DRY вместо дублирования constraints.

## 1. Preconditions — валидация на входе слоя

### SWERow (выход converter = вход preprocessor/bridge)

```python
from pydantic import BaseModel, model_validator

class SWERow(BaseModel):
    """Контракт: одна строка SWE dataset после converter."""
    pair_id: NonEmptyStr
    project_name: NonEmptyStr
    commit_id: NonEmptyStr
    refactoring_type: SWERefactoringType
    source_before: str
    source_after: str
    class_before: str
    class_after: str
    jdk_version: JDKVersion
    compile_command: str
    has_tests: bool
    file_path_before: str
    file_path_after: str

    @model_validator(mode="after")
    def source_not_empty(self):
        if not self.source_before.strip():
            raise ValueError(f"Empty source_before for {self.pair_id}")
        return self
```

### RMinerRow (выход converter)

```python
class RMinerRow(BaseModel):
    """Контракт: одна строка RMiner dataset после converter."""
    commit_id: int
    repository: NonEmptyStr
    commit_sha: CommitSha
    refactoring_type: NonEmptyStr
    description: str
    validation: RMinerValidation
    detection_tools: str
```

## 2. Postconditions — валидация выхода bridge

### GenAIRecord (выход bridge = вход workflow)

```python
class GenAIRecord(BaseModel):
    """Контракт: один MLflow GenAI record."""
    inputs: dict
    expectations: dict
    tags: dict

    @model_validator(mode="after")
    def inputs_not_empty(self):
        if not self.inputs:
            raise ValueError("Empty inputs in GenAI record")
        return self
```

## 3. Invariants через Annotated aliases

Вместо рантайм-проверок `if jdk < 0: raise` — декларативно через type aliases:

```python
# Все constraints в одном месте — contracts.py
JDKVersion  = Annotated[int, Field(gt=0, le=25)]       # JDK 1..25
NonEmptyStr = Annotated[str, Field(min_length=1)]       # не пустой
CommitSha   = Annotated[str, Field(min_length=6)]       # минимум 6 символов

# Используются в моделях без дублирования
class SWERow(BaseModel):
    jdk_version: JDKVersion
    commit_id: NonEmptyStr

class RMinerRow(BaseModel):
    commit_sha: CommitSha  # тот же контракт, без копипасты
```

Pydantic проверяет при создании объекта — zero boilerplate.

### Продвинутые паттерны из Pydantic v2 docs

**Generic constrained types с TypeVar:**
```python
from typing import TypeVar
from annotated_types import Len
T = TypeVar("T")

ShortList = Annotated[list[T], Len(max_length=10)]
# ShortList[str] — list of strings, max 10 elements
```

**Custom validation через AfterValidator:**
```python
from pydantic import AfterValidator

StripStr = Annotated[str, AfterValidator(lambda x: x.strip())]
# Автоматически strip() при создании
```

**Custom serialization через PlainSerializer:**
```python
from pydantic import PlainSerializer

TruncatedFloat = Annotated[
    float,
    AfterValidator(lambda x: round(x, 2)),
    PlainSerializer(lambda x: f"{x:.2f}", return_type=str),
]
```

## 4. Где ставить в коде

**Файл:** `smellai_datasets/contracts.py` (новый)

**Использование в converter:**
```python
# smellai_datasets/converter.py
from smellai_datasets.contracts import SWERow

def swe_refactor_to_hf(...) -> Dataset:
    ...
    for data in raw_jsons:
        row = _flatten_swe_record(data)
        SWERow(**row)  # validate — падает с понятной ошибкой если контракт нарушен
        rows.append(row)
```

**Использование в bridge:**
```python
# smellai_datasets/mlflow_bridge.py
from smellai_datasets.contracts import GenAIRecord

def hf_to_genai_records(ds, source) -> list[dict]:
    ...
    for row in ds:
        record = {...}
        GenAIRecord(**record)  # postcondition
        records.append(record)
```

## 5. Что это даёт

| Ситуация | Без контрактов | С контрактами |
|----------|---------------|---------------|
| Converter выдал пустой `source_before` | Тихо доходит до LLM, agent получает пустую строку | `ValidationError: Empty source_before for pair_123` на выходе converter |
| Bridge забыл колонку в MLFLOW_COLUMN_MAP | Пустой `inputs` в MLflow UI | `ValidationError: Empty inputs in GenAIRecord` |
| Новый dataset source, забыл колонку | `KeyError` где-то в agent | Pydantic говорит какое поле отсутствует |
| JDK version = -1 (битые данные) | Тихо прокидывается, `jenv` падает позже | `ValidationError: jdk_version > 0` |
| commit_sha = "" (пустой хеш) | `git checkout ""` — непонятная ошибка | `ValidationError: min_length=6` |

## 6. Чего НЕ делать

- Не ставить контракты внутри слоя (preprocessor deduplicate не нужен — он принимает и возвращает тот же тип)
- Не валидировать то, что уже проверяет HF `datasets` lib (типы колонок в Arrow)
- Не дублировать `RefactoringRecord` — он уже есть в `swe_refactor/dataset.py` и уже работает как контракт для agents
- Не использовать `annotated_types` напрямую — `Field()` уже есть в проекте, не добавлять лишний import style
