Использование ORM (особенно **SQLModel**) полностью соответствует современным best practices в экосистеме Python и LangGraph.

Вот анализ того, как мы можем оптимизировать ваш план.

### Почему ручной SQL — плохой выбор здесь?

1. **Избыточность кода:** Вам придется вручную мапить поля объектов Python на колонки БД и обратно.
2. **Хрупкость:** При изменении структуры (например, добавлении поля в `SmellEvent`) нужно править CREATE TABLE, INSERT-запросы и код извлечения в разных местах.
3. **Отсутствие типизации:** Ручной SQL не дает статических гарантий типов.

### Решение: SQLModel (Pydantic + SQLAlchemy)

Лучший современный подход для связки с LangGraph — **SQLModel**.

* **Совместимость:** LangGraph уже использует Pydantic для управления состоянием (`State`). SQLModel позволяет использовать одни и те же классы и как валидаторы данных, и как схемы БД.
* **Лаконичность:** Вы определяете класс один раз. Таблицы создаются автоматически.
* **Безопасность:** Защита от SQL-инъекций и ошибок типов "из коробки".

### Оптимизация плана (Best Practices LangGraph)

В контексте LangGraph важно разделить **Persistence (Состояние графа)** и **Analytics (Логирование событий)**.

1. **Состояние графа (Checkpoints):**
* *Не пишите это вручную.* Используйте встроенный `SqliteSaver` из `langgraph.checkpoint.sqlite`. Он уже умеет сохранять всё состояние агента, историю сообщений и позволяет делать "путешествия во времени" (replay).
* *Оптимизация:* Уберите таблицу `checkpoints` из вашего ручного плана. Доверьте это LangGraph.


2. **Аналитика (Ваши кастомные данные):**
* Для таблиц `smell_events`, `dependencies`, `tool_calls` и `token_usage` используйте ORM. Эти данные нужны для MLFlow и отчетов, их сложно доставать из сырого состояния графа.



### Обновленная структура (Минус 40% кода)

Вместо сложного `manager.py` с десятками SQL-строк, мы сделаем:

**1. `swe_refactor/persistence/models.py**` (Вместо `schema.py`)
Описываем таблицы как классы Python.

```python
from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class ToolCall(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    tool_name: str
    arguments: str  # JSON string
    duration_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SmellEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    iteration: int
    smell_type: str
    action: str  # "detected", "resolved", "created"
    # ... другие поля

```

**2. `swe_refactor/persistence/database.py**` (Упрощенный `manager.py`)
Вся логика сводится к созданию engine и простым сессиям.

```python
from sqlmodel import create_engine, SQLModel, Session

def init_db(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    return engine

def log_event(engine, event: SQLModel):
    with Session(engine) as session:
        session.add(event)
        session.commit()

```

### Итоговый вердикт по изменениям

* **Убираем:** Ручные SQL-строки, таблицу checkpoints (используем нативный `SqliteSaver`).
* **Добавляем:** Библиотеку `sqlmodel`.
* **Выигрыш:** Код сокращается в 2-3 раза, добавляется строгая типизация, упрощается рефакторинг.
