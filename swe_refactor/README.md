# swe_refactor

SWE-Refactor subsystem: dataset models, build/repo utilities, smell detection, and analytics persistence.

## Key Files

- **dataset.py** - `RefactoringRecord` (Pydantic model) and `SWERefactorDataset` loader for `pure_refactoring_data.json`
- **utils/** - Build compilation, JDK switching (jenv), repository cloning/checkout, code replacement
- **persistence/** - SQLModel ORM for analytics: `ToolCall`, `SmellEvent`, `RefactoringAttempt`, `TokenUsage`, `TestRun`
- **persistence/database.py** - `AnalyticsDB` manager with session summaries and query methods
- **smell_detection/** - SonarQube integration for detecting smells during refactoring workflow
- **analytics/** - Reporting utilities for evaluation sessions

## Usage

```python
from swe_refactor.dataset import load_swe_refactor_dataset

records = load_swe_refactor_dataset("/tmp/SWE-Refactor/pure_refactoring_data.json")
extract_methods = [r for r in records if r.type == "Extract Method"]
```

```python
from swe_refactor.utils import compile_project, switch_java_version, clone_repository

switch_java_version("17", project_path=path)
result = compile_project(path, command="mvn clean compile")
```

```python
from swe_refactor.persistence.database import AnalyticsDB

db = AnalyticsDB("test_analytics.db")
summary = db.get_session_summary(session_id)
```

## Database Schema

Analytics are stored in SQLite (`test_analytics.db`), separate from LangGraph checkpoints:

| Table | Purpose |
|-------|---------|
| `tool_calls` | Tool invocations with timing |
| `smell_events` | Smell lifecycle (detected/resolved/created) |
| `smell_dependencies` | Positive/negative smell relationships |
| `refactoring_attempts` | Refactoring outcomes per iteration |
| `token_usage` | LLM token consumption by node |
| `test_runs` | Test execution results per iteration |
