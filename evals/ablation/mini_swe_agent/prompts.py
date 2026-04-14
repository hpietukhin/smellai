"""Task prompt builder for mini-swe-agent ablation.

Reuses the same refactoring framing as agents/swe_eval/prompts.py but phrases
it as a bash-agent task (edit files in place, verify with compile command).
"""

from swe_refactor.dataset import RefactoringRecord


_TASK_TEMPLATE = """\
You are working on a Java project. Your task is to apply a **{refactoring_type}** refactoring.

## Target file
`{file_path_before}`

## Current source code
```java
{source_code}
```
{target_section}
## Instructions
1. Edit `{file_path_before}` in place to apply the refactoring.{extra_file_note}
2. After editing, verify compilation with:
   ```
   {compile_command}
   ```
3. Fix any compilation errors you introduced.
4. Once compilation succeeds, submit.

Rules:
- Preserve all existing behaviour and functionality.
- Follow Java naming conventions.
- Do NOT add features beyond the requested refactoring.
"""

_TARGET_SECTION_TEMPLATE = """\

## Target file (for move refactorings)
`{file_path_after}`
```java
{target_code}
```
"""


def build_refactoring_task(record: RefactoringRecord, project_path: str | None = None) -> str:
    """Build a free-form bash-agent task string from a RefactoringRecord.

    Args:
        record: The refactoring record (source code + metadata).
        project_path: Absolute path to the checked-out project root (for richer context).

    Returns:
        Task string suitable for DefaultAgent.run(task).
    """
    target_section = ""
    extra_file_note = ""

    is_move = record.type in (
        "Move Method",
        "Extract And Move Method",
        "Move And Rename Method",
        "Move And Inline Method",
    )

    if is_move and record.filePathAfter and record.filePathBefore != record.filePathAfter:
        target_code = ""
        if project_path:
            from pathlib import Path
            target_file = Path(project_path) / record.filePathAfter
            if target_file.exists():
                try:
                    target_code = target_file.read_text(encoding="utf-8")
                except OSError:
                    pass

        if target_code:
            target_section = _TARGET_SECTION_TEMPLATE.format(
                file_path_after=record.filePathAfter,
                target_code=target_code,
            )

        extra_file_note = (
            f"\n   Also update `{record.filePathAfter}` as required by the move."
        )

    return _TASK_TEMPLATE.format(
        refactoring_type=record.type,
        file_path_before=record.filePathBefore,
        source_code=record.sourceCodeBeforeForWhole,
        target_section=target_section,
        compile_command=record.compileCommand or "mvn -q compile",
        extra_file_note=extra_file_note,
    )
