"""DEPRECATED: Prompt templates for SWE refactoring generation by type.

Use Composite Refactorings 2020 extraction + planner evaluation instead.
"""

SYSTEM_PROMPT = """You are an expert Java refactoring assistant.

Your task is to apply a specific refactoring to Java code while preserving behavior.

Key requirements:
- Output ONLY valid Java code (the complete refactored file)
- Preserve all functionality and behavior
- Follow Java naming conventions
- Maintain proper indentation and formatting
- Do NOT add comments explaining the refactoring
- Do NOT add extra features or improvements beyond the refactoring

Tool/back-end preference (safe advisory, not forced):
- For structural refactorings (e.g., Rename Method, Move Method, Extract/Move Class), prefer Spoon-based execution (`run_spoon_refactor`).
- For local syntax-tree rewrites (e.g., Rename Local Variable, simple logger replacement), prefer ast-grep rewrites (`run_ast_grep_rewrite_git_safe`).
- If structural tooling is unavailable, fallback to simple textual patch (`replace_in_file_git_safe`) and report limitation.
- Prefer git-safe tools so every edit is rollback-able (`git reset --hard <previous_sha>`).
"""


def get_refactoring_prompt(
    refactoring_type: str,
    source_code_before: str,
    target_code_before: str | None,
    file_path_before: str,
    file_path_after: str,
) -> str:
    """Generate refactoring prompt for specific type.

    Args:
        refactoring_type: Type of refactoring to perform
        source_code_before: Full source file content
        target_code_before: Full target file content (for Move refactorings)
        file_path_before: Source file path
        file_path_after: Target file path

    Returns:
        Formatted prompt string
    """
    if refactoring_type == "Extract Method":
        return _extract_method_prompt(source_code_before, file_path_before)

    elif refactoring_type == "Inline Method":
        return _inline_method_prompt(source_code_before, file_path_before)

    elif refactoring_type in (
        "Move Method",
        "Move And Rename Method",
        "Move And Inline Method",
    ):
        return _move_method_prompt(
            source_code_before,
            target_code_before,
            file_path_before,
            file_path_after,
        )

    elif refactoring_type == "Extract And Move Method":
        return _extract_and_move_prompt(
            source_code_before,
            target_code_before,
            file_path_before,
            file_path_after,
        )

    raise ValueError(f"Unknown refactoring type: {refactoring_type}")


def _extract_method_prompt(source_code: str, file_path: str) -> str:
    """Prompt for Extract Method refactoring."""
    return f"""# Extract Method Refactoring

File: {file_path}

Analyze the code below and apply Extract Method refactoring to improve code organization.
Identify code that should be extracted into a separate method and perform the extraction.

Return the complete refactored file content.

## Source Code

```java
{source_code}
```

Output the refactored code:
"""


def _inline_method_prompt(source_code: str, file_path: str) -> str:
    """Prompt for Inline Method refactoring."""
    return f"""# Inline Method Refactoring

File: {file_path}

Analyze the code below and apply Inline Method refactoring.
Identify a method that should be inlined into its callers and perform the inlining.

Return the complete refactored file content.

## Source Code

```java
{source_code}
```

Output the refactored code:
"""


def _move_method_prompt(
    source_code: str,
    target_code: str | None,
    source_path: str,
    target_path: str,
) -> str:
    """Prompt for Move Method refactoring."""
    target_section = ""
    if target_code:
        target_section = f"""
## Target File (Before)

File: {target_path}

```java
{target_code}
```
"""

    return f"""# Move Method Refactoring

Move a method from the source class to the target class.

## Source File (Before)

File: {source_path}

```java
{source_code}
```
{target_section}

Identify which method should be moved from {source_path} to {target_path}.
Return TWO complete refactored files in this format:

// FILE: {source_path}
```java
<refactored source code>
```

// FILE: {target_path}
```java
<refactored target code>
```
"""


def _extract_and_move_prompt(
    source_code: str,
    target_code: str | None,
    source_path: str,
    target_path: str,
) -> str:
    """Prompt for Extract And Move Method refactoring."""
    target_section = ""
    if target_code:
        target_section = f"""
## Target File (Before)

File: {target_path}

```java
{target_code}
```
"""

    return f"""# Extract And Move Method Refactoring

Extract code into a new method AND move it to the target class (compound refactoring).

## Source File (Before)

File: {source_path}

```java
{source_code}
```
{target_section}

Identify code to extract into a method, then move that method to {target_path}.
Return TWO complete refactored files in this format:

// FILE: {source_path}
```java
<refactored source code>
```

// FILE: {target_path}
```java
<refactored target code>
```
"""
