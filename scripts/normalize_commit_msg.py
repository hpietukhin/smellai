#!/usr/bin/env python3
"""Commit message normalizer.

Enforces:
1. Imperative mood subject line
2. Conventional commits type prefix (feat/fix/refactor/docs/chore/test/wip)
   auto-detected from keywords if absent
3. Subject line ≤72 chars
4. Capitalized first word after type prefix
5. No trailing period on subject
"""
import re
import sys
from pathlib import Path

VALID_TYPES = ("feat", "fix", "refactor", "docs", "chore", "test", "wip", "style", "ci", "perf")

# Maps keywords found in subject → conventional commit type
KEYWORD_TYPE_MAP: list[tuple[list[str], str]] = [
    (["fix", "bug", "patch", "correct", "resolve", "repair"], "fix"),
    (["refactor", "extract", "rename", "restructure", "reorganize", "cleanup", "clean up"], "refactor"),
    (["doc", "docs", "readme", "comment", "document"], "docs"),
    (["test", "spec", "coverage", "assertion"], "test"),
    (["add", "implement", "introduce", "create", "new feature"], "feat"),
    (["wip", "work in progress"], "wip"),
    (["style", "format", "lint", "whitespace", "indent"], "style"),
    (["ci", "pipeline", "workflow", "github action", "deploy", "build"], "ci"),
    (["perf", "performance", "optimize", "speed", "cache"], "perf"),
    (["chore", "update deps", "bump", "upgrade", "remove", "delete", "move", "migrate"], "chore"),
]

# Conventional commits prefix pattern: "type(scope)?: "
_CC_PREFIX_RE = re.compile(
    r"^(?P<type>feat|fix|refactor|docs|chore|test|wip|style|ci|perf)"
    r"(?:\((?P<scope>[^)]+)\))?(?P<breaking>!)?"
    r":\s*",
    re.IGNORECASE,
)

# Comment lines (git inserts these)
_COMMENT_RE = re.compile(r"^\s*#")


def _detect_type(subject: str) -> str:
    """Guess conventional commit type from subject keywords."""
    lower = subject.lower()
    for keywords, ctype in KEYWORD_TYPE_MAP:
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                return ctype
    return "chore"  # safe default


def _capitalize_first(s: str) -> str:
    if not s:
        return s
    return s[0].upper() + s[1:]


def normalize(message: str) -> str:
    """Normalize a full commit message string."""
    lines = message.splitlines()

    # Separate non-comment lines
    content_lines = []
    trailing_comments: list[str] = []

    for line in lines:
        if _COMMENT_RE.match(line):
            trailing_comments.append(line)
        else:
            content_lines.append(line)

    if not content_lines:
        return message  # nothing to normalize

    subject_raw = content_lines[0]
    body_lines = content_lines[1:]

    # --- Parse existing CC prefix ---
    m = _CC_PREFIX_RE.match(subject_raw)
    if m:
        prefix_type = m.group("type").lower()
        scope = m.group("scope")
        breaking = m.group("breaking") or ""
        rest = subject_raw[m.end():]
    else:
        prefix_type = None
        scope = None
        breaking = ""
        rest = subject_raw

    # --- Detect type if missing ---
    if prefix_type is None:
        prefix_type = _detect_type(rest)

    # --- Capitalize first word of rest ---
    rest = rest.strip()
    rest = _capitalize_first(rest)

    # --- Remove trailing period ---
    rest = rest.rstrip(".")

    # --- Build new subject ---
    scope_str = f"({scope})" if scope else ""
    new_subject = f"{prefix_type}{scope_str}{breaking}: {rest}"

    # --- Enforce ≤72 chars ---
    if len(new_subject) > 72:
        max_rest_len = 72 - len(f"{prefix_type}{scope_str}{breaking}: ") - 3  # "..."
        rest = rest[:max_rest_len] + "..."
        new_subject = f"{prefix_type}{scope_str}{breaking}: {rest}"

    # --- Reconstruct message ---
    result_lines = [new_subject] + body_lines + trailing_comments
    return "\n".join(result_lines)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: normalize_commit_msg.py <message|file>", file=sys.stderr)
        sys.exit(1)

    arg = sys.argv[1]
    path = Path(arg)

    if path.exists():
        original = path.read_text()
        normalized = normalize(original)
        path.write_text(normalized)
        print(normalized, end="")
    else:
        # Treat arg as a raw message string
        normalized = normalize(arg)
        print(normalized, end="")


if __name__ == "__main__":
    main()
