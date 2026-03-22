"""Tests for scripts/normalize_commit_msg.py"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from normalize_commit_msg import normalize, _detect_type


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------

class TestDetectType:
    def test_add_keyword_maps_to_feat(self):
        assert _detect_type("add user authentication") == "feat"

    def test_fix_keyword_maps_to_fix(self):
        assert _detect_type("fix broken pipeline") == "fix"

    def test_refactor_keyword(self):
        assert _detect_type("refactor dataset loader") == "refactor"

    def test_docs_keyword(self):
        assert _detect_type("update readme for CI setup") == "docs"

    def test_test_keyword(self):
        assert _detect_type("add test coverage for agent") == "test"

    def test_wip_keyword(self):
        assert _detect_type("wip authentication flow") == "wip"

    def test_default_fallback_is_chore(self):
        assert _detect_type("some random change with no matching keyword") == "chore"


# ---------------------------------------------------------------------------
# Type prefix injection
# ---------------------------------------------------------------------------

class TestTypePrefixInjection:
    def test_injects_prefix_when_absent(self):
        result = normalize("add login endpoint")
        assert result.startswith("feat:")

    def test_preserves_existing_prefix(self):
        result = normalize("fix: broken import")
        assert result.startswith("fix:")

    def test_preserves_scope(self):
        result = normalize("feat(auth): add OAuth support")
        assert result.startswith("feat(auth):")

    def test_preserves_breaking_change_marker(self):
        result = normalize("feat!: drop Python 3.9")
        assert result.startswith("feat!:")

    def test_case_insensitive_prefix_normalised_to_lower(self):
        result = normalize("FIX: crash on startup")
        assert result.startswith("fix:")

    def test_fix_inferred_from_body_keyword(self):
        result = normalize("resolve null pointer in parser")
        assert result.startswith("fix:")

    def test_chore_inferred_for_remove(self):
        result = normalize("remove deprecated utils")
        assert result.startswith("chore:")


# ---------------------------------------------------------------------------
# Capitalization
# ---------------------------------------------------------------------------

class TestCapitalization:
    def test_capitalizes_first_word_after_prefix(self):
        result = normalize("fix: broken login")
        assert result == "fix: Broken login"

    def test_already_capitalised_unchanged(self):
        result = normalize("fix: Broken login")
        assert result == "fix: Broken login"

    def test_capitalizes_injected_prefix_rest(self):
        result = normalize("add new dataset loader")
        assert result.startswith("feat: Add")

    def test_capitalizes_after_scope(self):
        result = normalize("feat(db): add migration")
        assert "Add migration" in result


# ---------------------------------------------------------------------------
# Trailing period removal
# ---------------------------------------------------------------------------

class TestTrailingPeriod:
    def test_removes_trailing_period(self):
        result = normalize("fix: Remove trailing period.")
        assert not result.rstrip().endswith(".")

    def test_no_period_unchanged(self):
        result = normalize("fix: Remove trailing period")
        assert result == "fix: Remove trailing period"

    def test_ellipsis_not_stripped(self):
        # Ellipsis added by truncation should survive (it's "..." not ".")
        long_msg = "feat: " + "a" * 100
        result = normalize(long_msg)
        assert result.endswith("...")


# ---------------------------------------------------------------------------
# Subject line length
# ---------------------------------------------------------------------------

class TestSubjectLength:
    def test_short_subject_unchanged(self):
        msg = "fix: Short message"
        assert normalize(msg) == msg

    def test_long_subject_truncated_to_72(self):
        msg = "fix: " + "x" * 80
        result = normalize(msg)
        assert len(result.splitlines()[0]) <= 72

    def test_truncated_subject_ends_with_ellipsis(self):
        msg = "fix: " + "word " * 20
        result = normalize(msg)
        assert result.splitlines()[0].endswith("...")


# ---------------------------------------------------------------------------
# Multi-line message preservation
# ---------------------------------------------------------------------------

class TestMultilineMessages:
    def test_body_lines_preserved(self):
        msg = "fix: crash on init\n\nThis fixes the null ref error in loader.py"
        result = normalize(msg)
        assert "This fixes" in result

    def test_git_comment_lines_preserved(self):
        msg = "add feature\n# Please enter the commit message\n# Changes to be committed"
        result = normalize(msg)
        assert "# Please enter the commit message" in result
        assert result.startswith("feat:")

    def test_body_not_modified(self):
        msg = "fix: crash\n\nsome body text\nmore body"
        result = normalize(msg)
        lines = result.splitlines()
        assert lines[1] == ""
        assert lines[2] == "some body text"
        assert lines[3] == "more body"


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------

class TestWhitespace:
    def test_strips_leading_whitespace_in_subject_rest(self):
        result = normalize("fix:   extra spaces before")
        assert result == "fix: Extra spaces before"

    def test_preserves_blank_separator_line(self):
        msg = "feat: add thing\n\nbody here"
        result = normalize(msg)
        assert "\n\n" in result
