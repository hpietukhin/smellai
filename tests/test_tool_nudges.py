from agents.swe_eval.prompts import SYSTEM_PROMPT


def test_system_prompt_mentions_spoon_astgrep_and_git_safe_fallback():
    assert "run_spoon_refactor" in SYSTEM_PROMPT
    assert "run_ast_grep_rewrite_git_safe" in SYSTEM_PROMPT
    assert "replace_in_file_git_safe" in SYSTEM_PROMPT
