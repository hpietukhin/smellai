from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage
from smellai_datasets.schema import EvalSample

SWE_SAMPLE = EvalSample(
    source="swe",
    sample_id="swe:abc123",
    inputs={
        "project_name": "checkstyle",
        "commit_id": "abc123",
        "refactoring_type": "Extract Method",
        "file_path_before": "src/Foo.java",
        "file_path_after": "src/Foo.java",
        "class_before": "class Foo { void longMethod() {} }",
        "source_before": "void longMethod() {}",
        "jdk_version": 11,
        "compile_command": "mvn compile",
    },
    expectations={
        "class_after": "class Foo { void extracted() {} void longMethod() {} }",
        "source_after": "void longMethod() { extracted(); }",
    },
    tags={"is_pure": True, "is_compound": False, "has_tests": False},
)


class _FakeLLM:
    last_messages = None

    def __init__(self, model: str | None = None):
        self.model = model

    def invoke(self, messages):
        _FakeLLM.last_messages = messages
        return AIMessage(
            content="""```java\nclass Foo { void longMethod() {} }\n```""",
            response_metadata={"token_usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        )


def test_swe_agent_passes_tool_nudge_to_llm(tmp_path, monkeypatch):
    import agents.swe_eval.agent as mod

    # Patch LLM + runtime boundaries so invoke reaches A5_generate safely.
    monkeypatch.setattr(mod, "ChatLiteLLM", _FakeLLM)
    monkeypatch.setattr(
        mod,
        "setup_project_workspace",
        lambda record, workspace: SimpleNamespace(project_path=tmp_path, error=None),
    )
    monkeypatch.setattr(
        mod,
        "verify_refactoring",
        lambda record, project_path, refactored_code, refactored_target_code=None: SimpleNamespace(
            compile_success=True,
            test_success=True,
            error=None,
        ),
    )

    agent = mod.create_swe_eval_agent(enable_composite=False)
    out = mod.invoke_agent(agent, SWE_SAMPLE, workspace_path=tmp_path)

    assert out["compile_success"] is True
    assert _FakeLLM.last_messages is not None
    assert _FakeLLM.last_messages[0]["role"] == "system"

    system_prompt = _FakeLLM.last_messages[0]["content"]
    assert "run_spoon_refactor" in system_prompt
    assert "run_ast_grep_rewrite_git_safe" in system_prompt
    assert "replace_in_file_git_safe" in system_prompt
