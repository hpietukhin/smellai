"""Java test analysis functions for pipeline stages A, D, and J.

Stages (per conf.tex pipeline):
  A - load source, detect build system (Maven or Gradle)
  D - run full test suite, record pre-refactoring baseline
  J - run test suite after refactoring, compare before/after metrics

The normal path is deterministic shell execution. When verification fails,
the agent can invoke a bounded LangGraph ReAct repair attempt, then retry.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import json
import logging
import os
import re
import shlex
import subprocess
import uuid
from contextlib import contextmanager

import httpx
import javalang  # type: ignore[import-untyped]
from dataclasses import asdict, dataclass, field
from functools import wraps
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Sequence, TypeVar, cast

from agents.java_test.java_version import (
    detect_maven_compiler_source as _detect_maven_compiler_source,
    enhance_maven_command as _enhance_maven_command,
    parse_bool_env as _parse_bool_env,
    reset_java_version_caches,
    sdkman_maven_command_env as _sdkman_maven_command_env,
    with_sdkman_maven_setup as _with_sdkman_maven_setup,
)
from agents.litellm_config import current_datetime_context, load_openrouter_env, make_openrouter_chat_model, normalize_openrouter_model
from agents.observability import start_action
from agents.tools.java_inspection_tools import JavaInspectorUnavailableError, method_at_line_value, validate_java_source_value
from agents.tools.java_test_tools import (
    TestRunSummary,
    detect_build_system,
    run_cmd_and_parse,
    run_tests,
)

LOGGER = logging.getLogger(__name__)

try:
    from langchain.agents import create_agent as create_langchain_agent
    from langchain.agents.middleware import AgentMiddleware, FilesystemFileSearchMiddleware, ToolCallLimitMiddleware
    from langchain_core.messages import ToolMessage
    from langchain_core.tools import tool as _langchain_tool
    from langchain_litellm import ChatLiteLLM
    from langgraph.prebuilt import create_react_agent
    from langgraph.prebuilt.tool_node import ToolCallRequest
except ImportError:  # pragma: no cover - optional dependency
    create_langchain_agent = None  # type: ignore[assignment]
    AgentMiddleware = object  # type: ignore[assignment,misc]
    FilesystemFileSearchMiddleware = None  # type: ignore[assignment,misc]
    ToolCallLimitMiddleware = None  # type: ignore[assignment,misc]
    ToolMessage = None  # type: ignore[assignment,misc]
    _langchain_tool = None  # type: ignore[assignment]
    ChatLiteLLM = None  # type: ignore[assignment,misc]
    create_react_agent = None  # type: ignore[assignment]
    ToolCallRequest = object  # type: ignore[assignment,misc]

DEFAULT_CODE_AGENT_MODEL = normalize_openrouter_model(os.environ.get(
    "JAVA_TEST_CODE_AGENT_MODEL", "google/gemini-2.0-flash-001"
))
DEFAULT_CODE_AGENT_STEP_LIMIT = int(os.environ.get("JAVA_TEST_CODE_AGENT_MAX_STEPS", "4"))
DEFAULT_CODE_AGENT_TIMEOUT = int(os.environ.get("JAVA_TEST_CODE_AGENT_TIMEOUT", "2"))
DEFAULT_CODE_AGENT_MAX_ATTEMPTS = int(os.environ.get("JAVA_TEST_CODE_AGENT_MAX_ATTEMPTS", "3"))
DEFAULT_CODE_AGENT_COST_LIMIT = float(os.environ.get("JAVA_TEST_CODE_AGENT_COST_LIMIT", "0"))
DEFAULT_CODE_AGENT_TEST_TIMEOUT = int(os.environ.get("JAVA_TEST_CODE_AGENT_TEST_TIMEOUT", "2"))
JAVA_TEST_COMPILE_GATE_TIMEOUT = int(os.environ.get("JAVA_TEST_COMPILE_GATE_TIMEOUT", "2"))
TOOL_OUTPUT_TAIL_CHARS = 2500
AST_GREP_TEXT_PREVIEW_CHARS = 240
AST_GREP_CONTEXT_PREVIEW_CHARS = 400
PROMPT_OUTPUT_EXCERPT_CHARS = 12000
OKHTTP_CONTEXT_EXCERPT_CHARS = 3000
DEFAULT_EXCERPT_CHARS = 2500

F = TypeVar("F", bound=Callable[..., object])
BuildSystem = Literal["maven", "gradle"]


def _load_java_test_env(env_file: str | Path = ".env") -> None:
    load_openrouter_env(str(env_file))
    os.environ.setdefault("OR_APP_NAME", "smellai-java-test-agent")


_load_java_test_env()


_CODE_AGENT_PROJECT: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "java_test_code_agent_project",
    default=None,
)


@dataclass
class JavaTestAgentCaches:
    """Mutable caches owned by the Java test agent module."""

    warmed_maven_projects: set[Path] = field(default_factory=set)


_CACHES = JavaTestAgentCaches()


class _CompilerSourceCacheCompat:
    """Compatibility shim for older tests that clear the former dict cache."""

    def clear(self) -> None:
        _detect_maven_compiler_source.cache_clear()


_MAVEN_COMPILER_SOURCE_CACHE = _CompilerSourceCacheCompat()


def _reset_java_test_agent_caches() -> None:
    """Clear module caches for tests and deterministic ad-hoc runs."""
    _CACHES.warmed_maven_projects.clear()
    reset_java_version_caches()


class WriteGuardMiddleware(AgentMiddleware):  # type: ignore[misc]
    """Reject write-tool calls whose path is outside the allowed repair scope."""

    GUARDED_TOOLS = {"write_file", "replace_in_file"}

    def __init__(self, repo_root: str, allowed_paths: set[str]) -> None:
        super().__init__()
        self.repo_root = Path(repo_root).resolve()
        self.allowed_paths = frozenset((self.repo_root / p).resolve() for p in allowed_paths)

    def _is_allowed(self, path_arg: str) -> bool:
        if not path_arg.strip():
            return False
        resolved = (self.repo_root / path_arg).resolve()
        if resolved != self.repo_root and self.repo_root not in resolved.parents:
            return False
        return resolved in self.allowed_paths

    def _allowed_display(self) -> str:
        allowed = sorted(str(path.relative_to(self.repo_root)) for path in self.allowed_paths)
        return ", ".join(allowed) if allowed else "<none>"

    def _blocked_message(self, request: ToolCallRequest, path_arg: str):  # type: ignore[no-untyped-def]
        if ToolMessage is None:
            raise PermissionError(f"Write rejected for {path_arg}; allowed: {self._allowed_display()}")
        return ToolMessage(
            content=(
                f"BLOCKED: write to '{path_arg}' denied. "
                f"Allowed repair files: {self._allowed_display()}. "
                "Choose a permitted file or explain that repair is impossible within scope."
            ),
            tool_call_id=str(request.tool_call.get("id", "")),
            name=str(request.tool_call.get("name", "")),
            status="error",
        )

    def _path_arg(self, request: ToolCallRequest) -> str:
        args = request.tool_call.get("args", {})
        return str(args.get("path", "")) if isinstance(args, dict) else ""

    def wrap_tool_call(self, request: ToolCallRequest, handler):  # type: ignore[no-untyped-def,override]
        tool_name = str(request.tool_call.get("name", ""))
        if tool_name not in self.GUARDED_TOOLS:
            return handler(request)

        path_arg = self._path_arg(request)
        if self._is_allowed(path_arg):
            return handler(request)
        return self._blocked_message(request, path_arg)

    async def awrap_tool_call(self, request: ToolCallRequest, handler):  # type: ignore[no-untyped-def,override]
        tool_name = str(request.tool_call.get("name", ""))
        if tool_name not in self.GUARDED_TOOLS:
            return await handler(request)

        path_arg = self._path_arg(request)
        if self._is_allowed(path_arg):
            return await handler(request)
        return self._blocked_message(request, path_arg)



def _code_agent_project() -> Path:
    project = _CODE_AGENT_PROJECT.get()
    if project is None:
        raise RuntimeError("Repair agent project context is not initialized")
    return project.resolve()


def _resolve_code_agent_path(path: str) -> Path:
    project = _code_agent_project()
    target = Path(path)
    if not target.is_absolute():
        target = project / path
    target = target.resolve()
    if project not in (target, *target.parents):
        raise ValueError(f"Path escapes project root: {path}")
    return target


def _tool_log_context(tool_name: str, **fields: object) -> dict[str, object]:
    project = _CODE_AGENT_PROJECT.get()
    context: dict[str, object] = {
        "tool": tool_name,
        "project": str(project) if project is not None else "<unset>",
        **fields,
    }
    LOGGER.info("java_test repair tool call: %s", context)
    return context


def _log_tool_success(context: dict[str, object], result: str) -> None:
    LOGGER.info(
        "java_test repair tool success: tool=%s project=%s result_excerpt=%s",
        context.get("tool"),
        context.get("project"),
        result[:500].replace("\n", " "),
    )


def _log_tool_error(context: dict[str, object], exc: Exception) -> None:
    LOGGER.warning(
        "java_test repair tool error: tool=%s project=%s error=%s: %s",
        context.get("tool"),
        context.get("project"),
        type(exc).__name__,
        exc,
    )


def _read_file_impl(path: str) -> str:
    context = _tool_log_context("read_file", path=path)
    try:
        target = _resolve_code_agent_path(path)
        context["resolved_path"] = str(target)
        if not target.exists():
            raise FileNotFoundError(path)
        result = target.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError, RuntimeError) as exc:
        _log_tool_error(context, exc)
        raise
    _log_tool_success(context, f"read {len(result)} chars")
    return result


def _validate_java_source_impl(source: str) -> str:
    context = _tool_log_context("validate_java_source", source_chars=len(source))
    try:
        try:
            result = validate_java_source_value(source)
        except (JavaInspectorUnavailableError, httpx.HTTPError, OSError, RuntimeError) as inspector_exc:
            LOGGER.info("Java inspector validation unavailable; falling back to javalang: %s", inspector_exc)
            javalang.parse.parse(source)
            message = "valid Java source (javalang fallback)"
            _log_tool_success(context, message)
            return message
        if result.valid:
            message = "valid Java source"
            _log_tool_success(context, message)
            return message
        raise ValueError(result.error or "Java inspector rejected source")
    except javalang.parser.JavaSyntaxError as exc:
        error = ValueError(f"invalid Java syntax: {exc.description} at {exc.at}")
        _log_tool_error(context, error)
        raise error
    except javalang.tokenizer.LexerError as exc:
        error = ValueError(f"invalid Java lexer token: {exc}")
        _log_tool_error(context, error)
        raise error


def _ensure_valid_java_before_write(target: Path, content: str) -> None:
    if target.suffix == ".java":
        _validate_java_source_impl(content)


def _write_file_impl(path: str, content: str) -> str:
    context = _tool_log_context("write_file", path=path, content_chars=len(content))
    try:
        target = _resolve_code_agent_path(path)
        context["resolved_path"] = str(target)
        _ensure_valid_java_before_write(target, content)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        result = f"wrote {target.relative_to(_code_agent_project())} ({len(content)} chars)"
    except (OSError, ValueError, RuntimeError) as exc:
        _log_tool_error(context, exc)
        raise
    _log_tool_success(context, result)
    return result


def _replace_in_file_impl(path: str, old: str, new: str) -> str:
    context = _tool_log_context(
        "replace_in_file",
        path=path,
        old_chars=len(old),
        new_chars=len(new),
    )
    try:
        target = _resolve_code_agent_path(path)
        context["resolved_path"] = str(target)
        text = target.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count != 1:
            raise ValueError(f"Expected exactly one occurrence, found {count}")
        updated = text.replace(old, new)
        _ensure_valid_java_before_write(target, updated)
        target.write_text(updated, encoding="utf-8")
        result = f"replaced text in {target.relative_to(_code_agent_project())}"
    except (OSError, ValueError, RuntimeError) as exc:
        _log_tool_error(context, exc)
        raise
    _log_tool_success(context, result)
    return result


def _run_maven_impl(maven_args: str) -> str:
    project = _CODE_AGENT_PROJECT.get()
    context = _tool_log_context("run_maven", maven_args=maven_args)
    if project is None:
        missing_context = RuntimeError("Repair agent project context is not initialized")
        _log_tool_error(context, missing_context)
        raise missing_context
    if any(token in maven_args for token in [";", "&&", "||", "`", "$("]):
        unsafe_args = ValueError("Shell control operators are not allowed; pass Maven arguments only")
        _log_tool_error(context, unsafe_args)
        raise unsafe_args
    cmd = shlex.split(_enhance_maven_command(f"mvn {maven_args}"))
    try:
        result = subprocess.run(
            cmd,
            cwd=project,
            env=_sdkman_maven_command_env(),
            capture_output=True,
            text=True,
            timeout=DEFAULT_CODE_AGENT_TEST_TIMEOUT,
            check=False,
        )
    except OSError as exc:
        _log_tool_error(context, exc)
        return f"exit=1\nfailed to execute maven: {exc}"
    output = f"exit={result.returncode}\n{result.stdout[-TOOL_OUTPUT_TAIL_CHARS:]}{result.stderr[-TOOL_OUTPUT_TAIL_CHARS:]}"
    _log_tool_success(context, output)
    return output


def _run_test_impl(test_class: str) -> str:
    return _run_maven_impl(f"test -Dtest={test_class}")


def _list_java_files_impl(subdir: str = "") -> str:
    base = _resolve_code_agent_path(subdir or ".")
    if not base.exists():
        raise FileNotFoundError(subdir)
    root = _code_agent_project()
    files = base.glob("**/*.java") if base.is_dir() else [base]
    return "\n".join(
        str(f.relative_to(root))
        for f in files
        if f.is_file() and "target" not in f.parts
    )


def _ast_grep_search_impl(pattern: str, path: str = ".", max_matches: int = 20) -> str:
    """Run read-only ast-grep structural search in the checkout."""
    target = _resolve_code_agent_path(path or ".")
    if not target.exists():
        raise FileNotFoundError(path)
    capped = max(1, min(max_matches, 50))
    result = subprocess.run(
        ["sg", "-p", pattern, "--lang", "java", "--json=compact", str(target)],
        cwd=_code_agent_project(),
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )
    if result.returncode not in {0, 1}:
        return f"ast-grep failed: {result.stderr[-1000:]}"
    try:
        matches = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return result.stdout[-4000:]
    if not isinstance(matches, list):
        return "[]"
    rows: list[str] = []
    root = _code_agent_project()
    for item in matches[:capped]:
        if not isinstance(item, dict):
            continue
        file_name = str(item.get("file", ""))
        try:
            rel = str(Path(file_name).resolve().relative_to(root))
        except (OSError, ValueError):
            rel = file_name
        start = item.get("range", {}).get("start", {}) if isinstance(item.get("range"), dict) else {}
        line = int(start.get("line", 0)) + 1 if isinstance(start, dict) else 0
        text = str(item.get("text", item.get("lines", ""))).strip().replace("\n", " ")
        rows.append(f"{rel}:{line}: {text[:AST_GREP_TEXT_PREVIEW_CHARS]}")
    if len(matches) > capped:
        rows.append(f"... {len(matches) - capped} more matches omitted")
    return "\n".join(rows) if rows else "no matches"


def _java_method_at_line_impl(path: str, line: int) -> str:
    context = _tool_log_context("java_method_at_line", path=path, line=line)
    try:
        result = method_at_line_value(path, line)
    except (JavaInspectorUnavailableError, httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
        _log_tool_error(context, exc)
        raise
    if result.error:
        error = ValueError(result.error)
        _log_tool_error(context, error)
        raise error
    if result.start_line is None or result.end_line is None:
        return "no enclosing method/class found"
    message = f"{result.kind} {result.name or '<anonymous>'} lines {result.start_line}-{result.end_line}"
    _log_tool_success(context, message)
    return message


def _ast_grep_context_at_line_impl(path: str, line: int) -> str:
    """Return enclosing Java method/class near a 1-based line using inspector, then ast-grep fallback."""
    target = _resolve_code_agent_path(path)
    if not target.exists():
        raise FileNotFoundError(path)
    if line < 1:
        raise ValueError("line must be 1-based")
    try:
        return _java_method_at_line_impl(path, line)
    except (OSError, RuntimeError, ValueError):
        LOGGER.info("Java inspector method_at unavailable; falling back to ast-grep context")
    patterns = [
        "$RET $METHOD($$$ARGS) { $$$BODY }",
        "class $C { $$$BODY }",
        "interface $C { $$$BODY }",
        "enum $C { $$$BODY }",
    ]
    contexts: list[tuple[int, int, str]] = []
    for pattern in patterns:
        result = subprocess.run(
            ["sg", "-p", pattern, "--lang", "java", "--json=compact", str(target)],
            cwd=_code_agent_project(),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode not in {0, 1}:
            continue
        try:
            matches = json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            continue
        if not isinstance(matches, list):
            continue
        for item in matches:
            if not isinstance(item, dict):
                continue
            span = item.get("range", {})
            if not isinstance(span, dict):
                continue
            start = span.get("start", {})
            end = span.get("end", {})
            if not isinstance(start, dict) or not isinstance(end, dict):
                continue
            start_line = int(start.get("line", 0)) + 1
            end_line = int(end.get("line", 0)) + 1
            if start_line <= line <= end_line:
                text = str(item.get("text", item.get("lines", ""))).strip().replace("\n", " ")
                contexts.append((start_line, end_line, text[:AST_GREP_CONTEXT_PREVIEW_CHARS]))
    if not contexts:
        return "no enclosing method/class found"
    contexts.sort(key=lambda row: (row[1] - row[0], row[0]))
    return "\n".join(f"lines {start}-{end}: {text}" for start, end, text in contexts[:3])


def _repair_tool(fn: F) -> F:
    """Decorate a local repair tool when LangChain is installed, else stub it."""
    if _langchain_tool is not None:
        return _langchain_tool(fn)  # type: ignore[return-value]

    @wraps(fn)
    def stub(*args: object, **kwargs: object) -> object:
        raise RuntimeError("langchain tools are not installed")

    return stub  # type: ignore[return-value]


@_repair_tool
def read_file(path: str) -> str:
    """Read a source file from the Java checkout."""
    return _read_file_impl(path)


@_repair_tool
def write_file(path: str, content: str) -> str:
    """Write a complete file inside the Java checkout."""
    return _write_file_impl(path, content)


@_repair_tool
def replace_in_file(path: str, old: str, new: str) -> str:
    """Replace one exact text occurrence inside a file."""
    return _replace_in_file_impl(path, old, new)


@_repair_tool
def validate_java_source_text(source: str) -> str:
    """Validate complete Java source before writing it to a .java file."""
    return _validate_java_source_impl(source)


@_repair_tool
def list_java_files(subdir: str = "") -> str:
    """List Java files in the project. Optionally pass a relative subdirectory."""
    return _list_java_files_impl(subdir)


@_repair_tool
def run_maven(maven_args: str) -> str:
    """Run Maven with SDKMAN Java/Maven setup. Pass Maven args only, e.g. 'test -Dtest=FooTest'."""
    return _run_maven_impl(maven_args)


@_repair_tool
def run_test(test_class: str) -> str:
    """Run a Maven test class and return compact logs."""
    return _run_test_impl(test_class)


@_repair_tool
def ast_grep_search(pattern: str, path: str = ".", max_matches: int = 20) -> str:
    """Read-only Java structural search using ast-grep. Pattern examples: 'class $C { $$$ }', '$RET $M($$$ARGS) { $$$BODY }'."""
    return _ast_grep_search_impl(pattern, path, max_matches)


@_repair_tool
def ast_grep_context_at_line(path: str, line: int) -> str:
    """Read-only structural context for a Java file and 1-based line using the Java inspector, with ast-grep fallback."""
    return _ast_grep_context_at_line_impl(path, line)


@_repair_tool
def java_method_at_line(path: str, line: int) -> str:
    """Read-only Java inspector lookup for the enclosing method/class at a 1-based line."""
    return _java_method_at_line_impl(path, line)


@contextmanager
def _code_agent_context(project: Path) -> Iterator[None]:
    token = _CODE_AGENT_PROJECT.set(project)
    try:
        yield
    finally:
        _CODE_AGENT_PROJECT.reset(token)


@dataclass(frozen=True)
class MavenSetupCommand:
    """Project-specific Maven command curated in evals/maven_setup.md."""

    repo_slug: str
    command: str
    notes: str = ""


# Curated from evals/maven_setup.md, intentionally limited to the
# full-ready/low-friction Maven repos used by the eval candidate selector.
# Other Maven repos in that document are explicitly weak/problematic and should
# fall back to generic detection rather than receive special handling here.
MAVEN_SETUP_COMMANDS: dict[str, MavenSetupCommand] = {
    "phicode/philib": MavenSetupCommand("phicode/philib", "mvn test", "best candidate; real TestNG suite"),
    "jhalterman/lyra": MavenSetupCommand(
        "jhalterman/lyra",
        "mvn -Dtest=RetryPolicyTest,ChannelClosureTest,ConnectionFactoryInvocationTest,ChannelHandlerTest,ConnectionFactoryRecoveryTest,InterruptableWaiterTest,ConnectionRecoveryTest,RetryStatsTest,ChannelConfigTest,ChannelInvocationTest,ConnectionClosureTest,ReentrantCircuitTest,ConnectionInvocationTest test",
        "stable partial TestNG suite for historical eval commits; excludes ChannelRecoveryTest",
    ),
    "tupilabs/tap4j": MavenSetupCommand("tupilabs/tap4j", "mvn clean test", "Travis install command"),
    "junit-team/junit4": MavenSetupCommand("junit-team/junit4", "mvn test", "lower-friction test validation"),
    "square/okhttp": MavenSetupCommand(
        "square/okhttp",
        "mvn -Dalpn.jdk8.version=8.1.13.v20181017 test",
        "unmodified checkout needs explicit ALPN property on modern JDK8",
    ),
}


def _normalise_repo_slug(text: str) -> str | None:
    """Extract owner/repo from common GitHub remote URL forms."""
    raw = text.strip()
    if not raw:
        return None
    raw = raw.removesuffix(".git")

    patterns = [
        r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+)$",
        r"^(?P<owner>[^/]+)/(?P<repo>[^/]+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            return f"{match.group('owner')}/{match.group('repo')}".lower()
    return None


def _detect_repo_slug(project_path: str) -> str | None:
    """Detect GitHub owner/repo for a checkout, with path-name fallback."""
    project = Path(project_path)
    try:
        result = subprocess.run(
            ["git", "-C", str(project), "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            slug = _normalise_repo_slug(result.stdout)
            if slug:
                return slug
    except (OSError, subprocess.SubprocessError):
        LOGGER.debug("Could not read git remote for %s", project, exc_info=True)

    # Fallback for temporary fixtures/worktrees without a configured remote.
    leaf = project.name.lower()
    for slug in MAVEN_SETUP_COMMANDS:
        owner, repo = slug.split("/", 1)
        if leaf in {repo.lower(), slug.replace("/", "_")}:
            return slug
    return None


def _maven_setup_command_for(project_path: str) -> MavenSetupCommand | None:
    slug = _detect_repo_slug(project_path)
    if not slug:
        return None
    return MAVEN_SETUP_COMMANDS.get(slug)


def _dependency_warmup_command() -> str:
    return "dependency:go-offline -q"


def _maybe_warm_maven_dependencies(project_path: Path, timeout: int = 2) -> None:
    if not _parse_bool_env(os.getenv("JAVA_TEST_MAVEN_WARM_DEPS"), default=True):
        return
    if (not project_path.is_dir()):
        return
    marker = project_path.resolve()
    if marker in _CACHES.warmed_maven_projects:
        return
    command = _enhance_maven_command(f"mvn {_dependency_warmup_command()}", allow_test_optimizations=False)
    run_cmd_and_parse(
        ["bash", "-lc", _with_sdkman_maven_setup(command)],
        marker,
        "maven",
        timeout=max(timeout, JAVA_TEST_COMPILE_GATE_TIMEOUT),
    )
    _CACHES.warmed_maven_projects.add(marker)


def _java_version_prompt_context(project_path: str) -> str:
    """Compatibility wrapper around shared Java-version detection for tests/patching."""
    compiler_source = _detect_maven_compiler_source(project_path)
    if compiler_source is None:
        return "Determine Java version for project: maven.compiler.source could not be determined from Maven help:evaluate."
    return "\n".join(
        [
            f"Determine Java version for project: Maven reports maven.compiler.source={compiler_source}.",
            f"Java compiler source level: {compiler_source}.",
            f"Do not use Java language features newer than source level {compiler_source}.",
        ]
    )


def _combined_output(summary: TestRunSummary) -> str:
    return "\n".join(x for x in [summary.stdout, summary.stderr] if x)


def _read_excerpt(path: Path, *, max_chars: int = DEFAULT_EXCERPT_CHARS) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return f"<unreadable: {exc}>"
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n...\n{tail}"


def _extract_markdown_section(path: Path, heading: str) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return f"<missing markdown section: {exc}>"
    start = None
    for i, line in enumerate(lines):
        if line.strip() == heading.strip():
            start = i
            break
    if start is None:
        return ""
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("### "):
            end = i
            break
    return "\n".join(lines[start:end]).strip()


def _repair_prompt_context(project_path: str) -> str:
    project = Path(project_path).resolve()
    slug = _detect_repo_slug(str(project)) or ""
    sections: list[str] = [
        "## Java language level\n" + _java_version_prompt_context(str(project))
    ]

    if slug == "square/okhttp":
        maven_setup = _extract_markdown_section(
            Path("evals/maven_setup.md"), "### 9. OkHttp (`square/okhttp`) — interesting but more complex candidate"
        )
        if maven_setup:
            sections.append(f"## Relevant evals/maven_setup.md section\n{maven_setup}")

        old_platform = project / "okhttp/src/main/java/com/squareup/okhttp/internal/Platform.java"
        old_url_test = project / "okhttp-tests/src/test/java/com/squareup/okhttp/internal/http/URLConnectionTest.java"
        new_platform = project / "okhttp/src/main/java/okhttp3/internal/platform/Jdk9Platform.java"
        new_ssl = project / "mockwebserver/src/main/java/okhttp3/internal/tls/SslClient.java"
        sections.append(
            "## Layout facts\n"
            f"- old Platform.java exists: {old_platform.exists()} ({old_platform})\n"
            f"- old URLConnectionTest.java exists: {old_url_test.exists()} ({old_url_test})\n"
            f"- newer Jdk9Platform.java exists: {new_platform.exists()} ({new_platform})\n"
            f"- newer SslClient.java exists: {new_ssl.exists()} ({new_ssl})"
        )
        if old_platform.exists():
            sections.append(
                "## Platform.java excerpt\n```java\n"
                + _read_excerpt(old_platform, max_chars=OKHTTP_CONTEXT_EXCERPT_CHARS)
                + "\n```"
            )
        if old_url_test.exists():
            sections.append(
                "## URLConnectionTest excerpt\n```java\n"
                + _read_excerpt(old_url_test, max_chars=OKHTTP_CONTEXT_EXCERPT_CHARS)
                + "\n```"
            )
        pom = project / "pom.xml"
        if pom.exists():
            sections.append("## pom.xml excerpt\n```xml\n" + _read_excerpt(pom, max_chars=OKHTTP_CONTEXT_EXCERPT_CHARS) + "\n```")

    return "\n\n".join(s for s in sections if s)


def _code_agent_task(project_path: str, summary: TestRunSummary, verification_command: str) -> str:
    output_excerpt = _combined_output(summary)[-PROMPT_OUTPUT_EXCERPT_CHARS:]
    failed_tests = [
        f"{item.name}: {item.status}"
        for item in summary.tests
        if item.status in {"FAIL", "ERROR", "FAILURE"}
    ]
    failing = "\n".join(failed_tests)
    context = _repair_prompt_context(project_path)
    return f"""Fix this Java test failure with minimal changes.

Project root: {project_path}
Verification command:
{verification_command}

Failing tests:
{failing}

Current output excerpt:
{output_excerpt}

{context}

Rules:
- Act as a repair bot, not a refactoring bot: fix only what makes the verification command fail.
- Make only minimal compatibility/build/test-fix changes.
- Respect the detected Java language level; do not introduce newer syntax than the Maven compiler source level.
- You may edit pom.xml, Java sources, and tests when needed.
- Prefer read_file/list_java_files + replace_in_file for surgical edits; use write_file only when replacing/creating a whole file is safer.
- write_file and replace_in_file validate complete .java files before writing; use validate_java_source_text explicitly when unsure.
- Local tools resolve relative paths from the project root. Prefer local read_file/replace_in_file/write_file for edits.
- MCP tools are prefixed (for example filesystem__read_file) and run with cwd set to the project root; pass project-relative or absolute paths inside the checkout.
- Run the provided test command or the closest failing test through run_maven/run_test when needed.
- Return a complete patch strategy and apply edits through tool calls.
"""


def _code_agent_compile_task(project_path: str, summary: TestRunSummary, verification_command: str) -> str:
    output_excerpt = _combined_output(summary)[-PROMPT_OUTPUT_EXCERPT_CHARS:]
    context = _repair_prompt_context(project_path)
    return f"""Fix this Java compilation failure with minimal edits.

Project root: {project_path}
Verification command:
{verification_command}

Current output excerpt:
{output_excerpt}

{context}

Rules:
- Act as a compile-only repair bot: fix only what is required to make compilation pass.
- Do not change test logic unless it directly fixes a compile error.
- Respect the detected Java language level; do not introduce newer syntax than the Maven compiler source level.
- Prefer read_file + replace_in_file for minimal changes.
- write_file and replace_in_file reject invalid .java content before writing.
- Avoid running full test suites unless compilation succeeds.
- Use run_maven for local checks and return exact edits.
"""

def _make_tool_errors_observable(tools: Sequence[object]) -> list[object]:
    """Return tools configured to report failures to the agent instead of aborting."""
    for tool_obj in tools:
        if hasattr(tool_obj, "handle_tool_error"):
            tool_obj.handle_tool_error = True
        if hasattr(tool_obj, "handle_validation_error"):
            tool_obj.handle_validation_error = True
    return list(tools)


@dataclass(frozen=True)
class RepairAgentRun:
    """Structured result from a single LLM repair-agent invocation."""

    output: str
    steps: int
    tool_calls: int
    reached_limit: bool


async def _run_react_repair_agent(
    project: Path,
    task: str,
    *,
    model_name: str,
    step_limit: int,
    allowed_write_paths: set[str] | None = None,
) -> RepairAgentRun:
    local_tools = _make_tool_errors_observable([
        read_file,
        write_file,
        replace_in_file,
        validate_java_source_text,
        list_java_files,
        ast_grep_search,
        ast_grep_context_at_line,
        java_method_at_line,
        run_maven,
        run_test,
    ])
    model = make_openrouter_chat_model(model_name, temperature=0)
    system_prompt = (
        f"{current_datetime_context()}\n"
        "You are a minimal Java build repair agent. Fix only verification failures. "
        "Use local tools for edits. Use tools to inspect/edit files and rerun Maven. "
        "Do not refactor unrelated code. If a write tool returns BLOCKED, do not try "
        "to bypass it; choose an allowed file or explain that repair is impossible."
    )
    tools = local_tools
    if create_langchain_agent is not None:
        middleware: list[object] = []
        if FilesystemFileSearchMiddleware is not None:
            middleware.append(FilesystemFileSearchMiddleware(root_path=str(project), use_ripgrep=True, max_file_size_mb=5))
        if ToolCallLimitMiddleware is not None:
            middleware.append(ToolCallLimitMiddleware(run_limit=max(1, step_limit * 2), exit_behavior="continue"))
        if allowed_write_paths is not None:
            middleware.append(WriteGuardMiddleware(str(project), allowed_write_paths))
        result = await create_langchain_agent(
            model=model,
            tools=cast(Sequence[Any], tools),
            system_prompt=system_prompt,
            middleware=middleware,  # type: ignore[arg-type]
        ).ainvoke(
            {"messages": [{"role": "user", "content": task}]},
            config={"recursion_limit": max(4, step_limit * 2), "configurable": {"thread_id": uuid.uuid4().hex}},
        )
    else:
        result = await create_react_agent(
            model=model,
            tools=cast(Sequence[Any], tools),
            prompt=system_prompt,
        ).ainvoke(
            {"messages": [("user", task)]},
            config={"recursion_limit": max(4, step_limit * 2)},
            durability="async",
        )
    messages = result.get("messages", []) if isinstance(result, dict) else []
    output = "\n".join(str(getattr(m, "content", m)) for m in messages[-4:])
    tool_calls = sum(
        len(message_tool_calls)
        for message in messages
        if isinstance((message_tool_calls := getattr(message, "tool_calls", None)), list)
    )
    return RepairAgentRun(
        output=output,
        steps=len(messages),
        tool_calls=tool_calls,
        reached_limit="Sorry, need more steps to process this request." in output,
    )


def _git_changed_files(project: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(project), "status", "--short"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line[3:] for line in result.stdout.splitlines() if len(line) >= 4]


def _restore_git_paths(project: Path, paths: list[str]) -> bool:
    if not paths:
        return True
    result = subprocess.run(
        ["git", "checkout", "--", *paths],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )
    return result.returncode == 0


@dataclass
class RepairResult:
    """Stable dictionary-compatible result for checkout repair attempts."""

    attempted: bool = False
    applied: bool = False
    reverted: bool = False
    error: str | None = None
    changed_files: list[str] = field(default_factory=list)
    steps: int | None = None
    tool_calls: int = 0
    stopped_early: bool = False
    output_excerpt: str = ""
    token_usage: dict[str, object] = field(default_factory=dict)
    model: str = ""
    command: str = "LangChain create_agent + WriteGuardMiddleware + in-place execution"
    mcp_errors: list[str] = field(default_factory=list)
    attempt: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _run_async_from_sync(coro: object) -> object:
    """Run a coroutine from sync code, even if caller already owns an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future: concurrent.futures.Future[object] = executor.submit(asyncio.run, coro)  # type: ignore[arg-type]
        return future.result()


def _code_agent_repair_checkout(
    project_path: str,
    summary: TestRunSummary,
    *,
    verification_command: str,
    model_name: str | None = None,
    step_limit: int = DEFAULT_CODE_AGENT_STEP_LIMIT,
    cost_limit: float = DEFAULT_CODE_AGENT_COST_LIMIT,
    timeout: int = DEFAULT_CODE_AGENT_TIMEOUT,
    allowed_write_paths: set[str] | None = None,
    compile_mode: bool = False,
) -> dict:
    with start_action(
        action_type="code_agent_repair_checkout",
        summary_exit_code=summary.exit_code,
        timeout=timeout,
        step_limit=step_limit,
    ):
        del cost_limit  # compatibility only
        project = Path(project_path).resolve()
        normalized_model = normalize_openrouter_model(model_name or DEFAULT_CODE_AGENT_MODEL)
        if create_react_agent is None or ChatLiteLLM is None or _langchain_tool is None:
            return RepairResult(
                attempted=False,
                applied=False,
                error="LangGraph/LangChain repair dependencies are not installed",
                model=normalized_model,
            ).to_dict()

        task = _code_agent_compile_task(str(project), summary, verification_command) if compile_mode else _code_agent_task(str(project), summary, verification_command)
        before_files = set(_git_changed_files(project))
        try:
            with _code_agent_context(project):
                repair_run = cast(
                    RepairAgentRun,
                    _run_async_from_sync(
                        _run_react_repair_agent(
                            project,
                            task,
                            model_name=model_name or DEFAULT_CODE_AGENT_MODEL,
                            step_limit=max(1, step_limit),
                            allowed_write_paths=allowed_write_paths,
                        )
                    ),
                )
        except (OSError, RuntimeError, ValueError, asyncio.TimeoutError) as exc:
            return RepairResult(
                attempted=False,
                applied=False,
                error=str(exc),
                model=normalized_model,
            ).to_dict()

        after_files = set(_git_changed_files(project))
        changed_files = sorted(after_files - before_files)
        if allowed_write_paths is not None:
            outside_allowed = [path for path in changed_files if path not in allowed_write_paths]
            if outside_allowed:
                _restore_git_paths(project, outside_allowed)
                return RepairResult(
                    attempted=True,
                    applied=False,
                    reverted=True,
                    error=f"Repair touched files outside allowlist: {outside_allowed}",
                    steps=repair_run.steps,
                    tool_calls=repair_run.tool_calls,
                    stopped_early=True,
                    output_excerpt=repair_run.output[-PROMPT_OUTPUT_EXCERPT_CHARS:],
                    model=normalized_model,
                    command="LangChain create_agent + WriteGuardMiddleware + rollback_paths",
                ).to_dict()
        stopped_early = repair_run.reached_limit or (repair_run.tool_calls >= max(1, step_limit * 2) and not changed_files)
        return RepairResult(
            attempted=True,
            applied=bool(changed_files),
            changed_files=changed_files,
            steps=repair_run.steps,
            tool_calls=repair_run.tool_calls,
            stopped_early=stopped_early,
            output_excerpt=repair_run.output[-PROMPT_OUTPUT_EXCERPT_CHARS:],
            model=normalized_model,
        ).to_dict()


def _allowed_write_paths_from_targets(project_path: str, target_files: list[str] | None) -> set[str] | None:
    """Convert target files to project-relative write allowlist for repair.

    ``None`` means unrestricted legacy repair (used for baseline repair where no
    refactor target exists). A non-empty set means write tools may only touch
    those paths.
    """
    if target_files is None:
        return None
    project_root = Path(project_path).resolve()
    allowed: set[str] = set()
    for target_file in target_files:
        if not target_file:
            continue
        path = Path(target_file)
        if not path.is_absolute():
            path = project_root / path
        try:
            resolved = path.resolve()
            if resolved == project_root or project_root not in resolved.parents:
                continue
            allowed.add(resolved.relative_to(project_root).as_posix())
        except (OSError, ValueError):
            continue
    return allowed


def _resolve_targeted_maven_tests(
    project_path: str,
    target_files: list[str] | None,
) -> list[str]:
    """Derive Maven -Dtest class selectors from changed Java files.

    The heuristic prefers direct test files (under src/test/java) and then
    falls back to name-based matching when a production class changes.
    """
    if not target_files:
        return []

    project_root = Path(project_path)
    candidates: list[str] = []

    with start_action(
        action_type="resolve_targeted_maven_tests",
        project=str(project_root),
        target_file_count=len(target_files),
    ):
        def _add_target(target: str) -> None:
            target = target.strip()
            if not target or target in candidates:
                return
            candidates.append(target)

        def _to_test_class(file_path: Path) -> str | None:
            try:
                rel = file_path.relative_to(project_root)
            except ValueError:
                rel = file_path

            rel_posix = rel.as_posix()
            lower = rel_posix.lower()
            if not lower.endswith(".java"):
                return None

            stem = rel.stem
            if not stem.endswith("Test") and not stem.endswith("Tests"):
                return None

            marker = "src/test/java/"
            if marker in lower:
                class_part = rel_posix[lower.index(marker) + len(marker) : -5]
                return class_part.replace("/", ".")

            marker2 = "test/"
            if marker2 in lower:
                # Best-effort fallback for non-standard layouts.
                idx = lower.index(marker2)
                class_part = rel_posix[idx + len(marker2) : -5]
                return class_part.replace("/", ".")
            return stem

        def _find_tests_for_source_class(class_stem: str) -> list[str]:
            globs = (
                f"**/{class_stem}Test.java",
                f"**/{class_stem}Tests.java",
                f"**/*{class_stem}*Test*.java",
            )
            test_roots = (
                root_name
                for root_name in ("src/test/java", "test/java", "src/test", "test")
                if (project_root / root_name).exists()
            )
            test_files = (
                test_file
                for root_name, pattern in product(test_roots, globs)
                for test_file in (project_root / root_name).glob(pattern)
            )
            return list(dict.fromkeys(
                test_class
                for test_file in test_files
                if (test_class := _to_test_class(test_file)) is not None
            ))

        for target_file in target_files:
            if not target_file:
                continue
            file_path = Path(target_file)
            if not file_path.is_absolute():
                file_path = (project_root / file_path).resolve()
            direct = _to_test_class(file_path)
            if direct:
                _add_target(direct)
                continue

            if file_path.suffix != ".java":
                continue

            class_stem = file_path.stem
            for candidate in _find_tests_for_source_class(class_stem):
                _add_target(candidate)

        if not candidates:
            # If the heuristic produced no direct/derived tests, keep empty list and
            # let caller fall back to baseline setup/default path.
            pass

    return candidates


@dataclass
class JavaTestAnalysisResult:
    """Stable dictionary-compatible result for Java test analysis."""

    project_path: str
    build_system: BuildSystem | None
    summary: TestRunSummary | None
    command: str | None = None
    command_source: str = "default"
    verification_command: str | None = None
    code_agent_repair: dict[str, object] = field(default_factory=lambda: RepairResult().to_dict())
    pre_code_agent_exit_code: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "project_path": self.project_path,
            "build_system": self.build_system,
            "summary": self.summary,
            "command": self.command,
            "verification_command": self.verification_command,
            "command_source": self.command_source,
            "code_agent_repair": self.code_agent_repair,
            "pre_code_agent_exit_code": self.pre_code_agent_exit_code,
            "error": self.error,
        }


@dataclass(frozen=True)
class RepairConfig:
    """Configuration for bounded LLM repair attempts."""

    enabled: bool
    model: str | None
    step_limit: int
    max_attempts: int
    cost_limit: float
    timeout: int
    target_files: list[str] | None


@dataclass(frozen=True)
class JavaTestCommand:
    """Selected verification command for a Java test run."""

    source: str
    raw_command: str | None
    verification_command: str


@dataclass(frozen=True)
class JavaTestRunConfig:
    """Runtime configuration for Java build/test verification."""

    project_path: str
    build_system: BuildSystem
    clean: bool
    timeout: int
    setup_command: MavenSetupCommand | None
    targeted_tests: list[str]


def _select_java_test_command(config: JavaTestRunConfig) -> JavaTestCommand:
    if config.build_system != "maven":
        verification_command = "gradle clean test" if config.clean else "gradle test"
        return JavaTestCommand("default", None, verification_command)

    if config.targeted_tests:
        raw_command = f"mvn {'clean ' if config.clean else ''}-Dtest={','.join(config.targeted_tests)} test"
        return JavaTestCommand("targeted-maven-tests", raw_command, _with_sdkman_maven_setup(raw_command))

    if config.setup_command is not None:
        return JavaTestCommand("evals/maven_setup.md", config.setup_command.command, _with_sdkman_maven_setup(config.setup_command.command))

    raw_command = "mvn clean test" if config.clean else "mvn test"
    return JavaTestCommand("default", None, _with_sdkman_maven_setup(raw_command))


def _to_maven_compile_command(selected_raw_command: str, *, force_clean: bool = True) -> str:
    try:
        tokens = shlex.split(selected_raw_command)
    except ValueError:
        return "mvn -q -DskipTests compile"
    if not tokens or tokens[0] != "mvn":
        return "mvn -q -DskipTests compile"

    normalized: list[str] = ["mvn"]
    for token in tokens[1:]:
        if token.startswith("-Dtest="):
            continue
        if token == "test":
            continue
        if token == "compile":
            continue
        normalized.append(token)

    if force_clean and "clean" not in normalized:
        normalized.insert(1, "clean")

    if "compile" not in normalized:
        normalized.append("compile")
    return " ".join(normalized)


def _run_java_tests_once(config: JavaTestRunConfig, selected: JavaTestCommand, *, compile_gated: bool = False) -> TestRunSummary:
    if config.build_system != "maven":
        with start_action(action_type="java_test_run_once", command_source="default", build_system=config.build_system):
            return run_tests(config.project_path, config.build_system, clean=config.clean, timeout=config.timeout)

    _maybe_warm_maven_dependencies(Path(config.project_path), timeout=config.timeout)
    if selected.raw_command is not None:
        if compile_gated and selected.source == "targeted-maven-tests":
            compile_command = _to_maven_compile_command(selected.raw_command, force_clean=config.clean)
            with start_action(
                action_type="java_test_compile_once",
                command_source=selected.source,
                command=compile_command,
                clean=config.clean,
            ):
                compile_summary = run_cmd_and_parse(
                    ["bash", "-lc", _with_sdkman_maven_setup(compile_command)],
                    Path(config.project_path),
                    config.build_system,
                    timeout=max(JAVA_TEST_COMPILE_GATE_TIMEOUT, config.timeout),
                )
            if not compile_summary.success:
                return compile_summary

        with start_action(
            action_type="java_test_run_once",
            command_source=selected.source,
            command=selected.raw_command,
            clean=config.clean,
        ):
            return run_cmd_and_parse(
                ["bash", "-lc", _with_sdkman_maven_setup(selected.raw_command)],
                Path(config.project_path),
                config.build_system,
                timeout=config.timeout,
            )

    with start_action(action_type="java_test_run_once", command_source="default"):
        return run_tests(config.project_path, config.build_system, clean=config.clean, timeout=config.timeout)


def _attempt_code_agent_repair(
    config: JavaTestRunConfig,
    repair_config: RepairConfig,
    selected: JavaTestCommand,
    summary: TestRunSummary,
    *,
    attempt: int,
    compile_mode: bool,
) -> dict:
    effective_timeout = min(config.timeout, repair_config.timeout)
    with start_action(
        action_type="code_agent_repair_attempt",
        attempt=attempt,
        command_source=selected.source,
        timeout=effective_timeout,
        mode="compile_fix" if compile_mode else "full_fix",
    ):
        return _code_agent_repair_checkout(
            config.project_path,
            summary,
            verification_command=selected.verification_command,
            model_name=repair_config.model,
            step_limit=max(1, min(2, repair_config.step_limit)) if compile_mode else repair_config.step_limit,
            cost_limit=repair_config.cost_limit,
            timeout=effective_timeout,
            allowed_write_paths=_allowed_write_paths_from_targets(config.project_path, repair_config.target_files),
            compile_mode=compile_mode,
        )


def run_java_test_analysis(
    project_path: str,
    *,
    clean: bool = True,
    timeout: int = 2,
    enable_code_agent_repair: bool = True,
    code_agent_model: str | None = None,
    llm_repair_model: str | None = None,
    code_agent_step_limit: int = DEFAULT_CODE_AGENT_STEP_LIMIT,
    code_agent_max_attempts: int = DEFAULT_CODE_AGENT_MAX_ATTEMPTS,
    code_agent_cost_limit: float = DEFAULT_CODE_AGENT_COST_LIMIT,
    code_agent_timeout: int | None = None,
    target_files: list[str] | None = None,
) -> dict:
    """Detect build system and run Java tests (stages A + D/J).

    Args:
        project_path: Path to the Java project directory.
        clean: Whether to run clean before tests.
        timeout: Timeout in seconds for the test command.
        enable_code_agent_repair: Try a bounded LangGraph ReAct repair fallback.
        code_agent_model: Optional repair model override.
        llm_repair_model: Backward-compatible alias for code_agent_model.
        code_agent_step_limit: Max agent steps per repair attempt.
        code_agent_max_attempts: Max bounded repair attempts.
        code_agent_cost_limit: Deprecated compatibility option; ignored.
        code_agent_timeout: Per-attempt timeout (seconds) for repair attempt.
        target_files: Optional list of files changed before this test run.

    Returns:
        dict with keys: project_path, build_system, summary (TestRunSummary).
    """
    if code_agent_model is None:
        code_agent_model = llm_repair_model

    with start_action(
        action_type="run_java_test_analysis",
        project_path=project_path,
        clean=clean,
        timeout=timeout,
        target_file_count=len(target_files or []),
        enable_code_agent_repair=enable_code_agent_repair,
    ):
        with start_action(action_type="detect_build_system"):
            build_system = detect_build_system(project_path)
        if build_system is None:
            return JavaTestAnalysisResult(
                project_path=project_path,
                build_system=None,
                summary=None,
                error=f"No Java build system detected in {project_path}",
            ).to_dict()

        setup_command = _maven_setup_command_for(project_path) if build_system == "maven" else None
        targeted_tests = _resolve_targeted_maven_tests(project_path, target_files)
        run_config = JavaTestRunConfig(
            project_path=project_path,
            build_system=cast(BuildSystem, build_system),
            clean=clean,
            timeout=timeout,
            setup_command=setup_command,
            targeted_tests=targeted_tests,
        )
        resolved_code_agent_timeout = timeout if code_agent_timeout is None else code_agent_timeout
        repair_config = RepairConfig(
            enabled=enable_code_agent_repair,
            model=code_agent_model,
            step_limit=code_agent_step_limit,
            max_attempts=code_agent_max_attempts,
            cost_limit=code_agent_cost_limit,
            timeout=resolved_code_agent_timeout,
            target_files=target_files,
        )

        with start_action(
            action_type="select_java_test_path",
            build_system=build_system,
            targeted_request=bool(targeted_tests),
            has_setup_command=setup_command is not None,
        ):
            selected = _select_java_test_command(run_config)

        compile_gated = bool(target_files)
        summary = _run_java_tests_once(run_config, selected, compile_gated=compile_gated)

        code_agent_result = RepairResult().to_dict()
        pre_code_agent_exit_code = None
        if repair_config.enabled and not summary.success:
            compile_mode = summary.exit_code != 0 and summary.counts.total == 0
            max_attempts = 1 if compile_mode else max(1, repair_config.max_attempts)
            for attempt in range(1, max_attempts + 1):
                attempt_result = _attempt_code_agent_repair(
                    run_config,
                    repair_config,
                    selected,
                    summary,
                    attempt=attempt,
                    compile_mode=compile_mode,
                )
                attempt_result["attempt"] = attempt
                code_agent_result = attempt_result
                if not attempt_result.get("attempted"):
                    break
                if not attempt_result.get("applied"):
                    break
                pre_code_agent_exit_code = summary.exit_code
                summary = _run_java_tests_once(run_config, selected, compile_gated=compile_gated)
                if compile_mode or summary.success or attempt_result.get("stopped_early"):
                    break

        return JavaTestAnalysisResult(
            project_path=project_path,
            build_system=cast(BuildSystem, build_system),
            summary=summary,
            command=selected.raw_command,
            verification_command=selected.verification_command,
            command_source=selected.source,
            code_agent_repair=code_agent_result,
            pre_code_agent_exit_code=pre_code_agent_exit_code,
        ).to_dict()
