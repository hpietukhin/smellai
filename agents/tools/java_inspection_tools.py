"""HTTP-backed Java code inspection tools.

The preferred setup is one inspector service per checkout/worktree. Set
``JAVA_INSPECTOR_URL`` to the service URL, or use ``java_inspector_process``
when launching a per-case service from Python.
"""

from __future__ import annotations

import atexit
import logging
import os
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger(__name__)
DEFAULT_INSPECTOR_URL = "http://127.0.0.1:7070"
DEFAULT_STARTUP_TIMEOUT_SECONDS = 2.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 1.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 2.0

_INSPECTOR_URL_CONTEXT: ContextVar[str | None] = ContextVar("java_inspector_url", default=None)
_SHARED_CLIENTS: dict[str, JavaInspectorClient] = {}
_AVAILABILITY_BY_URL: dict[str, bool] = {}
_CLIENT_LOCK = threading.RLock()


class JavaInspectorUnavailableError(RuntimeError):
    """Raised when no healthy Java inspector is available for the active URL."""


class ResolveSmellLocationResult(BaseModel):
    """Result returned by the Java inspector resolver."""

    model_config = ConfigDict(populate_by_name=True)

    file: str | None = None
    line_range: list[int] | None = Field(default=None, alias="lineRange")
    matched_by: str = Field(default="unknown", alias="matchedBy")
    candidates: list[str] = Field(default_factory=list)
    error: str | None = None

    def to_tool_dict(self) -> dict[str, object]:
        """Return legacy camelCase dict expected by existing LangChain tools/tests."""
        return self.model_dump(by_alias=True, exclude_none=True)


class ParseJavaSourceResult(BaseModel):
    """Java source parser response."""

    valid: bool = False
    error: str | None = None
    package_name: str | None = Field(default=None, alias="package")


class MethodAtLineResult(BaseModel):
    """Enclosing method/class location returned by the Java inspector."""

    file: str | None = None
    name: str | None = None
    kind: str = "unknown"
    start_line: int | None = Field(default=None, alias="startLine")
    end_line: int | None = Field(default=None, alias="endLine")
    error: str | None = None


class JavaInspectorClient:
    """Typed synchronous client for the long-lived Java inspector service."""

    def __init__(self, base_url: str | None = None) -> None:
        url = (base_url or _inspector_url()).rstrip("/")
        self.base_url = url
        self._client = httpx.Client(
            base_url=url,
            timeout=httpx.Timeout(
                connect=DEFAULT_CONNECT_TIMEOUT_SECONDS,
                read=DEFAULT_REQUEST_TIMEOUT_SECONDS,
                write=DEFAULT_REQUEST_TIMEOUT_SECONDS,
                pool=1.0,
            ),
            transport=httpx.HTTPTransport(retries=1),
            event_hooks={"request": [_log_request], "response": [_log_response]},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> JavaInspectorClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def check_health(self) -> None:
        """Fail fast if the inspector is not reachable."""
        response = self._client.get("/health")
        response.raise_for_status()

    def wait_for_health(self, timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS) -> None:
        """Wait for startup without making each connection attempt slow."""
        started = time.monotonic()
        last_error: Exception | None = None
        while time.monotonic() - started <= timeout_seconds:
            try:
                self.check_health()
                return
            except (httpx.HTTPError, OSError) as exc:
                last_error = exc
                time.sleep(0.2)
        raise TimeoutError(f"Java inspector did not become healthy at {self.base_url}: {last_error}")

    def resolve_smell_location(self, smell_path: str, class_name: str | None = None) -> ResolveSmellLocationResult:
        params = {"path": smell_path}
        if class_name:
            params["class"] = class_name
        response = self._client.get("/resolve", params=params)
        response.raise_for_status()
        return ResolveSmellLocationResult.model_validate(response.json())

    def validate_java_source(self, source: str) -> ParseJavaSourceResult:
        response = self._client.post("/parse", json={"source": source})
        response.raise_for_status()
        return ParseJavaSourceResult.model_validate(response.json())

    def method_at_line(self, file: str, line: int) -> MethodAtLineResult:
        response = self._client.get("/method_at", params={"file": file, "line": line})
        response.raise_for_status()
        return MethodAtLineResult.model_validate(response.json())


@dataclass
class JavaInspectorHandle:
    """Owned Java inspector subprocess and client."""

    process: subprocess.Popen[str]
    url: str
    repo_path: Path
    client: JavaInspectorClient

    def __enter__(self) -> JavaInspectorHandle:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.client.close()
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)


def _inspector_url() -> str:
    context_url = _INSPECTOR_URL_CONTEXT.get()
    if context_url:
        return context_url.rstrip("/")
    return os.environ.get("JAVA_INSPECTOR_URL", DEFAULT_INSPECTOR_URL).rstrip("/")


def set_java_inspector_url(url: str | None) -> Token[str | None]:
    """Set the active inspector URL for the current context/thread."""
    normalized = url.rstrip("/") if url else None
    token = _INSPECTOR_URL_CONTEXT.set(normalized)
    if normalized:
        mark_java_inspector_availability(normalized, True)
    return token


def reset_java_inspector_url(token: Token[str | None]) -> None:
    """Restore the previous active inspector URL for the current context/thread."""
    _INSPECTOR_URL_CONTEXT.reset(token)


@contextmanager
def java_inspector_url_context(url: str | None) -> Iterator[None]:
    """Temporarily bind an inspector URL to the current execution context."""
    token = set_java_inspector_url(url)
    try:
        yield
    finally:
        reset_java_inspector_url(token)


def _log_request(request: httpx.Request) -> None:
    LOGGER.debug("java-inspector request: %s %s", request.method, request.url)


def _log_response(response: httpx.Response) -> None:
    """Log inspector responses without risking hook-time exceptions."""
    try:
        LOGGER.debug(
            "java-inspector response: status=%s url=%s",
            response.status_code,
            response.url,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        LOGGER.debug("Could not log Java inspector response: %s", exc)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _shared_inspector_client(url: str) -> JavaInspectorClient:
    with _CLIENT_LOCK:
        client = _SHARED_CLIENTS.get(url)
        if client is None:
            client = JavaInspectorClient(url)
            _SHARED_CLIENTS[url] = client
        return client


def close_shared_inspector_clients() -> None:
    """Close process-wide cached inspector clients."""
    with _CLIENT_LOCK:
        clients = list(_SHARED_CLIENTS.values())
        _SHARED_CLIENTS.clear()
        _AVAILABILITY_BY_URL.clear()
    for client in clients:
        client.close()


def mark_java_inspector_availability(url: str | None = None, available: bool = True) -> None:
    """Record known inspector availability for the active or supplied URL."""
    target_url = (url or _inspector_url()).rstrip("/")
    with _CLIENT_LOCK:
        _AVAILABILITY_BY_URL[target_url] = available


def java_inspector_available(url: str | None = None) -> bool:
    """Return cached availability, probing /health only on first use per URL."""
    target_url = (url or _inspector_url()).rstrip("/")
    with _CLIENT_LOCK:
        cached = _AVAILABILITY_BY_URL.get(target_url)
    if cached is not None:
        return cached
    try:
        _shared_inspector_client(target_url).check_health()
    except (httpx.HTTPError, OSError) as exc:
        LOGGER.info("Java inspector unavailable at %s: %s", target_url, exc)
        mark_java_inspector_availability(target_url, False)
        return False
    mark_java_inspector_availability(target_url, True)
    return True


def _available_inspector_client() -> JavaInspectorClient:
    url = _inspector_url()
    if not java_inspector_available(url):
        raise JavaInspectorUnavailableError(f"Java inspector is not available at {url}")
    return _shared_inspector_client(url)


def _mark_unavailable_on_http_error(url: str, exc: httpx.HTTPError) -> None:
    """Only blacklist the inspector for transport/pool failures, not HTTP status errors."""
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.ReadTimeout)):
        LOGGER.warning("Marking Java inspector unavailable at %s after transport error: %s", url, exc)
        mark_java_inspector_availability(url, False)
        return
    LOGGER.info("Transient Java inspector HTTP error at %s: %s", url, exc)


@contextmanager
def java_inspector_process(repo_path: Path, jar_path: Path) -> Iterator[JavaInspectorHandle]:
    """Start a per-checkout Java inspector and stop it on context exit."""
    repo = repo_path.resolve()
    jar = jar_path.resolve()
    if not jar.exists():
        raise FileNotFoundError(f"Java inspector jar not found: {jar}")
    port = _find_free_port()
    proc = subprocess.Popen(
        ["java", "-jar", str(jar), str(repo), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"
    client = JavaInspectorClient(url)
    handle = JavaInspectorHandle(process=proc, url=url, repo_path=repo, client=client)
    try:
        while proc.poll() is None:
            try:
                client.wait_for_health(DEFAULT_STARTUP_TIMEOUT_SECONDS)
                break
            except (httpx.HTTPError, OSError, TimeoutError):
                if proc.poll() is not None:
                    break
                raise
        if proc.poll() is not None:
            stderr = proc.stderr.read() if proc.stderr is not None else ""
            raise RuntimeError(f"Java inspector exited early with {proc.returncode}: {stderr[-1000:]}")
        mark_java_inspector_availability(url, True)
        yield handle
    finally:
        handle.__exit__(None, None, None)
        mark_java_inspector_availability(url, False)


def resolve_smell_location_value(smell_path: str, class_name: str | None = None) -> ResolveSmellLocationResult:
    """Resolve a smell location with typed return value for Python callers."""
    url = _inspector_url()
    try:
        return _available_inspector_client().resolve_smell_location(smell_path, class_name)
    except httpx.HTTPError as exc:
        _mark_unavailable_on_http_error(url, exc)
        raise


def validate_java_source_value(source: str) -> ParseJavaSourceResult:
    """Validate Java source with typed return value for Python callers."""
    url = _inspector_url()
    try:
        return _available_inspector_client().validate_java_source(source)
    except httpx.HTTPError as exc:
        _mark_unavailable_on_http_error(url, exc)
        raise


def method_at_line_value(file: str, line: int) -> MethodAtLineResult:
    """Return enclosing method/class at line with typed return value for Python callers."""
    url = _inspector_url()
    try:
        return _available_inspector_client().method_at_line(file, line)
    except httpx.HTTPError as exc:
        _mark_unavailable_on_http_error(url, exc)
        raise


@tool
def resolve_smell_location(smell_path: str, class_name: str | None = None) -> dict[str, object]:
    """Resolve an OrganicDetector Java smell path to the real source file."""
    try:
        return resolve_smell_location_value(smell_path, class_name).to_tool_dict()
    except Exception as exc:
        LOGGER.exception("Java inspector resolve request failed")
        return ResolveSmellLocationResult(
            file=None,
            lineRange=None,
            matchedBy="inspector_error",
            candidates=[],
            error=f"{type(exc).__name__}: {exc}",
        ).to_tool_dict()


@tool
def validate_java_source(source: str) -> dict[str, object]:
    """Parse Java source with the inspector to verify syntactic validity before writing."""
    try:
        return validate_java_source_value(source).model_dump(by_alias=True, exclude_none=True)
    except Exception as exc:
        LOGGER.exception("Java inspector parse request failed")
        return ParseJavaSourceResult(valid=False, error=f"{type(exc).__name__}: {exc}").model_dump(by_alias=True, exclude_none=True)


@tool
def find_method_at_line(file: str, line: int) -> dict[str, object]:
    """Return the enclosing method or class for a Java file and 1-based line."""
    try:
        return method_at_line_value(file, line).model_dump(by_alias=True, exclude_none=True)
    except Exception as exc:
        LOGGER.exception("Java inspector method_at request failed")
        return MethodAtLineResult(file=file, error=f"{type(exc).__name__}: {exc}").model_dump(by_alias=True, exclude_none=True)


atexit.register(close_shared_inspector_clients)
