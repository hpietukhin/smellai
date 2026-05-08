from __future__ import annotations

import httpx

from agents.tools import java_inspection_tools as tools


def test_log_response_never_accesses_elapsed() -> None:
    response = httpx.Response(200, request=httpx.Request("GET", "http://inspector/health"))

    tools._log_response(response)


def test_http_status_error_does_not_mark_inspector_unavailable() -> None:
    url = "http://inspector-status-error"
    tools.mark_java_inspector_availability(url, True)
    request = httpx.Request("GET", f"{url}/resolve")
    response = httpx.Response(400, request=request)
    exc = httpx.HTTPStatusError("bad request", request=request, response=response)

    tools._mark_unavailable_on_http_error(url, exc)

    assert tools.java_inspector_available(url) is True


def test_connect_error_marks_inspector_unavailable() -> None:
    url = "http://inspector-connect-error"
    tools.mark_java_inspector_availability(url, True)
    request = httpx.Request("GET", f"{url}/resolve")
    exc = httpx.ConnectError("connection refused", request=request)

    tools._mark_unavailable_on_http_error(url, exc)

    assert tools.java_inspector_available(url) is False
