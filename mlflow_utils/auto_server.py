"""Automatic MLflow server management for workflows.

This module provides utilities to automatically start/manage MLflow server
when needed, so workflows can run without manual server setup.
"""

import re
from contextlib import contextmanager
from typing import Optional

import mlflow
from mlflow_utils.server import MLflowServer


def parse_tracking_uri(uri: str) -> tuple[bool, Optional[int]]:
    """Parse tracking URI to determine if it's a local HTTP server.

    Returns:
        Tuple of (is_local_http, port_number)
    """
    if not uri.startswith("http://"):
        return False, None

    # Match localhost or 127.0.0.1 with optional port
    match = re.match(r"http://(localhost|127\.0\.0\.1):(\d+)", uri)
    if match:
        return True, int(match.group(2))

    return False, None


@contextmanager
def ensure_mlflow_server(tracking_uri: str, backend_uri: str = "sqlite:///mlflow.db"):
    """Context manager that ensures MLflow server is running if needed.

    Args:
        tracking_uri: The tracking URI to use (e.g., "http://localhost:5000" or "sqlite:///mlflow.db")
        backend_uri: Backend storage URI if we need to start the server

    Yields:
        None

    Example:
        with ensure_mlflow_server("http://localhost:5000"):
            mlflow.set_tracking_uri("http://localhost:5000")
            # ... run MLflow operations ...
    """
    is_local_http, port = parse_tracking_uri(tracking_uri)

    if not is_local_http:
        # Not a local HTTP server, just use as-is (SQLite, remote server, etc.)
        yield
        return

    # It's a local HTTP server - ensure it's running
    server = MLflowServer(port=port, backend_uri=backend_uri)

    existing_pid = server.is_running()

    if not existing_pid:
        print(f"Starting MLflow server on port {port}...")
        server.start(background=True)

        if not server.wait_for_ready(timeout=30):
            raise RuntimeError(f"Failed to start MLflow server on port {port}")
    else:
        print(f"Using existing MLflow server on port {port} (PID: {existing_pid})")

    try:
        yield
    finally:
        # Note: We don't stop the server even if we started it, because:
        # 1. Other processes might be using it
        # 2. It's lightweight and can be reused for subsequent runs
        # 3. User can manually stop it if needed
        pass


def setup_mlflow_tracking(
    tracking_uri: str = "http://localhost:5000",
    backend_uri: str = "sqlite:///mlflow.db",
    experiment_name: Optional[str] = None,
    auto_start_server: bool = True,
) -> None:
    """Set up MLflow tracking with automatic server management.

    Args:
        tracking_uri: The tracking URI (default: http://localhost:5000)
        backend_uri: Backend storage if starting a new server (default: sqlite:///mlflow.db)
        experiment_name: Optional experiment name to set
        auto_start_server: Whether to automatically start server if needed (default: True)

    Example:
        setup_mlflow_tracking(
            tracking_uri="http://localhost:5000",
            experiment_name="my-experiment"
        )
    """
    if auto_start_server:
        is_local_http, port = parse_tracking_uri(tracking_uri)

        if is_local_http:
            server = MLflowServer(port=port, backend_uri=backend_uri)

            if not server.is_running():
                print(f"Starting MLflow server on port {port}...")
                server.start(background=True)

                if not server.wait_for_ready(timeout=30):
                    raise RuntimeError(f"Failed to start MLflow server on port {port}")
            else:
                existing_pid = server.is_running()
                print(f"Using existing MLflow server on port {port} (PID: {existing_pid})")

    print(f"Setting tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    if experiment_name:
        print(f"Setting experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)
