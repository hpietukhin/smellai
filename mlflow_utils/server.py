import subprocess
import time
import os
import signal
import requests
from typing import Optional


class MLflowServer:
    def __init__(
        self,
        port: int = 5000,
        backend_uri: str = "sqlite:///mlflow.db",
        host: str = "0.0.0.0",
    ):
        self.port = port
        self.backend_uri = backend_uri
        self.host = host

    def is_running(self) -> Optional[int]:
        """Checks if the port is in use and returns the PID."""
        try:
            # Use lsof to find the PID listening on the port
            output = (
                subprocess.check_output(
                    ["lsof", "-ti", f":{self.port}"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
            if not output:
                return None
            # Return the first PID if multiple are found
            return int(output.split("\n")[0])
        except subprocess.CalledProcessError:
            return None
        except FileNotFoundError:
            # Fallback if lsof is not available (e.g. some minimal containers), though macOS usually has it.
            # Could try pgrep -f "mlflow ui" but that's less precise on port.
            return None

    def start(self, background: bool = True) -> int:
        pid = self.is_running()
        if pid:
            print(f"MLflow UI is already running on port {self.port} (PID: {pid})")
            return pid

        print("Starting MLflow UI...")
        print(f"  Port: {self.port}")
        print(f"  Backend: {self.backend_uri}")

        cmd = [
            "mlflow",
            "ui",
            "--backend-store-uri",
            self.backend_uri,
            "--port",
            str(self.port),
            "--host",
            self.host,
        ]

        if background:
            log_file = open("mlflow_ui.log", "w")
            # preexec_fn=os.setpgrp is used to detach the process group, similar to nohup behavior
            process = subprocess.Popen(
                cmd, stdout=log_file, stderr=log_file, preexec_fn=os.setpgrp
            )
            print(f"Started MLflow UI in background (PID: {process.pid})")
            return process.pid
        else:
            subprocess.run(cmd)
            return 0

    def stop(self):
        pid = self.is_running()
        if not pid:
            print(f"MLflow UI is not running on port {self.port}")
            return

        print(f"Stopping MLflow UI (PID: {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def wait_for_ready(self, timeout: int = 30) -> bool:
        print("Waiting for server to start", end="", flush=True)
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{self.port}")
                if response.status_code == 200:
                    print("\n✓ MLflow UI started successfully")
                    return True
            except requests.ConnectionError:
                pass

            print(".", end="", flush=True)
            time.sleep(1)

        print("\n✗ Failed to start MLflow UI")
        return False
