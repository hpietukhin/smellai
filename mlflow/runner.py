from .server import MLflowServer
from .datasets import DatasetManager
import subprocess
import os


class EvaluationRunner:
    def run(self, config: dict):
        # config keys: manifest, experiment, tracking_uri, model, limit, dataset_limit, skip_dataset, skip_ui

        tracking_uri = config.get("tracking_uri", "http://localhost:5000")
        # Extract port from URI if possible, else default to 5000
        port = 5000
        if tracking_uri.startswith("http://localhost:"):
            try:
                port = int(tracking_uri.split(":")[-1])
            except ValueError:
                pass

        # Step 1: Server
        if not config.get("skip_ui", False):
            server = MLflowServer(port=port, backend_uri="sqlite:///mlflow.db")
            if not server.is_running():
                server.start()
                server.wait_for_ready()
            else:
                print(f"MLflow UI is already running on port {port}")

        # Step 2: Dataset
        if not config.get("skip_dataset", False):
            manager = DatasetManager(tracking_uri=tracking_uri)
            manager.create_dataset(
                manifest_path=config.get("manifest", "rminer_data/manifest.json"),
                limit=config.get("dataset_limit", 20),
                experiment=config.get("experiment", "rminer-evaluation"),
                tracking_uri=tracking_uri,
            )

        # Step 3: Evaluation
        print("Running evaluation pipeline...")
        # The script calls smellai/pipelines/rminer_eval.py
        # I'll assume this path is correct relative to repo root.
        cmd = [
            "uv",
            "run",
            "smellai/pipelines/rminer_eval.py",
            "--manifest",
            config.get("manifest", "rminer_data/manifest.json"),
            "--experiment",
            config.get("experiment", "rminer-evaluation"),
            "--tracking-uri",
            tracking_uri,
            "--model",
            config.get("model", "gpt-4o-mini"),
            "--limit",
            str(config.get("limit", 5)),
        ]
        try:
            subprocess.run(cmd, check=True)
            print("Evaluation completed successfully")
        except subprocess.CalledProcessError:
            print("Evaluation failed")
            # Don't exit here, let the caller handle it or just return
