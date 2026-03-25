"""CLI for managing MLflow server, datasets, and evaluation runs."""

import argparse
import os
import sys
from .server import MLflowServer
from .datasets.manager import DatasetManager
from .runner import EvaluationRunner


def main():
    parser = argparse.ArgumentParser(description="SmellAI MLflow Infrastructure CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Server commands
    server_parser = subparsers.add_parser("server", help="Manage MLflow server")
    server_parser.add_argument(
        "action", choices=["start", "stop", "status", "restart"], help="Action"
    )
    server_parser.add_argument("--port", type=int, default=5000, help="Port")
    server_parser.add_argument(
        "--backend-uri", default="sqlite:///mlflow.db", help="Backend URI"
    )

    # Dataset commands
    dataset_parser = subparsers.add_parser("datasets", help="Manage datasets")
    dataset_parser.add_argument(
        "action", choices=["list", "get", "create", "delete"], help="Action"
    )
    dataset_parser.add_argument("--name", help="Dataset name")
    dataset_parser.add_argument(
        "--manifest", default=os.environ.get("RMINER_MANIFEST_PATH", "rminer_data/manifest.json"), help="Manifest path"
    )
    dataset_parser.add_argument("--limit", type=int, default=20, help="Limit")
    dataset_parser.add_argument(
        "--experiment", default="rminer-evaluation", help="Experiment name"
    )
    dataset_parser.add_argument(
        "--tracking-uri", default="http://localhost:5000", help="Tracking URI"
    )

    # Evaluate commands
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.add_argument("--manifest", default=os.environ.get("RMINER_MANIFEST_PATH", "rminer_data/manifest.json"))
    eval_parser.add_argument("--experiment", default="rminer-evaluation")
    eval_parser.add_argument("--tracking-uri", default="http://localhost:5000")
    eval_parser.add_argument("--model", default="gpt-4o-mini")
    eval_parser.add_argument("--limit", type=int, default=5)
    eval_parser.add_argument("--dataset-limit", type=int, default=20)
    eval_parser.add_argument("--skip-dataset", action="store_true")
    eval_parser.add_argument("--skip-ui", action="store_true")

    args = parser.parse_args()

    if args.command == "server":
        server = MLflowServer(port=args.port, backend_uri=args.backend_uri)
        if args.action == "start":
            server.start()
            server.wait_for_ready()
        elif args.action == "stop":
            server.stop()
        elif args.action == "status":
            pid = server.is_running()
            if pid:
                print(f"Running (PID: {pid})")
            else:
                print("Stopped")
        elif args.action == "restart":
            server.stop()
            server.start()
            server.wait_for_ready()

    elif args.command == "datasets":
        manager = DatasetManager(tracking_uri=args.tracking_uri)
        if args.action == "list":
            print(manager.list_datasets(experiment_name=args.experiment))
        elif args.action == "get":
            if not args.name:
                print("Error: --name required for get")
                sys.exit(1)
            print(manager.get_dataset(args.name))
        elif args.action == "create":
            from rminer.create_rminer_dataset import build_genai_records
            from pathlib import Path

            records = build_genai_records(Path(args.manifest), limit=args.limit)
            manager.create_dataset_from_records(
                records=records,
                name=f"rminer-dataset-{args.limit or 'all'}",
                experiment=args.experiment,
                tags={"source": "RefactoringMiner", "total_pairs": str(len(records))},
            )
        elif args.action == "delete":
            if not args.name:
                print("Error: --name required for delete")
                sys.exit(1)
            manager.delete_dataset(args.name)

    elif args.command == "evaluate":
        runner = EvaluationRunner()
        config = vars(args)
        runner.run(config)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
