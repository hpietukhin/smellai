import argparse
import os
import sys
from pathlib import Path

from .datasets.manager import DatasetManager
from .runner import EvaluationRunner
from .server import MLflowServer

DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_MANIFEST_PATH = os.environ.get("RMINER_MANIFEST_PATH", "rminer_data/manifest.json")


def _add_server_parser(subparsers):
    server_parser = subparsers.add_parser("server", help="Manage MLflow server")
    server_parser.add_argument(
        "action", choices=["start", "stop", "status", "restart"], help="Action"
    )
    server_parser.add_argument("--port", type=int, default=5000, help="Port")
    server_parser.add_argument(
        "--backend-uri", default="sqlite:///mlflow.db", help="Backend URI"
    )


def _add_dataset_parser(subparsers):
    dataset_parser = subparsers.add_parser("datasets", help="Manage datasets")
    dataset_parser.add_argument(
        "action", choices=["list", "get", "create", "delete"], help="Action"
    )
    dataset_parser.add_argument("--name", help="Dataset name")
    dataset_parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH, help="Manifest path")
    dataset_parser.add_argument("--limit", type=int, default=20, help="Limit")
    dataset_parser.add_argument(
        "--experiment", default="rminer-evaluation", help="Experiment name"
    )
    dataset_parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI, help="Tracking URI")


def _add_evaluate_parser(subparsers):
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    eval_parser.add_argument("--experiment", default="rminer-evaluation")
    eval_parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    eval_parser.add_argument("--model", default="gpt-4o-mini")
    eval_parser.add_argument("--limit", type=int, default=5)
    eval_parser.add_argument("--dataset-limit", type=int, default=20)
    eval_parser.add_argument("--skip-dataset", action="store_true")
    eval_parser.add_argument("--skip-ui", action="store_true")


def _build_parser():
    parser = argparse.ArgumentParser(description="SmellAI MLflow Infrastructure CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    _add_server_parser(subparsers)
    _add_dataset_parser(subparsers)
    _add_evaluate_parser(subparsers)
    return parser


def _handle_server(args):
    server = MLflowServer(port=args.port, backend_uri=args.backend_uri)
    if args.action == "start":
        server.start()
        server.wait_for_ready()
    elif args.action == "stop":
        server.stop()
    elif args.action == "status":
        pid = server.is_running()
        print(f"Running (PID: {pid})" if pid else "Stopped")
    elif args.action == "restart":
        server.stop()
        server.start()
        server.wait_for_ready()


def _require_name(args):
    if not args.name:
        print(f"Error: --name required for {args.action}")
        sys.exit(1)


def _handle_datasets(args):
    manager = DatasetManager(tracking_uri=args.tracking_uri)
    if args.action == "list":
        print(manager.list_datasets(experiment_name=args.experiment))
    elif args.action == "get":
        _require_name(args)
        print(manager.get_dataset(args.name))
    elif args.action == "create":
        from rminer.create_rminer_dataset import build_genai_records

        records = build_genai_records(Path(args.manifest), limit=args.limit)
        manager.create_dataset_from_records(
            records=records,
            name=f"rminer-dataset-{args.limit or 'all'}",
            experiment=args.experiment,
            tags={"source": "RefactoringMiner", "total_pairs": str(len(records))},
        )
    elif args.action == "delete":
        _require_name(args)
        manager.delete_dataset(args.name)


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "server":
        _handle_server(args)
    elif args.command == "datasets":
        _handle_datasets(args)
    elif args.command == "evaluate":
        EvaluationRunner().run(vars(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
