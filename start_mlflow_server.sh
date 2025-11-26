#!/bin/bash
# Start MLflow tracking server with SQLite backend.

set -euo pipefail

PORT=${PORT:-5000}
export MLFLOW_TRACKING_URI=http://localhost:5000

echo "Starting MLflow server..."
echo "Database: $(pwd)/mlflow.db"
echo "URL: http://localhost:${PORT}"

uv run mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port "${PORT}" --host 0.0.0.0
