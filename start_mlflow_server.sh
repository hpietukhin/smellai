#!/bin/bash
# Start MLflow tracking server with SQLite backend.

set -euo pipefail

cd "$(dirname "$0")"

PORT=${PORT:-5000}
DB_PATH=${DB_PATH:-"$(pwd)/mlflow.db"}
BACKEND_URI="sqlite:///${DB_PATH}"

cat <<EOF
Starting MLflow server...
Database: ${DB_PATH}
Backend URI: ${BACKEND_URI}
URL: http://localhost:${PORT}
EOF

if command -v uv >/dev/null 2>&1; then
  uv run mlflow ui \
    --backend-store-uri "${BACKEND_URI}" \
    --port "${PORT}"
else
  if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi

  mlflow ui \
    --backend-store-uri "${BACKEND_URI}" \
    --port "${PORT}"
fi
