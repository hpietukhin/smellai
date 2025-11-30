#!/bin/bash
# Start MLflow tracking server with basic health checks.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PORT=5000
HOST=0.0.0.0
WAIT_SECONDS=30
RUN_MODE=foreground
CHECK_ONLY=false
DB_PATH="${SCRIPT_DIR}/mlflow.db"
BACKEND_URI="sqlite:///${DB_PATH}"
LOG_FILE="${SCRIPT_DIR}/mlflow_ui.log"

usage() {
	cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --port N          Port to bind (default: ${PORT})
  --host HOST       Host interface (default: ${HOST})
  --backend URI     Backend store URI (default: sqlite file in repo root)
  --background      Run MLflow UI in background and wait for readiness
  --wait N          Seconds to wait for readiness checks (default: ${WAIT_SECONDS})
  --log PATH        Log file for background mode (default: ${LOG_FILE})
  --check           Only check whether MLflow UI is reachable
  --help            Show this message
EOF
}

require_cmd() {
	if ! command -v "$1" >/dev/null 2>&1; then
		echo "Error: Required command '$1' not found" >&2
		exit 1
	fi
}

server_running() {
	curl --silent --fail --max-time 2 "http://localhost:${PORT}/" >/dev/null 2>&1
}

port_in_use() {
	lsof -Pi :"${PORT}" -sTCP:LISTEN -t >/dev/null 2>&1
}

wait_for_server() {
	for ((i = 0; i < WAIT_SECONDS; i++)); do
		if server_running; then
			return 0
		fi
		sleep 1
	done
	return 1
}

while [[ $# -gt 0 ]]; do
	case $1 in
		--port)
			PORT="$2"
			shift 2
			;;
		--host)
			HOST="$2"
			shift 2
			;;
		--backend)
			BACKEND_URI="$2"
			shift 2
			;;
		--background)
			RUN_MODE=background
			shift
			;;
		--wait)
			WAIT_SECONDS="$2"
			shift 2
			;;
		--log)
			LOG_FILE="$2"
			shift 2
			;;
		--check)
			CHECK_ONLY=true
			shift
			;;
		--help|-h)
			usage
			exit 0
			;;
		*)
			echo "Error: Unknown option '$1'" >&2
			usage
			exit 1
			;;
	esac
done

require_cmd curl

if [[ "${CHECK_ONLY}" == true ]]; then
	if server_running; then
		exit 0
	fi
	exit 1
fi

require_cmd uv
require_cmd lsof

export MLFLOW_TRACKING_URI="http://localhost:${PORT}"

if server_running; then
	echo "MLflow UI already reachable on http://localhost:${PORT}"
	exit 0
fi

if port_in_use; then
	echo "Error: Port ${PORT} is in use but MLflow UI is not reachable" >&2
	exit 1
fi

if ! uv run python -c "import mlflow" >/dev/null 2>&1; then
	echo "Error: mlflow package is not available to uv" >&2
	exit 1
fi

CMD=(uv run mlflow ui --backend-store-uri "${BACKEND_URI}" --port "${PORT}" --host "${HOST}")

echo "Starting MLflow UI on http://localhost:${PORT}"
echo "Backend store: ${BACKEND_URI}"

if [[ "${RUN_MODE}" == background ]]; then
	nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
	ML_PID=$!
	echo "MLflow UI logs: ${LOG_FILE}"
	if wait_for_server; then
		echo "MLflow UI started (PID ${ML_PID})"
		exit 0
	fi
	echo "Error: MLflow UI did not become ready in ${WAIT_SECONDS} seconds" >&2
	kill "${ML_PID}" >/dev/null 2>&1 || true
	exit 1
fi

exec "${CMD[@]}"
