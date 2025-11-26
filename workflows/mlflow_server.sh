#!/usr/bin/env bash
#
# MLflow UI Management Script
#

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PORT=5000
BACKEND_URI="sqlite:///mlflow.db"

show_help() {
    cat << EOF
Usage: $(basename "$0") COMMAND [OPTIONS]

Manage MLflow UI server.

Commands:
  start          Start MLflow UI server
  stop           Stop MLflow UI server
  restart        Restart MLflow UI server
  status         Check MLflow UI server status
  logs           Show MLflow UI logs

Options:
  --port PORT           Port number (default: ${PORT})
  --backend-uri URI     Backend store URI (default: ${BACKEND_URI})

Examples:
  # Start server
  $(basename "$0") start

  # Check status
  $(basename "$0") status

  # View logs
  $(basename "$0") logs

  # Stop server
  $(basename "$0") stop
EOF
    exit 0
}

get_mlflow_pid() {
    lsof -ti :${PORT} 2>/dev/null || true
}

start_server() {
    local pid=$(get_mlflow_pid)
    
    if [[ -n "$pid" ]]; then
        echo -e "${YELLOW}MLflow UI is already running on port ${PORT} (PID: ${pid})${NC}"
        echo "Use '$(basename "$0") stop' to stop it first"
        exit 1
    fi
    
    echo "Starting MLflow UI..."
    echo "  Port: ${PORT}"
    echo "  Backend: ${BACKEND_URI}"
    
    nohup mlflow ui --backend-store-uri "${BACKEND_URI}" --port ${PORT} --host 0.0.0.0 > mlflow_ui.log 2>&1 &
    local new_pid=$!
    
    # Wait for server to start
    echo -n "Waiting for server to start"
    for i in {1..30}; do
        if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo ""
            echo -e "${GREEN}✓ MLflow UI started successfully${NC}"
            echo "  PID: ${new_pid}"
            echo "  URL: http://localhost:${PORT}"
            echo "  Logs: mlflow_ui.log"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    
    echo ""
    echo -e "${RED}✗ Failed to start MLflow UI${NC}"
    echo "Check mlflow_ui.log for details"
    exit 1
}

stop_server() {
    local pid=$(get_mlflow_pid)
    
    if [[ -z "$pid" ]]; then
        echo -e "${YELLOW}MLflow UI is not running on port ${PORT}${NC}"
        exit 0
    fi
    
    echo "Stopping MLflow UI (PID: ${pid})..."
    kill "$pid" 2>/dev/null || true
    
    # Wait for process to stop
    for i in {1..10}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo -e "${GREEN}✓ MLflow UI stopped${NC}"
            return 0
        fi
        sleep 1
    done
    
    # Force kill if still running
    echo "Process still running, forcing stop..."
    kill -9 "$pid" 2>/dev/null || true
    echo -e "${GREEN}✓ MLflow UI stopped (forced)${NC}"
}

status_server() {
    local pid=$(get_mlflow_pid)
    
    if [[ -z "$pid" ]]; then
        echo -e "${RED}✗ MLflow UI is not running on port ${PORT}${NC}"
        exit 1
    else
        echo -e "${GREEN}✓ MLflow UI is running${NC}"
        echo "  PID: ${pid}"
        echo "  Port: ${PORT}"
        echo "  URL: http://localhost:${PORT}"
        
        # Show process details
        echo ""
        echo "Process details:"
        ps -p "$pid" -o pid,ppid,user,%cpu,%mem,etime,command | tail -n +2
    fi
}

show_logs() {
    if [[ ! -f mlflow_ui.log ]]; then
        echo -e "${RED}Log file not found: mlflow_ui.log${NC}"
        exit 1
    fi
    
    tail -f mlflow_ui.log
}

if [[ $# -eq 0 ]]; then
    show_help
fi

COMMAND="$1"
shift

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --backend-uri)
            BACKEND_URI="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

case "$COMMAND" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        echo ""
        start_server
        ;;
    status)
        status_server
        ;;
    logs)
        show_logs
        ;;
    --help|-h|help)
        show_help
        ;;
    *)
        echo "Unknown command: ${COMMAND}"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
