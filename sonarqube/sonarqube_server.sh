#!/usr/bin/env bash
#
# SonarQube server management script
#

set -euo pipefail

COMPOSE_FILE="infra/sonarqube/docker-compose.yml"

show_help() {
    cat << EOF
Usage: $(basename "$0") COMMAND

Manage SonarQube server.

Commands:
  start          Start SonarQube server
  stop           Stop SonarQube server
  restart        Restart SonarQube server
  status         Check SonarQube server status
  logs           Show SonarQube logs
EOF
    exit 0
}

start_server() {
    docker compose -f "${COMPOSE_FILE}" up -d
    echo "SonarQube starting at http://localhost:9000"
}

stop_server() {
    docker compose -f "${COMPOSE_FILE}" down
}

status_server() {
    docker compose -f "${COMPOSE_FILE}" ps
}

show_logs() {
    docker compose -f "${COMPOSE_FILE}" logs -f sonarqube
}

if [[ $# -eq 0 ]]; then
    show_help
fi

case "$1" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
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
        echo "Unknown command: $1"
        exit 1
        ;;
esac
