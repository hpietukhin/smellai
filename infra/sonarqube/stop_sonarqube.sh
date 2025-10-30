#!/usr/bin/env bash

# SonarQube Shutdown Script
# Cleanly stops SonarQube Docker container
# Exit codes: 0 = success, 1 = failure

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Main execution
main() {
    log_info "========================================="
    log_info "  SonarQube Shutdown Script"
    log_info "========================================="
    echo ""

    log_info "Stopping SonarQube container..."

    if docker-compose -f "${COMPOSE_FILE}" down; then
        echo ""
        log_info "========================================="
        log_info "✓ SonarQube stopped successfully"
        log_info "========================================="
        exit 0
    else
        log_error "Failed to stop SonarQube"
        exit 1
    fi
}

# Run main function
main
