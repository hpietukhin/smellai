#!/usr/bin/env bash

# SonarQube Startup Script
# Starts SonarQube Docker container and waits for it to be ready
# Exit codes: 0 = success, 1 = failure

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
MAX_RETRIES=30
RETRY_INTERVAL=10
SONAR_URL="http://localhost:9000"
HEALTH_ENDPOINT="${SONAR_URL}/api/system/health"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if docker-compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "docker-compose is not installed or not in PATH"
        exit 1
    fi
    log_info "Docker Compose is available"
}

# Start SonarQube container
start_container() {
    log_info "Starting SonarQube container..."
    docker-compose -f "${COMPOSE_FILE}" up -d
    if [ $? -eq 0 ]; then
        log_info "Container started successfully"
    else
        log_error "Failed to start container"
        exit 1
    fi
}

# Wait for port to be open
wait_for_port() {
    log_info "Waiting for port 9000 to be open..."
    local retry_count=0

    while [ $retry_count -lt $MAX_RETRIES ]; do
        if command -v nc &> /dev/null; then
            # Use netcat if available
            if nc -z localhost 9000 2>/dev/null; then
                log_info "Port 9000 is open"
                return 0
            fi
        else
            # Fall back to curl
            if curl --output /dev/null --silent --fail --connect-timeout 1 "${SONAR_URL}" 2>/dev/null; then
                log_info "Port 9000 is open"
                return 0
            fi
        fi

        retry_count=$((retry_count + 1))
        log_info "Attempt $retry_count/$MAX_RETRIES - Port not ready yet, waiting ${RETRY_INTERVAL}s..."
        sleep $RETRY_INTERVAL
    done

    log_error "Timeout waiting for port 9000 to open"
    return 1
}

# Check SonarQube health
check_health() {
    log_info "Checking SonarQube health endpoint..."
    local retry_count=0

    while [ $retry_count -lt $MAX_RETRIES ]; do
        # Try to get health status
        local health_response=$(curl --silent --fail "${HEALTH_ENDPOINT}" 2>/dev/null || echo "FAILED")

        if [ "$health_response" != "FAILED" ]; then
            # Check if response contains "GREEN" or "YELLOW" health status
            if echo "$health_response" | grep -q '"health":"GREEN"'; then
                log_info "✓ SonarQube is healthy (GREEN status)"
                return 0
            elif echo "$health_response" | grep -q '"health":"YELLOW"'; then
                log_warn "SonarQube is healthy but with warnings (YELLOW status)"
                return 0
            fi
        fi

        retry_count=$((retry_count + 1))
        log_info "Attempt $retry_count/$MAX_RETRIES - Health check not passing, waiting ${RETRY_INTERVAL}s..."
        sleep $RETRY_INTERVAL
    done

    log_error "Timeout waiting for SonarQube to become healthy"
    return 1
}

# Main execution
main() {
    log_info "========================================="
    log_info "  SonarQube Startup Script"
    log_info "========================================="
    echo ""

    # Check prerequisites
    check_docker_compose

    # Start container
    start_container
    echo ""

    # Wait for port
    if ! wait_for_port; then
        log_error "Failed to start SonarQube: port check failed"
        exit 1
    fi
    echo ""

    # Check health
    if ! check_health; then
        log_error "Failed to start SonarQube: health check failed"
        exit 1
    fi
    echo ""

    log_info "========================================="
    log_info "✓ SonarQube is ready!"
    log_info "  URL: ${SONAR_URL}"
    log_info "  Default credentials: admin/admin"
    log_info "========================================="

    exit 0
}

# Run main function
main
