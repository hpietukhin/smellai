#!/usr/bin/env bash
#
# Dataset Management Helper Script
#
# Provides convenient commands for managing MLflow datasets
#

set -euo pipefail

# Color output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

TRACKING_URI="http://localhost:5000"

show_help() {
    cat << EOF
Usage: $(basename "$0") COMMAND [OPTIONS]

Dataset management helper for RefactoringMiner evaluation.

Commands:
  list                    List all datasets
  get NAME                Get dataset by name
  get-id ID              Get dataset by ID
  create                 Create a new dataset from manifest
  inspect NAME           Inspect dataset records

Options:
  --tracking-uri URI     MLflow tracking URI (default: ${TRACKING_URI})
  --json                 Output as JSON (for list command)
  --show-records         Show records (for get commands)
  --limit N              Limit number of records (for create command)

Examples:
  # List all datasets
  $(basename "$0") list

  # List with JSON output
  $(basename "$0") list --json

  # Get specific dataset and show records
  $(basename "$0") get rminer-eval-dataset --show-records

  # Create new dataset with 50 records
  $(basename "$0") create --limit 50

  # Use custom tracking URI
  $(basename "$0") list --tracking-uri sqlite:///mlflow.db
EOF
    exit 0
}

if [[ $# -eq 0 ]]; then
    show_help
fi

COMMAND="$1"
shift

case "$COMMAND" in
    list)
        uv run infra/rminer_dataset_cli.py list --tracking-uri "${TRACKING_URI}" "$@"
        ;;
    get)
        if [[ $# -eq 0 ]]; then
            echo "Error: Dataset name required"
            exit 1
        fi
        NAME="$1"
        shift
        uv run infra/rminer_dataset_cli.py get --name "${NAME}" --tracking-uri "${TRACKING_URI}" "$@"
        ;;
    get-id)
        if [[ $# -eq 0 ]]; then
            echo "Error: Dataset ID required"
            exit 1
        fi
        ID="$1"
        shift
        uv run infra/rminer_dataset_cli.py get --id "${ID}" --tracking-uri "${TRACKING_URI}" "$@"
        ;;
    create)
        uv run infra/mlflow/rminer_dataset.py \
            --manifest rminer_data/manifest.json \
            --experiment rminer-evaluation \
            --tracking-uri "${TRACKING_URI}" \
            "$@"
        ;;
    inspect)
        if [[ $# -eq 0 ]]; then
            echo "Error: Dataset name required"
            exit 1
        fi
        NAME="$1"
        shift
        uv run infra/rminer_dataset_cli.py get --name "${NAME}" --show-records --tracking-uri "${TRACKING_URI}" "$@"
        ;;
    --help|-h|help)
        show_help
        ;;
    *)
        echo "Error: Unknown command '${COMMAND}'"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
