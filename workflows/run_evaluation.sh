#!/usr/bin/env bash
#
# RefactoringMiner Evaluation Pipeline
#
# This script automates the complete MLflow evaluation workflow:
# 1. Ensures MLflow UI is running
# 2. Creates an MLflow dataset from manifest
# 3. Runs the evaluation pipeline
# 4. Opens results in browser
#

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
START_MLFLOW_SCRIPT="${REPO_ROOT}/start_mlflow_server.sh"

# Default configuration
MANIFEST="rminer_data/manifest.json"
EXPERIMENT="rminer-evaluation"
TRACKING_URI="http://localhost:5000"
MODEL="gpt-4o-mini"
LIMIT=5
DATASET_LIMIT=20
SKIP_DATASET=false
SKIP_UI=false
MLFLOW_PORT=5000

# Print colored message
print_step() {
    echo -e "${BLUE}==>${NC} ${1}"
}

print_success() {
    echo -e "${GREEN}✓${NC} ${1}"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} ${1}"
}

print_error() {
    echo -e "${RED}✗${NC} ${1}"
}

# Show help
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

RefactoringMiner evaluation pipeline automation script.

Options:
  --manifest PATH         Path to manifest.json (default: ${MANIFEST})
  --experiment NAME       Experiment name (default: ${EXPERIMENT})
  --tracking-uri URI      MLflow tracking URI (default: ${TRACKING_URI})
  --model NAME           Model to use (default: ${MODEL})
  --limit N              Limit number of records for evaluation (default: ${LIMIT})
  --dataset-limit N      Limit for dataset creation (default: ${DATASET_LIMIT})
  --skip-dataset         Skip dataset creation step
  --skip-ui              Skip starting MLflow UI
  --help                 Show this help message

Examples:
  # Quick test with 3 records
  $(basename "$0") --limit 3 --dataset-limit 3

  # Full evaluation with Claude
  $(basename "$0") --model claude-sonnet-4-5 --limit 50 --dataset-limit 100

  # Use existing dataset
  $(basename "$0") --skip-dataset

  # Use local SQLite database
  $(basename "$0") --tracking-uri sqlite:///mlflow.db
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --manifest)
            MANIFEST="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --tracking-uri)
            TRACKING_URI="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --dataset-limit)
            DATASET_LIMIT="$2"
            shift 2
            ;;
        --skip-dataset)
            SKIP_DATASET=true
            shift
            ;;
        --skip-ui)
            SKIP_UI=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo ""
print_step "RefactoringMiner Evaluation Pipeline"
echo ""
echo "Configuration:"
echo "  Manifest:      ${MANIFEST}"
echo "  Experiment:    ${EXPERIMENT}"
echo "  Tracking URI:  ${TRACKING_URI}"
echo "  Model:         ${MODEL}"
echo "  Eval Limit:    ${LIMIT}"
echo "  Dataset Limit: ${DATASET_LIMIT}"
echo ""

# Check if manifest exists
if [[ ! -f "$MANIFEST" ]]; then
    print_error "Manifest file not found: ${MANIFEST}"
    exit 1
fi

# Step 1: Check/Start MLflow UI
if [[ "$SKIP_UI" == false ]] && [[ "$TRACKING_URI" == http://* ]]; then
    print_step "Checking MLflow UI on port ${MLFLOW_PORT}..."
    
    if lsof -Pi :${MLFLOW_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_success "MLflow UI is already running on port ${MLFLOW_PORT}"
    else
        print_warning "MLflow UI is not running. Starting it now..."
        
        # Determine backend store URI from tracking URI
        if [[ "$TRACKING_URI" == "http://localhost:${MLFLOW_PORT}" ]]; then
            BACKEND_URI="sqlite:///mlflow.db"
        else
            BACKEND_URI="sqlite:///mlflow.db"
        fi
        
        # Start MLflow UI in background using uv to ensure version consistency
        nohup uv run mlflow ui --backend-store-uri "${BACKEND_URI}" --port ${MLFLOW_PORT} --host 0.0.0.0 > mlflow_ui.log 2>&1 &
        MLFLOW_PID=$!
        
        # Wait for server to start
        echo -n "Waiting for MLflow UI to start"
        for i in {1..30}; do
            if lsof -Pi :${MLFLOW_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
                echo ""
                print_success "MLflow UI started (PID: ${MLFLOW_PID})"
                echo "  Log file: mlflow_ui.log"
                echo "  URL: http://localhost:${MLFLOW_PORT}"
                break
            fi
            echo -n "."
            sleep 1
        done
        
        if ! lsof -Pi :${MLFLOW_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo ""
            print_error "Failed to start MLflow UI. Check mlflow_ui.log for details."
            exit 1
        fi
    fi
    echo ""
fi

# Step 2: Create MLflow Dataset
if [[ "$SKIP_DATASET" == false ]]; then
    print_step "Creating MLflow dataset..."
    echo "  Limit: ${DATASET_LIMIT} records"
    
    if uv run scripts/create_rminer_dataset.py \
        --manifest "${MANIFEST}" \
        --limit "${DATASET_LIMIT}" \
        --experiment "${EXPERIMENT}" \
        --tracking-uri "${TRACKING_URI}"; then
        print_success "Dataset created successfully"
    else
        print_error "Dataset creation failed"
        exit 1
    fi
    echo ""
else
    print_warning "Skipping dataset creation (--skip-dataset)"
    echo ""
fi

# Step 3: Run Evaluation
print_step "Running evaluation pipeline..."
echo "  Model: ${MODEL}"
echo "  Limit: ${LIMIT} records"
echo ""

if uv run smellai/pipelines/rminer_eval.py \
    --manifest "${MANIFEST}" \
    --experiment "${EXPERIMENT}" \
    --tracking-uri "${TRACKING_URI}" \
    --model "${MODEL}" \
    --limit "${LIMIT}"; then
    print_success "Evaluation completed successfully"
else
    print_error "Evaluation failed"
    exit 1
fi

echo ""

# Step 4: Summary and Next Steps
print_success "Pipeline completed!"
echo ""
echo "Next steps:"
echo "  1. View results in MLflow UI: http://localhost:${MLFLOW_PORT}"
echo "  2. Navigate to the '${EXPERIMENT}' experiment"
echo "  3. Click on the latest run to see metrics and artifacts"
echo ""

# List available datasets
print_step "Available datasets:"
uv run cli/datasets/rminer_dataset_cli.py list --tracking-uri "${TRACKING_URI}" 2>/dev/null || true

echo ""
print_step "To re-run evaluation with existing dataset:"
echo "  uv run src/pipelines/rminer_eval.py \\"
echo "      --dataset-name <dataset-name> \\"
echo "      --experiment ${EXPERIMENT} \\"
echo "      --tracking-uri ${TRACKING_URI}"
echo ""
