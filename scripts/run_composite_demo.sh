#!/usr/bin/env bash
# Run composite evaluation demo with visualization

set -e

echo "=== Composite Evaluation Demo ==="
echo ""

# Wait for SonarQube to be ready
echo "Checking SonarQube status..."
for i in {1..12}; do
    STATUS=$(curl -s http://localhost:9000/api/system/status 2>/dev/null | jq -r '.status' 2>/dev/null || echo "STARTING")
    if [ "$STATUS" = "UP" ]; then
        echo "✅ SonarQube is ready"
        break
    fi
    echo "⏳ SonarQube status: $STATUS (attempt $i/12)"
    sleep 5
done

if [ "$STATUS" != "UP" ]; then
    echo "❌ SonarQube not ready after 60 seconds"
    echo "Check manually: http://localhost:9000"
    exit 1
fi

echo ""
echo "=== Running Composite Evaluation ==="
echo "Dataset: compound_demo.json (2 records)"
echo "Mode: Composite (A0 → A1 → A3 → [A5 → A6] loop)"
echo "Max iterations: 3"
echo ""

uv run workflows/swe_eval_workflow.py \
  --dataset compound_demo.json \
  --enable-composite \
  --max-refactorings 3 \
  --analytics-db demo_composite.db \
  --model claude-sonnet-4-5-20250929 \
  --limit 1

echo ""
echo "=== Evaluation Complete ==="
echo ""
echo "Results saved to:"
echo "  - Analytics DB: demo_composite.db"
echo "  - MLflow: sqlite:///mlflow.db"
echo ""
echo "Next steps:"
echo "  1. View MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db"
echo "  2. Visualize results: ./scripts/run_visualizer.sh"
echo "     Then load demo_composite.db in the UI"
echo ""
