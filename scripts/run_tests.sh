#!/bin/bash
# run_tests.sh - hoboQRAG Test Runner Script
set -e

# Resolve paths relative to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"

cd "$BACKEND_DIR"

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment (.venv) not found in $BACKEND_DIR."
    exit 1
fi

# Export test mode indicator env var
export TESTING=true

STAGE=${1:-unit}

case "$STAGE" in
    unit)
        echo "Running Unit Tests..."
        .venv/bin/pytest tests/unit/test_document_processing.py tests/unit/test_embedding.py tests/unit/test_qdrant.py tests/unit/test_reranker.py tests/unit/test_prompt_construction.py tests/unit/test_rag_pipeline.py
        ;;
    api)
        echo "Running API Endpoint Tests..."
        .venv/bin/pytest tests/unit/test_api_endpoints.py
        ;;
    integration)
        echo "Running Integration Tests..."
        .venv/bin/pytest tests/integration
        ;;
    e2e)
        echo "Running End-to-End Tests..."
        .venv/bin/pytest tests/e2e
        ;;
    all)
        echo "Running All Tests..."
        .venv/bin/pytest tests/
        ;;
    *)
        echo "Usage: $0 {unit|api|integration|e2e|all}"
        exit 1
        ;;
esac

echo "Tests execution complete!"
