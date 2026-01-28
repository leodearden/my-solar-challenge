#!/bin/bash
# Solar Challenge Energy Flow Simulator - Environment Setup

set -e

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Navigate to project root
cd "$PROJECT_ROOT"

echo "=== Setting up Solar Challenge Simulator ==="
echo "Project root: $PROJECT_ROOT"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    if command -v uv &> /dev/null; then
        uv venv venv
    else
        python3 -m venv venv
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
if command -v uv &> /dev/null; then
    uv pip install -r requirements.txt
else
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Run basic smoke test
echo "Running smoke test..."
python -c "import pvlib; import pandas; print('Core imports OK')"

# If tests exist, run them
if [ -f "pytest.ini" ] || [ -d "tests" ]; then
    echo "Running tests..."
    pytest -v --tb=short || echo "Some tests failed - review before proceeding"
fi

echo "=== Setup complete ==="
echo ""
echo "Quick reference:"
echo "   - Progress log:   cat $SCRIPT_DIR/progress.txt"
echo "   - Feature list:   cat $SCRIPT_DIR/feature_list.json"
echo "   - Git history:    git log --oneline -10"
echo "   - Git status:     git status"
