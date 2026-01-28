#!/bin/bash
# Solar Challenge Energy Flow Simulator - Environment Setup

set -e

echo "=== Setting up Solar Challenge Simulator ==="

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
