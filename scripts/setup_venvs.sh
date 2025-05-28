#!/bin/bash

# Check for Python 3.11
PYTHON_CMD="python3.11"

# Check if Python 3.11 is available
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Python 3.11 is required but not found in PATH"
    echo "Please install Python 3.11 and ensure it's available in your PATH"
    exit 1
fi

# Create alias for python3.11 as python
alias python=$PYTHON_CMD

echo "Using Python command: $PYTHON_CMD (aliased as 'python')"

# Create virtual environments for each service using Python 3.11
python -m venv .venv/model
python -m venv .venv/app

# Install model dependencies
source .venv/model/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r model/requirements.txt
pip install -r app/requirements.txt  # Install app dependencies in model venv for testing
pip install -r requirements.txt      # Install development dependencies
deactivate

# Install app dependencies
source .venv/app/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r app/requirements.txt
deactivate

echo "Virtual environments created and dependencies installed!" 