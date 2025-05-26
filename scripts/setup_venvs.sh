#!/bin/bash

# Use Python 3.11 from Homebrew
PYTHON_CMD="/opt/homebrew/bin/python3.11"

if [ ! -f "$PYTHON_CMD" ]; then
    echo "Python 3.11 is required but not found at $PYTHON_CMD"
    echo "Please ensure Python 3.11 is installed and in your PATH."
    echo "You can install it using: brew install python@3.11"
    exit 1
fi

echo "Using Python command: $PYTHON_CMD"

# Create virtual environments for each service using Python 3.11
$PYTHON_CMD -m venv .venv/model
$PYTHON_CMD -m venv .venv/app

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