#!/bin/bash

echo "Setting up Python environment with uv..."

# Create virtual environment
uv venv

# Install dependencies
echo "Installing dependencies..."
uv sync --no-dev

echo "Installation complete!"
echo "To activate the environment, run: source .venv/bin/activate"
echo "To run tests: uv run pytest tests/"