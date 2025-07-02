#!/usr/bin/env python3
"""
DRAT Development Environment Setup Script

This script helps contributors quickly set up their development environment
for working on the DRAT project.
"""

import os
import subprocess
import sys
import venv
from pathlib import Path

def main():
    """Main setup function."""
    print("🚀 Setting up DRAT development environment...")
    print("   For full setup, please refer to CONTRIBUTING.md")
    print("   Basic setup: pip install -e .")
    print("   Dev dependencies: pip install pytest black isort flake8 mypy")

if __name__ == "__main__":
    main() 