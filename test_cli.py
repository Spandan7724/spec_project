#!/usr/bin/env python3
"""
Test script for the CLI before full installation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    from src.cli.main import app
    app()