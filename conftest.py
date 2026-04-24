"""Pytest configuration — adds project root to path so `from src...` works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
