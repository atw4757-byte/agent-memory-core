"""Pytest rootdir shim — makes ``benchmark`` importable without PYTHONPATH hacks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
