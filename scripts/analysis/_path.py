"""Put scripts/ on sys.path so analysis scripts can reuse scripts/_bootstrap.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401
