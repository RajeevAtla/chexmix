"""Pytest configuration and import path setup."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable for tests when running from repo root.
src_path = (Path(__file__).resolve().parents[1] / "src").as_posix()
if src_path not in sys.path:
    sys.path.insert(0, src_path)
