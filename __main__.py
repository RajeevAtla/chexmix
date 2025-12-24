"""Module entrypoint that ensures src/ is on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """Add the src/ directory to sys.path for local module imports."""
    # Resolve repo root and prepend src for local imports.
    src_path = (Path(__file__).resolve().parent / "src").as_posix()
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main() -> int:
    """Run the CLI entrypoint with src/ on sys.path."""
    _ensure_src_on_path()
    # Import after sys.path update to resolve local modules.
    import cli

    return cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
