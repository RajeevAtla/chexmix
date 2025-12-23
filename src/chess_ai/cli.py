"""
Command-line entrypoints.

Commands:
- train: run self-play + learning loop
- eval: evaluate checkpoints (arena matches)
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="chexmix")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training loop")
    train_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to TOML config",
    )

    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to TOML config",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entrypoint.

    Returns:
        Process exit code (0 for success).
    """
    parser = build_parser()
    parser.parse_args(argv)
    return 0
