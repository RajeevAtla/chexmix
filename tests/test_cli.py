from __future__ import annotations

from pathlib import Path

import pytest

from chess_ai.cli import build_parser, main


def test_build_parser_train_args() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--config", "config/default.toml"])
    assert args.command == "train"
    assert isinstance(args.config, Path)


def test_main_returns_zero() -> None:
    assert main(["eval", "--config", "config/default.toml"]) == 0
