"""
Command-line entrypoints.

Commands:
- train: run self-play + learning loop
- eval: evaluate checkpoints (arena matches)
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from pathlib import Path

from paths import RunPaths
from pgn.writer import PgnHeaders, format_pgn, write_pgn_file
from toml_io import TomlValue, load_toml, save_toml
from train.logging import Metrics, write_metrics_snapshot


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


def _get_table(data: dict[str, TomlValue], key: str) -> dict[str, TomlValue]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing TOML table: {key}")
    return value


def _get_str(table: dict[str, TomlValue], key: str) -> str:
    value = table.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Missing string key: {key}")
    return value


def _run_id(run_name: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{run_name}"


def _write_events(paths: RunPaths, run_id: str, run_name: str) -> None:
    events: dict[str, TomlValue] = {
        "event": "start",
        "run_id": run_id,
        "run_name": run_name,
        "started_utc": datetime.now(UTC).isoformat(),
    }
    save_toml(paths.events_toml, events)


def _write_bootstrap_artifacts(paths: RunPaths, run_name: str) -> None:
    metrics = Metrics(
        step=0,
        loss_total=0.0,
        loss_policy=0.0,
        loss_value=0.0,
        value_mean=0.0,
        entropy_mean=0.0,
    )
    write_metrics_snapshot(paths, metrics)

    headers = PgnHeaders(
        event=run_name,
        site="local",
        date=date.today(),
        round="1",
        white="selfplay",
        black="selfplay",
        result="*",
    )
    pgn = format_pgn(headers, moves=[])
    write_pgn_file(paths.games_dir / "game_0000000001.pgn", pgn)


def main(argv: list[str] | None = None) -> int:
    """Main entrypoint.

    Returns:
        Process exit code (0 for success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_toml(args.config)
    run_table = _get_table(config, "run")
    run_name = _get_str(run_table, "name")
    run_id = _run_id(run_name)
    paths = RunPaths.create(run_id)
    save_toml(paths.config_toml, config)
    _write_events(paths, run_id, run_name)
    if args.command == "train":
        _write_bootstrap_artifacts(paths, run_name)
    return 0
