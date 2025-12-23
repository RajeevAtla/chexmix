from __future__ import annotations

from pathlib import Path

import pytest

from cli import build_parser, main
from toml_io import save_toml


def test_build_parser_train_args() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--config", "config/default.toml"])
    assert args.command == "train"
    assert isinstance(args.config, Path)


def test_main_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = (
        Path(__file__).resolve().parents[1] / "config" / "default.toml"
    )
    assert main(["eval", "--config", str(config_path)]) == 0


def test_main_train_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.toml"
    save_toml(config_path, {"run": {"name": "demo"}})

    assert main(["train", "--config", str(config_path)]) == 0

    runs_dir = tmp_path / "runs"
    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_root = run_dirs[0]
    assert (run_root / "config.toml").exists()
    assert (run_root / "events.toml").exists()
    assert (run_root / "metrics").exists()
    assert (run_root / "games" / "game_0000000001.pgn").exists()


def test_main_missing_run_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.toml"
    save_toml(config_path, {"env": {"id": "chess"}})
    with pytest.raises(ValueError, match="Missing TOML table"):
        _ = main(["eval", "--config", str(config_path)])


def test_main_missing_run_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.toml"
    save_toml(config_path, {"run": {"seed": 42}})
    with pytest.raises(ValueError, match="Missing string key"):
        _ = main(["eval", "--config", str(config_path)])
