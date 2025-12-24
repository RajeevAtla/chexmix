"""Tests for TOML metrics logging."""

from __future__ import annotations

from pathlib import Path

from paths import RunPaths
from train.logging import Metrics, write_metrics_snapshot


def test_write_metrics_snapshot(tmp_path: Path) -> None:
    """Metrics snapshots are written with expected filename."""
    # Build a RunPaths pointing at a temp metrics directory.
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    paths = RunPaths(
        root=tmp_path,
        checkpoints=tmp_path / "checkpoints",
        metrics_dir=metrics_dir,
        games_dir=tmp_path / "games",
        events_toml=tmp_path / "events.toml",
        config_toml=tmp_path / "config.toml",
    )
    # Write a simple metrics payload.
    metrics = Metrics(
        step=5,
        loss_total=1.0,
        loss_policy=0.5,
        loss_value=0.25,
        value_mean=0.1,
        entropy_mean=0.2,
    )
    write_metrics_snapshot(paths, metrics)

    # Validate the file and content.
    path = metrics_dir / "step_0000000005.toml"
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "loss_total" in content
