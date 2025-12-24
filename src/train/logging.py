"""
Run directory creation + TOML metrics snapshots.

All metrics artifacts must be TOML and stored in /runs/<run_id>/metrics/.
"""

from __future__ import annotations

from dataclasses import dataclass

from paths import RunPaths
from toml_io import TomlValue, save_toml


@dataclass(frozen=True, slots=True)
class Metrics:
    """A minimal metrics bundle suitable for TOML serialization."""

    step: int
    loss_total: float
    loss_policy: float
    loss_value: float
    value_mean: float
    entropy_mean: float


def write_metrics_snapshot(paths: RunPaths, metrics: Metrics) -> None:
    """Write a metrics snapshot TOML file named by step."""
    # Use zero-padded filenames for lexicographic ordering.
    filename = f"step_{metrics.step:010d}.toml"
    path = paths.metrics_dir / filename
    # Serialize metrics into TOML-compatible dict.
    data: dict[str, TomlValue] = {
        "step": metrics.step,
        "loss_total": metrics.loss_total,
        "loss_policy": metrics.loss_policy,
        "loss_value": metrics.loss_value,
        "value_mean": metrics.value_mean,
        "entropy_mean": metrics.entropy_mean,
    }
    save_toml(path, data)
