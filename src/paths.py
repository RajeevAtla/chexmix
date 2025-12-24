"""
Centralized path conventions for /runs outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Default base directories for configs and run artifacts.
RUNS_DIR: Final[Path] = Path("runs")
CONFIG_DIR: Final[Path] = Path("config")


@dataclass(frozen=True, slots=True)
class RunPaths:
    """All filesystem paths for a single run."""

    root: Path
    checkpoints: Path
    metrics_dir: Path
    games_dir: Path
    events_toml: Path
    config_toml: Path

    @staticmethod
    def create(run_id: str) -> RunPaths:
        """Create run directories under /runs/<run_id>."""
        # Compute all run subdirectories and files.
        # Resolve to an absolute path for Orbax/Tensorstore compatibility.
        root = (RUNS_DIR / run_id).resolve()
        checkpoints = root / "checkpoints"
        metrics_dir = root / "metrics"
        games_dir = root / "games"
        events_toml = root / "events.toml"
        config_toml = root / "config.toml"

        # Ensure all directories exist.
        checkpoints.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        games_dir.mkdir(parents=True, exist_ok=True)

        return RunPaths(
            root=root,
            checkpoints=checkpoints,
            metrics_dir=metrics_dir,
            games_dir=games_dir,
            events_toml=events_toml,
            config_toml=config_toml,
        )
