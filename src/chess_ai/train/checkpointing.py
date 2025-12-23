"""
Orbax checkpointing utilities.

Hard requirement:
- All checkpoints must live under /runs/<run_id>/checkpoints/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import orbax.checkpoint as ocp

from chess_ai.train.state import TrainState


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Checkpoint schedule."""

    every_steps: int
    max_to_keep: int


def make_checkpoint_manager(
    checkpoints_dir: Path, cfg: CheckpointConfig
) -> ocp.CheckpointManager:
    """Create an Orbax CheckpointManager."""
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(
        max_to_keep=cfg.max_to_keep, create=True
    )
    return ocp.CheckpointManager(checkpoints_dir, checkpointer, options)


def save_checkpoint(
    manager: ocp.CheckpointManager,
    state: TrainState,
) -> None:
    """Save TrainState via Orbax."""
    manager.save(step=int(state.step), items=state)
    manager.wait_until_finished()


def restore_latest(
    manager: ocp.CheckpointManager,
) -> TrainState:
    """Restore the latest TrainState.

    Raises:
        FileNotFoundError: If no checkpoint exists.
    """
    step = manager.latest_step()
    if step is None:
        raise FileNotFoundError("No checkpoint found.")
    restored = manager.restore(step)
    if not isinstance(restored, TrainState):
        raise ValueError("Restored object is not a TrainState.")
    return restored
