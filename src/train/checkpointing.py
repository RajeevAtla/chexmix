"""
Orbax checkpointing utilities.

Hard requirement:
- All checkpoints must live under /runs/<run_id>/checkpoints/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import orbax.checkpoint as ocp

from train.state import TrainState


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Checkpoint schedule."""

    every_steps: int
    max_to_keep: int


def make_checkpoint_manager(
    checkpoints_dir: Path, cfg: CheckpointConfig
) -> ocp.CheckpointManager:
    """Create an Orbax CheckpointManager."""
    registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    registry.add(
        "default", ocp.args.StandardRestore, ocp.StandardCheckpointHandler
    )
    registry.add(
        "default", ocp.args.StandardSave, ocp.StandardCheckpointHandler
    )
    options = ocp.CheckpointManagerOptions(
        max_to_keep=cfg.max_to_keep,
        create=True,
        enable_async_checkpointing=False,
    )
    return ocp.CheckpointManager(
        checkpoints_dir,
        options=options,
        handler_registry=registry,
    )


def save_checkpoint(
    manager: ocp.CheckpointManager,
    state: TrainState,
) -> None:
    """Save TrainState via Orbax."""
    manager.save(
        step=int(state.step),
        args=ocp.args.StandardSave(state),
    )
    manager.wait_until_finished()


def restore_latest(
    manager: ocp.CheckpointManager,
    target_state: TrainState,
) -> TrainState:
    """Restore the latest TrainState.

    Raises:
        FileNotFoundError: If no checkpoint exists.
    """
    step = manager.latest_step()
    if step is None:
        raise FileNotFoundError("No checkpoint found.")
    restored = manager.restore(
        step, args=ocp.args.StandardRestore(target_state)
    )
    if not isinstance(restored, TrainState):
        raise ValueError("Restored checkpoint is not a TrainState.")
    return restored
