"""Checkpointing and restore behavior tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import chex
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest
from flax import nnx

from chex_types import Array, Step
from train.checkpointing import (
    CheckpointConfig,
    make_checkpoint_manager,
    restore_latest,
    save_checkpoint,
)
from train.optimizer import OptimConfig, make_optimizer
from train.state import TrainState


class TinyModel(nnx.Module):
    """Minimal model used for checkpoint tests."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        """Initialize a single weight parameter.

        Args:
            rngs: NNX RNGs (unused).
        """
        # Deterministic scalar parameter for roundtrip testing.
        self.w = nnx.Param(jnp.array([1.0], dtype=jnp.float32))

    def __call__(self, x: Array) -> Array:
        """Apply the scalar weight.

        Args:
            x: Input array.

        Returns:
            Scaled array.
        """
        return x * self.w[...]


def _dummy_state() -> TrainState:
    """Create a minimal TrainState for restore tests."""
    return TrainState(
        step=Step(0),
        params=nnx.State({}),
        opt_state=(),
        rng_key=jax.random.PRNGKey(0),
    )


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Checkpoint save/restore roundtrips a TrainState."""
    # Build a tiny model and optimizer state.
    model = TinyModel(rngs=nnx.Rngs(0))
    params = nnx.state(model)

    tx, _ = make_optimizer(
        OptimConfig(
            learning_rate=1e-3,
            warmup_steps=1,
            total_steps=2,
            grad_clip_norm=1.0,
            weight_decay=0.0,
        )
    )
    opt_state = tx.init(params)
    state = TrainState(
        step=Step(0),
        params=params,
        opt_state=opt_state,
        rng_key=jax.random.PRNGKey(0),
    )

    # Create a manager and write a checkpoint.
    checkpoints_dir = tmp_path / "checkpoints"
    manager = make_checkpoint_manager(
        checkpoints_dir, CheckpointConfig(every_steps=1, max_to_keep=1)
    )

    save_checkpoint(manager, state)
    restored = restore_latest(manager, state)

    # All fields should roundtrip exactly.
    chex.assert_trees_all_equal(restored.params, state.params)
    chex.assert_trees_all_equal(restored.opt_state, state.opt_state)
    chex.assert_trees_all_equal(restored.rng_key, state.rng_key)
    assert restored.step == state.step


def test_restore_latest_missing(tmp_path: Path) -> None:
    """restore_latest raises when no checkpoint exists."""
    manager = make_checkpoint_manager(
        tmp_path / "checkpoints",
        CheckpointConfig(every_steps=1, max_to_keep=1),
    )
    with pytest.raises(FileNotFoundError):
        _ = restore_latest(manager, _dummy_state())


def test_restore_latest_invalid_mapping() -> None:
    """restore_latest rejects non-TrainState restore payloads."""

    class DummyManager:
        """CheckpointManager stub returning invalid data."""

        def latest_step(self) -> int:
            """Return a fixed latest step."""
            return 0

        def restore(
            self, step: int, **_kwargs: ocp.args.StandardRestore
        ) -> int:
            """Return an invalid payload instead of a TrainState."""
            del step
            return 123

    with pytest.raises(ValueError, match="TrainState"):
        _ = restore_latest(
            cast(ocp.CheckpointManager, DummyManager()),
            _dummy_state(),
        )
