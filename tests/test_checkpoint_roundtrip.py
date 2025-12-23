from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from chess_ai.train.checkpointing import CheckpointConfig, make_checkpoint_manager, restore_latest, save_checkpoint
from chess_ai.train.optimizer import OptimConfig, make_optimizer
from chess_ai.train.state import TrainState
from chess_ai.types import Array, Step


class TinyModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.w = nnx.Param(jnp.array([1.0], dtype=jnp.float32))

    def __call__(self, x: Array) -> Array:
        return x * self.w.value


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
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

    checkpoints_dir = tmp_path / "checkpoints"
    manager = make_checkpoint_manager(
        checkpoints_dir, CheckpointConfig(every_steps=1, max_to_keep=1)
    )

    save_checkpoint(manager, state)
    restored = restore_latest(manager)

    chex.assert_trees_all_equal(restored.params, state.params)
    chex.assert_trees_all_equal(restored.opt_state, state.opt_state)
    chex.assert_trees_all_equal(restored.rng_key, state.rng_key)
    assert restored.step == state.step


def test_restore_latest_missing(tmp_path: Path) -> None:
    manager = make_checkpoint_manager(
        tmp_path / "checkpoints",
        CheckpointConfig(every_steps=1, max_to_keep=1),
    )
    with pytest.raises(FileNotFoundError):
        _ = restore_latest(manager)
