"""Checkpointing and restore behavior tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import chex
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pytest
from flax import nnx
from orbax.checkpoint import args as args_lib

from chex_types import Array, Step
from train.checkpointing import (
    CheckpointConfig,
    CheckpointTree,
    _as_state_if_dict,
    _is_value_dict,
    _normalize_pytree,
    _restore_args_from_metadata,
    _restore_opt_state,
    _tree_to_state,
    make_checkpoint_manager,
    restore_latest,
    save_checkpoint,
    ShardingLike,
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
    restored = restore_latest(manager)

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
        _ = restore_latest(manager)


def test_tree_to_state_missing_step() -> None:
    """_tree_to_state rejects missing step field."""
    tree = cast(
        CheckpointTree,
        {
            "params": nnx.State({}),
            "opt_state": (),
            "rng_key": jax.random.PRNGKey(0),
        },
    )
    with pytest.raises(ValueError, match="integer step"):
        _ = _tree_to_state(tree)


def test_tree_to_state_invalid_params_structure() -> None:
    """_tree_to_state rejects invalid params dict structure."""
    tree = cast(
        CheckpointTree,
        {
            "step": 0,
            "params": {"value": jnp.array([1.0], dtype=jnp.float32)},
            "opt_state": (),
            "rng_key": jax.random.PRNGKey(0),
        },
    )
    with pytest.raises(ValueError, match="invalid structure"):
        _ = _tree_to_state(tree)


def test_tree_to_state_missing_params() -> None:
    """_tree_to_state rejects missing params state."""
    tree = cast(
        CheckpointTree,
        {
            "step": 0,
            "params": 1,
            "opt_state": (),
            "rng_key": jax.random.PRNGKey(0),
        },
    )
    with pytest.raises(ValueError, match="params state"):
        _ = _tree_to_state(tree)


def test_tree_to_state_missing_opt_state() -> None:
    """_tree_to_state rejects missing optimizer state."""
    tree = cast(
        CheckpointTree,
        {
            "step": 0,
            "params": nnx.State({}),
            "opt_state": None,
            "rng_key": jax.random.PRNGKey(0),
        },
    )
    with pytest.raises(ValueError, match="optimizer state"):
        _ = _tree_to_state(tree)


def test_tree_to_state_invalid_params_state_structure() -> None:
    """_tree_to_state rejects malformed nnx.State."""
    tree = cast(
        CheckpointTree,
        {
            "step": 0,
            "params": nnx.State({"value": jnp.array([1.0], dtype=jnp.float32)}),
            "opt_state": (),
            "rng_key": jax.random.PRNGKey(0),
        },
    )
    with pytest.raises(ValueError, match="invalid structure"):
        _ = _tree_to_state(tree)


def test_tree_to_state_missing_rng_key() -> None:
    """_tree_to_state rejects missing RNG key."""
    tree = cast(
        CheckpointTree,
        {
            "step": 0,
            "params": nnx.State({}),
            "opt_state": (),
            "rng_key": 123,
        },
    )
    with pytest.raises(ValueError, match="RNG key"):
        _ = _tree_to_state(tree)


def test_restore_latest_invalid_mapping() -> None:
    """restore_latest rejects non-mapping restore payloads."""

    class DummyManager:
        """CheckpointManager stub returning invalid data."""

        def latest_step(self) -> int:
            """Return a fixed latest step."""
            return 0

        def restore(self, step: int, **_kwargs: args_lib.PyTreeRestore) -> int:
            """Return an invalid payload instead of a mapping."""
            del step
            return 123

        def item_metadata(self, step: int) -> None:
            """Return dummy metadata for restore args."""
            del step
            return None

    with pytest.raises(ValueError, match="not a mapping"):
        _ = restore_latest(cast(ocp.CheckpointManager, DummyManager()))


def test_checkpoint_helpers_paths() -> None:
    """Helper utilities handle edge cases for pytrees and opt state."""

    # Validate restore args generation from sharding metadata.
    class FakeLeaf:
        """Leaf container with sharding metadata."""

        def __init__(self, sharding: ShardingLike | None) -> None:
            """Store sharding metadata on the leaf.

            Args:
                sharding: Sharding metadata object.
            """
            self.sharding = sharding

    class FakeMeta:
        """Metadata wrapper holding a tree attribute."""

        def __init__(self, tree: dict[str, FakeLeaf | None]) -> None:
            """Store tree metadata structure.

            Args:
                tree: Tree structure to wrap.
            """
            self.tree = tree

    restore_args = cast(
        dict[str, ocp.ArrayRestoreArgs | None],
        _restore_args_from_metadata(
            FakeMeta({"x": FakeLeaf(sharding=cast(ShardingLike, object())), "y": None})
        ),
    )
    assert isinstance(restore_args["x"], ocp.ArrayRestoreArgs)
    assert restore_args["y"] is None

    # Validate helper behavior for edge cases.
    assert not _is_value_dict(1)
    assert _normalize_pytree([1, 2]) == (1, 2)
    assert _normalize_pytree({"a": 1}) == {"a": 1}
    assert _as_state_if_dict(1) == 1
    restored = _restore_opt_state((None, None))
    assert isinstance(restored, tuple)
    assert isinstance(restored[0], optax.EmptyState)
    fallback = _restore_opt_state({"foo": 1})
    fallback_dict = cast(dict[str, int], fallback)
    assert fallback_dict["foo"] == 1
