"""
Orbax checkpointing utilities.

Hard requirement:
- All checkpoints must live under /runs/<run_id>/checkpoints/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeGuard, cast, runtime_checkable

import jax
import optax
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Sharding
from orbax.checkpoint import args as args_lib
from orbax.checkpoint._src.metadata.sharding import ShardingMetadata

from chex_types import Step
from train.state import TrainState


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Checkpoint schedule."""

    every_steps: int
    max_to_keep: int


type ShardingLike = Sharding | ShardingMetadata


@runtime_checkable
class _HasSharding(Protocol):
    """Protocol for metadata leaves that expose sharding information."""

    sharding: ShardingLike | None


@runtime_checkable
class _HasTree(Protocol):
    """Protocol for metadata objects that wrap a tree attribute."""

    tree: MetadataTree


type PyTree = (
    jax.Array
    | nnx.Param
    | nnx.State
    | optax.OptState
    | int
    | float
    | bool
    | str
    | None
    | tuple["PyTree", ...]
    | list["PyTree"]
    | dict[str, "PyTree"]
)
type CheckpointTree = dict[str, PyTree]
type MetadataLeaf = _HasSharding | None
type MetadataTree = (
    MetadataLeaf
    | dict[str, "MetadataTree"]
    | list["MetadataTree"]
    | tuple["MetadataTree", ...]
)
type RestoreArgsLeaf = ocp.ArrayRestoreArgs | None
type RestoreArgsTree = (
    RestoreArgsLeaf
    | dict[str, "RestoreArgsTree"]
    | list["RestoreArgsTree"]
    | tuple["RestoreArgsTree", ...]
)


def make_checkpoint_manager(
    checkpoints_dir: Path, cfg: CheckpointConfig
) -> ocp.CheckpointManager:
    """Create an Orbax CheckpointManager."""
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(
        max_to_keep=cfg.max_to_keep,
        create=True,
        enable_async_checkpointing=False,
    )
    return ocp.CheckpointManager(checkpoints_dir, checkpointer, options)


def _restore_args_from_metadata(
    metadata: MetadataTree | _HasTree,
) -> RestoreArgsTree:
    """Build a restore_args pytree from Orbax metadata.

    Args:
        metadata: TreeMetadata or a tree-like structure containing leaves
            with optional sharding info.

    Returns:
        A pytree matching the metadata structure with ArrayRestoreArgs
        where sharding metadata is available, otherwise None leaves.
    """
    # TreeMetadata stores the actual tree under a `.tree` attribute.
    tree = metadata.tree if isinstance(metadata, _HasTree) else metadata

    def _leaf_to_args(leaf: MetadataLeaf) -> RestoreArgsLeaf:
        if isinstance(leaf, _HasSharding) and leaf.sharding is not None:
            return ocp.ArrayRestoreArgs(sharding=leaf.sharding)
        return None

    return jax.tree_util.tree_map(_leaf_to_args, tree)


def _state_to_tree(state: TrainState) -> CheckpointTree:
    """Convert TrainState to a checkpointable dict.

    Args:
        state: TrainState to serialize.

    Returns:
        Dict of checkpoint fields.
    """
    # Use simple primitives for Orbax PyTree serialization.
    return {
        "step": int(state.step),
        "params": state.params,
        "opt_state": state.opt_state,
        "rng_key": state.rng_key,
    }


def _is_value_dict(value: PyTree) -> TypeGuard[dict[str, PyTree]]:
    """Check for a serialized nnx.Param wrapper dict.

    Args:
        value: Candidate object.

    Returns:
        True if the dict only contains a "value" key.
    """
    if not isinstance(value, dict):
        return False
    return set(value.keys()) == {"value"}


def _normalize_pytree(value: PyTree) -> PyTree:
    """Normalize a pytree to restore nnx.Param leaves.

    Args:
        value: Serialized pytree value.

    Returns:
        Normalized pytree with nnx.Param where appropriate.
    """
    # Re-wrap serialized Params so NNX has the correct leaf types.
    if _is_value_dict(value):
        value_dict = cast(dict[str, PyTree], value)
        leaf = value_dict["value"]
        if isinstance(leaf, jax.Array):
            return nnx.Param(leaf)
    # Recursively normalize containers.
    if isinstance(value, list):
        items = cast(list[PyTree], value)
        return tuple(_normalize_pytree(item) for item in items)
    if isinstance(value, dict):
        items = cast(dict[str, PyTree], value)
        return {key: _normalize_pytree(item) for key, item in items.items()}
    return value


def _as_state_if_dict(value: PyTree) -> PyTree:
    """Convert dicts to nnx.State for optimizer state restoration.

    Args:
        value: Candidate object.

    Returns:
        nnx.State if value is a dict, otherwise value.
    """
    # Orbax may restore params as raw dicts; wrap them if needed.
    if isinstance(value, dict):
        return nnx.State(value)
    return value


def _restore_opt_state(value: PyTree) -> PyTree:
    """Restore Optax optimizer state from a serialized structure.

    Args:
        value: Serialized optimizer state.

    Returns:
        Restored optimizer state object.
    """
    # Handle the absence of optimizer state explicitly.
    if value is None:
        return optax.EmptyState()
    # Recursively rebuild tuples/lists and Optax state structs.
    if isinstance(value, list):
        items = cast(list[PyTree], value)
        return tuple(_restore_opt_state(item) for item in items)
    if isinstance(value, tuple):
        items = cast(tuple[PyTree, ...], value)
        return tuple(_restore_opt_state(item) for item in items)
    if isinstance(value, dict):
        value_dict = cast(dict[str, PyTree], value)
        if set(value_dict.keys()) == {"count", "mu", "nu"}:
            mu = _as_state_if_dict(_normalize_pytree(value_dict["mu"]))
            nu = _as_state_if_dict(_normalize_pytree(value_dict["nu"]))
            return optax.ScaleByAdamState(
                count=cast(jax.Array, value_dict["count"]),
                mu=cast(nnx.State, mu),
                nu=cast(nnx.State, nu),
            )
        if set(value_dict.keys()) == {"count"}:
            return optax.ScaleByScheduleState(
                count=cast(jax.Array, value_dict["count"])
            )
        # Fall back to restoring nested mappings.
        return {
            key: _restore_opt_state(item) for key, item in value_dict.items()
        }
    return value


def _tree_to_state(tree: CheckpointTree) -> TrainState:
    """Convert a checkpoint tree back into a TrainState.

    Args:
        tree: Mapping loaded from Orbax.

    Returns:
        Reconstructed TrainState.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    # Validate required fields and normalize parameter containers.
    step_value = tree.get("step")
    params = tree.get("params")
    opt_state = tree.get("opt_state")
    rng_key = tree.get("rng_key")
    if not isinstance(step_value, int):
        raise ValueError("Checkpoint missing integer step.")
    if isinstance(params, dict):
        normalized = _normalize_pytree(params)
        if not isinstance(normalized, dict):
            raise ValueError("Checkpoint params have invalid structure.")
        params_state = nnx.State(normalized)
    elif isinstance(params, nnx.State):
        normalized = _normalize_pytree(params.raw_mapping)
        if not isinstance(normalized, dict):
            raise ValueError("Checkpoint params have invalid structure.")
        params_state = nnx.State(normalized)
    else:
        raise ValueError("Checkpoint missing params state.")
    # Rebuild optimizer state and RNG key.
    if opt_state is None:
        raise ValueError("Checkpoint missing optimizer state.")
    opt_state = cast(optax.OptState, _restore_opt_state(opt_state))
    if not isinstance(rng_key, jax.Array):
        raise ValueError("Checkpoint missing RNG key.")
    return TrainState(
        step=Step(step_value),
        params=params_state,
        opt_state=opt_state,
        rng_key=rng_key,
    )


def save_checkpoint(
    manager: ocp.CheckpointManager,
    state: TrainState,
) -> None:
    """Save TrainState via Orbax."""
    manager.save(step=int(state.step), items=_state_to_tree(state))
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
    # Use metadata-backed restore args to provide explicit sharding info.
    metadata = manager.item_metadata(step)
    restore_args = _restore_args_from_metadata(metadata)
    restored = manager.restore(
        step, args=args_lib.PyTreeRestore(restore_args=restore_args)
    )
    if not isinstance(restored, dict):
        raise ValueError("Restored checkpoint is not a mapping.")
    return _tree_to_state(cast(CheckpointTree, restored))
