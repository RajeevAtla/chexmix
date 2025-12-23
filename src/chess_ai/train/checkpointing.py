"""
Orbax checkpointing utilities.

Hard requirement:
- All checkpoints must live under /runs/<run_id>/checkpoints/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard, cast

import jax
import optax
import orbax.checkpoint as ocp
from flax import nnx

from chess_ai.train.state import TrainState
from chess_ai.types import PRNGKey, Step


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Checkpoint schedule."""

    every_steps: int
    max_to_keep: int


type CheckpointTree = dict[str, nnx.State | optax.OptState | PRNGKey | int]


def make_checkpoint_manager(
    checkpoints_dir: Path, cfg: CheckpointConfig
) -> ocp.CheckpointManager:
    """Create an Orbax CheckpointManager."""
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(
        max_to_keep=cfg.max_to_keep, create=True
    )
    return ocp.CheckpointManager(checkpoints_dir, checkpointer, options)


def _state_to_tree(state: TrainState) -> CheckpointTree:
    return {
        "step": int(state.step),
        "params": state.params,
        "opt_state": state.opt_state,
        "rng_key": state.rng_key,
    }


def _is_value_dict(value: object) -> TypeGuard[dict[str, object]]:
    if not isinstance(value, dict):
        return False
    return set(value.keys()) == {"value"}


def _normalize_pytree(value: object) -> object:
    if _is_value_dict(value):
        value_dict = cast(dict[str, object], value)
        leaf = value_dict["value"]
        if isinstance(leaf, jax.Array):
            return nnx.Param(leaf)
    if isinstance(value, list):
        return tuple(_normalize_pytree(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalize_pytree(item) for key, item in value.items()}
    return value


def _as_state_if_dict(value: object) -> object:
    if isinstance(value, dict):
        return nnx.State(value)
    return value


def _restore_opt_state(value: object) -> object:
    if value is None:
        return optax.EmptyState()
    if isinstance(value, list):
        return tuple(_restore_opt_state(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_restore_opt_state(item) for item in value)
    if isinstance(value, dict):
        value_dict = cast(dict[str, object], value)
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
        return {
            key: _restore_opt_state(item) for key, item in value_dict.items()
        }
    return value


def _tree_to_state(tree: CheckpointTree) -> TrainState:
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
    restored = manager.restore(step)
    if not isinstance(restored, dict):
        raise ValueError("Restored checkpoint is not a mapping.")
    return _tree_to_state(cast(CheckpointTree, restored))
