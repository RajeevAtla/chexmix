"""
Self-play rollout using MCTS at each move.

Hard requirements:
- Fully vectorized across games within each device shard
- Deterministic given the RNG stream
- Produces Trajectory with policy targets from MCTS action_weights
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pgx
from flax import nnx

from chess_ai.mcts.planner import MctsConfig, MctsOutput, run_mcts
from chess_ai.selfplay.trajectory import Trajectory
from chess_ai.types import Array, PRNGKey


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    """Self-play settings."""

    games_per_device: int
    max_moves: int


def _tree_where(mask: Array, new: pgx.State, old: pgx.State) -> pgx.State:
    """Select between two PGX states using a per-batch mask."""

    def _select(new_leaf: Array, old_leaf: Array) -> Array:
        if new_leaf.ndim == 0:
            return jnp.where(mask, new_leaf, old_leaf)
        expanded = mask.reshape((mask.shape[0],) + (1,) * (new_leaf.ndim - 1))
        return jnp.where(expanded, new_leaf, old_leaf)

    return jax.tree_util.tree_map(_select, new, old)


def _safe_policy_targets(action_weights: Array, valid: Array) -> Array:
    """Mask policy targets for invalid timesteps."""
    return jnp.where(valid[:, None], action_weights, jnp.zeros_like(action_weights))


def _zero_outcome(
    outcome: Array, valid: Array, dtype: jnp.dtype
) -> Array:
    """Zero invalid outcomes."""
    return jnp.where(valid, outcome, jnp.zeros_like(outcome, dtype=dtype))


def generate_selfplay_trajectories(
    *,
    env: pgx.Env,
    model: nnx.Module,
    params: nnx.State,
    rng_key: PRNGKey,
    selfplay_cfg: SelfPlayConfig,
    mcts_cfg: MctsConfig,
) -> Trajectory:
    """Run batched self-play and return a Trajectory batch."""
    batch_size = selfplay_cfg.games_per_device
    max_moves = selfplay_cfg.max_moves

    init_keys = jax.random.split(rng_key, batch_size)
    state = jax.vmap(env.init)(init_keys)

    def step_fn(
        carry: pgx.State, step_idx: Array
    ) -> tuple[pgx.State, tuple[Array, Array, Array, Array]]:
        obs = carry.observation
        player_id = carry.current_player
        valid = jnp.logical_not(carry.terminated)

        step_key = jax.random.fold_in(rng_key, step_idx)
        mcts_out: MctsOutput = run_mcts(
            env=env,
            model=model,
            params=params,
            rng_key=step_key,
            state=carry,
            cfg=mcts_cfg,
        )
        policy_targets = _safe_policy_targets(mcts_out.action_weights, valid)

        next_state = jax.vmap(env.step)(carry, mcts_out.action)
        carry = _tree_where(valid, next_state, carry)

        return carry, (obs, policy_targets, player_id, valid)

    steps = jnp.arange(max_moves)
    final_state, history = jax.lax.scan(step_fn, state, steps)
    obs, policy_targets, player_id, valid = history
    obs = jnp.swapaxes(obs, 0, 1)
    policy_targets = jnp.swapaxes(policy_targets, 0, 1)
    player_id = jnp.swapaxes(player_id, 0, 1)
    valid = jnp.swapaxes(valid, 0, 1)

    rewards = final_state.rewards
    rewards_expanded = rewards[:, None, :]
    player_id_expanded = player_id[..., None]
    outcome = jnp.take_along_axis(
        rewards_expanded, player_id_expanded, axis=-1
    ).squeeze(-1)
    outcome = _zero_outcome(outcome, valid, rewards.dtype)

    return Trajectory(
        obs=obs,
        policy_targets=policy_targets,
        player_id=player_id,
        valid=valid,
        outcome=outcome,
    )
