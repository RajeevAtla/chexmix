from __future__ import annotations

from typing import cast

import chex
import jax
import jax.numpy as jnp
import pgx
import pytest
from flax import nnx

from chess_ai.mcts.planner import MctsConfig, MctsOutput
from chess_ai.selfplay.buffer import ReplayBuffer, ReplayConfig
from chess_ai.selfplay.rollout import (
    SelfPlayConfig,
    _tree_where,
    generate_selfplay_trajectories,
)
from chess_ai.selfplay.trajectory import Trajectory
from chess_ai.types import Array, PolicyValue


class DummyModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.bias = nnx.Param(jnp.array(0.0))

    def __call__(self, obs: Array) -> PolicyValue:
        batch = obs.shape[0]
        policy = jnp.zeros((batch, 4672), dtype=jnp.float32)
        value = jnp.zeros((batch,), dtype=jnp.float32)
        return PolicyValue(policy_logits=policy, value=value)


def _fake_run_mcts(
    *,
    env: pgx.Env,
    model: nnx.Module,
    params: nnx.State,
    rng_key: Array,
    state: pgx.State,
    cfg: MctsConfig,
) -> MctsOutput:
    del env, model, params, rng_key, cfg
    legal = state.legal_action_mask
    fallback = jax.nn.one_hot(
        jnp.zeros((legal.shape[0],), dtype=jnp.int32), legal.shape[1]
    )
    legal_sum = jnp.sum(legal, axis=-1, keepdims=True)
    safe_legal = jnp.where(legal_sum > 0, legal, fallback)
    action = jnp.argmax(safe_legal, axis=-1)
    weights = safe_legal / jnp.sum(safe_legal, axis=-1, keepdims=True)
    return MctsOutput(action=action, action_weights=weights)


def test_selfplay_rollout_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "chess_ai.selfplay.rollout.run_mcts",
        _fake_run_mcts,
    )
    env = pgx.make("chess")
    model = DummyModel(rngs=nnx.Rngs(0))
    params = nnx.state(model)

    selfplay_cfg = SelfPlayConfig(games_per_device=2, max_moves=4)
    mcts_cfg = MctsConfig(
        num_simulations=1,
        max_depth=1,
        c_puct=1.0,
        gumbel_scale=0.0,
    )
    rng_key = jax.random.PRNGKey(0)

    traj1 = generate_selfplay_trajectories(
        env=env,
        model=model,
        params=params,
        rng_key=rng_key,
        selfplay_cfg=selfplay_cfg,
        mcts_cfg=mcts_cfg,
    )
    traj2 = generate_selfplay_trajectories(
        env=env,
        model=model,
        params=params,
        rng_key=rng_key,
        selfplay_cfg=selfplay_cfg,
        mcts_cfg=mcts_cfg,
    )

    chex.assert_trees_all_equal(traj1.obs, traj2.obs)
    chex.assert_trees_all_equal(traj1.policy_targets, traj2.policy_targets)
    chex.assert_trees_all_equal(traj1.player_id, traj2.player_id)
    chex.assert_trees_all_equal(traj1.valid, traj2.valid)
    chex.assert_trees_all_equal(traj1.outcome, traj2.outcome)
    assert traj1.obs.shape == (2, 4, 8, 8, 119)
    assert traj1.policy_targets.shape == (2, 4, 4672)
    assert traj1.player_id.shape == (2, 4)
    assert traj1.valid.shape == (2, 4)
    assert traj1.outcome.shape == (2, 4)


def test_replay_buffer_sample() -> None:
    obs = jnp.zeros((1, 2, 8, 8, 119), dtype=jnp.float32)
    policy = jnp.full((1, 2, 4672), 1.0 / 4672.0, dtype=jnp.float32)
    player_id = jnp.zeros((1, 2), dtype=jnp.int32)
    valid = jnp.array([[1, 0]], dtype=jnp.bool_)
    outcome = jnp.array([[1.0, 0.0]], dtype=jnp.float32)
    traj = Trajectory(
        obs=obs,
        policy_targets=policy,
        player_id=player_id,
        valid=valid,
        outcome=outcome,
    )

    buffer = ReplayBuffer(ReplayConfig(capacity=4, min_to_sample=1))
    buffer.add(traj)
    assert buffer.can_sample()
    batch = buffer.sample_batch(jax.random.PRNGKey(0), batch_size=1)

    assert batch["obs"].shape == (1, 8, 8, 119)
    assert batch["policy_targets"].shape == (1, 4672)
    assert batch["outcome"].shape == (1,)
    assert batch["valid"].shape == (1,)


def _make_traj(
    *,
    steps: int,
    valid_mask: Array,
) -> Trajectory:
    obs = jnp.zeros((1, steps, 8, 8, 119), dtype=jnp.float32)
    policy = jnp.full((1, steps, 4672), 1.0 / 4672.0, dtype=jnp.float32)
    player_id = jnp.zeros((1, steps), dtype=jnp.int32)
    outcome = jnp.zeros((1, steps), dtype=jnp.float32)
    return Trajectory(
        obs=obs,
        policy_targets=policy,
        player_id=player_id,
        valid=valid_mask,
        outcome=outcome,
    )


def test_replay_buffer_empty_valid_add() -> None:
    buffer = ReplayBuffer(ReplayConfig(capacity=4, min_to_sample=1))
    valid = jnp.zeros((1, 2), dtype=jnp.bool_)
    traj = _make_traj(steps=2, valid_mask=valid)
    buffer.add(traj)
    assert not buffer.can_sample()


def test_replay_buffer_sample_empty_raises() -> None:
    buffer = ReplayBuffer(ReplayConfig(capacity=4, min_to_sample=1))
    with pytest.raises(ValueError, match="ReplayBuffer is empty"):
        _ = buffer.sample_batch(jax.random.PRNGKey(0), batch_size=1)


def test_replay_buffer_size_zero_raises() -> None:
    buffer = ReplayBuffer(ReplayConfig(capacity=4, min_to_sample=1))
    valid = jnp.ones((1, 1), dtype=jnp.bool_)
    traj = _make_traj(steps=1, valid_mask=valid)
    buffer.add(traj)
    buffer._size = 0
    with pytest.raises(ValueError, match="ReplayBuffer is empty"):
        _ = buffer.sample_batch(jax.random.PRNGKey(0), batch_size=1)


def test_replay_buffer_wrap_and_capacity() -> None:
    buffer = ReplayBuffer(ReplayConfig(capacity=3, min_to_sample=1))
    valid = jnp.ones((1, 2), dtype=jnp.bool_)
    buffer.add(_make_traj(steps=2, valid_mask=valid))
    buffer.add(_make_traj(steps=2, valid_mask=valid))
    assert buffer.can_sample()

    buffer_full = ReplayBuffer(ReplayConfig(capacity=2, min_to_sample=1))
    valid_full = jnp.ones((1, 3), dtype=jnp.bool_)
    buffer_full.add(_make_traj(steps=3, valid_mask=valid_full))
    assert buffer_full.can_sample()


def test_replay_buffer_insert_uninitialized() -> None:
    buffer = ReplayBuffer(ReplayConfig(capacity=2, min_to_sample=1))
    obs = jnp.zeros((1, 8, 8, 119), dtype=jnp.float32)
    policy = jnp.zeros((1, 4672), dtype=jnp.float32)
    outcome = jnp.zeros((1,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="storage not initialized"):
        buffer._insert(obs, policy, outcome, count=1)


def test_tree_where_scalar_branch() -> None:
    mask = jnp.array([True, False])
    new = {"x": jnp.array(1.0), "y": jnp.array([1.0, 2.0])}
    old = {"x": jnp.array(2.0), "y": jnp.array([3.0, 4.0])}
    out = _tree_where(
        mask, cast(pgx.State, new), cast(pgx.State, old)
    )
    out_dict = cast(dict[str, Array], out)
    assert out_dict["x"].shape == (2,)
