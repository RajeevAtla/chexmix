from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pgx
import pytest
from flax import nnx

from chess_ai.mcts.planner import MctsConfig, MctsOutput
from chess_ai.selfplay.buffer import ReplayBuffer, ReplayConfig
from chess_ai.selfplay.rollout import SelfPlayConfig, generate_selfplay_trajectories
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

    chex.assert_trees_all_equal(traj1, traj2)
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
