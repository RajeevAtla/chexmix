from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from chess_ai.train.learner import train_step
from chess_ai.train.losses import LossConfig, Losses
from chess_ai.train.optimizer import OptimConfig, make_optimizer
from chess_ai.train.state import TrainState
from chess_ai.types import Array, PolicyValue, Step


class LinearModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.w = nnx.Param(jnp.array(0.1, dtype=jnp.float32))

    def __call__(self, obs: Array) -> PolicyValue:
        batch = obs.shape[0]
        logits = jnp.ones((batch, 2), dtype=jnp.float32) * self.w.value
        value = jnp.ones((batch,), dtype=jnp.float32) * self.w.value
        return PolicyValue(policy_logits=logits, value=value)


def _make_state() -> tuple[LinearModel, TrainState, jax.Array]:
    model = LinearModel(rngs=nnx.Rngs(0))
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
    return model, state, tx


def _make_batch(batch_size: int) -> dict[str, Array]:
    obs = jnp.zeros((batch_size, 1), dtype=jnp.float32)
    policy_targets = jnp.zeros((batch_size, 2), dtype=jnp.float32).at[:, 0].set(1.0)
    outcome = jnp.zeros((batch_size,), dtype=jnp.float32)
    valid = jnp.ones((batch_size,), dtype=jnp.float32)
    return {
        "obs": obs,
        "policy_targets": policy_targets,
        "outcome": outcome,
        "valid": valid,
    }


def test_train_step_pmap_equivalence() -> None:
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("Need at least 2 devices for pmap equivalence test.")

    model, state, tx = _make_state()
    batch = _make_batch(batch_size=2)
    loss_cfg = LossConfig(value_loss_weight=1.0, weight_decay=0.0)

    def step_fn(s: TrainState, b: dict[str, Array]) -> tuple[TrainState, Losses]:
        return train_step(model=model, tx=tx, state=s, batch=b, loss_cfg=loss_cfg)

    p_step_multi = jax.pmap(step_fn, axis_name="data", devices=devices)
    p_step_single = jax.pmap(step_fn, axis_name="data", devices=[devices[0]])

    state_multi = jax.device_put_replicated(state, devices)
    batch_multi = jax.device_put_replicated(batch, devices)
    state_single = jax.device_put_replicated(state, [devices[0]])
    batch_single = jax.device_put_replicated(batch, [devices[0]])

    out_multi = p_step_multi(state_multi, batch_multi)
    out_single = p_step_single(state_single, batch_single)

    state_multi_0, losses_multi_0 = jax.tree_util.tree_map(lambda x: x[0], out_multi)
    state_single_0, losses_single_0 = jax.tree_util.tree_map(lambda x: x[0], out_single)

    chex.assert_trees_all_close(state_multi_0.params, state_single_0.params)
    chex.assert_trees_all_close(losses_multi_0.total, losses_single_0.total)
