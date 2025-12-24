"""Tests for training step and TrainState utilities."""

from __future__ import annotations

from typing import cast

import chex
import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from chex_types import Array, PolicyValue, Step
from train.learner import _params_l2, train_step
from train.losses import LossConfig, Losses
from train.optimizer import OptimConfig, make_optimizer
from train.state import TrainState


class LinearModel(nnx.Module):
    """Tiny linear model for training step tests."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        """Initialize a single scalar weight.

        Args:
            rngs: NNX RNGs (unused).
        """
        # Deterministic parameter for easy expectation checks.
        self.w = nnx.Param(jnp.array(0.1, dtype=jnp.float32))

    def __call__(self, obs: Array) -> PolicyValue:
        """Return constant logits and values.

        Args:
            obs: Observation batch.

        Returns:
            PolicyValue with constant logits and values.
        """
        batch = obs.shape[0]
        logits = jnp.ones((batch, 2), dtype=jnp.float32) * self.w.value
        value = jnp.ones((batch,), dtype=jnp.float32) * self.w.value
        return PolicyValue(policy_logits=logits, value=value)


def _make_state() -> (
    tuple[LinearModel, TrainState, optax.GradientTransformation]
):
    """Construct a model, TrainState, and optimizer for tests.

    Returns:
        Tuple of model, TrainState, and optimizer transformation.
    """
    # Build model and optimizer with minimal config.
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
    """Create a minimal training batch.

    Args:
        batch_size: Number of samples to create.

    Returns:
        Batch dict with obs, policy_targets, outcome, and valid.
    """
    # Use trivial values for deterministic losses.
    obs = jnp.zeros((batch_size, 1), dtype=jnp.float32)
    policy_targets = (
        jnp.zeros((batch_size, 2), dtype=jnp.float32).at[:, 0].set(1.0)
    )
    outcome = jnp.zeros((batch_size,), dtype=jnp.float32)
    valid = jnp.ones((batch_size,), dtype=jnp.float32)
    return {
        "obs": obs,
        "policy_targets": policy_targets,
        "outcome": outcome,
        "valid": valid,
    }


def test_train_step_pmap_equivalence() -> None:
    """pmap training matches single-device results."""
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("Need at least 2 devices for pmap equivalence test.")

    # Compare multi-device and single-device pmap outputs.
    model, state, tx = _make_state()
    batch = _make_batch(batch_size=2)
    loss_cfg = LossConfig(value_loss_weight=1.0, weight_decay=0.0)

    def step_fn(
        s: TrainState, b: dict[str, Array]
    ) -> tuple[TrainState, Losses]:
        """Wrapper to call train_step with closed-over config."""
        return train_step(
            model=model, tx=tx, state=s, batch=b, loss_cfg=loss_cfg
        )

    p_step_multi = jax.pmap(step_fn, axis_name="data", devices=devices)
    p_step_single = jax.pmap(step_fn, axis_name="data", devices=[devices[0]])

    state_multi = jax.device_put_replicated(state, devices)
    batch_multi = jax.device_put_replicated(batch, devices)
    state_single = jax.device_put_replicated(state, [devices[0]])
    batch_single = jax.device_put_replicated(batch, [devices[0]])

    out_multi = p_step_multi(state_multi, batch_multi)
    out_single = p_step_single(state_single, batch_single)

    # Compare the first replica across outputs.
    state_multi_0, losses_multi_0 = jax.tree_util.tree_map(
        lambda x: x[0], out_multi
    )
    state_single_0, losses_single_0 = jax.tree_util.tree_map(
        lambda x: x[0], out_single
    )

    chex.assert_trees_all_close(state_multi_0.params, state_single_0.params)
    chex.assert_trees_all_close(losses_multi_0.total, losses_single_0.total)


def test_train_step_single_device() -> None:
    """train_step works with a single-device pmap."""
    devices = jax.devices()
    if not devices:
        pytest.skip("No JAX devices available.")

    # Run a single-device pmap and validate loss finiteness.
    model, state, tx = _make_state()
    batch = _make_batch(batch_size=1)
    loss_cfg = LossConfig(value_loss_weight=1.0, weight_decay=0.0)

    def step_fn(
        s: TrainState, b: dict[str, Array]
    ) -> tuple[TrainState, Losses]:
        """Wrapper to call train_step with closed-over config."""
        return train_step(
            model=model, tx=tx, state=s, batch=b, loss_cfg=loss_cfg
        )

    p_step = jax.pmap(
        step_fn,
        axis_name="data",
        in_axes=(None, 0),
        out_axes=(None, None),
        devices=[devices[0]],
    )
    batch_repl = jax.tree_util.tree_map(lambda x: x[None, ...], batch)
    state_out, losses_out = p_step(state, batch_repl)
    chex.assert_trees_all_close(state_out.params, state.params)
    assert jnp.isfinite(losses_out.total).all()


def test_train_state_tree_unflatten_invalid_step() -> None:
    """TrainState.tree_unflatten rejects invalid step types."""
    _model, state, _ = _make_state()
    with pytest.raises(TypeError, match="Invalid step type"):
        _ = TrainState.tree_unflatten(
            cast(int, "bad"),
            (state.params, state.opt_state, state.rng_key),
        )


def test_params_l2_empty_state() -> None:
    """_params_l2 returns zero for empty state."""
    empty = nnx.State({})
    value = _params_l2(empty)
    assert value == 0.0
