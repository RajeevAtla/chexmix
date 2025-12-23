"""
Core training loop pieces.

Design:
- `train_step` is jitted and pmapped
- self-play runs on-device, replay sampling on host (initially)
- checkpoint + metrics writing on host
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from chess_ai.types import Array, PRNGKey, Step
from chess_ai.train.losses import LossConfig, Losses, compute_losses
from chess_ai.train.state import TrainState


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Training settings."""

    batch_size_per_device: int


def _apply_model(model: nnx.Module, params: nnx.State, obs: Array) -> tuple[Array, Array]:
    """Apply an NNX model with its parameters."""
    if hasattr(nnx, "apply"):
        output = nnx.apply(model, params)(obs)
        return output.policy_logits, output.value
    if hasattr(nnx, "merge"):
        try:
            merged = nnx.merge(model, params)
        except TypeError:
            merged = nnx.merge(params, model)
        output = merged(obs)
        return output.policy_logits, output.value
    output = model(obs)
    return output.policy_logits, output.value


def _params_l2(params: nnx.State) -> Array:
    """Compute L2 norm of parameter leaves."""
    leaves = jax.tree_util.tree_leaves(params)
    terms = [jnp.sum(jnp.square(x)) for x in leaves if isinstance(x, jax.Array)]
    if not terms:
        return jnp.array(0.0, dtype=jnp.float32)
    return jnp.sum(jnp.stack(terms))


def train_step(
    *,
    model: nnx.Module,
    tx: optax.GradientTransformation,
    state: TrainState,
    batch: dict[str, Array],
    loss_cfg: LossConfig,
) -> tuple[TrainState, Losses]:
    """One optimization step.

    Must be compatible with pmap:
    - batch is per-device shard
    - grads aggregated by lax.pmean(axis_name="data")
    """

    def loss_fn(params: nnx.State) -> tuple[Array, Losses]:
        policy_logits, value_pred = _apply_model(model, params, batch["obs"])
        losses = compute_losses(
            policy_logits=policy_logits,
            value_pred=value_pred,
            policy_targets=batch["policy_targets"],
            outcome=batch["outcome"],
            valid=batch["valid"],
            params_l2=_params_l2(params),
            cfg=loss_cfg,
        )
        return losses.total, losses

    (_, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name="data")
    losses = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, "data"), losses)

    updates, opt_state = tx.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)

    rng_key, next_rng = jax.random.split(state.rng_key)
    new_state = TrainState(
        step=Step(int(state.step) + 1),
        params=params,
        opt_state=opt_state,
        rng_key=next_rng,
    )
    _ = rng_key

    return new_state, losses
