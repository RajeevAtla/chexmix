"""
Loss functions for policy/value training.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from chess_ai.types import Array


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Loss weights and regularization strengths."""

    value_loss_weight: float
    weight_decay: float


@dataclass(frozen=True, slots=True)
class Losses:
    """Computed losses."""

    total: Array
    policy: Array
    value: Array
    l2: Array


def compute_losses(
    *,
    policy_logits: Array,
    value_pred: Array,
    policy_targets: Array,
    outcome: Array,
    valid: Array,
    params_l2: Array,
    cfg: LossConfig,
) -> Losses:
    """Compute total, policy, value, and L2 losses.

    Args:
        policy_logits: (B, 4672)
        value_pred: (B,)
        policy_targets: (B, 4672)
        outcome: (B,)
        valid: (B,) boolean/float mask
        params_l2: scalar L2 norm term (already computed with correct param masking)
        cfg: LossConfig

    Returns:
        Losses struct.
    """
    log_probs = jax.nn.log_softmax(policy_logits, axis=-1)
    policy_loss = -jnp.sum(policy_targets * log_probs, axis=-1)
    value_loss = jnp.square(value_pred - outcome)

    mask = valid.astype(jnp.float32)
    denom = jnp.maximum(mask.sum(), 1.0)

    policy_mean = jnp.sum(policy_loss * mask) / denom
    value_mean = jnp.sum(value_loss * mask) / denom
    l2 = cfg.weight_decay * params_l2

    total = policy_mean + cfg.value_loss_weight * value_mean + l2
    return Losses(total=total, policy=policy_mean, value=value_mean, l2=l2)
