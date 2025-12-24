"""Tests for loss computation utilities."""

from __future__ import annotations

import chex
import jax.numpy as jnp

from train.losses import LossConfig, compute_losses


def test_compute_losses_masking_and_values() -> None:
    """compute_losses applies masking and weights correctly."""
    # Build a tiny deterministic batch.
    policy_logits = jnp.array([[0.0, 0.0], [0.0, 0.0]], dtype=jnp.float32)
    policy_targets = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    value_pred = jnp.array([0.5, -0.5], dtype=jnp.float32)
    outcome = jnp.array([1.0, -1.0], dtype=jnp.float32)
    valid = jnp.array([1.0, 0.0], dtype=jnp.float32)
    params_l2 = jnp.array(2.0, dtype=jnp.float32)
    cfg = LossConfig(value_loss_weight=2.0, weight_decay=0.1)

    # Compute losses using the provided config.
    losses = compute_losses(
        policy_logits=policy_logits,
        value_pred=value_pred,
        policy_targets=policy_targets,
        outcome=outcome,
        valid=valid,
        params_l2=params_l2,
        cfg=cfg,
    )

    # Expected values from hand calculation.
    expected_policy = jnp.log(jnp.array(2.0))
    expected_value = jnp.array(0.25)
    expected_l2 = jnp.array(0.2)
    expected_total = expected_policy + 2.0 * expected_value + expected_l2

    chex.assert_trees_all_close(
        losses.policy, expected_policy, rtol=1e-6, atol=1e-6
    )
    chex.assert_trees_all_close(
        losses.value, expected_value, rtol=1e-6, atol=1e-6
    )
    chex.assert_trees_all_close(losses.l2, expected_l2, rtol=1e-6, atol=1e-6)
    chex.assert_trees_all_close(
        losses.total, expected_total, rtol=1e-6, atol=1e-6
    )
