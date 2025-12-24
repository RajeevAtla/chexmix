"""Tests for deterministic RNG stream helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chex_types import Step
from rng import RngStream


def test_rng_stream_determinism() -> None:
    """RngStream yields deterministic keys for the same step."""
    # Same step should yield identical keys.
    base_key = jax.random.PRNGKey(42)
    stream = RngStream(base_key=base_key)

    key_a1 = stream.key_for_step(Step(1))
    key_a2 = stream.key_for_step(Step(1))
    key_b = stream.key_for_step(Step(2))

    assert jnp.array_equal(key_a1, key_a2)
    assert not jnp.array_equal(key_a1, key_b)


def test_rng_stream_device_key() -> None:
    """RngStream yields deterministic device keys per axis index."""
    # Axis-specific fold-in should differ by device index.
    base_key = jax.random.PRNGKey(42)
    stream = RngStream(base_key=base_key)
    step_key = stream.key_for_step(Step(3))

    key0 = stream.key_for_device(step_key, 0)
    key0_again = stream.key_for_device(step_key, 0)
    key1 = stream.key_for_device(step_key, 1)

    assert jnp.array_equal(key0, key0_again)
    assert not jnp.array_equal(key0, key1)
