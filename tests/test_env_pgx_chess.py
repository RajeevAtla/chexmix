"""Tests for PGX chess environment helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from env.pgx_chess import (
    compile_pgx_fns,
    make_chess_env,
    mask_illegal_logits,
)


def test_compile_pgx_fns_shapes() -> None:
    """Compiled PGX fns preserve expected batch shapes."""
    # Initialize two batched environments.
    env = make_chess_env()
    fns = compile_pgx_fns(env)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    state = fns.init_fn(keys)

    # Ensure legal action mask shapes are stable.
    assert state.legal_action_mask.shape == (2, 4672)

    legal_actions = jnp.argmax(state.legal_action_mask, axis=-1)
    next_state = fns.step_fn(state, legal_actions)
    # Step preserves mask shape.
    assert next_state.legal_action_mask.shape == (2, 4672)


def test_mask_illegal_logits() -> None:
    """mask_illegal_logits sets illegal logits to -inf."""
    # Use a tiny logits/mask example for determinism.
    logits = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)
    mask = jnp.array([[True, False, True], [False, False, True]])
    masked = mask_illegal_logits(logits, mask)
    neg_inf = jnp.finfo(logits.dtype).min

    assert masked.shape == logits.shape
    assert masked[0, 0] == 1.0
    assert masked[0, 1] == neg_inf
    assert masked[1, 0] == neg_inf
    assert masked[1, 2] == 6.0
