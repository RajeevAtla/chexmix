"""Tests for MCTS legality and normalization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pgx
from flax import nnx

from mcts.planner import MctsConfig, run_mcts
from model.chess_transformer import ChessTransformer
from model.nnx_blocks import TransformerConfig


def test_mcts_outputs_legal_actions_and_weights() -> None:
    """MCTS returns legal actions and normalized weights."""
    # Build a small model to run MCTS on CPU.
    env = pgx.make("chess")
    cfg = TransformerConfig(d_model=32, n_heads=4, mlp_ratio=2, n_layers=2)
    model = ChessTransformer(cfg, rngs=nnx.Rngs(0))
    params = nnx.state(model)

    batch = 2
    keys = jax.random.split(jax.random.PRNGKey(0), batch)
    state = jax.vmap(env.init)(keys)
    mcts_cfg = MctsConfig(
        num_simulations=2, max_depth=2, c_puct=1.5, gumbel_scale=1.0
    )
    # Run MCTS and validate output shapes and legality.
    out = run_mcts(
        env=env,
        model=model,
        params=params,
        rng_key=jax.random.PRNGKey(1),
        state=state,
        cfg=mcts_cfg,
    )
    assert out.action.shape == (batch,)
    assert out.action_weights.shape == (batch, 64 * 73)
    # Ensure selected actions are legal and weights sum to 1.
    legal = state.legal_action_mask
    chosen_legal = jnp.take_along_axis(legal, out.action[:, None], axis=1)
    assert jnp.all(chosen_legal)
    weights_sum = jnp.sum(out.action_weights, axis=-1)
    assert jnp.allclose(weights_sum, 1.0, atol=1e-5)


def test_mcts_per_game_rng_keys() -> None:
    """MCTS accepts per-game RNG keys and returns legal outputs."""
    env = pgx.make("chess")
    cfg = TransformerConfig(d_model=32, n_heads=4, mlp_ratio=2, n_layers=2)
    model = ChessTransformer(cfg, rngs=nnx.Rngs(0))
    params = nnx.state(model)

    batch = 2
    keys = jax.random.split(jax.random.PRNGKey(0), batch)
    state = jax.vmap(env.init)(keys)
    mcts_cfg = MctsConfig(
        num_simulations=2, max_depth=2, c_puct=1.5, gumbel_scale=1.0
    )
    out = run_mcts(
        env=env,
        model=model,
        params=params,
        rng_key=keys,
        state=state,
        cfg=mcts_cfg,
    )
    assert out.action.shape == (batch,)
    assert out.action_weights.shape == (batch, 64 * 73)
    legal = state.legal_action_mask
    chosen_legal = jnp.take_along_axis(legal, out.action[:, None], axis=1)
    assert jnp.all(chosen_legal)


def test_mcts_rng_key_batch_mismatch() -> None:
    """MCTS raises when RNG key batch does not match state batch."""
    env = pgx.make("chess")
    cfg = TransformerConfig(d_model=32, n_heads=4, mlp_ratio=2, n_layers=2)
    model = ChessTransformer(cfg, rngs=nnx.Rngs(0))
    params = nnx.state(model)

    batch = 2
    keys = jax.random.split(jax.random.PRNGKey(0), batch)
    state = jax.vmap(env.init)(keys)
    mcts_cfg = MctsConfig(
        num_simulations=2, max_depth=2, c_puct=1.5, gumbel_scale=1.0
    )
    bad_keys = jax.random.split(jax.random.PRNGKey(1), batch + 1)
    try:
        _ = run_mcts(
            env=env,
            model=model,
            params=params,
            rng_key=bad_keys,
            state=state,
            cfg=mcts_cfg,
        )
    except ValueError as exc:
        assert "rng_key batch does not match state batch" in str(exc)
    else:
        raise AssertionError("expected ValueError for RNG key batch mismatch")
