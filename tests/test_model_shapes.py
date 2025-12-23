from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from model.chess_transformer import ChessTransformer
from model.nnx_blocks import MultiHeadSelfAttention, TransformerConfig


def test_chess_transformer_shapes_and_determinism() -> None:
    cfg = TransformerConfig(d_model=32, n_heads=4, mlp_ratio=2, n_layers=2)
    model = ChessTransformer(cfg, rngs=nnx.Rngs(0))
    obs = jnp.zeros((2, 8, 8, 119), dtype=jnp.float32)
    out1 = model(obs)
    out2 = model(obs)
    assert out1.policy_logits.shape == (2, 64 * 73)
    assert out1.value.shape == (2,)
    assert jnp.allclose(out1.policy_logits, out2.policy_logits)
    assert jnp.allclose(out1.value, out2.value)


def test_mhsa_head_divisibility() -> None:
    try:
        _ = MultiHeadSelfAttention(30, 8, rngs=nnx.Rngs(0))
    except ValueError as exc:
        assert "d_model must be divisible" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid head config")
