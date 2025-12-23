"""
Chess transformer policy/value network.

Input:
- observation: (B, 8, 8, 119)

Tokenization:
- flatten board squares => 64 tokens
- per-token features = 119
- linear projection to d_model
- add learned positional embeddings for 64 squares

Outputs:
- policy_logits: (B, 4672) where 4672 = 64 * 73
- value: (B,) in [-1, 1]
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from chex_types import Array, PolicyValue
from model.nnx_blocks import TransformerBlock, TransformerConfig


class ChessTransformer(nnx.Module):
    """Transformer encoder with policy and value heads."""

    def __init__(self, cfg: TransformerConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize embeddings, blocks, and heads."""
        self.cfg = cfg
        self.token_embed = nnx.Linear(119, cfg.d_model, rngs=rngs)
        self.pos_embed = nnx.Param(
            jnp.zeros((64, cfg.d_model), dtype=jnp.float32)
        )
        self.blocks = nnx.List(
            [TransformerBlock(cfg, rngs=rngs) for _ in range(cfg.n_layers)]
        )
        self.policy_head = nnx.Linear(cfg.d_model, 73, rngs=rngs)
        self.value_head = nnx.Linear(cfg.d_model, 1, rngs=rngs)

    def __call__(self, obs: Array) -> PolicyValue:
        """Forward pass.

        Args:
            obs: PGX observation (B, 8, 8, 119)

        Returns:
            PolicyValue containing:
              - policy_logits: (B, 4672)
              - value: (B,)
        """
        batch = obs.shape[0]
        tokens = obs.reshape(batch, 64, 119)
        x = self.token_embed(tokens)
        x = x + self.pos_embed.value
        for block in self.blocks:
            x = block(x)
        policy_logits = self.policy_head(x)
        policy_logits = policy_logits.reshape(batch, 64 * 73)
        pooled = jnp.mean(x, axis=1)
        value = self.value_head(pooled).squeeze(-1)
        value = jnp.tanh(value)
        return PolicyValue(policy_logits=policy_logits, value=value)
