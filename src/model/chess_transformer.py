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
        # Store config for downstream inspection/debugging.
        self.cfg = cfg
        # Token projection from 119 planes to model dimension.
        self.token_embed = nnx.Linear(119, cfg.d_model, rngs=rngs)
        # Learned positional embedding for each square.
        self.pos_embed = nnx.Param(
            jnp.zeros((64, cfg.d_model), dtype=jnp.float32)
        )
        # Stack of transformer blocks.
        self.blocks = nnx.List(
            [TransformerBlock(cfg, rngs=rngs) for _ in range(cfg.n_layers)]
        )
        # Policy head predicts per-square 73-plane logits.
        self.policy_head = nnx.Linear(cfg.d_model, 73, rngs=rngs)
        # Value head predicts a scalar value.
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
        # Flatten board into 64 tokens with 119 features each.
        tokens = obs.reshape(batch, 64, 119)
        x = self.token_embed(tokens)
        # Add positional embeddings for board squares.
        x = x + self.pos_embed[...]
        # Apply transformer blocks sequentially.
        for block in self.blocks:
            x = block(x)
        # Produce policy logits and flatten to 4672 actions.
        policy_logits = self.policy_head(x)
        policy_logits = policy_logits.reshape(batch, 64 * 73)
        # Pool token embeddings for value head.
        pooled = jnp.mean(x, axis=1)
        value = self.value_head(pooled).squeeze(-1)
        # Tanh keeps value in [-1, 1].
        value = jnp.tanh(value)
        return PolicyValue(policy_logits=policy_logits, value=value)
