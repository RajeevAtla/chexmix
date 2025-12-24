"""
NNX transformer building blocks.

Hard requirements:
- Use flax.nnx ONLY (no flax.linen)
- Fully typed
- No dropout by default (determinism)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from chex_types import Array


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """Transformer hyperparameters."""

    d_model: int
    n_heads: int
    mlp_ratio: int
    n_layers: int


class RMSNorm(nnx.Module):
    """RMSNorm layer."""

    def __init__(self, d_model: int, *, rngs: nnx.Rngs) -> None:
        """Initialize RMSNorm scale parameters.

        Args:
            d_model: Model hidden size.
            rngs: NNX RNGs (unused, required by API).
        """
        del rngs
        # Trainable scale vector with deterministic init.
        self.scale = nnx.Param(jnp.ones((d_model,), dtype=jnp.float32))
        self._eps = 1e-6

    def __call__(self, x: Array) -> Array:
        """Apply RMSNorm."""
        # Normalize by root-mean-square across features.
        mean_sq = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        rms = jnp.sqrt(mean_sq + self._eps)
        return x / rms * self.scale.value


class MultiHeadSelfAttention(nnx.Module):
    """Multi-head self-attention (MHSA)."""

    def __init__(self, d_model: int, n_heads: int, *, rngs: nnx.Rngs) -> None:
        """Initialize MHSA projection matrices."""
        if d_model % n_heads != 0:
            msg = "d_model must be divisible by n_heads"
            raise ValueError(msg)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nnx.Linear(d_model, 3 * d_model, use_bias=False, rngs=rngs)
        self.proj = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        """Compute MHSA output.

        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        # Project to queries, keys, and values.
        bsz, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        # Reshape for multi-head attention.
        q = q.reshape(bsz, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(bsz, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(bsz, seq_len, self.n_heads, self.head_dim)
        # Scaled dot-product attention.
        scale = jnp.sqrt(jnp.array(self.head_dim, dtype=x.dtype))
        attn_logits = jnp.einsum("bthd,bshd->bhts", q / scale, k)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_out = jnp.einsum("bhts,bshd->bthd", attn_weights, v)
        attn_out = attn_out.reshape(bsz, seq_len, d_model)
        # Final projection back to model dimension.
        return self.proj(attn_out)


class MlpBlock(nnx.Module):
    """Transformer MLP block."""

    def __init__(self, d_model: int, mlp_ratio: int, *, rngs: nnx.Rngs) -> None:
        """Initialize MLP projections.

        Args:
            d_model: Model hidden size.
            mlp_ratio: Expansion ratio for hidden layer.
            rngs: NNX RNGs used for parameter init.
        """
        # Expand then project back to model dimension.
        hidden = d_model * mlp_ratio
        self.fc1 = nnx.Linear(d_model, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, d_model, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        """Apply MLP block."""
        # GELU activation matches transformer defaults.
        return self.fc2(jax.nn.gelu(self.fc1(x)))


class TransformerBlock(nnx.Module):
    """Pre-norm transformer block."""

    def __init__(self, cfg: TransformerConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize attention/MLP sublayers.

        Args:
            cfg: TransformerConfig with dimensions.
            rngs: NNX RNGs used for parameter init.
        """
        # Pre-norm residual layout improves stability.
        self.norm1 = RMSNorm(cfg.d_model, rngs=rngs)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, rngs=rngs)
        self.norm2 = RMSNorm(cfg.d_model, rngs=rngs)
        self.mlp = MlpBlock(cfg.d_model, cfg.mlp_ratio, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        """Apply attention + MLP with residual connections."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
