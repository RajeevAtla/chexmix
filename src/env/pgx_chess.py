"""
PGX chess wrapper: vectorized init/step and small helpers.

The wrapper exists to:
- centralize vmap/jit compilation
- enforce consistent dtype/shape expectations
- provide utility conversions (masking, terminal checks)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pgx

from chex_types import Array


@dataclass(frozen=True, slots=True)
class PgxFns:
    """JIT-compiled, vectorized PGX callables."""

    init_fn: callable
    step_fn: callable


def make_chess_env() -> pgx.Env:
    """Create the PGX chess environment."""
    return pgx.make("chess")


def compile_pgx_fns(env: pgx.Env) -> PgxFns:
    """Compile vmap+jit wrappers for init and step.

    Args:
        env: PGX environment.

    Returns:
        PgxFns containing compiled init_fn and step_fn.
    """
    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))
    return PgxFns(init_fn=init_fn, step_fn=step_fn)


def mask_illegal_logits(policy_logits: Array, legal_action_mask: Array) -> Array:
    """Mask illegal actions by setting logits to a large negative value.

    Args:
        policy_logits: (B, 4672)
        legal_action_mask: (B, 4672) boolean

    Returns:
        Masked logits (B, 4672)
    """
    neg_inf = jnp.finfo(policy_logits.dtype).min
    return jnp.where(legal_action_mask, policy_logits, neg_inf)



