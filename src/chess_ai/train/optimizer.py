"""
Optax optimizer and LR schedule creation.
"""

from __future__ import annotations

from dataclasses import dataclass

import optax


@dataclass(frozen=True, slots=True)
class OptimConfig:
    """Optimizer hyperparameters."""

    learning_rate: float
    warmup_steps: int
    total_steps: int
    grad_clip_norm: float
    weight_decay: float


def make_optimizer(
    cfg: OptimConfig,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """Create (optimizer, schedule) tuple."""
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        decay_steps=cfg.total_steps,
        end_value=0.0,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(schedule, weight_decay=cfg.weight_decay),
    )
    return tx, schedule
