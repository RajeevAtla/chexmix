"""
Training state (params, optimizer state, step, rng).

Use dataclass(frozen=True) and return new instances to keep functional style.
"""

from __future__ import annotations

from dataclasses import dataclass

import optax
from flax import nnx

from chess_ai.types import PRNGKey, Step


@dataclass(frozen=True, slots=True)
class TrainState:
    """Immutable training state."""

    step: Step
    params: nnx.State
    opt_state: optax.OptState
    rng_key: PRNGKey
