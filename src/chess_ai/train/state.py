"""
Training state (params, optimizer state, step, rng).

Use dataclass(frozen=True) and return new instances to keep functional style.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from chess_ai.types import PRNGKey, Step


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class TrainState:
    """Immutable training state."""

    step: Step
    params: nnx.State
    opt_state: optax.OptState
    rng_key: PRNGKey

    def tree_flatten(self) -> tuple[tuple[object, ...], None]:
        children = (int(self.step), self.params, self.opt_state, self.rng_key)
        return children, None

    @classmethod
    def tree_unflatten(
        cls, aux_data: None, children: tuple[object, ...]
    ) -> TrainState:
        del aux_data
        step, params, opt_state, rng_key = children
        if isinstance(step, jax.Array):
            step_value = int(jnp.ravel(step)[0])
        elif isinstance(step, int):
            step_value = step
        else:
            raise TypeError("Invalid step type in TrainState pytree.")
        return cls(
            step=Step(step_value),
            params=cast(nnx.State, params),
            opt_state=cast(optax.OptState, opt_state),
            rng_key=cast(PRNGKey, rng_key),
        )
