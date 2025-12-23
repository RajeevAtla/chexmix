"""
Training state (params, optimizer state, step, rng).

Use dataclass(frozen=True) and return new instances to keep functional style.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import optax
from flax import nnx

from chex_types import PRNGKey, Step


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class TrainState:
    """Immutable training state."""

    step: Step
    params: nnx.State
    opt_state: optax.OptState
    rng_key: PRNGKey

    def tree_flatten(self) -> tuple[tuple[object, ...], int]:
        children = (self.params, self.opt_state, self.rng_key)
        aux_data = int(self.step)
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: int, children: tuple[object, ...]
    ) -> TrainState:
        if not isinstance(aux_data, int):
            raise TypeError("Invalid step type in TrainState pytree.")
        params, opt_state, rng_key = children
        return cls(
            step=Step(aux_data),
            params=cast(nnx.State, params),
            opt_state=cast(optax.OptState, opt_state),
            rng_key=cast(PRNGKey, rng_key),
        )
