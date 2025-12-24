"""
Deterministic RNG utilities.

All randomness MUST flow through these helpers and explicit PRNGKey passing.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax

from chex_types import PRNGKey, Step


@dataclass(frozen=True, slots=True)
class RngStream:
    """A deterministic RNG stream derived from a base key.

    This class is intentionally small and purely functional: calling methods
    returns new keys without mutating state.
    """

    base_key: PRNGKey

    def key_for_step(self, step: Step) -> PRNGKey:
        """Derive a deterministic key for a given global step.

        Args:
            step: Global training step counter.

        Returns:
            A PRNGKey derived via fold_in.
        """
        # Fold in the step to keep deterministic per-step keys.
        return jax.random.fold_in(self.base_key, int(step))

    def key_for_device(self, step_key: PRNGKey, axis_index: int) -> PRNGKey:
        """Derive a deterministic device-specific key.

        Args:
            step_key: The per-step key.
            axis_index: The integer device index on the pmap axis.

        Returns:
            A PRNGKey derived via fold_in.
        """
        # Fold in the device index to shard randomness across devices.
        return jax.random.fold_in(step_key, axis_index)
