"""
Trajectory dataclasses for self-play.

All arrays are padded to fixed length T = env.max_moves.
A boolean mask indicates valid timesteps.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax

from chex_types import Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class Trajectory:
    """A batch of self-play games stored as fixed-length tensors.

    Shapes:
        obs: (B, T, 8, 8, 119)
        policy_targets: (B, T, 4672)
        actions: (B, T)
        legal_action_mask: (B, T, 4672)
        player_id: (B, T)  -- player-to-move at time t
        valid: (B, T)      -- timestep validity mask
        outcome: (B, T)    -- z in {-1,0,1} from perspective of player-to-move
    """

    obs: Array
    policy_targets: Array
    actions: Array
    legal_action_mask: Array
    player_id: Array
    valid: Array
    outcome: Array

    def tree_flatten(self) -> tuple[tuple[Array, ...], None]:
        """Flatten Trajectory for JAX pytree registration.

        Returns:
            Tuple of children arrays and None aux data.
        """
        # Preserve a stable field ordering for pytree ops.
        return (
            (
                self.obs,
                self.policy_targets,
                self.actions,
                self.legal_action_mask,
                self.player_id,
                self.valid,
                self.outcome,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls, aux_data: None, children: tuple[Array, ...]
    ) -> Trajectory:
        """Reconstruct a Trajectory from pytree children.

        Args:
            aux_data: Unused auxiliary data.
            children: Tuple of arrays in field order.

        Returns:
            Reconstructed Trajectory instance.
        """
        del aux_data
        # Unpack in the same order as tree_flatten.
        (
            obs,
            policy_targets,
            actions,
            legal_action_mask,
            player_id,
            valid,
            outcome,
        ) = children
        return cls(
            obs=obs,
            policy_targets=policy_targets,
            actions=actions,
            legal_action_mask=legal_action_mask,
            player_id=player_id,
            valid=valid,
            outcome=outcome,
        )
