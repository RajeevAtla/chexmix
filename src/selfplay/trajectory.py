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
        player_id: (B, T)  -- player-to-move at time t
        valid: (B, T)      -- timestep validity mask
        outcome: (B, T)    -- z in {-1,0,1} from perspective of player-to-move
    """

    obs: Array
    policy_targets: Array
    player_id: Array
    valid: Array
    outcome: Array

    def tree_flatten(self) -> tuple[tuple[Array, ...], None]:
        return (
            (
                self.obs,
                self.policy_targets,
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
        del aux_data
        obs, policy_targets, player_id, valid, outcome = children
        return cls(
            obs=obs,
            policy_targets=policy_targets,
            player_id=player_id,
            valid=valid,
            outcome=outcome,
        )
