"""
Replay buffer for trajectories.

Design constraint:
- Keep host-side buffer metadata minimal
- Store arrays in a JAX-friendly layout where possible
- Deterministic sampling given RNG
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from chex_types import Array, PRNGKey
from selfplay.trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Replay buffer capacity and sampling parameters."""

    capacity: int
    min_to_sample: int


class ReplayBuffer:
    """A simple ring buffer for trajectory timesteps."""

    def __init__(self, cfg: ReplayConfig) -> None:
        """Initialize empty buffer."""
        # Basic ring buffer bookkeeping.
        self._capacity = cfg.capacity
        self._min_to_sample = cfg.min_to_sample
        self._size = 0
        self._write_idx = 0
        self._obs: np.ndarray | None = None
        self._policy: np.ndarray | None = None
        self._outcome: np.ndarray | None = None

    def add(self, traj: Trajectory) -> None:
        """Insert a batch of trajectories into the buffer."""
        # Flatten time dimension and filter by valid mask.
        obs_flat = jnp.reshape(traj.obs, (-1, *traj.obs.shape[2:]))
        policy_flat = jnp.reshape(
            traj.policy_targets, (-1, traj.policy_targets.shape[-1])
        )
        outcome_flat = jnp.reshape(traj.outcome, (-1,))
        valid_flat = jnp.reshape(traj.valid, (-1,))

        # Pull validity to host for filtering.
        valid_np = np.asarray(jax.device_get(valid_flat)).astype(bool)
        if not valid_np.any():
            return

        # Keep only valid entries and truncate to capacity.
        obs_np = np.asarray(jax.device_get(obs_flat))[valid_np]
        policy_np = np.asarray(jax.device_get(policy_flat))[valid_np]
        outcome_np = np.asarray(jax.device_get(outcome_flat))[valid_np]
        count = int(obs_np.shape[0])
        if count >= self._capacity:
            obs_np = obs_np[-self._capacity :]
            policy_np = policy_np[-self._capacity :]
            outcome_np = outcome_np[-self._capacity :]
            count = self._capacity
            self._write_idx = 0
            self._size = self._capacity

        # Lazily initialize storage on first insert.
        if self._obs is None:
            self._obs = np.zeros(
                (self._capacity, *obs_np.shape[1:]), dtype=obs_np.dtype
            )
            self._policy = np.zeros(
                (self._capacity, *policy_np.shape[1:]), dtype=policy_np.dtype
            )
            self._outcome = np.zeros((self._capacity,), dtype=outcome_np.dtype)

        # Insert into ring buffer and update pointers.
        self._insert(obs_np, policy_np, outcome_np, count)
        self._size = min(self._capacity, self._size + count)
        self._write_idx = (self._write_idx + count) % self._capacity

    def can_sample(self) -> bool:
        """Return True if buffer has enough data to sample."""
        # Require minimum number of samples to start training.
        return self._size >= self._min_to_sample

    def sample_batch(
        self, rng_key: PRNGKey, batch_size: int
    ) -> dict[str, Array]:
        """Sample a training batch.

        Returns:
            A dict with strictly-typed arrays:
              - obs: (B, 8, 8, 119)
              - policy_targets: (B, 4672)
              - outcome: (B,)
              - valid: (B,)
        """
        # Guard against sampling before any data is added.
        if self._obs is None or self._policy is None or self._outcome is None:
            raise ValueError("ReplayBuffer is empty.")
        if self._size == 0:
            raise ValueError("ReplayBuffer is empty.")

        # Sample indices deterministically from RNG key.
        indices = jax.random.randint(rng_key, (batch_size,), 0, self._size)
        indices_np = np.asarray(jax.device_get(indices))

        # Gather samples into contiguous arrays.
        obs = jnp.asarray(self._obs[indices_np])
        policy = jnp.asarray(self._policy[indices_np])
        outcome = jnp.asarray(self._outcome[indices_np])
        valid = jnp.ones((batch_size,), dtype=jnp.bool_)

        return {
            "obs": obs,
            "policy_targets": policy,
            "outcome": outcome,
            "valid": valid,
        }

    def _insert(
        self,
        obs_np: np.ndarray,
        policy_np: np.ndarray,
        outcome_np: np.ndarray,
        count: int,
    ) -> None:
        """Insert numpy data into the ring buffer."""
        # Ensure storage is initialized before insert.
        if self._obs is None or self._policy is None or self._outcome is None:
            raise ValueError("ReplayBuffer storage not initialized.")

        # Write in two segments if wrapping around the ring buffer.
        first = min(self._capacity - self._write_idx, count)
        second = count - first

        self._obs[self._write_idx : self._write_idx + first] = obs_np[:first]
        self._policy[self._write_idx : self._write_idx + first] = policy_np[
            :first
        ]
        self._outcome[self._write_idx : self._write_idx + first] = outcome_np[
            :first
        ]

        if second > 0:
            self._obs[:second] = obs_np[first:]
            self._policy[:second] = policy_np[first:]
            self._outcome[:second] = outcome_np[first:]
