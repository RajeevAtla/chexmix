"""
Core type aliases and small immutable dataclasses.

Hard requirements:
- No Any
- No object-typed containers to bypass type checking
- Prefer explicit TypeAlias, NamedTuple/dataclass(frozen=True), and protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

import jax

type Array = jax.Array
type PRNGKey = jax.Array  # shape (2,), dtype uint32

Step = NewType("Step", int)
GameId = NewType("GameId", int)


@dataclass(frozen=True, slots=True)
class PolicyValue:
    """Model output.

    Attributes:
        policy_logits: Unnormalized logits for all actions (4672).
        value: Scalar value estimate in [-1, 1].
    """

    policy_logits: Array  # (B, 4672)
    value: Array  # (B,)
