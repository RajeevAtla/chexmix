"""
Action encoding/decoding utilities for the 64x73 AlphaZero-style plane encoding.

This module MUST be tested extensively against PGX legal_action_mask.
"""

from __future__ import annotations

from dataclasses import dataclass

# Directions for queen-like moves (7 distances each).
_QUEEN_DIRS = (
    (0, 1),  # N
    (0, -1),  # S
    (1, 0),  # E
    (-1, 0),  # W
    (1, 1),  # NE
    (-1, 1),  # NW
    (1, -1),  # SE
    (-1, -1),  # SW
)

# Knight move offsets.
_KNIGHT_DIRS = (
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
)

# Promotion pieces encoded in underpromotion planes.
_PROMO_PIECES = ("r", "b", "n")
# Promotion directions: forward and diagonals.
_PROMO_DIRS = (
    (0, 1),  # forward
    (-1, 1),  # forward-left
    (1, 1),  # forward-right
)


@dataclass(frozen=True, slots=True)
class DecodedMove:
    """A decoded move in board coordinates.

    Coordinates:
        file: 0..7 (a..h)
        rank: 0..7 (1..8) in white's perspective

    Promotions:
        promo is one of: "q", "r", "b", "n" or "" for none.
    """

    from_file: int
    from_rank: int
    to_file: int
    to_rank: int
    promo: str


def decode_action(action: int) -> DecodedMove:
    """Decode an action index (0..4671) into a move.

    Raises:
        ValueError: if action is out of range.
    """
    # Validate range before decoding.
    if action < 0 or action >= 64 * 73:
        raise ValueError("action out of range")
    # Decode from-square and plane index.
    from_square = action // 73
    plane = action % 73
    from_rank = from_square // 8
    from_file = from_square % 8
    if plane < 56:
        # Queen-like sliding moves are encoded in 8 dirs * 7 distances.
        dir_idx = plane // 7
        dist = (plane % 7) + 1
        dx, dy = _QUEEN_DIRS[dir_idx]
        to_file = from_file + dx * dist
        to_rank = from_rank + dy * dist
        return DecodedMove(
            from_file=from_file,
            from_rank=from_rank,
            to_file=to_file,
            to_rank=to_rank,
            promo="",
        )
    if plane < 64:
        # Knight moves are stored in 8 planes after sliding moves.
        knight_idx = plane - 56
        dx, dy = _KNIGHT_DIRS[knight_idx]
        return DecodedMove(
            from_file=from_file,
            from_rank=from_rank,
            to_file=from_file + dx,
            to_rank=from_rank + dy,
            promo="",
        )
    # Promotion planes cover forward and diagonal promotions.
    promo_plane = plane - 64
    promo_idx = promo_plane // 3
    dir_idx = promo_plane % 3
    dx, dy = _PROMO_DIRS[dir_idx]
    return DecodedMove(
        from_file=from_file,
        from_rank=from_rank,
        to_file=from_file + dx,
        to_rank=from_rank + dy,
        promo=_PROMO_PIECES[promo_idx],
    )
