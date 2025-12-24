"""
Minimal Elo utilities.
"""

from __future__ import annotations


def expected_score(rating_a: float, rating_b: float) -> float:
    """Compute expected score for player A against player B."""
    # Standard Elo expected score formula.
    exponent = (rating_b - rating_a) / 400.0
    return 1.0 / (1.0 + 10.0**exponent)


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k_factor: float = 32.0,
) -> tuple[float, float]:
    """Return updated Elo ratings for a single game.

    Args:
        rating_a: Current rating for player A.
        rating_b: Current rating for player B.
        score_a: Game score for player A (1.0 win, 0.5 draw, 0.0 loss).
        k_factor: Elo update factor.
    """
    # Compute expected score and apply the K-factor update.
    expected_a = expected_score(rating_a, rating_b)
    delta = k_factor * (score_a - expected_a)
    return rating_a + delta, rating_b - delta
