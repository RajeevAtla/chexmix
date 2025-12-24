"""Tests for Elo expected score and update logic."""

from __future__ import annotations

from eval.elo import expected_score, update_elo


def test_expected_score_symmetry() -> None:
    """Expected score is 0.5 for equal ratings."""
    # Symmetry check for identical ratings.
    assert expected_score(1000.0, 1000.0) == 0.5


def test_update_elo_win() -> None:
    """Winning increases rating and decreases opponent rating."""
    # Score of 1.0 means A wins.
    rating_a, rating_b = update_elo(1000.0, 1000.0, score_a=1.0)
    assert rating_a > 1000.0
    assert rating_b < 1000.0
