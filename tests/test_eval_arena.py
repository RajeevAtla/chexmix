"""Tests for evaluation arena statistics."""

from __future__ import annotations

from eval.arena import MatchResult


def test_match_result_summary() -> None:
    """MatchResult aggregates wins/draws/losses correctly."""
    # Use a small example to validate math helpers.
    result = MatchResult(wins=3, draws=2, losses=1)
    assert result.total_games() == 6
    assert result.score() == 4.0
    assert result.win_rate() == 0.5


def test_match_result_empty() -> None:
    """Empty MatchResult yields zero totals and win rate."""
    # Zeroed results should not divide by zero.
    result = MatchResult(wins=0, draws=0, losses=0)
    assert result.total_games() == 0
    assert result.win_rate() == 0.0
