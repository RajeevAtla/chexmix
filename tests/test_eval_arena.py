from __future__ import annotations

from eval.arena import MatchResult


def test_match_result_summary() -> None:
    result = MatchResult(wins=3, draws=2, losses=1)
    assert result.total_games() == 6
    assert result.score() == 4.0
    assert result.win_rate() == 0.5


def test_match_result_empty() -> None:
    result = MatchResult(wins=0, draws=0, losses=0)
    assert result.total_games() == 0
    assert result.win_rate() == 0.0
