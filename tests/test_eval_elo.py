from __future__ import annotations

from eval.elo import expected_score, update_elo


def test_expected_score_symmetry() -> None:
    assert expected_score(1000.0, 1000.0) == 0.5


def test_update_elo_win() -> None:
    rating_a, rating_b = update_elo(1000.0, 1000.0, score_a=1.0)
    assert rating_a > 1000.0
    assert rating_b < 1000.0
