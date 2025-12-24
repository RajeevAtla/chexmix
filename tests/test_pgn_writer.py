"""Tests for PGN encoding and formatting."""

from __future__ import annotations

from datetime import date

from pgn.encode import DecodedMove, decode_action
from pgn.writer import PgnHeaders, format_pgn, write_pgn_file


def test_decode_action_basic_planes() -> None:
    """decode_action handles basic move planes."""
    # Action 0 should be a simple forward move.
    move = decode_action(0)
    assert move == DecodedMove(
        from_file=0,
        from_rank=0,
        to_file=0,
        to_rank=1,
        promo="",
    )
    knight_move = decode_action(56)
    assert knight_move.from_file == 0
    assert knight_move.from_rank == 0
    assert (knight_move.to_file, knight_move.to_rank) == (2, 1)


def test_decode_action_promotion_plane() -> None:
    """decode_action handles promotion plane offsets."""
    # Promotion plane should set promo to rook.
    promo = decode_action(64)
    assert promo.promo == "r"
    assert promo.from_file == 0
    assert promo.from_rank == 0


def test_decode_action_invalid() -> None:
    """decode_action raises on invalid indices."""
    # Negative action should raise ValueError.
    try:
        _ = decode_action(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-range action")


def test_format_and_write_pgn(tmp_path) -> None:
    """format_pgn and write_pgn_file output valid PGN."""
    # Build headers and a short move list.
    headers = PgnHeaders(
        event="SelfPlay",
        site="Local",
        date=date(2025, 1, 1),
        round="1",
        white="White",
        black="Black",
        result="1-0",
    )
    moves = ["e2e4", "e7e5", "g1f3"]
    pgn = format_pgn(headers, moves)
    # Validate headers and movetext content.
    assert '[Event "SelfPlay"]' in pgn
    assert '[Result "1-0"]' in pgn
    assert "1. e2e4 e7e5 2. g1f3 1-0" in pgn
    path = tmp_path / "game.pgn"
    write_pgn_file(path, pgn)
    assert path.read_text(encoding="utf-8") == pgn
