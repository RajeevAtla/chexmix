"""
PGN writer for logged self-play games.

Outputs a PGN string and writes to /runs/<run_id>/games/.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PgnHeaders:
    """Standard PGN headers."""

    event: str
    site: str
    date: date
    round: str
    white: str
    black: str
    result: str  # "1-0", "0-1", "1/2-1/2", "*"


def _format_header_line(key: str, value: str) -> str:
    """Format a single PGN header line.

    Args:
        key: PGN header key.
        value: PGN header value.

    Returns:
        Formatted header line.
    """
    # Wrap values in quotes per PGN spec conventions.
    return f'[{key} "{value}"]'


def _format_movetext(moves: list[str], result: str) -> str:
    """Format movetext lines with move numbers and result.

    Args:
        moves: List of move tokens (SAN or coordinate).
        result: Game result token (e.g., "1-0").

    Returns:
        Movetext string suitable for PGN output.
    """
    # Group moves into white/black pairs with move numbers.
    parts: list[str] = []
    for idx in range(0, len(moves), 2):
        move_no = (idx // 2) + 1
        white_move = moves[idx]
        if idx + 1 < len(moves):
            black_move = moves[idx + 1]
            parts.append(f"{move_no}. {white_move} {black_move}")
        else:
            parts.append(f"{move_no}. {white_move}")
    parts.append(result)
    return " ".join(parts)


def format_pgn(headers: PgnHeaders, moves: list[str]) -> str:
    """Format headers + movetext into a PGN string."""
    # Emit required headers in deterministic order.
    header_lines = [
        _format_header_line("Event", headers.event),
        _format_header_line("Site", headers.site),
        _format_header_line("Date", headers.date.isoformat()),
        _format_header_line("Round", headers.round),
        _format_header_line("White", headers.white),
        _format_header_line("Black", headers.black),
        _format_header_line("Result", headers.result),
    ]
    # Movetext always ends with the result token.
    movetext = _format_movetext(moves, headers.result)
    return "\n".join(header_lines) + "\n\n" + movetext + "\n"


def write_pgn_file(path: Path, pgn: str) -> None:
    """Write PGN to disk (UTF-8)."""
    path.write_text(pgn, encoding="utf-8")
