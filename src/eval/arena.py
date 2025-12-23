"""
Arena helpers for evaluation summaries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Simple match outcome summary."""

    wins: int
    draws: int
    losses: int

    def total_games(self) -> int:
        """Return total number of games."""
        return self.wins + self.draws + self.losses

    def score(self) -> float:
        """Return score with draws worth 0.5."""
        return float(self.wins) + 0.5 * float(self.draws)

    def win_rate(self) -> float:
        """Return win rate across all games."""
        total = self.total_games()
        if total == 0:
            return 0.0
        return float(self.wins) / float(total)
