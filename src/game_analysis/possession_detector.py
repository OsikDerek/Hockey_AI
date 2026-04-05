"""Possession detection for game analysis.

Determines which player has the puck based on proximity,
movement direction consistency, and temporal smoothing.
"""

import numpy as np
from typing import Optional
from collections import deque

from .game_context import TrackedObject


class PossessionDetector:
    """Determine which player has the puck.

    Uses puck-to-player proximity weighted by movement direction
    consistency. Applies hysteresis to prevent flickering on
    noisy broadcast footage.

    Usage:
        pd = PossessionDetector()
        player_id = pd.update(puck, players)
    """

    def __init__(
        self,
        proximity_threshold: float = 120,
        hysteresis_window: int = 8,
        momentum_weight: float = 0.3,
        coast_frames: int = 10,
    ):
        """
        Args:
            proximity_threshold: Max pixel distance for puck-to-player possession.
            hysteresis_window: Rolling window for majority-vote smoothing.
            momentum_weight: Weight for movement direction consistency (0-1).
            coast_frames: Frames to hold last possession when puck is lost.
        """
        self.proximity_threshold = proximity_threshold
        self.hysteresis_window = hysteresis_window
        self.momentum_weight = momentum_weight
        self.coast_frames = coast_frames

        self._prev_puck_pos = None
        self._prev_player_positions = {}   # track_id -> (x, y)
        self._possession_history = deque(maxlen=hysteresis_window)
        self._last_possession = None
        self._no_puck_streak = 0

    def update(
        self,
        puck: Optional[TrackedObject],
        players: list,
    ) -> Optional[int]:
        """Update possession estimate.

        Args:
            puck: Puck TrackedObject (or None if not detected).
            players: List of player TrackedObjects.

        Returns:
            track_id of possessing player, or None.
        """
        if puck is None:
            self._no_puck_streak += 1
            # Coast: hold last possession for a few frames
            if self._no_puck_streak <= self.coast_frames and self._last_possession is not None:
                return self._last_possession
            return None

        self._no_puck_streak = 0

        if not players:
            self._prev_puck_pos = puck.center
            return None

        # Score each player
        best_id = None
        best_score = float("inf")

        puck_pos = np.array(puck.center)

        for player in players:
            if player.track_id < 0:
                continue

            player_pos = np.array(player.center)
            dist = np.linalg.norm(puck_pos - player_pos)

            if dist > self.proximity_threshold:
                continue

            # Base score is distance (lower = better)
            score = dist

            # Bonus: movement direction consistency
            if (
                self.momentum_weight > 0
                and self._prev_puck_pos is not None
                and player.track_id in self._prev_player_positions
            ):
                puck_vel = puck_pos - np.array(self._prev_puck_pos)
                prev_player = np.array(self._prev_player_positions[player.track_id])
                player_vel = player_pos - prev_player

                # Cosine similarity of movement vectors
                puck_speed = np.linalg.norm(puck_vel)
                player_speed = np.linalg.norm(player_vel)

                if puck_speed > 1 and player_speed > 1:
                    cos_sim = np.dot(puck_vel, player_vel) / (puck_speed * player_speed)
                    # cos_sim in [-1, 1], higher = more aligned
                    # Reduce score for aligned movement
                    score -= cos_sim * self.momentum_weight * self.proximity_threshold

            if score < best_score:
                best_score = score
                best_id = player.track_id

        # Update history
        self._prev_puck_pos = puck.center
        self._prev_player_positions = {
            p.track_id: p.center for p in players if p.track_id >= 0
        }

        # Add to rolling window
        self._possession_history.append(best_id)

        # Majority vote for smoothing
        smoothed = self._majority_vote()

        if smoothed is not None:
            self._last_possession = smoothed

        return smoothed

    def _majority_vote(self) -> Optional[int]:
        """Return majority possession from recent history."""
        if not self._possession_history:
            return None

        # Filter out Nones
        valid = [p for p in self._possession_history if p is not None]
        if not valid:
            return None

        # Count votes
        counts = {}
        for pid in valid:
            counts[pid] = counts.get(pid, 0) + 1

        winner = max(counts, key=counts.get)

        # Require at least 30% of window to claim possession
        if counts[winner] < len(self._possession_history) * 0.3:
            return None

        return winner

    def get_possession_change(self) -> bool:
        """Check if possession changed on the most recent update."""
        if len(self._possession_history) < 2:
            return False

        recent = list(self._possession_history)
        if len(recent) >= 2:
            # Simple check: current != previous non-None
            curr = recent[-1]
            for prev in reversed(recent[:-1]):
                if prev is not None:
                    return curr != prev
        return False

    def reset(self):
        """Reset all state."""
        self._prev_puck_pos = None
        self._prev_player_positions = {}
        self._possession_history.clear()
        self._last_possession = None
        self._no_puck_streak = 0
