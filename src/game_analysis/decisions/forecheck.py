"""Forecheck decision detector.

Detects forechecking situations when the puck is in the defensive
zone (from the opponent's perspective) and classifies the pressure
strategy being applied.
"""

import numpy as np
from typing import Optional

from ..game_context import FrameContext, GameEvent


class ForecheckDetector:
    """Detect forechecking decisions.

    Trigger: Puck is in the defensive zone (treated as 'we are
    forechecking in their zone' since we don't have team
    differentiation yet).

    Classification:
    - Aggressive: Multiple players pressuring the puck carrier closely
    - Moderate: Standard forecheck, one hard on puck, one supporting
    - Passive: Sitting back in a trap-style structure (1-2-2 / 1-3-1)
    """

    def __init__(
        self,
        forecheck_cooldown: int = 60,
        event_window: int = 20,
        aggressive_distance: float = 100.0,
        min_defensive_frames: int = 10,
    ):
        """
        Args:
            forecheck_cooldown: Frames between consecutive forecheck detections.
            event_window: Frames of context around each event.
            aggressive_distance: Distance (px) threshold for 'pressuring' the puck.
            min_defensive_frames: Puck must be in defensive zone for N frames
                                  before we evaluate the forecheck.
        """
        self.forecheck_cooldown = forecheck_cooldown
        self.event_window = event_window
        self.aggressive_distance = aggressive_distance
        self.min_defensive_frames = min_defensive_frames

        self._frames: list[FrameContext] = []
        self._defensive_streak = 0
        self._cooldown = 0

    def add_frame(self, context: FrameContext) -> None:
        """Process one frame."""
        self._frames.append(context)

        if context.zone == "defensive":
            self._defensive_streak += 1
        else:
            self._defensive_streak = 0

        if self._cooldown > 0:
            self._cooldown -= 1

    def analyze(self, fps: float) -> list:
        """Detect all forecheck events."""
        events = []
        defensive_streak = 0

        for i in range(len(self._frames)):
            ctx = self._frames[i]

            if ctx.zone == "defensive":
                defensive_streak += 1
            else:
                defensive_streak = 0
                continue

            if defensive_streak < self.min_defensive_frames:
                continue

            # Only evaluate once per cooldown window
            if (defensive_streak - self.min_defensive_frames) % self.forecheck_cooldown != 0:
                continue

            event = self._evaluate_forecheck(i, fps)
            if event is not None:
                events.append(event)

        return events

    def _evaluate_forecheck(self, idx: int, fps: float) -> Optional[GameEvent]:
        """Evaluate the forecheck strategy at this frame."""
        ctx = self._frames[idx]

        if ctx.puck is None:
            return None

        puck_pos = np.array(ctx.puck.center)

        # Count players in the zone and measure their distance to the puck
        players_in_zone = len(ctx.players)
        pressure_distances = []

        for player in ctx.players:
            player_pos = np.array(player.center)
            dist = np.linalg.norm(player_pos - puck_pos)
            pressure_distances.append(dist)

        if not pressure_distances:
            return None

        pressure_distances.sort()
        closest_distance = pressure_distances[0]

        # Count how many players are within aggressive range
        pressuring_count = sum(
            1 for d in pressure_distances if d < self.aggressive_distance
        )

        # Classify forecheck strategy
        decision = self._classify_forecheck(
            pressuring_count, closest_distance, players_in_zone
        )

        frame_start = max(0, idx - self.event_window)
        frame_end = min(len(self._frames) - 1, idx + self.event_window)

        return GameEvent(
            event_type="forecheck",
            frame_idx=idx,
            frame_start=frame_start,
            frame_end=frame_end,
            timestamp_sec=idx / fps,
            decision_made=decision,
            player_id=ctx.possession_player_id,
            context={
                "num_players_in_zone": players_in_zone,
                "pressure_distance": round(closest_distance, 1),
                "pressuring_count": pressuring_count,
                "zone": ctx.zone,
            },
        )

    def _classify_forecheck(
        self,
        pressuring_count: int,
        closest_distance: float,
        players_in_zone: int,
    ) -> str:
        """Classify the forecheck strategy.

        - Aggressive: 2+ players pressuring hard, closest is very tight
        - Moderate: 1 player pressuring, others supporting
        - Passive: Nobody pressuring closely, sitting back (trap)
        """
        if pressuring_count >= 2 and closest_distance < self.aggressive_distance * 0.5:
            return "aggressive"
        elif pressuring_count >= 1 and closest_distance < self.aggressive_distance:
            return "moderate"
        else:
            return "passive"
