"""Defensive play decision detector.

Detects defensive decisions when a puck carrier is approaching
with speed and classifies the defender's response.
"""

import numpy as np
from typing import Optional

from ..game_context import FrameContext, GameEvent


class DefensivePlayDetector:
    """Detect defensive decisions under pressure.

    Trigger: Puck carrier is approaching with speed in the
    neutral or offensive zone (opponent coming toward goal).

    Classification:
    - Gap close: Defender aggressively closes the gap on the attacker
    - Contain: Defender maintains distance, skating backward to contain
    - Pinch: Defender aggressively steps up / pinches from the blue line
    """

    def __init__(
        self,
        defense_cooldown: int = 45,
        event_window: int = 20,
        approach_speed_threshold: float = 5.0,
        gap_close_threshold: float = -3.0,
        pinch_forward_threshold: float = 5.0,
    ):
        """
        Args:
            defense_cooldown: Frames between consecutive defensive detections.
            event_window: Frames of context around each event.
            approach_speed_threshold: Minimum puck approach speed (px/frame)
                                      to count as 'under pressure'.
            gap_close_threshold: If gap change per frame is below this (negative =
                                  closing), classify as gap_close.
            pinch_forward_threshold: If the nearest defender moves forward by this
                                      much per frame, classify as pinch.
        """
        self.defense_cooldown = defense_cooldown
        self.event_window = event_window
        self.approach_speed_threshold = approach_speed_threshold
        self.gap_close_threshold = gap_close_threshold
        self.pinch_forward_threshold = pinch_forward_threshold

        self._frames: list[FrameContext] = []
        self._cooldown = 0

    def add_frame(self, context: FrameContext) -> None:
        """Process one frame."""
        self._frames.append(context)
        if self._cooldown > 0:
            self._cooldown -= 1

    def analyze(self, fps: float) -> list:
        """Detect all defensive play events."""
        events = []

        for i in range(5, len(self._frames)):
            if self._cooldown > 0:
                self._cooldown -= 1
                continue

            event = self._check_frame(i, fps)
            if event is not None:
                events.append(event)
                self._cooldown = self.defense_cooldown

        return events

    def _check_frame(self, idx: int, fps: float) -> Optional[GameEvent]:
        """Check if a defensive decision is needed at this frame."""
        ctx = self._frames[idx]

        # Need possession and puck visible
        if ctx.possession_player_id is None:
            return None
        if ctx.puck is None:
            return None

        # Should be in neutral or offensive zone (attacker coming toward goal)
        if ctx.zone not in ("neutral", "offensive"):
            return None

        # Check puck approach speed (moving toward defensive end)
        approach_speed = self._get_approach_speed(idx)
        if approach_speed < self.approach_speed_threshold:
            return None

        # Find the nearest defender (player closest to the goal that
        # isn't the puck carrier). Since we don't have team labels,
        # use positional heuristic: players between the puck and the
        # goal area are likely defenders.
        defender_info = self._find_nearest_defender(idx, ctx)
        if defender_info is None:
            return None

        defender_id, gap_distance = defender_info

        # Measure gap change over recent frames to classify decision
        gap_change = self._measure_gap_change(idx, defender_id)
        closing_speed = -gap_change if gap_change is not None else 0.0

        # Classify the defensive decision
        decision = self._classify_defense(gap_change, idx, defender_id)

        # Count defenders in the area
        num_defenders = self._count_defenders(idx, ctx)

        frame_start = max(0, idx - self.event_window)
        frame_end = min(len(self._frames) - 1, idx + self.event_window)

        return GameEvent(
            event_type="defensive_play",
            frame_idx=idx,
            frame_start=frame_start,
            frame_end=frame_end,
            timestamp_sec=idx / fps,
            decision_made=decision,
            player_id=defender_id,
            context={
                "gap_distance": round(gap_distance, 1),
                "closing_speed": round(closing_speed, 1),
                "num_defenders": num_defenders,
                "approach_speed": round(approach_speed, 1),
                "zone": ctx.zone,
            },
        )

    def _get_approach_speed(self, idx: int) -> float:
        """Measure how fast the puck is moving (general speed)."""
        if idx < 3:
            return 0.0

        speeds = []
        for j in range(idx - 2, idx + 1):
            if j < 1:
                continue
            curr_puck = self._frames[j].puck
            prev_puck = self._frames[j - 1].puck
            if curr_puck and prev_puck:
                dist = np.linalg.norm(
                    np.array(curr_puck.center) - np.array(prev_puck.center)
                )
                speeds.append(dist)

        return float(np.mean(speeds)) if speeds else 0.0

    def _find_nearest_defender(
        self, idx: int, ctx: FrameContext
    ) -> Optional[tuple]:
        """Find the nearest player between puck and goal (likely defender).

        Returns (defender_track_id, gap_distance) or None.
        """
        if ctx.puck is None:
            return None

        puck_pos = np.array(ctx.puck.center)
        carrier_id = ctx.possession_player_id

        # Determine 'backward' direction (toward own goal)
        # Use puck movement direction -- defenders are ahead in opposite dir
        if idx >= 2 and self._frames[idx - 2].puck is not None:
            prev_pos = np.array(self._frames[idx - 2].puck.center)
            forward_dir = puck_pos - prev_pos
            forward_mag = np.linalg.norm(forward_dir)
            if forward_mag < 1e-6:
                return None
            forward_dir = forward_dir / forward_mag
        else:
            return None

        best_defender = None
        best_dist = float("inf")

        for player in ctx.players:
            if player.track_id == carrier_id:
                continue

            player_pos = np.array(player.center)
            to_player = player_pos - puck_pos

            # Player should be ahead of the puck (in the direction of travel)
            # meaning they're between the attacker and the goal
            proj = np.dot(to_player, forward_dir)
            if proj <= 0:
                continue  # Behind the puck, not a defender in this context

            dist = np.linalg.norm(to_player)
            if dist < best_dist:
                best_dist = dist
                best_defender = player.track_id

        if best_defender is None:
            return None

        return best_defender, best_dist

    def _measure_gap_change(self, idx: int, defender_id: int) -> Optional[float]:
        """Measure how the gap between puck and defender changes.

        Returns average gap change per frame (negative = closing).
        """
        lookback = min(idx, 5)
        gaps = []

        for j in range(idx - lookback, idx + 1):
            ctx = self._frames[j]
            if ctx.puck is None:
                continue

            puck_pos = np.array(ctx.puck.center)

            for player in ctx.players:
                if player.track_id == defender_id:
                    player_pos = np.array(player.center)
                    gap = np.linalg.norm(player_pos - puck_pos)
                    gaps.append(gap)
                    break

        if len(gaps) < 2:
            return None

        # Average gap change per frame
        changes = [gaps[i + 1] - gaps[i] for i in range(len(gaps) - 1)]
        return float(np.mean(changes))

    def _classify_defense(
        self, gap_change: Optional[float], idx: int, defender_id: int
    ) -> str:
        """Classify the defensive decision.

        - gap_close: Defender aggressively closing gap (gap shrinking fast)
        - pinch: Defender stepping up / moving forward aggressively
        - contain: Defender maintaining distance, controlled retreat
        """
        if gap_change is None:
            return "contain"

        # Check if defender is moving forward aggressively (pinch)
        defender_forward = self._get_defender_forward_speed(idx, defender_id)

        if defender_forward is not None and defender_forward > self.pinch_forward_threshold:
            return "pinch"

        if gap_change < self.gap_close_threshold:
            return "gap_close"

        return "contain"

    def _get_defender_forward_speed(
        self, idx: int, defender_id: int
    ) -> Optional[float]:
        """Measure how fast the defender is moving forward (toward puck direction)."""
        if idx < 3:
            return None

        positions = []
        for j in range(idx - 3, idx + 1):
            ctx = self._frames[j]
            for player in ctx.players:
                if player.track_id == defender_id:
                    positions.append(np.array(player.center))
                    break

        if len(positions) < 2:
            return None

        # Get puck movement direction as 'forward'
        if self._frames[idx].puck and self._frames[idx - 2].puck:
            puck_now = np.array(self._frames[idx].puck.center)
            puck_prev = np.array(self._frames[idx - 2].puck.center)
            forward_dir = puck_now - puck_prev
            mag = np.linalg.norm(forward_dir)
            if mag < 1e-6:
                return None
            forward_dir = forward_dir / mag
        else:
            return None

        # Measure defender movement projected onto forward direction
        # Positive = moving toward the puck carrier (pinching)
        movement = positions[-1] - positions[0]
        # Defender moving AGAINST the puck direction (toward the attacker)
        # means they're closing; same direction means retreating
        # For a pinch, defender moves in the OPPOSITE direction of the puck
        # (i.e., toward the puck)
        forward_speed = -np.dot(movement, forward_dir) / len(positions)

        return float(forward_speed)

    def _count_defenders(self, idx: int, ctx: FrameContext) -> int:
        """Count players between puck and goal (approximate defender count)."""
        if ctx.puck is None:
            return 0

        puck_pos = np.array(ctx.puck.center)
        carrier_id = ctx.possession_player_id

        if idx >= 2 and self._frames[idx - 2].puck is not None:
            prev_pos = np.array(self._frames[idx - 2].puck.center)
            forward_dir = puck_pos - prev_pos
            mag = np.linalg.norm(forward_dir)
            if mag < 1e-6:
                return 0
            forward_dir = forward_dir / mag
        else:
            return 0

        count = 0
        for player in ctx.players:
            if player.track_id == carrier_id:
                continue
            player_pos = np.array(player.center)
            to_player = player_pos - puck_pos
            proj = np.dot(to_player, forward_dir)
            if proj > 0:
                count += 1

        return count
