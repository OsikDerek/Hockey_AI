"""Odd-man rush decision detector.

Detects 2-on-1, 3-on-2, and breakaway situations where the attacking
team has a numerical advantage and classifies the decision made.
"""

import numpy as np
from typing import Optional

from ..game_context import FrameContext, GameEvent


class OddManRushDetector:
    """Detect odd-man rush situations (2-on-1, 3-on-2, breakaway).

    Trigger: Fewer defenders than attackers ahead of the puck carrier
    in the offensive or neutral zone, with forward momentum.

    Classification:
    - Shot: Puck carrier shoots
    - Pass: Puck carrier passes to open teammate
    - Deke: Puck carrier attempts to deke past defender/goalie
    """

    def __init__(
        self,
        rush_cooldown: int = 45,
        event_window: int = 25,
        speed_threshold: float = 5.0,
    ):
        """
        Args:
            rush_cooldown: Frames between consecutive rush detections.
            event_window: Frames of context around each event.
            speed_threshold: Minimum puck carrier speed (px/frame) to qualify as a rush.
        """
        self.rush_cooldown = rush_cooldown
        self.event_window = event_window
        self.speed_threshold = speed_threshold

        self._frames: list[FrameContext] = []
        self._cooldown = 0

    def add_frame(self, context: FrameContext) -> None:
        """Process one frame."""
        self._frames.append(context)
        if self._cooldown > 0:
            self._cooldown -= 1

    def analyze(self, fps: float) -> list:
        """Detect all odd-man rush events."""
        events = []

        for i in range(2, len(self._frames)):
            if self._cooldown > 0:
                self._cooldown -= 1
                continue

            event = self._check_frame(i, fps)
            if event is not None:
                events.append(event)
                self._cooldown = self.rush_cooldown

        return events

    def _check_frame(self, idx: int, fps: float) -> Optional[GameEvent]:
        """Check if an odd-man rush is happening at this frame."""
        ctx = self._frames[idx]

        # Must have possession and be in offensive or neutral zone
        if ctx.possession_player_id is None:
            return None
        if ctx.zone not in ("offensive", "neutral"):
            return None

        # Need a puck to measure direction
        if ctx.puck is None:
            return None

        # Determine puck carrier speed (proxy for rush)
        carrier_speed = self._get_carrier_speed(idx)
        if carrier_speed < self.speed_threshold:
            return None

        # Count attackers and defenders relative to the puck
        attackers_ahead, defenders_ahead = self._count_players_ahead(idx, ctx)

        # Classify the rush type
        rush_type = self._classify_rush_type(attackers_ahead, defenders_ahead)
        if rush_type is None:
            return None  # Not an odd-man rush

        # Classify the decision (look ahead a few frames for resolution)
        decision = self._classify_decision(idx)

        frame_start = max(0, idx - self.event_window)
        frame_end = min(len(self._frames) - 1, idx + self.event_window)

        return GameEvent(
            event_type="odd_man_rush",
            frame_idx=idx,
            frame_start=frame_start,
            frame_end=frame_end,
            timestamp_sec=idx / fps,
            decision_made=decision,
            player_id=ctx.possession_player_id,
            context={
                "rush_type": rush_type,
                "attackers_ahead": attackers_ahead,
                "defenders_back": defenders_ahead,
                "speed": round(carrier_speed, 1),
                "zone": ctx.zone,
                "num_players": len(ctx.players),
            },
        )

    def _get_carrier_speed(self, idx: int) -> float:
        """Estimate puck carrier speed from puck position delta."""
        if idx < 2:
            return 0.0

        curr_puck = self._frames[idx].puck
        prev_puck = self._frames[idx - 2].puck

        if curr_puck is None or prev_puck is None:
            return 0.0

        dist = np.linalg.norm(
            np.array(curr_puck.center) - np.array(prev_puck.center)
        )
        # Divide by 2 frames to get per-frame speed
        return dist / 2.0

    def _count_players_ahead(self, idx: int, ctx: FrameContext) -> tuple:
        """Count attackers and defenders ahead of the puck.

        'Ahead' is defined as further toward the offensive end of the rink.
        In broadcast view this roughly means further right (or left, depending
        on the period). Since we don't know orientation, we use the puck
        carrier's movement direction as a proxy for 'forward'.

        Returns:
            (attackers_ahead, defenders_ahead) -- attackers includes the
            carrier itself.
        """
        if ctx.puck is None:
            return 0, 0

        puck_pos = np.array(ctx.puck.center)

        # Determine forward direction from recent puck movement
        if idx >= 2 and self._frames[idx - 2].puck is not None:
            prev_pos = np.array(self._frames[idx - 2].puck.center)
            forward_dir = puck_pos - prev_pos
            forward_mag = np.linalg.norm(forward_dir)
            if forward_mag < 1e-6:
                return 0, 0
            forward_dir = forward_dir / forward_mag
        else:
            return 0, 0

        attackers_ahead = 1  # Count the puck carrier
        defenders_ahead = 0

        carrier_id = ctx.possession_player_id

        for player in ctx.players:
            if player.track_id == carrier_id:
                continue

            player_pos = np.array(player.center)
            to_player = player_pos - puck_pos

            # Project onto forward direction
            proj = np.dot(to_player, forward_dir)

            if proj > 0:
                # Player is ahead of the puck in the direction of travel
                # We don't have team labels, so use proximity heuristic:
                # players near the carrier's lateral position are likely
                # teammates; those far to the side or directly in the path
                # may be defenders.
                perp = np.linalg.norm(to_player - proj * forward_dir)

                # Simple heuristic: players spread wide are likely
                # teammates (attackers), those centered in the lane
                # are more likely defenders
                if perp > 80:
                    attackers_ahead += 1
                else:
                    defenders_ahead += 1

        # Also count goalies as defenders
        for goalie in ctx.goalies:
            goalie_pos = np.array(goalie.center)
            to_goalie = goalie_pos - puck_pos
            proj = np.dot(to_goalie, forward_dir)
            if proj > 0:
                defenders_ahead += 1

        return attackers_ahead, defenders_ahead

    def _classify_rush_type(self, attackers: int, defenders: int) -> Optional[str]:
        """Classify the rush type based on player counts.

        Returns None if it's not an odd-man rush.
        """
        advantage = attackers - defenders

        if advantage <= 0:
            return None  # No numerical advantage

        if defenders == 0 and attackers == 1:
            return "breakaway"
        elif attackers == 2 and defenders == 1:
            return "2on1"
        elif attackers == 3 and defenders == 2:
            return "3on2"
        elif attackers == 2 and defenders == 0:
            return "2on0"
        elif attackers == 3 and defenders <= 1:
            return "3on1"
        elif advantage >= 1:
            return f"{attackers}on{defenders}"
        else:
            return None

    def _classify_decision(self, idx: int) -> str:
        """Classify what the puck carrier decided to do.

        Look ahead a few frames to see if possession changed (pass),
        puck sped up toward goal (shot), or carrier maintained close
        possession through a defender (deke).
        """
        ctx = self._frames[idx]
        carrier_id = ctx.possession_player_id

        lookahead = min(len(self._frames) - 1 - idx, 15)

        # Track puck velocity toward goal and possession changes
        for j in range(idx + 1, idx + lookahead + 1):
            future = self._frames[j]

            # Possession changed -> pass
            if (future.possession_player_id is not None
                    and future.possession_player_id != carrier_id):
                return "pass"

            # Large velocity spike -> shot
            if j >= 2:
                curr_puck = future.puck
                prev_puck = self._frames[j - 1].puck
                if curr_puck and prev_puck:
                    speed = np.linalg.norm(
                        np.array(curr_puck.center) - np.array(prev_puck.center)
                    )
                    if speed > 20:
                        return "shot"

        # If carrier kept possession through the whole window -> deke attempt
        return "deke"
