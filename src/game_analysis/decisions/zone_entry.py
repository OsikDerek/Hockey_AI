"""Zone entry decision detector.

Detects when the puck transitions from neutral zone to offensive zone
and classifies whether it was a carry-in, dump-in, or pass-in.
"""

import numpy as np
from typing import Optional

from ..game_context import FrameContext, GameEvent


class ZoneEntryDetector:
    """Detect zone entry decisions.

    Trigger: Puck transitions from neutral to offensive zone.

    Classification:
    - Carry: Possessing player crosses with the puck
    - Dump: Puck sent in without player (no possession on entry)
    - Pass-in: Puck passed to a teammate already in the zone
    """

    def __init__(
        self,
        entry_cooldown: int = 30,
        event_window: int = 20,
    ):
        """
        Args:
            entry_cooldown: Frames between consecutive entry detections.
            event_window: Frames of context around each event.
        """
        self.entry_cooldown = entry_cooldown
        self.event_window = event_window

        self._frames = []
        self._prev_zone = None
        self._cooldown = 0

    def add_frame(self, context: FrameContext) -> None:
        """Process one frame."""
        self._frames.append(context)
        if self._cooldown > 0:
            self._cooldown -= 1

    def analyze(self, fps: float) -> list:
        """Detect all zone entry events."""
        events = []

        for i in range(1, len(self._frames)):
            prev = self._frames[i - 1]
            curr = self._frames[i]

            # Detect neutral -> offensive transition
            if prev.zone == "neutral" and curr.zone == "offensive":
                if self._cooldown <= 0:
                    event = self._classify_entry(i, fps)
                    if event is not None:
                        events.append(event)
                        self._cooldown = self.entry_cooldown

        return events

    def _classify_entry(self, idx: int, fps: float) -> Optional[GameEvent]:
        """Classify the type of zone entry."""
        ctx = self._frames[idx]

        # Look at possession state around the entry
        # Check a few frames before and after
        lookback = min(idx, 10)
        pre_possession = None
        post_possession = None

        for j in range(idx - lookback, idx):
            if self._frames[j].possession_player_id is not None:
                pre_possession = self._frames[j].possession_player_id
                break

        lookahead = min(len(self._frames) - 1 - idx, 10)
        for j in range(idx, idx + lookahead):
            if self._frames[j].possession_player_id is not None:
                post_possession = self._frames[j].possession_player_id
                break

        # Classify
        if pre_possession is not None and pre_possession == post_possession:
            decision = "carry"
        elif pre_possession is not None and post_possession is not None and pre_possession != post_possession:
            decision = "pass_in"
        elif pre_possession is not None and post_possession is None:
            decision = "dump"
        else:
            decision = "dump"  # Default when unsure

        # Determine entry speed (puck velocity around entry point)
        entry_speed = self._calc_entry_speed(idx)

        frame_start = max(0, idx - self.event_window)
        frame_end = min(len(self._frames) - 1, idx + self.event_window)

        return GameEvent(
            event_type="zone_entry",
            frame_idx=idx,
            frame_start=frame_start,
            frame_end=frame_end,
            timestamp_sec=idx / fps,
            decision_made=decision,
            player_id=pre_possession,
            context={
                "pre_possession": pre_possession,
                "post_possession": post_possession,
                "entry_speed": entry_speed,
                "num_players_in_zone": len(ctx.players),
            },
        )

    def _calc_entry_speed(self, idx: int) -> float:
        """Calculate puck speed around the entry frame."""
        speeds = []
        for j in range(max(0, idx - 3), min(len(self._frames), idx + 3)):
            if j == 0:
                continue
            curr_puck = self._frames[j].puck
            prev_puck = self._frames[j - 1].puck
            if curr_puck and prev_puck:
                dist = np.linalg.norm(
                    np.array(curr_puck.center) - np.array(prev_puck.center)
                )
                speeds.append(dist)

        return float(np.mean(speeds)) if speeds else 0.0
