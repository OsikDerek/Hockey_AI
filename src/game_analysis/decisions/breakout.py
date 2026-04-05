"""Breakout decision detector.

Detects when a team exits the defensive zone and classifies
the breakout strategy used.
"""

import numpy as np
from typing import Optional

from ..game_context import FrameContext, GameEvent


class BreakoutDetector:
    """Detect defensive zone breakout decisions.

    Trigger: Puck transitions from defensive to neutral zone.

    Classification:
    - Carry: Player skates puck out of the zone
    - Rim: Puck sent along the boards
    - Direct pass: Clean pass to a teammate up ice
    - Chip: Puck chipped off the boards/glass to clear
    """

    def __init__(
        self,
        breakout_cooldown: int = 45,
        event_window: int = 25,
    ):
        """
        Args:
            breakout_cooldown: Frames between consecutive breakout detections.
            event_window: Frames of context around each event.
        """
        self.breakout_cooldown = breakout_cooldown
        self.event_window = event_window

        self._frames = []
        self._cooldown = 0

    def add_frame(self, context: FrameContext) -> None:
        """Process one frame."""
        self._frames.append(context)
        if self._cooldown > 0:
            self._cooldown -= 1

    def analyze(self, fps: float) -> list:
        """Detect all breakout events."""
        events = []

        for i in range(1, len(self._frames)):
            prev = self._frames[i - 1]
            curr = self._frames[i]

            # Detect defensive -> neutral transition
            if prev.zone == "defensive" and curr.zone in ("neutral", "offensive"):
                if self._cooldown <= 0:
                    event = self._classify_breakout(i, fps)
                    if event is not None:
                        events.append(event)
                        self._cooldown = self.breakout_cooldown

        return events

    def _classify_breakout(self, idx: int, fps: float) -> Optional[GameEvent]:
        """Classify the breakout strategy."""
        ctx = self._frames[idx]

        # Look at what happened in the defensive zone
        lookback = min(idx, 30)
        dzone_frames = []
        for j in range(idx - lookback, idx):
            if self._frames[j].zone == "defensive":
                dzone_frames.append(self._frames[j])

        # Time spent in defensive zone
        dzone_time = len(dzone_frames) / fps if fps > 0 else 0

        # Who had possession at exit
        exit_possession = ctx.possession_player_id

        # Check possession changes before exit
        pre_possession = None
        for j in range(idx - min(lookback, 10), idx):
            if self._frames[j].possession_player_id is not None:
                pre_possession = self._frames[j].possession_player_id
                break

        # Puck velocity at exit
        exit_speed = self._calc_exit_speed(idx)

        # Classify based on speed and possession
        if exit_speed > 20:
            # High speed exit
            if pre_possession == exit_possession and exit_possession is not None:
                decision = "carry"
            else:
                decision = "chip"  # Fast, different possession = cleared
        else:
            # Controlled exit
            if pre_possession == exit_possession and exit_possession is not None:
                decision = "carry"
            elif exit_possession is not None and pre_possession != exit_possession:
                decision = "direct_pass"
            else:
                decision = "rim"

        # Check if breakout was successful (maintained possession for 1+ seconds after)
        success = self._check_breakout_success(idx, fps)

        frame_start = max(0, idx - self.event_window)
        frame_end = min(len(self._frames) - 1, idx + self.event_window)

        return GameEvent(
            event_type="breakout",
            frame_idx=idx,
            frame_start=frame_start,
            frame_end=frame_end,
            timestamp_sec=idx / fps,
            decision_made=decision,
            player_id=exit_possession or pre_possession,
            context={
                "dzone_time": round(dzone_time, 1),
                "exit_speed": round(exit_speed, 1),
                "exit_possession": exit_possession,
                "pre_possession": pre_possession,
                "success": success,
            },
        )

    def _calc_exit_speed(self, idx: int) -> float:
        """Puck speed around the exit frame."""
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

    def _check_breakout_success(self, idx: int, fps: float) -> bool:
        """Check if the breakout maintained possession for 1+ seconds."""
        check_frames = int(fps)  # 1 second
        end = min(len(self._frames), idx + check_frames)

        possession_frames = 0
        for j in range(idx, end):
            if self._frames[j].possession_player_id is not None:
                possession_frames += 1
            # If we re-enter defensive zone, breakout failed
            if self._frames[j].zone == "defensive":
                return False

        # Success if possession maintained for >50% of the check window
        return possession_frames > (end - idx) * 0.5
