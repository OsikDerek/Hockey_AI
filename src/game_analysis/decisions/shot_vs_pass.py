"""Shot vs Pass decision detector.

Detects when a player in the offensive zone releases the puck
(velocity spike) and classifies whether it was a shot or pass
based on puck trajectory relative to the goal and other players.
"""

import numpy as np
from typing import Optional
from collections import deque

from ..game_context import FrameContext, GameEvent, TrackedObject


class ShotVsPassDetector:
    """Detect shot vs pass decisions in the offensive zone.

    Trigger: Possessing player in offensive zone, puck velocity spikes
    (indicating release).

    Classification:
    - Shot: puck moves toward goal area
    - Pass: puck moves toward another player
    - Dump: puck sent into corner/boards without clear target
    """

    def __init__(
        self,
        velocity_threshold: float = 15.0,
        min_offensive_frames: int = 5,
        event_window: int = 15,
    ):
        """
        Args:
            velocity_threshold: Minimum puck velocity (px/frame) for a "release".
            min_offensive_frames: Must be in offensive zone for N frames before triggering.
            event_window: Frames of context around each event.
        """
        self.velocity_threshold = velocity_threshold
        self.min_offensive_frames = min_offensive_frames
        self.event_window = event_window

        self._frames = []
        self._puck_positions = []       # (x, y) or None per frame
        self._offensive_streak = 0
        self._cooldown = 0

    def add_frame(self, context: FrameContext) -> None:
        """Process one frame."""
        self._frames.append(context)

        # Track puck position
        if context.puck is not None:
            self._puck_positions.append(context.puck.center)
        else:
            self._puck_positions.append(None)

        # Track offensive zone streak
        if context.zone == "offensive":
            self._offensive_streak += 1
        else:
            self._offensive_streak = 0

        if self._cooldown > 0:
            self._cooldown -= 1

    def analyze(self, fps: float) -> list:
        """Detect all shot/pass events."""
        events = []

        for i in range(2, len(self._puck_positions)):
            event = self._check_frame(i, fps)
            if event is not None:
                events.append(event)

        return events

    def _check_frame(self, idx: int, fps: float) -> Optional[GameEvent]:
        """Check if a shot/pass event occurs at this frame."""
        ctx = self._frames[idx]

        # Must be in offensive zone with possession
        if ctx.zone != "offensive":
            return None
        if ctx.possession_player_id is None:
            return None

        # Check puck velocity
        curr_pos = self._puck_positions[idx]
        prev_pos = self._puck_positions[idx - 1]
        prev2_pos = self._puck_positions[idx - 2]

        if curr_pos is None or prev_pos is None or prev2_pos is None:
            return None

        curr = np.array(curr_pos)
        prev = np.array(prev_pos)
        prev2 = np.array(prev2_pos)

        velocity = np.linalg.norm(curr - prev)
        prev_velocity = np.linalg.norm(prev - prev2)

        # Detect velocity spike (puck was slow, now fast = release)
        if velocity < self.velocity_threshold:
            return None
        if prev_velocity > self.velocity_threshold * 0.7:
            return None  # Already moving fast, not a new release

        # Classify: shot vs pass vs dump
        decision = self._classify_release(idx, curr, prev, ctx)

        # Count defenders in the lane
        defenders_blocking = self._count_lane_blockers(idx, ctx)

        # Count open teammates
        open_teammates = self._count_open_teammates(idx, ctx)

        frame_start = max(0, idx - self.event_window)
        frame_end = min(len(self._frames) - 1, idx + self.event_window)

        return GameEvent(
            event_type="shot_vs_pass",
            frame_idx=idx,
            frame_start=frame_start,
            frame_end=frame_end,
            timestamp_sec=idx / fps,
            decision_made=decision,
            player_id=ctx.possession_player_id,
            context={
                "puck_velocity": float(velocity),
                "zone": ctx.zone,
                "defenders_blocking": defenders_blocking,
                "open_teammates": open_teammates,
                "num_players": len(ctx.players),
            },
        )

    def _classify_release(
        self, idx: int, puck_pos: np.ndarray, prev_pos: np.ndarray, ctx: FrameContext
    ) -> str:
        """Classify whether the release was a shot, pass, or dump."""
        puck_direction = puck_pos - prev_pos
        puck_dir_norm = puck_direction / (np.linalg.norm(puck_direction) + 1e-8)

        # Check if puck is moving toward goal
        # In broadcast view, goals are typically at the edges of the frame
        # If we have goalies detected, use their position
        if ctx.goalies:
            goalie_pos = np.array(ctx.goalies[0].center)
            to_goal = goalie_pos - prev_pos
            to_goal_norm = to_goal / (np.linalg.norm(to_goal) + 1e-8)

            goal_alignment = np.dot(puck_dir_norm, to_goal_norm)

            if goal_alignment > 0.5:
                return "shot"

        # Check if puck is moving toward another player
        possessing_id = ctx.possession_player_id
        for player in ctx.players:
            if player.track_id == possessing_id:
                continue
            player_pos = np.array(player.center)
            to_player = player_pos - prev_pos
            dist_to_player = np.linalg.norm(to_player)
            if dist_to_player < 10:
                continue
            to_player_norm = to_player / (dist_to_player + 1e-8)

            player_alignment = np.dot(puck_dir_norm, to_player_norm)

            if player_alignment > 0.6:
                return "pass"

        return "dump"

    def _count_lane_blockers(self, idx: int, ctx: FrameContext) -> int:
        """Count players between puck and goal (rough shooting lane check)."""
        if not ctx.goalies or ctx.puck is None:
            return 0

        puck_pos = np.array(ctx.puck.center)
        goalie_pos = np.array(ctx.goalies[0].center)

        # Line from puck to goal
        lane_vec = goalie_pos - puck_pos
        lane_len = np.linalg.norm(lane_vec)
        if lane_len < 10:
            return 0

        lane_dir = lane_vec / lane_len
        blockers = 0

        for player in ctx.players:
            if player.track_id == ctx.possession_player_id:
                continue

            player_pos = np.array(player.center)
            to_player = player_pos - puck_pos

            # Project onto lane
            proj = np.dot(to_player, lane_dir)
            if proj < 0 or proj > lane_len:
                continue  # Not between puck and goal

            # Perpendicular distance to lane
            perp = np.linalg.norm(to_player - proj * lane_dir)
            if perp < 50:  # Within ~50px of the lane
                blockers += 1

        return blockers

    def _count_open_teammates(self, idx: int, ctx: FrameContext) -> int:
        """Count teammates who are relatively open (not closely guarded)."""
        if ctx.possession_player_id is None:
            return 0

        open_count = 0
        for player in ctx.players:
            if player.track_id == ctx.possession_player_id:
                continue
            if player.track_id < 0:
                continue

            # Check if any other player is very close (guarding)
            player_pos = np.array(player.center)
            is_guarded = False
            for other in ctx.players:
                if other.track_id == player.track_id:
                    continue
                if other.track_id == ctx.possession_player_id:
                    continue
                other_pos = np.array(other.center)
                if np.linalg.norm(player_pos - other_pos) < 60:
                    is_guarded = True
                    break

            if not is_guarded:
                open_count += 1

        return open_count
