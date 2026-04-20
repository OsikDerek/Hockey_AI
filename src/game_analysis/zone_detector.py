"""Ice zone detection from broadcast footage.

Determines whether play is in the offensive, neutral, or defensive zone
using detected rink landmarks (goals, faceoff dots, center ice) and
player distribution heuristics as fallback.
"""

import numpy as np
from typing import Optional
from collections import deque

from .game_context import TrackedObject


class ZoneDetector:
    """Classify the current zone from tracked objects.

    Primary method: Use rink landmarks (goal, faceoff, centroid detections)
    to estimate blue line positions, then classify puck/player positions.

    Fallback: Use player distribution heuristic when landmarks aren't visible.

    Usage:
        zd = ZoneDetector()
        zone = zd.update(objects, frame_width=1920)
    """

    def __init__(self, smoothing_window: int = 15, recalibrate_every: int = 300):
        """
        Args:
            smoothing_window: Rolling window for zone classification smoothing.
            recalibrate_every: Rebuild blue-line estimates every N frames to
                track follow-cam drift. ~10s at 30fps.
        """
        self.smoothing_window = smoothing_window
        self.recalibrate_every = recalibrate_every

        # Rink calibration state (bounded histories so old detections decay)
        self._goal_positions = deque(maxlen=100)    # x-positions of detected goals
        self._center_positions = deque(maxlen=100)  # x-positions of center ice
        self._blue_lines = None         # (left_x, right_x) once calibrated
        self._calibrated = False
        self._calibration_frame = 0     # tick counter for periodic rebuild

        # Zone history for smoothing
        self._zone_history = deque(maxlen=smoothing_window)

        # Which side is "our" goal (for offensive/defensive)
        # Default: left = defensive, right = offensive
        # Can be flipped if we detect team direction
        self._defensive_side = "left"

    def update(
        self,
        objects: list,
        puck: Optional[TrackedObject],
        frame_width: int,
    ) -> Optional[str]:
        """Update zone estimate from current frame's detections.

        Args:
            objects: All tracked objects this frame.
            puck: Puck TrackedObject (or None).
            frame_width: Width of the frame in pixels.

        Returns:
            "offensive", "neutral", "defensive", or None.
        """
        # Try to calibrate from rink landmarks
        self._update_calibration(objects, frame_width)

        # Determine raw zone
        raw_zone = None
        if self._calibrated and puck is not None:
            raw_zone = self._zone_from_landmarks(puck, frame_width)
        elif puck is not None:
            raw_zone = self._zone_from_heuristic(objects, puck, frame_width)

        # Smooth with history
        if raw_zone is not None:
            self._zone_history.append(raw_zone)

        return self._smoothed_zone()

    def _update_calibration(self, objects: list, frame_width: int):
        """Update rink landmark estimates from detected objects.

        Called every frame. Accumulates goal/centroid x-positions (bounded to
        last 100 samples), periodically rebuilds blue-line estimates to track
        follow-cam drift, and uses NHL-accurate symmetric geometry.
        """
        self._calibration_frame += 1

        # Periodic rebuild so a moving camera can re-calibrate
        if (
            self._calibration_frame % self.recalibrate_every == 0
            and self._calibration_frame > 0
        ):
            self._calibrated = False
            self._blue_lines = None

        for obj in objects:
            if obj.class_name == "goal" and obj.confidence > 0.4:
                self._goal_positions.append(obj.center[0])
            elif obj.class_name == "centroid" and obj.confidence > 0.4:
                self._center_positions.append(obj.center[0])

        # Need at least some goal and center detections to calibrate
        if len(self._goal_positions) >= 5 and len(self._center_positions) >= 3:
            # Goals should cluster at two x-positions (left and right)
            goal_x = np.array(list(self._goal_positions))
            center_x = np.median(list(self._center_positions))

            # Split goals into left and right clusters
            left_goals = goal_x[goal_x < center_x]
            right_goals = goal_x[goal_x >= center_x]

            if len(left_goals) > 0 and len(right_goals) > 0:
                left_goal_x = np.median(left_goals)
                right_goal_x = np.median(right_goals)

                # NHL geometry: goal line to blue line = 64 ft; goal line to
                # center = 89 ft; so blue line is ~0.72 of the way from goal
                # toward center. Apply the same 0.72 ratio on both sides
                # (mirrored) so left/right zones are symmetric.
                left_blue = left_goal_x + (center_x - left_goal_x) * 0.72
                right_blue = center_x + (right_goal_x - center_x) * 0.28

                self._blue_lines = (left_blue, right_blue)
                self._calibrated = True

    def _zone_from_landmarks(
        self, puck: TrackedObject, frame_width: int
    ) -> str:
        """Classify zone using calibrated blue line positions."""
        px = puck.center[0]
        left_blue, right_blue = self._blue_lines

        if px < left_blue:
            return "defensive" if self._defensive_side == "left" else "offensive"
        elif px > right_blue:
            return "offensive" if self._defensive_side == "left" else "defensive"
        else:
            return "neutral"

    def _zone_from_heuristic(
        self,
        objects: list,
        puck: TrackedObject,
        frame_width: int,
    ) -> str:
        """Fallback: classify zone from puck position relative to frame."""
        # Simple thirds heuristic
        px = puck.center[0]
        third = frame_width / 3.0

        if px < third:
            return "defensive" if self._defensive_side == "left" else "offensive"
        elif px > 2 * third:
            return "offensive" if self._defensive_side == "left" else "defensive"
        else:
            return "neutral"

    def _smoothed_zone(self) -> Optional[str]:
        """Return majority-vote zone from recent history."""
        if not self._zone_history:
            return None

        # Majority vote
        counts = {}
        for z in self._zone_history:
            counts[z] = counts.get(z, 0) + 1

        return max(counts, key=counts.get)

    def reset(self):
        """Reset all state."""
        self._goal_positions.clear()
        self._center_positions.clear()
        self._blue_lines = None
        self._calibrated = False
        self._calibration_frame = 0
        self._zone_history.clear()
        self._defensive_side = "left"
