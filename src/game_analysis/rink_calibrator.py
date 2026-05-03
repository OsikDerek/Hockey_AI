"""Pixel <-> ice coordinate calibration from detected rink landmarks.

Phase B0 uses a 2D similarity transform (translation + rotation + uniform
scale) anchored on the two goal x-positions and the centroid (center ice).
This is less accurate than a full perspective homography (it can't correct
for camera tilt that compresses the far side of the rink) but it's robust
when only a handful of landmarks are visible per frame, which is the
common case in broadcast follow-cam footage.

Phase B1 will upgrade to a full 8-DoF homography once we solve the
faceoff-dot identification problem (the YOLO model labels all 8 faceoff
dots as a single class).

Ice coordinates use the standard NHL convention: x in [0, 200] running
goal-line to goal-line, y in [0, 85] running board-to-board, units feet.
"""

from collections import deque
from typing import Optional

import numpy as np


# NHL rink geometry (feet)
RINK_LENGTH_FT = 200.0
RINK_WIDTH_FT = 85.0
GOAL_LINE_FROM_END_FT = 11.0           # goal line is 11 ft from each end board
NHL_GOAL_WIDTH_FT = 6.0                # NHL regulation goal is 6 ft wide
LEFT_GOAL_X_FT = GOAL_LINE_FROM_END_FT
RIGHT_GOAL_X_FT = RINK_LENGTH_FT - GOAL_LINE_FROM_END_FT  # 189
CENTER_X_FT = RINK_LENGTH_FT / 2.0     # 100
CENTER_Y_FT = RINK_WIDTH_FT / 2.0      # 42.5
BLUE_LINE_LEFT_X_FT = 75.0
BLUE_LINE_RIGHT_X_FT = 125.0


class RinkCalibrator:
    """Maintain a running pixel<->ice similarity transform.

    Update once per frame with the rink landmarks dict (FrameContext format).
    Once enough samples accumulate, builds a transform that can map between
    pixel coordinates and ice (feet) coordinates either way.
    """

    def __init__(self, recalibrate_every: int = 300, history_size: int = 100):
        self.recalibrate_every = recalibrate_every
        self._left_goal_xs: deque = deque(maxlen=history_size)
        self._right_goal_xs: deque = deque(maxlen=history_size)
        self._goal_ys: deque = deque(maxlen=history_size)        # average goal y, for rotation
        self._center_xs: deque = deque(maxlen=history_size)
        self._center_ys: deque = deque(maxlen=history_size)

        # Similarity transform parameters: ice (x_ft, y_ft) -> pixel (px, py)
        # px = origin_px[0] + scale * (cos*x_ft - sin*y_ft)
        # py = origin_px[1] + scale * (sin*x_ft + cos*y_ft)
        # Where origin is the pixel position of ice (0, 0).
        self._origin_px: Optional[np.ndarray] = None
        self._scale_px_per_ft: Optional[float] = None
        self._cos: float = 1.0
        self._sin: float = 0.0
        self._calibrated: bool = False
        self._frame_count: int = 0

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def reset(self) -> None:
        """Clear all calibration state — call on hard scene change."""
        self._left_goal_xs.clear()
        self._right_goal_xs.clear()
        self._goal_ys.clear()
        self._center_xs.clear()
        self._center_ys.clear()
        self._origin_px = None
        self._scale_px_per_ft = None
        self._cos = 1.0
        self._sin = 0.0
        self._calibrated = False

    def update(self, rink_landmarks: dict, frame_width: int, frame_height: int) -> None:
        """Per-frame calibration update.

        Broadcast follow-cam rarely shows both goals at once, so we cannot
        accumulate left/right goal positions across frames (they'd come from
        different camera angles and produce nonsense scale). Instead, we
        try to fit a transform from THIS frame's landmarks alone, and hold
        the previous transform when this frame doesn't have enough.

        Per-frame strategies, in order of preference:
        1. Both left + right goal visible: scale from goal-to-goal pixel
           distance (most accurate when available).
        2. One goal + centroid: scale from goal-to-centroid pixel distance
           (the common broadcast case — camera follows the play to one end).
        3. Hold previous transform.
        """
        self._frame_count += 1

        goals = [g for g in rink_landmarks.get("goal", []) if g.confidence >= 0.4]
        centroids = [c for c in rink_landmarks.get("centroid", []) if c.confidence >= 0.4]

        # Split goals into left/right of frame center (best we can do
        # without explicit handedness)
        left_goals = [g for g in goals if g.center[0] < frame_width / 2.0]
        right_goals = [g for g in goals if g.center[0] >= frame_width / 2.0]

        if left_goals and right_goals:
            self._fit_from_two_goals(left_goals[0], right_goals[0], centroids, frame_height)
        elif goals and centroids:
            self._fit_from_goal_and_centroid(goals[0], centroids[0])
        elif goals:
            # Last resort: a single goal alone. Use the known regulation
            # goal-frame width (6 ft) to derive scale, place the goal at its
            # known ice x-coord, anchor y=42.5 ft to the goal's pixel y.
            self._fit_from_single_goal(goals[0], frame_width)
        # else: hold whatever transform we already have

    def _fit_from_two_goals(self, left_goal, right_goal, centroids: list, frame_height: int) -> None:
        left_x, left_y = left_goal.center
        right_x, right_y = right_goal.center
        goal_axis_px = right_x - left_x
        if goal_axis_px <= 1.0:
            return
        ice_axis_ft = RIGHT_GOAL_X_FT - LEFT_GOAL_X_FT  # 178 ft
        self._scale_px_per_ft = goal_axis_px / ice_axis_ft
        self._cos = 1.0
        self._sin = 0.0
        # Use centroid y if visible; else average of the two goal ys
        line_y_px = float(centroids[0].center[1]) if centroids else (left_y + right_y) / 2.0
        origin_x = left_x - LEFT_GOAL_X_FT * self._scale_px_per_ft
        origin_y = line_y_px - CENTER_Y_FT * self._scale_px_per_ft
        self._origin_px = np.array([origin_x, origin_y], dtype=np.float64)
        self._calibrated = True

    def _fit_from_single_goal(self, goal, frame_width: int) -> None:
        gx, gy = goal.center
        gx1, _, gx2, _ = goal.bbox
        goal_width_px = float(gx2 - gx1)
        if goal_width_px <= 1.0:
            return
        self._scale_px_per_ft = goal_width_px / NHL_GOAL_WIDTH_FT
        self._cos = 1.0
        self._sin = 0.0
        # Decide which goal this is by frame-half. This is a heuristic:
        # if the camera is mostly looking at one end, the goal there will
        # appear roughly centered; we still need a side guess. Use frame
        # half — same convention as elsewhere.
        goal_ice_x = LEFT_GOAL_X_FT if gx < frame_width / 2.0 else RIGHT_GOAL_X_FT
        origin_x = gx - goal_ice_x * self._scale_px_per_ft
        origin_y = gy - CENTER_Y_FT * self._scale_px_per_ft
        self._origin_px = np.array([origin_x, origin_y], dtype=np.float64)
        self._calibrated = True

    def _fit_from_goal_and_centroid(self, goal, centroid) -> None:
        gx, gy = goal.center
        cx, cy = centroid.center
        # Determine which side this goal is on by comparing x to centroid x
        if gx < cx:
            goal_ice_x = LEFT_GOAL_X_FT
        else:
            goal_ice_x = RIGHT_GOAL_X_FT
        ice_axis_ft = abs(CENTER_X_FT - goal_ice_x)  # 89 ft
        goal_axis_px = abs(cx - gx)
        if goal_axis_px <= 1.0:
            return
        self._scale_px_per_ft = goal_axis_px / ice_axis_ft
        self._cos = 1.0
        self._sin = 0.0
        # The line through goal and centroid is the rink centerline (y=42.5).
        # Use centroid's y as the centerline.
        line_y_px = float(cy)
        # Origin in pixel space: subtract LEFT_GOAL or RIGHT_GOAL contribution
        # appropriately. Both reference points lie on the centerline, so the
        # origin x simply offsets from the goal we have.
        if goal_ice_x == LEFT_GOAL_X_FT:
            origin_x = gx - LEFT_GOAL_X_FT * self._scale_px_per_ft
        else:
            origin_x = gx - RIGHT_GOAL_X_FT * self._scale_px_per_ft
        origin_y = line_y_px - CENTER_Y_FT * self._scale_px_per_ft
        self._origin_px = np.array([origin_x, origin_y], dtype=np.float64)
        self._calibrated = True

    def pixel_to_ice(self, pt) -> Optional[tuple]:
        """Map a pixel (px, py) to ice (x_ft, y_ft). None if not calibrated."""
        if not self._calibrated or self._origin_px is None or self._scale_px_per_ft is None:
            return None
        px, py = float(pt[0]), float(pt[1])
        dx = px - self._origin_px[0]
        dy = py - self._origin_px[1]
        # Inverse rotation (transpose for the rotation portion of the
        # similarity), then divide out the scale.
        x_ft = (self._cos * dx + self._sin * dy) / self._scale_px_per_ft
        y_ft = (-self._sin * dx + self._cos * dy) / self._scale_px_per_ft
        return (float(x_ft), float(y_ft))

    def ice_to_pixel(self, pt) -> Optional[tuple]:
        """Map ice (x_ft, y_ft) to pixel (px, py). None if not calibrated."""
        if not self._calibrated or self._origin_px is None or self._scale_px_per_ft is None:
            return None
        x_ft, y_ft = float(pt[0]), float(pt[1])
        rx = self._cos * x_ft - self._sin * y_ft
        ry = self._sin * x_ft + self._cos * y_ft
        px = self._origin_px[0] + self._scale_px_per_ft * rx
        py = self._origin_px[1] + self._scale_px_per_ft * ry
        return (float(px), float(py))
