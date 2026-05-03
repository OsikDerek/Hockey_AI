"""Pixel <-> ice coordinate calibration from detected rink landmarks.

Phase B0 uses a 2D similarity transform (translation + rotation + uniform
scale) anchored on the goal positions and centroid. Phase B0.5 stabilizes
that transform across frames so player positions on the minimap (and the
3D viewer in Phase C) move smoothly instead of teleporting.

Stability strategies (B0.5):
1. **EMA smoothing**: blend each new fit into the previous transform
   instead of replacing it outright.
2. **Outlier rejection**: drop a fit if its scale changes >50% or its
   origin jumps >200 px from the previous transform — broadcast cameras
   don't actually do that.
3. **Side disambiguation by prediction**: when a single goal is visible
   and we already have a previous transform, pick the goal-side guess
   whose predicted pixel position is closest to the observed goal.
4. **Hard reset on camera cuts**: caller must invoke .reset() when
   `is_camera_cut` is true so we don't blend across angle changes.

Phase B1 will upgrade to a full 8-DoF homography once we solve the
faceoff-dot identification problem.

Ice coordinates use the standard NHL convention: x in [0, 200] running
goal-line to goal-line, y in [0, 85] running board-to-board, units feet.
"""

from typing import Optional

import numpy as np


# NHL rink geometry (feet)
RINK_LENGTH_FT = 200.0
RINK_WIDTH_FT = 85.0
GOAL_LINE_FROM_END_FT = 11.0
NHL_GOAL_WIDTH_FT = 6.0
LEFT_GOAL_X_FT = GOAL_LINE_FROM_END_FT
RIGHT_GOAL_X_FT = RINK_LENGTH_FT - GOAL_LINE_FROM_END_FT  # 189
CENTER_X_FT = RINK_LENGTH_FT / 2.0     # 100
CENTER_Y_FT = RINK_WIDTH_FT / 2.0      # 42.5
BLUE_LINE_LEFT_X_FT = 75.0
BLUE_LINE_RIGHT_X_FT = 125.0


class RinkCalibrator:
    """Maintain a smoothed pixel<->ice similarity transform across frames."""

    def __init__(
        self,
        ema_alpha: float = 0.15,                 # weight of new fit (0=hold, 1=replace)
        max_scale_change: float = 0.5,           # reject fit if scale changes by >50%
        max_origin_jump_px: float = 200.0,       # reject fit if origin moves >200 px
        min_scale_px_per_ft: float = 2.0,        # plausibility floor (rink 4x frame width)
        max_scale_px_per_ft: float = 25.0,       # plausibility ceiling (rink 0.4x frame width)
    ):
        self.ema_alpha = ema_alpha
        self.max_scale_change = max_scale_change
        self.max_origin_jump_px = max_origin_jump_px
        self.min_scale_px_per_ft = min_scale_px_per_ft
        self.max_scale_px_per_ft = max_scale_px_per_ft

        # Similarity transform parameters: ice (x_ft, y_ft) -> pixel (px, py)
        # px = origin_px[0] + scale * (cos*x_ft - sin*y_ft)
        # py = origin_px[1] + scale * (sin*x_ft + cos*y_ft)
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
        """Clear all calibration state — call on camera cut."""
        self._origin_px = None
        self._scale_px_per_ft = None
        self._cos = 1.0
        self._sin = 0.0
        self._calibrated = False

    def update(self, rink_landmarks: dict, frame_width: int, frame_height: int) -> None:
        """Per-frame calibration update with smoothing + outlier rejection."""
        self._frame_count += 1

        goals = [g for g in rink_landmarks.get("goal", []) if g.confidence >= 0.4]
        centroids = [c for c in rink_landmarks.get("centroid", []) if c.confidence >= 0.4]

        # Reject obviously bad goal detections (the bbox aspect ratio of a
        # real net is roughly square; far-zoom misdetections are often
        # vertical slivers or thin strips).
        goals = [g for g in goals if self._goal_bbox_plausible(g)]

        # Side-bin goals using the previous transform if we have one;
        # otherwise fall back to frame-half.
        left_goals, right_goals = self._bin_goals_by_side(goals, frame_width)

        candidate = None
        if left_goals and right_goals:
            candidate = self._fit_from_two_goals(left_goals[0], right_goals[0], centroids)
        elif goals and centroids:
            candidate = self._fit_from_goal_and_centroid(goals[0], centroids[0])
        elif goals:
            candidate = self._fit_from_single_goal(goals[0], frame_width)

        if candidate is None:
            return  # No new info this frame; hold previous transform

        # Plausibility floor / ceiling on the absolute scale — rejects
        # fits that would imply an absurd rink size in the frame.
        cand_scale = candidate[1]
        if not (self.min_scale_px_per_ft <= cand_scale <= self.max_scale_px_per_ft):
            return

        # Outlier rejection (only meaningful once we have a previous fit)
        if self._calibrated and not self._fit_is_plausible(candidate):
            return

        # Apply the candidate. First fit replaces; subsequent fits blend via EMA.
        if not self._calibrated:
            self._apply_fit(candidate, alpha=1.0)
        else:
            self._apply_fit(candidate, alpha=self.ema_alpha)
        self._calibrated = True

    # ── Fit candidates: each returns a tuple (origin_px, scale, cos, sin) ──
    def _fit_from_two_goals(self, left_goal, right_goal, centroids: list):
        left_x, left_y = left_goal.center
        right_x, right_y = right_goal.center
        goal_axis_px = right_x - left_x
        if goal_axis_px <= 1.0:
            return None
        scale = goal_axis_px / (RIGHT_GOAL_X_FT - LEFT_GOAL_X_FT)
        line_y_px = float(centroids[0].center[1]) if centroids else (left_y + right_y) / 2.0
        origin_x = left_x - LEFT_GOAL_X_FT * scale
        origin_y = line_y_px - CENTER_Y_FT * scale
        return (np.array([origin_x, origin_y], dtype=np.float64), float(scale), 1.0, 0.0)

    def _fit_from_goal_and_centroid(self, goal, centroid):
        gx, gy = goal.center
        cx, cy = centroid.center
        goal_ice_x = LEFT_GOAL_X_FT if gx < cx else RIGHT_GOAL_X_FT
        ice_axis_ft = abs(CENTER_X_FT - goal_ice_x)
        goal_axis_px = abs(cx - gx)
        if goal_axis_px <= 1.0:
            return None
        scale = goal_axis_px / ice_axis_ft
        origin_x = gx - goal_ice_x * scale
        origin_y = float(cy) - CENTER_Y_FT * scale
        return (np.array([origin_x, origin_y], dtype=np.float64), float(scale), 1.0, 0.0)

    def _fit_from_single_goal(self, goal, frame_width: int):
        """Single-goal observations only refine the origin, never scale.

        Goal-bbox width is too noisy (varies heavily with camera zoom and
        partial occlusion) to derive scale reliably. We only use single-goal
        observations to track camera pan — translating the origin without
        changing the scale that was set by a more reliable fit.
        """
        if self._scale_px_per_ft is None or not self._calibrated:
            # No previous scale to anchor to — refuse to fit at all.
            return None
        gx, gy = goal.center
        scale = self._scale_px_per_ft
        goal_ice_x = self._best_goal_side_guess(gx, frame_width)
        origin_x = gx - goal_ice_x * scale
        origin_y = gy - CENTER_Y_FT * scale
        return (np.array([origin_x, origin_y], dtype=np.float64), float(scale), 1.0, 0.0)

    # ── Helpers ──
    def _bin_goals_by_side(self, goals: list, frame_width: int):
        """Split detected goals into left-side / right-side groups."""
        if not self._calibrated or self._origin_px is None or self._scale_px_per_ft is None:
            half = frame_width / 2.0
            left = [g for g in goals if g.center[0] < half]
            right = [g for g in goals if g.center[0] >= half]
            return left, right
        # Use the existing transform to predict where each goal-side SHOULD
        # appear, then assign each detection to whichever predicted side is
        # closer.
        left_pred = self.ice_to_pixel((LEFT_GOAL_X_FT, CENTER_Y_FT))
        right_pred = self.ice_to_pixel((RIGHT_GOAL_X_FT, CENTER_Y_FT))
        left, right = [], []
        for g in goals:
            gx = g.center[0]
            d_left = abs(gx - left_pred[0]) if left_pred else float("inf")
            d_right = abs(gx - right_pred[0]) if right_pred else float("inf")
            if d_left <= d_right:
                left.append(g)
            else:
                right.append(g)
        return left, right

    def _best_goal_side_guess(self, gx: float, frame_width: int) -> float:
        """Pick LEFT_GOAL_X_FT or RIGHT_GOAL_X_FT for a single observed goal pixel."""
        if self._calibrated and self._origin_px is not None and self._scale_px_per_ft is not None:
            left_pred = self.ice_to_pixel((LEFT_GOAL_X_FT, CENTER_Y_FT))
            right_pred = self.ice_to_pixel((RIGHT_GOAL_X_FT, CENTER_Y_FT))
            if left_pred is not None and right_pred is not None:
                if abs(gx - left_pred[0]) <= abs(gx - right_pred[0]):
                    return LEFT_GOAL_X_FT
                return RIGHT_GOAL_X_FT
        # Fallback: frame-half heuristic
        return LEFT_GOAL_X_FT if gx < frame_width / 2.0 else RIGHT_GOAL_X_FT

    @staticmethod
    def _goal_bbox_plausible(goal) -> bool:
        x1, y1, x2, y2 = goal.bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        ratio = w / h
        return 0.6 <= ratio <= 3.0

    def _fit_is_plausible(self, candidate) -> bool:
        new_origin, new_scale, _, _ = candidate
        if self._scale_px_per_ft is None or self._origin_px is None:
            return True
        scale_change = abs(new_scale - self._scale_px_per_ft) / max(1e-6, self._scale_px_per_ft)
        if scale_change > self.max_scale_change:
            return False
        origin_jump = float(np.linalg.norm(new_origin - self._origin_px))
        if origin_jump > self.max_origin_jump_px:
            return False
        return True

    def _apply_fit(self, candidate, alpha: float) -> None:
        new_origin, new_scale, new_cos, new_sin = candidate
        if self._origin_px is None or self._scale_px_per_ft is None or alpha >= 1.0:
            self._origin_px = new_origin.copy()
            self._scale_px_per_ft = new_scale
            self._cos = new_cos
            self._sin = new_sin
            return
        self._origin_px = (1.0 - alpha) * self._origin_px + alpha * new_origin
        self._scale_px_per_ft = (1.0 - alpha) * self._scale_px_per_ft + alpha * new_scale
        self._cos = (1.0 - alpha) * self._cos + alpha * new_cos
        self._sin = (1.0 - alpha) * self._sin + alpha * new_sin

    # ── Coordinate transforms ──
    def pixel_to_ice(self, pt) -> Optional[tuple]:
        if not self._calibrated or self._origin_px is None or self._scale_px_per_ft is None:
            return None
        px, py = float(pt[0]), float(pt[1])
        dx = px - self._origin_px[0]
        dy = py - self._origin_px[1]
        x_ft = (self._cos * dx + self._sin * dy) / self._scale_px_per_ft
        y_ft = (-self._sin * dx + self._cos * dy) / self._scale_px_per_ft
        return (float(x_ft), float(y_ft))

    def ice_to_pixel(self, pt) -> Optional[tuple]:
        if not self._calibrated or self._origin_px is None or self._scale_px_per_ft is None:
            return None
        x_ft, y_ft = float(pt[0]), float(pt[1])
        rx = self._cos * x_ft - self._sin * y_ft
        ry = self._sin * x_ft + self._cos * y_ft
        px = self._origin_px[0] + self._scale_px_per_ft * rx
        py = self._origin_px[1] + self._scale_px_per_ft * ry
        return (float(px), float(py))
