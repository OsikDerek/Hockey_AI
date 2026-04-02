"""Video frame annotation for skating analysis.

Draws skeleton overlays, angle measurements, color-coded feedback,
and a HUD panel on video frames.
"""

import cv2
import numpy as np
from typing import Optional
from src.pose_estimator import SKELETON_CONNECTIONS, LANDMARK_NAMES
from src.mechanics_engine import MechanicResult


# Color scheme (BGR format for OpenCV)
COLORS = {
    "good": (0, 220, 0),       # Green
    "warning": (0, 220, 220),   # Yellow
    "poor": (0, 0, 220),        # Red
    "skeleton": (200, 200, 200),  # Light gray
    "keypoint": (255, 165, 0),    # Orange
    "left": (255, 200, 50),       # Light blue
    "right": (50, 200, 255),      # Orange-ish
    "text_bg": (40, 40, 40),      # Dark gray
}

# Landmarks where we draw angle arcs
ANGLE_DRAW_POINTS = {
    "knee_angle": {"left": "left_knee", "right": "right_knee"},
    "hip_angle": {"left": "left_hip", "right": "right_hip"},
    "forward_lean": {"left": "left_shoulder", "right": "right_shoulder"},
    "ankle_dorsiflexion": {"left": "left_ankle", "right": "right_ankle"},
}


class SkatingAnnotator:
    """Draws skating analysis overlays on video frames."""

    def __init__(
        self,
        skeleton_thickness: int = 2,
        keypoint_radius: int = 4,
        font_scale: float = 0.5,
        show_skeleton: bool = True,
        show_angles: bool = True,
        show_hud: bool = True,
    ):
        """Initialize annotator settings.

        Args:
            skeleton_thickness: Line width for skeleton connections.
            keypoint_radius: Circle radius for keypoint dots.
            font_scale: Text size multiplier.
            show_skeleton: Whether to draw the skeleton.
            show_angles: Whether to draw angle measurements at joints.
            show_hud: Whether to draw the HUD feedback panel.
        """
        self.skeleton_thickness = skeleton_thickness
        self.keypoint_radius = keypoint_radius
        self.font_scale = font_scale
        self.show_skeleton = show_skeleton
        self.show_angles = show_angles
        self.show_hud = show_hud
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(
        self,
        frame: np.ndarray,
        landmarks: Optional[dict],
        mechanic_results: Optional[list[MechanicResult]] = None,
    ) -> np.ndarray:
        """Render all annotations onto a frame.

        Args:
            frame: BGR frame (will be modified in-place and returned).
            landmarks: Dict of smoothed landmarks, or None if no detection.
            mechanic_results: List of MechanicResult objects, or None.

        Returns:
            Annotated BGR frame.
        """
        annotated = frame.copy()

        if landmarks is None:
            # No detection — draw "No skater detected" message
            cv2.putText(
                annotated,
                "No skater detected",
                (20, 40),
                self.font,
                0.8,
                (0, 0, 255),
                2,
            )
            return annotated

        if self.show_skeleton:
            self._draw_skeleton(annotated, landmarks)

        if self.show_angles and mechanic_results:
            self._draw_angle_labels(annotated, landmarks, mechanic_results)

        if self.show_hud and mechanic_results:
            self._draw_hud(annotated, mechanic_results)

        return annotated

    def _draw_skeleton(self, frame: np.ndarray, landmarks: dict):
        """Draw skeleton connections and keypoint dots."""
        # Draw connections
        for lm_a_name, lm_b_name in SKELETON_CONNECTIONS:
            if lm_a_name not in landmarks or lm_b_name not in landmarks:
                continue

            lm_a = landmarks[lm_a_name]
            lm_b = landmarks[lm_b_name]

            if lm_a["visibility"] < 0.5 or lm_b["visibility"] < 0.5:
                continue

            pt_a = (int(lm_a["x"]), int(lm_a["y"]))
            pt_b = (int(lm_b["x"]), int(lm_b["y"]))

            # Color by side
            if "left" in lm_a_name:
                color = COLORS["left"]
            elif "right" in lm_a_name:
                color = COLORS["right"]
            else:
                color = COLORS["skeleton"]

            cv2.line(frame, pt_a, pt_b, color, self.skeleton_thickness)

        # Draw keypoints (only body landmarks, skip face)
        body_start = LANDMARK_NAMES.index("left_shoulder")
        for name in LANDMARK_NAMES[body_start:]:
            if name not in landmarks:
                continue
            lm = landmarks[name]
            if lm["visibility"] < 0.5:
                continue

            pt = (int(lm["x"]), int(lm["y"]))
            cv2.circle(frame, pt, self.keypoint_radius, COLORS["keypoint"], -1)

    def _draw_angle_labels(
        self,
        frame: np.ndarray,
        landmarks: dict,
        results: list[MechanicResult],
    ):
        """Draw angle values near their corresponding joints with color coding."""
        for result in results:
            # Find the landmark position to draw near
            draw_info = ANGLE_DRAW_POINTS.get(result.name)
            if draw_info is None:
                continue

            if result.side and result.side in draw_info:
                lm_name = draw_info[result.side]
            else:
                continue

            if lm_name not in landmarks:
                continue

            lm = landmarks[lm_name]
            if lm["visibility"] < 0.5:
                continue

            color = COLORS.get(result.rating, COLORS["skeleton"])
            pt = (int(lm["x"]) + 10, int(lm["y"]) - 10)

            # Draw text with background for readability
            text = f"{result.value:.0f}\xb0"
            text_size = cv2.getTextSize(text, self.font, self.font_scale, 1)[0]

            # Background rectangle
            bg_pt1 = (pt[0] - 2, pt[1] - text_size[1] - 2)
            bg_pt2 = (pt[0] + text_size[0] + 2, pt[1] + 4)
            cv2.rectangle(frame, bg_pt1, bg_pt2, COLORS["text_bg"], -1)

            cv2.putText(frame, text, pt, self.font, self.font_scale, color, 1)

    def _draw_hud(self, frame: np.ndarray, results: list[MechanicResult]):
        """Draw a semi-transparent HUD panel with all mechanic results."""
        if not results:
            return

        h, w = frame.shape[:2]

        # Panel dimensions
        line_height = 24
        panel_width = 320
        panel_height = 30 + len(results) * line_height
        panel_x = w - panel_width - 10
        panel_y = 10

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            COLORS["text_bg"],
            -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(
            frame,
            "SKATING MECHANICS",
            (panel_x + 10, panel_y + 20),
            self.font,
            0.5,
            (255, 255, 255),
            1,
        )

        # Each mechanic result
        for i, result in enumerate(results):
            y = panel_y + 40 + i * line_height
            color = COLORS.get(result.rating, (255, 255, 255))

            # Rating indicator dot
            cv2.circle(frame, (panel_x + 15, y - 4), 5, color, -1)

            # Mechanic name and value
            text = f"{result.display_name}: {result.value:.1f}\xb0"
            cv2.putText(
                frame, text, (panel_x + 28, y), self.font, 0.42, (255, 255, 255), 1
            )

            # Rating text
            rating_text = result.rating.upper()
            cv2.putText(
                frame,
                rating_text,
                (panel_x + panel_width - 60, y),
                self.font,
                0.38,
                color,
                1,
            )
