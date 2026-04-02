"""Video frame annotation for skating analysis.

Draws skeleton overlays with color-coded error highlighting,
a compact HUD panel, and visual callouts on problem joints.

Design philosophy (per coach feedback):
  - Raw angle numbers are NOT useful on joints — keep them in HUD only
  - Problem joints get circled/highlighted in red so issues are obvious at a glance
  - Skeleton segments turn red when connected joints have poor ratings
  - HUD shows compact status with color dots
"""

import cv2
import numpy as np
from typing import Optional
from src.pose_estimator import SKELETON_CONNECTIONS, LANDMARK_NAMES
from src.mechanics_engine import MechanicResult


# Color scheme (BGR format for OpenCV)
COLORS = {
    "good": (0, 200, 0),         # Green
    "warning": (0, 200, 220),    # Yellow
    "poor": (0, 0, 240),         # Bright red
    "skeleton": (180, 180, 180), # Light gray
    "keypoint": (255, 165, 0),   # Orange
    "left": (255, 200, 50),      # Light blue
    "right": (50, 200, 255),     # Orange-ish
    "text_bg": (30, 30, 30),     # Dark gray
    "error_ring": (0, 0, 255),   # Pure red for error circles
    "error_glow": (0, 50, 200),  # Darker red for glow effect
}

# Map mechanic names to the joint landmark they apply to
MECHANIC_JOINT_MAP = {
    "knee_angle": {"left": "left_knee", "right": "right_knee"},
    "hip_angle": {"left": "left_hip", "right": "right_hip"},
    "forward_lean": {"left": "left_shoulder", "right": "right_shoulder"},
    "ankle_dorsiflexion": {"left": "left_ankle", "right": "right_ankle"},
}

# Skeleton segments associated with each mechanic (for coloring)
MECHANIC_SEGMENTS = {
    "knee_angle": {
        "left": [("left_hip", "left_knee"), ("left_knee", "left_ankle")],
        "right": [("right_hip", "right_knee"), ("right_knee", "right_ankle")],
    },
    "hip_angle": {
        "left": [("left_shoulder", "left_hip"), ("left_hip", "left_knee")],
        "right": [("right_shoulder", "right_hip"), ("right_hip", "right_knee")],
    },
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
        self.skeleton_thickness = skeleton_thickness
        self.keypoint_radius = keypoint_radius
        self.font_scale = font_scale
        self.show_skeleton = show_skeleton
        self.show_angles = show_angles
        self.show_hud = show_hud
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self._frame_counter = 0

    def render(
        self,
        frame: np.ndarray,
        landmarks: Optional[dict],
        mechanic_results: Optional[list[MechanicResult]] = None,
    ) -> np.ndarray:
        """Render all annotations onto a frame."""
        annotated = frame.copy()
        self._frame_counter += 1

        if landmarks is None:
            cv2.putText(
                annotated, "No skater detected",
                (20, 40), self.font, 0.8, (0, 0, 255), 2,
            )
            return annotated

        # Build a rating lookup for joints
        joint_ratings = self._build_joint_ratings(mechanic_results)

        if self.show_skeleton:
            self._draw_skeleton(annotated, landmarks, joint_ratings, mechanic_results)

        # Draw error highlights on problem joints
        if mechanic_results:
            self._draw_error_highlights(annotated, landmarks, mechanic_results)

        if self.show_hud and mechanic_results:
            self._draw_hud(annotated, mechanic_results)

        return annotated

    def _build_joint_ratings(
        self, results: Optional[list[MechanicResult]]
    ) -> dict[str, str]:
        """Build mapping of landmark_name -> worst rating affecting it."""
        ratings = {}
        if not results:
            return ratings
        for r in results:
            joint_map = MECHANIC_JOINT_MAP.get(r.name)
            if joint_map and r.side and r.side in joint_map:
                lm_name = joint_map[r.side]
                # Keep the worst rating
                current = ratings.get(lm_name, "good")
                if r.rating == "poor" or (r.rating == "warning" and current == "good"):
                    ratings[lm_name] = r.rating
        return ratings

    def _draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks: dict,
        joint_ratings: dict,
        mechanic_results: Optional[list[MechanicResult]],
    ):
        """Draw skeleton with color-coded segments based on ratings."""
        # Build segment rating lookup
        segment_ratings = {}
        if mechanic_results:
            for r in mechanic_results:
                seg_map = MECHANIC_SEGMENTS.get(r.name, {})
                if r.side and r.side in seg_map:
                    for seg in seg_map[r.side]:
                        key = tuple(sorted(seg))
                        current = segment_ratings.get(key, "good")
                        if r.rating == "poor" or (r.rating == "warning" and current == "good"):
                            segment_ratings[key] = r.rating

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

            # Check if this segment has a rating
            seg_key = tuple(sorted((lm_a_name, lm_b_name)))
            seg_rating = segment_ratings.get(seg_key)

            if seg_rating == "poor":
                color = COLORS["poor"]
                thickness = self.skeleton_thickness + 2
            elif seg_rating == "warning":
                color = COLORS["warning"]
                thickness = self.skeleton_thickness + 1
            elif "left" in lm_a_name:
                color = COLORS["left"]
                thickness = self.skeleton_thickness
            elif "right" in lm_a_name:
                color = COLORS["right"]
                thickness = self.skeleton_thickness
            else:
                color = COLORS["skeleton"]
                thickness = self.skeleton_thickness

            cv2.line(frame, pt_a, pt_b, color, thickness)

        # Draw keypoints with rating-based colors
        body_start = LANDMARK_NAMES.index("left_shoulder")
        for name in LANDMARK_NAMES[body_start:]:
            if name not in landmarks:
                continue
            lm = landmarks[name]
            if lm["visibility"] < 0.5:
                continue

            pt = (int(lm["x"]), int(lm["y"]))
            rating = joint_ratings.get(name)

            if rating == "poor":
                cv2.circle(frame, pt, self.keypoint_radius + 2, COLORS["poor"], -1)
            elif rating == "warning":
                cv2.circle(frame, pt, self.keypoint_radius + 1, COLORS["warning"], -1)
            else:
                cv2.circle(frame, pt, self.keypoint_radius, COLORS["keypoint"], -1)

    def _draw_error_highlights(
        self,
        frame: np.ndarray,
        landmarks: dict,
        results: list[MechanicResult],
    ):
        """Draw attention-grabbing circles around joints with poor ratings.

        Poor joints get a pulsing red ring + brief text callout.
        Warning joints get a smaller yellow ring.
        """
        for result in results:
            if result.rating == "good":
                continue

            joint_map = MECHANIC_JOINT_MAP.get(result.name)
            if not joint_map or not result.side or result.side not in joint_map:
                continue

            lm_name = joint_map[result.side]
            if lm_name not in landmarks:
                continue

            lm = landmarks[lm_name]
            if lm["visibility"] < 0.5:
                continue

            pt = (int(lm["x"]), int(lm["y"]))

            if result.rating == "poor":
                # Pulsing red ring — radius oscillates based on frame count
                pulse = int(8 + 6 * abs(np.sin(self._frame_counter * 0.15)))
                cv2.circle(frame, pt, pulse + 15, COLORS["error_ring"], 3)
                cv2.circle(frame, pt, pulse + 20, COLORS["error_glow"], 1)

                # Short callout label
                label = self._short_label(result.name)
                label_pt = (pt[0] + 25, pt[1] - 5)
                text_size = cv2.getTextSize(label, self.font, 0.45, 1)[0]
                bg1 = (label_pt[0] - 3, label_pt[1] - text_size[1] - 3)
                bg2 = (label_pt[0] + text_size[0] + 3, label_pt[1] + 5)
                cv2.rectangle(frame, bg1, bg2, COLORS["poor"], -1)
                cv2.putText(frame, label, label_pt, self.font, 0.45, (255, 255, 255), 1)

            elif result.rating == "warning":
                # Subtle yellow ring
                cv2.circle(frame, pt, 18, COLORS["warning"], 2)

    def _short_label(self, mechanic_name: str) -> str:
        """Return a very short label for on-screen callout."""
        labels = {
            "knee_angle": "KNEE",
            "hip_angle": "HIP",
            "forward_lean": "LEAN",
            "ankle_dorsiflexion": "ANKLE",
            "trunk_alignment": "BALANCE",
        }
        return labels.get(mechanic_name, mechanic_name.upper())

    def _draw_hud(self, frame: np.ndarray, results: list[MechanicResult]):
        """Draw a compact HUD panel with mechanic results.

        Shows: colored dot + name + value + rating.
        Numbers stay here in the HUD, NOT floating on joints.
        """
        if not results:
            return

        h, w = frame.shape[:2]

        line_height = 22
        panel_width = 280
        panel_height = 28 + len(results) * line_height
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
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Title
        cv2.putText(
            frame, "MECHANICS",
            (panel_x + 10, panel_y + 18),
            self.font, 0.45, (200, 200, 200), 1,
        )

        for i, result in enumerate(results):
            y = panel_y + 36 + i * line_height
            color = COLORS.get(result.rating, (255, 255, 255))

            # Colored dot
            cv2.circle(frame, (panel_x + 12, y - 4), 4, color, -1)

            # Name + value
            text = f"{result.display_name}: {result.value:.0f}"
            cv2.putText(
                frame, text, (panel_x + 24, y),
                self.font, 0.38, (220, 220, 220), 1,
            )

            # Rating label
            cv2.putText(
                frame, result.rating.upper(),
                (panel_x + panel_width - 55, y),
                self.font, 0.35, color, 1,
            )
