"""Video frame annotation for skating analysis.

Design principles:
  - The skater should be clearly visible — skeleton/keypoints are subtle
  - Only highlight what's WRONG — good mechanics get minimal annotation
  - During crossover events, show crossover-specific feedback ONLY (no generic angles)
  - Numbers stay in the HUD panel, never floating on joints
"""

import cv2
import numpy as np
from typing import Optional
from src.pose_estimator import SKELETON_CONNECTIONS, LANDMARK_NAMES
from src.mechanics_engine import MechanicResult


# Color scheme (BGR)
COLORS = {
    "good": (0, 200, 0),
    "warning": (0, 200, 220),
    "poor": (0, 0, 240),
    "skeleton": (120, 120, 120),     # Subtle gray
    "skeleton_good": (100, 140, 100),  # Muted green-gray
    "keypoint": (160, 140, 100),     # Muted blue-gray
    "left": (180, 160, 80),          # Subtle blue
    "right": (80, 160, 180),         # Subtle orange
    "text_bg": (30, 30, 30),
    "error_ring": (0, 0, 255),
    "error_glow": (0, 50, 200),
}

# Map mechanic names to joint landmarks
MECHANIC_JOINT_MAP = {
    "knee_angle": {"left": "left_knee", "right": "right_knee"},
    "hip_angle": {"left": "left_hip", "right": "right_hip"},
    "forward_lean": {"left": "left_shoulder", "right": "right_shoulder"},
    "ankle_dorsiflexion": {"left": "left_ankle", "right": "right_ankle"},
}

# Skeleton segments associated with each mechanic
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
    """Draws skating analysis overlays on video frames.

    Keeps annotations minimal unless there's an actual problem to show.
    """

    def __init__(
        self,
        show_skeleton: bool = True,
        show_angles: bool = True,
        show_hud: bool = True,
        technique_engine=None,
    ):
        self.show_skeleton = show_skeleton
        self.show_angles = show_angles
        self.show_hud = show_hud
        self.technique_engine = technique_engine
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self._frame_counter = 0

    def render(
        self,
        frame: np.ndarray,
        landmarks: Optional[dict],
        mechanic_results: Optional[list[MechanicResult]] = None,
        crossover_events: Optional[list] = None,
    ) -> np.ndarray:
        annotated = frame.copy()
        self._frame_counter += 1

        if landmarks is None:
            cv2.putText(
                annotated, "No skater detected",
                (20, 40), self.font, 0.7, (0, 0, 200), 1, cv2.LINE_AA,
            )
            return annotated

        # Crossover mode (crossover_events is a list, possibly empty)
        if crossover_events is not None:
            if self.show_skeleton:
                self._draw_skeleton_minimal(annotated, landmarks)
            if crossover_events:  # Non-empty = active crossover
                self._draw_crossover_feedback(annotated, landmarks, crossover_events)
                if self.show_hud:
                    self._draw_crossover_hud(annotated, crossover_events)
            # Empty list = crossover mode but between events — just minimal skeleton
            return annotated

        # General mode: subtle skeleton + only highlight problems
        if self.show_skeleton:
            self._draw_skeleton_rated(annotated, landmarks, mechanic_results)

        if mechanic_results:
            self._draw_issue_highlights(annotated, landmarks, mechanic_results)

        if self.show_hud and mechanic_results:
            self._draw_hud(annotated, mechanic_results)

        return annotated

    # ------------------------------------------------------------------
    # Skeleton drawing
    # ------------------------------------------------------------------

    def _draw_skeleton_minimal(self, frame: np.ndarray, landmarks: dict):
        """Draw a very subtle skeleton — just enough to see the pose."""
        for lm_a_name, lm_b_name in SKELETON_CONNECTIONS:
            if lm_a_name not in landmarks or lm_b_name not in landmarks:
                continue
            lm_a, lm_b = landmarks[lm_a_name], landmarks[lm_b_name]
            if lm_a["visibility"] < 0.5 or lm_b["visibility"] < 0.5:
                continue
            pt_a = (int(lm_a["x"]), int(lm_a["y"]))
            pt_b = (int(lm_b["x"]), int(lm_b["y"]))
            cv2.line(frame, pt_a, pt_b, COLORS["skeleton"], 1, cv2.LINE_AA)

    def _draw_skeleton_rated(
        self, frame: np.ndarray, landmarks: dict,
        mechanic_results: Optional[list[MechanicResult]],
    ):
        """Draw skeleton that only stands out where there are issues.

        Good mechanics = very subtle lines. Problems = thicker, colored.
        """
        # Build segment ratings
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

        for lm_a_name, lm_b_name in SKELETON_CONNECTIONS:
            if lm_a_name not in landmarks or lm_b_name not in landmarks:
                continue
            lm_a, lm_b = landmarks[lm_a_name], landmarks[lm_b_name]
            if lm_a["visibility"] < 0.5 or lm_b["visibility"] < 0.5:
                continue
            pt_a = (int(lm_a["x"]), int(lm_a["y"]))
            pt_b = (int(lm_b["x"]), int(lm_b["y"]))

            seg_key = tuple(sorted((lm_a_name, lm_b_name)))
            seg_rating = segment_ratings.get(seg_key)

            if seg_rating == "poor":
                cv2.line(frame, pt_a, pt_b, COLORS["poor"], 3, cv2.LINE_AA)
            elif seg_rating == "warning":
                cv2.line(frame, pt_a, pt_b, COLORS["warning"], 2, cv2.LINE_AA)
            else:
                # Subtle — barely visible
                cv2.line(frame, pt_a, pt_b, COLORS["skeleton"], 1, cv2.LINE_AA)

        # Keypoints: only draw on problem joints, skip the rest
        joint_ratings = self._build_joint_ratings(mechanic_results)
        body_start = LANDMARK_NAMES.index("left_shoulder")
        for name in LANDMARK_NAMES[body_start:]:
            if name not in landmarks:
                continue
            lm = landmarks[name]
            if lm["visibility"] < 0.5:
                continue
            rating = joint_ratings.get(name)
            pt = (int(lm["x"]), int(lm["y"]))

            if rating == "poor":
                cv2.circle(frame, pt, 5, COLORS["poor"], -1, cv2.LINE_AA)
            elif rating == "warning":
                cv2.circle(frame, pt, 4, COLORS["warning"], -1, cv2.LINE_AA)
            else:
                # Tiny dot — just a reference point
                cv2.circle(frame, pt, 2, COLORS["keypoint"], -1, cv2.LINE_AA)

    def _build_joint_ratings(
        self, results: Optional[list[MechanicResult]]
    ) -> dict[str, str]:
        ratings = {}
        if not results:
            return ratings
        for r in results:
            joint_map = MECHANIC_JOINT_MAP.get(r.name)
            if joint_map and r.side and r.side in joint_map:
                lm_name = joint_map[r.side]
                current = ratings.get(lm_name, "good")
                if r.rating == "poor" or (r.rating == "warning" and current == "good"):
                    ratings[lm_name] = r.rating
        return ratings

    # ------------------------------------------------------------------
    # Issue highlights (non-crossover frames)
    # ------------------------------------------------------------------

    def _draw_issue_highlights(
        self, frame: np.ndarray, landmarks: dict,
        results: list[MechanicResult],
    ):
        """Only highlight joints with warning or poor ratings.

        Good joints get NO extra annotation — the viewer doesn't need
        to know things are fine. Only problems get attention.
        """
        for result in results:
            if result.rating == "good":
                continue

            joint_map = MECHANIC_JOINT_MAP.get(result.name)
            if not joint_map or not result.side or result.side not in joint_map:
                continue

            lm_name = joint_map[result.side]
            if lm_name not in landmarks or landmarks[lm_name]["visibility"] < 0.5:
                continue

            pt = (int(landmarks[lm_name]["x"]), int(landmarks[lm_name]["y"]))

            if result.rating == "poor":
                pulse = int(6 + 4 * abs(np.sin(self._frame_counter * 0.15)))
                cv2.circle(frame, pt, pulse + 12, COLORS["error_ring"], 2, cv2.LINE_AA)

                label = self._short_label(result.name)
                lpt = (pt[0] + 18, pt[1] - 5)
                tsz = cv2.getTextSize(label, self.font, 0.4, 1)[0]
                cv2.rectangle(frame, (lpt[0]-2, lpt[1]-tsz[1]-2),
                              (lpt[0]+tsz[0]+2, lpt[1]+4), COLORS["poor"], -1)
                cv2.putText(frame, label, lpt, self.font, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)

            elif result.rating == "warning":
                cv2.circle(frame, pt, 12, COLORS["warning"], 1, cv2.LINE_AA)

    def _short_label(self, mechanic_name: str) -> str:
        labels = {
            "knee_angle": "KNEE",
            "hip_angle": "HIP",
            "forward_lean": "LEAN",
            "ankle_dorsiflexion": "ANKLE",
            "trunk_alignment": "BALANCE",
        }
        return labels.get(mechanic_name, mechanic_name.upper())

    # ------------------------------------------------------------------
    # HUD panel
    # ------------------------------------------------------------------

    def _draw_hud(self, frame: np.ndarray, results: list[MechanicResult]):
        """Compact HUD — only shows metrics that aren't 'good'."""
        # Filter to only show issues
        issues = [r for r in results if r.rating != "good"]
        if not issues:
            # Everything good — show a small green indicator
            cv2.putText(frame, "OK", (frame.shape[1] - 40, 25),
                        self.font, 0.5, COLORS["good"], 1, cv2.LINE_AA)
            return

        h, w = frame.shape[:2]
        line_height = 20
        panel_width = 240
        panel_height = 24 + len(issues) * line_height
        panel_x = w - panel_width - 8
        panel_y = 8

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      COLORS["text_bg"], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        for i, result in enumerate(issues):
            y = panel_y + 18 + i * line_height
            color = COLORS.get(result.rating, (200, 200, 200))
            cv2.circle(frame, (panel_x + 10, y - 3), 3, color, -1)
            text = f"{result.display_name}: {result.value:.0f}"
            cv2.putText(frame, text, (panel_x + 20, y),
                        self.font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, result.rating.upper(),
                        (panel_x + panel_width - 55, y),
                        self.font, 0.3, color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Crossover-specific annotations
    # ------------------------------------------------------------------

    def _draw_crossover_feedback(
        self, frame: np.ndarray, landmarks: dict, events: list,
    ):
        """Draw crossover-specific feedback — replaces all generic annotations."""
        h, w = frame.shape[:2]

        for event in events:
            cl = event.crossing_leg
            knee_name = f"{cl}_knee"
            ankle_name = f"{cl}_ankle"

            # Knee drive feedback — the main coaching point
            if knee_name in landmarks and landmarks[knee_name]["visibility"] > 0.4:
                knee_pt = (int(landmarks[knee_name]["x"]),
                           int(landmarks[knee_name]["y"]))

                if event.knee_drive_rating == "poor":
                    pulse = int(8 + 5 * abs(np.sin(self._frame_counter * 0.2)))
                    cv2.circle(frame, knee_pt, pulse + 14, COLORS["error_ring"], 2, cv2.LINE_AA)
                    label = "DRIVE KNEE!"
                    lpt = (knee_pt[0] + 22, knee_pt[1] - 8)
                    tsz = cv2.getTextSize(label, self.font, 0.45, 1)[0]
                    cv2.rectangle(frame, (lpt[0]-2, lpt[1]-tsz[1]-3),
                                  (lpt[0]+tsz[0]+2, lpt[1]+4), COLORS["poor"], -1)
                    cv2.putText(frame, label, lpt, self.font, 0.45,
                                (255, 255, 255), 1, cv2.LINE_AA)

                elif event.knee_drive_rating == "warning":
                    cv2.circle(frame, knee_pt, 16, COLORS["warning"], 2, cv2.LINE_AA)
                    label = "MORE KNEE DRIVE"
                    lpt = (knee_pt[0] + 20, knee_pt[1] - 6)
                    tsz = cv2.getTextSize(label, self.font, 0.35, 1)[0]
                    cv2.rectangle(frame, (lpt[0]-2, lpt[1]-tsz[1]-2),
                                  (lpt[0]+tsz[0]+2, lpt[1]+3),
                                  COLORS["text_bg"], -1)
                    cv2.putText(frame, label, lpt, self.font, 0.35,
                                COLORS["warning"], 1, cv2.LINE_AA)

                elif event.knee_drive_rating == "good":
                    cv2.circle(frame, knee_pt, 14, COLORS["good"], 1, cv2.LINE_AA)

            # Rotation feedback on ankle
            if (ankle_name in landmarks
                    and landmarks[ankle_name]["visibility"] > 0.4
                    and event.rotation_rating == "poor"):
                apt = (int(landmarks[ankle_name]["x"]),
                       int(landmarks[ankle_name]["y"]))
                cv2.circle(frame, apt, 12, COLORS["poor"], 2, cv2.LINE_AA)
                cv2.putText(frame, "ROTATE", (apt[0] - 22, apt[1] + 20),
                            self.font, 0.3, COLORS["poor"], 1, cv2.LINE_AA)

        # Top banner
        if events:
            evt = events[0]
            banner_color = COLORS.get(evt.overall_rating, COLORS["skeleton"])
            banner_text = f"CROSSOVER: {evt.crossing_leg.upper()} over {evt.stance_leg.upper()}"
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 28), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, banner_text, (10, 20),
                        self.font, 0.5, banner_color, 1, cv2.LINE_AA)

    def _draw_crossover_hud(self, frame: np.ndarray, events: list):
        """Crossover-specific HUD — only shows crossover metrics."""
        if not events:
            return

        evt = events[0]
        h, w = frame.shape[:2]

        metrics = [
            ("Knee Drive", evt.knee_drive_rating),
            ("Rotation", evt.rotation_rating),
            ("Step-Out", evt.step_out_rating),
        ]
        # Only show non-good metrics
        issues = [(n, r) for n, r in metrics if r not in ("good", "unknown")]

        if not issues:
            cv2.putText(frame, "GOOD", (w - 55, 22),
                        self.font, 0.45, COLORS["good"], 1, cv2.LINE_AA)
            return

        line_height = 20
        panel_width = 180
        panel_height = 22 + len(issues) * line_height
        panel_x = w - panel_width - 8
        panel_y = 32  # Below banner

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      COLORS["text_bg"], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        for i, (name, rating) in enumerate(issues):
            y = panel_y + 16 + i * line_height
            color = COLORS.get(rating, (150, 150, 150))
            cv2.circle(frame, (panel_x + 10, y - 3), 3, color, -1)
            cv2.putText(frame, name, (panel_x + 20, y),
                        self.font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, rating.upper(),
                        (panel_x + panel_width - 60, y),
                        self.font, 0.3, color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Technique-based rendering (new knowledge base system)
    # ------------------------------------------------------------------

    def render_technique(
        self,
        frame: np.ndarray,
        landmarks: Optional[dict],
        event=None,
        events=None,
    ) -> np.ndarray:
        """Render using technique engine's config-driven annotations.

        For frame-by-frame techniques: pass event (single TechniqueEvent or None).
        For temporal techniques: pass events (list of TechniqueEvent or None).
        """
        annotated = frame.copy()
        self._frame_counter += 1

        if landmarks is None:
            cv2.putText(
                annotated, "No skater detected",
                (20, 40), self.font, 0.7, (0, 0, 200), 1, cv2.LINE_AA,
            )
            return annotated

        # Temporal technique (crossover, shot, etc.)
        if events is not None:
            if self.show_skeleton:
                self._draw_skeleton_minimal(annotated, landmarks)
            if events:
                self._draw_technique_events(annotated, landmarks, events)
                if self.show_hud:
                    self._draw_technique_hud(annotated, events)
            return annotated

        # Frame-by-frame technique
        if event is not None:
            self._draw_technique_frame(annotated, landmarks, event)
        else:
            if self.show_skeleton:
                self._draw_skeleton_minimal(annotated, landmarks)

        return annotated

    def _draw_technique_events(self, frame: np.ndarray, landmarks: dict, events: list):
        """Draw annotations for temporal technique events using YAML config."""
        h, w = frame.shape[:2]

        for event in events:
            context = event.context

            for result in event.check_results:
                if result.rating == "good":
                    continue

                # Get annotation config from technique engine
                ann_config = None
                if self.technique_engine:
                    ann_config = self.technique_engine.get_check_annotation(result.check_name)

                if ann_config and result.rating in ann_config:
                    rating_config = ann_config[result.rating]
                    # Resolve landmark template
                    target_template = ann_config.get("target_landmark", "")
                    try:
                        target_lm = target_template.format(**context)
                    except (KeyError, IndexError):
                        continue

                    if target_lm not in landmarks or landmarks[target_lm]["visibility"] < 0.4:
                        continue

                    pt = (int(landmarks[target_lm]["x"]), int(landmarks[target_lm]["y"]))
                    color = COLORS.get(result.rating, COLORS["skeleton"])
                    style = rating_config.get("style", "ring")
                    label = rating_config.get("label", "")

                    if style == "pulsing_ring":
                        pulse = int(8 + 5 * abs(np.sin(self._frame_counter * 0.2)))
                        cv2.circle(frame, pt, pulse + 14, color, 2, cv2.LINE_AA)
                    else:
                        cv2.circle(frame, pt, 14, color, 2, cv2.LINE_AA)

                    if label:
                        lpt = (pt[0] + 20, pt[1] - 6)
                        tsz = cv2.getTextSize(label, self.font, 0.4, 1)[0]
                        cv2.rectangle(frame, (lpt[0]-2, lpt[1]-tsz[1]-3),
                                      (lpt[0]+tsz[0]+2, lpt[1]+4),
                                      COLORS["text_bg"], -1)
                        cv2.putText(frame, label, lpt, self.font, 0.4,
                                    color, 1, cv2.LINE_AA)

            # Banner
            if self.technique_engine:
                banner_template = self.technique_engine.get_banner_template()
                if banner_template:
                    try:
                        banner_text = banner_template.format(**context).upper()
                    except (KeyError, IndexError):
                        banner_text = event.overall_rating.upper()

                    banner_color = COLORS.get(event.overall_rating, COLORS["skeleton"])
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, 28), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    cv2.putText(frame, banner_text, (10, 20),
                                self.font, 0.5, banner_color, 1, cv2.LINE_AA)

    def _draw_technique_hud(self, frame: np.ndarray, events: list):
        """HUD for technique events — only shows non-good checks."""
        if not events:
            return

        evt = events[0]
        h, w = frame.shape[:2]

        issues = [(r.display_name, r.rating) for r in evt.check_results
                  if r.rating not in ("good", "unknown")]

        if not issues:
            cv2.putText(frame, "GOOD", (w - 55, 22),
                        self.font, 0.45, COLORS["good"], 1, cv2.LINE_AA)
            return

        line_height = 20
        panel_width = 180
        panel_height = 22 + len(issues) * line_height
        panel_x = w - panel_width - 8
        panel_y = 32

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      COLORS["text_bg"], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        for i, (name, rating) in enumerate(issues):
            y = panel_y + 16 + i * line_height
            color = COLORS.get(rating, (150, 150, 150))
            cv2.circle(frame, (panel_x + 10, y - 3), 3, color, -1)
            cv2.putText(frame, name, (panel_x + 20, y),
                        self.font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, rating.upper(),
                        (panel_x + panel_width - 60, y),
                        self.font, 0.3, color, 1, cv2.LINE_AA)

    def _draw_technique_frame(self, frame: np.ndarray, landmarks: dict, event):
        """Draw annotations for a frame-by-frame technique event."""
        # Build ratings from check results
        from src.mechanics_engine import MechanicResult

        # Convert TechniqueEvent check_results to MechanicResult-compatible format
        # for reuse of existing rendering functions
        mechanic_results = []
        for r in event.check_results:
            # Extract side from check_name (e.g., "knee_angle_left" -> "left")
            side = None
            base_name = r.check_name
            for s in ("_left", "_right"):
                if r.check_name.endswith(s):
                    side = s[1:]  # Remove leading underscore
                    base_name = r.check_name[:-len(s)]
                    break

            mechanic_results.append(MechanicResult(
                name=base_name,
                display_name=r.display_name,
                value=r.metric_value if r.metric_value is not None else 0.0,
                rating=r.rating,
                feedback=r.feedback,
                side=side,
            ))

        # Use existing rated skeleton + issue highlight rendering
        if self.show_skeleton:
            self._draw_skeleton_rated(frame, landmarks, mechanic_results)

        if mechanic_results:
            self._draw_issue_highlights(frame, landmarks, mechanic_results)

        if self.show_hud and mechanic_results:
            self._draw_hud(frame, mechanic_results)
