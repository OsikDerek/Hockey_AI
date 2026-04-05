"""Game analysis video annotator.

Renders bounding boxes, track IDs, zone labels, and game events
on video frames for game analysis mode.
"""

import cv2
import numpy as np
from typing import Optional

from .game_context import FrameContext, GameEvent


# Colors by class (BGR)
CLASS_COLORS = {
    "player": (255, 180, 50),     # Light blue
    "goalie": (0, 255, 255),      # Yellow
    "puck": (0, 255, 0),          # Green
    "referee": (128, 128, 128),   # Gray
    "centroid": (200, 200, 200),  # Light gray
    "faceoff": (200, 200, 200),
    "goal": (200, 200, 200),
}

ZONE_COLORS = {
    "offensive": (0, 100, 255),   # Orange
    "neutral": (200, 200, 200),   # Gray
    "defensive": (255, 100, 100), # Blue
}


class GameAnnotator:
    """Render game analysis overlays on video frames.

    Draws:
    - Bounding boxes around tracked objects with class labels and IDs
    - Zone label banner at top
    - Possession highlight
    - Camera cut indicator
    - Game event overlays (Phase 2+)
    """

    def __init__(
        self,
        show_boxes: bool = True,
        show_zone: bool = True,
        show_ids: bool = True,
        show_rink_landmarks: bool = False,
        box_thickness: int = 2,
    ):
        self.show_boxes = show_boxes
        self.show_zone = show_zone
        self.show_ids = show_ids
        self.show_rink_landmarks = show_rink_landmarks
        self.box_thickness = box_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(
        self,
        frame: np.ndarray,
        context: FrameContext,
        events: Optional[list] = None,
    ) -> np.ndarray:
        """Render all game analysis overlays on a frame.

        Args:
            frame: BGR video frame.
            context: FrameContext with tracked objects and game state.
            events: List of active GameEvents at this frame (or None).

        Returns:
            Annotated frame.
        """
        annotated = frame.copy()

        if not context.is_gameplay:
            self._draw_non_gameplay(annotated)
            return annotated

        if context.is_camera_cut:
            self._draw_camera_cut(annotated)

        # Draw bounding boxes
        if self.show_boxes:
            self._draw_objects(annotated, context)

        # Draw zone banner
        if self.show_zone and context.zone:
            self._draw_zone_banner(annotated, context.zone)

        # Draw possession highlight
        if context.possession_player_id is not None:
            self._draw_possession(annotated, context)

        # Draw active events (Phase 2+)
        if events:
            self._draw_events(annotated, events)

        # Frame info
        self._draw_frame_info(annotated, context)

        return annotated

    def _draw_objects(self, frame: np.ndarray, context: FrameContext):
        """Draw bounding boxes with class labels and track IDs."""
        for obj in context.objects:
            # Skip rink landmarks unless enabled
            if obj.class_name in ("centroid", "faceoff", "goal"):
                if not self.show_rink_landmarks:
                    continue

            color = CLASS_COLORS.get(obj.class_name, (200, 200, 200))
            x1, y1, x2, y2 = [int(v) for v in obj.bbox]

            # Bounding box
            thickness = self.box_thickness
            if obj.class_name == "puck":
                thickness = 3  # Puck gets thicker box

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label
            if self.show_ids and obj.track_id >= 0:
                label = f"{obj.class_name}#{obj.track_id}"
            else:
                label = obj.class_name

            # Add confidence for puck
            if obj.class_name == "puck":
                label += f" {obj.confidence:.0%}"

            label_size = cv2.getTextSize(label, self.font, 0.45, 1)[0]
            label_y = max(y1 - 6, label_size[1] + 4)

            # Label background
            cv2.rectangle(
                frame,
                (x1, label_y - label_size[1] - 4),
                (x1 + label_size[0] + 4, label_y + 2),
                color, -1,
            )
            cv2.putText(
                frame, label,
                (x1 + 2, label_y - 2),
                self.font, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
            )

    def _draw_zone_banner(self, frame: np.ndarray, zone: str):
        """Draw zone label banner at top of frame."""
        h, w = frame.shape[:2]
        color = ZONE_COLORS.get(zone, (200, 200, 200))

        # Semi-transparent banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), color, -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Zone text
        zone_text = f"{zone.upper()} ZONE"
        text_size = cv2.getTextSize(zone_text, self.font, 0.7, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(
            frame, zone_text,
            (text_x, 24),
            self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

    def _draw_possession(self, frame: np.ndarray, context: FrameContext):
        """Highlight the player with possession."""
        for obj in context.players:
            if obj.track_id == context.possession_player_id:
                x1, y1, x2, y2 = [int(v) for v in obj.bbox]
                # Thick highlight box
                cv2.rectangle(
                    frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                    (0, 255, 0), 3,
                )
                # "PUCK" label
                cv2.putText(
                    frame, "PUCK",
                    (x1, y2 + 18),
                    self.font, 0.5, (0, 255, 0), 2, cv2.LINE_AA,
                )
                break

    def _draw_camera_cut(self, frame: np.ndarray):
        """Flash indicator for camera cut."""
        h, w = frame.shape[:2]
        cv2.putText(
            frame, "CAMERA CUT",
            (w // 2 - 80, h - 20),
            self.font, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )

    def _draw_non_gameplay(self, frame: np.ndarray):
        """Overlay for non-gameplay frames."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(
            frame, "NON-GAMEPLAY",
            (w // 2 - 100, h // 2),
            self.font, 1.0, (100, 100, 100), 2, cv2.LINE_AA,
        )

    def _draw_events(self, frame: np.ndarray, events: list):
        """Draw active game event banners."""
        h, w = frame.shape[:2]
        for i, event in enumerate(events):
            y_offset = 40 + i * 30

            # Event banner
            text = f"{event.event_type.upper().replace('_', ' ')}"
            if event.decision_made:
                text += f": {event.decision_made}"

            # Banner background
            text_size = cv2.getTextSize(text, self.font, 0.6, 2)[0]
            cv2.rectangle(
                frame,
                (10, y_offset - 18),
                (20 + text_size[0], y_offset + 6),
                (0, 80, 200), -1,
            )
            cv2.putText(
                frame, text,
                (15, y_offset),
                self.font, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )

    def _draw_frame_info(self, frame: np.ndarray, context: FrameContext):
        """Draw frame counter and object count."""
        h, w = frame.shape[:2]
        info = (
            f"F:{context.frame_idx} "
            f"P:{len(context.players)} "
            f"G:{len(context.goalies)} "
            f"{'PUCK' if context.puck else 'no puck'}"
        )
        cv2.putText(
            frame, info,
            (10, h - 10),
            self.font, 0.4, (180, 180, 180), 1, cv2.LINE_AA,
        )
