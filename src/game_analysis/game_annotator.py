"""Game analysis video annotator with coaching overlays.

Renders bounding boxes, team colors, zone labels, passing/shooting lanes,
decision freeze frames, and success probabilities on video frames.
"""

import cv2
import numpy as np
from typing import Optional

from .game_context import FrameContext, GameEvent
from .lane_calculator import LaneCalculator
from .space_detector import SpaceDetector


# Colors by class (BGR)
CLASS_COLORS = {
    "player": (255, 180, 50),     # Light blue
    "goalie": (0, 255, 255),      # Yellow
    "puck": (0, 255, 0),          # Green
    "referee": (128, 128, 128),   # Gray
    "centroid": (200, 200, 200),
    "faceoff": (200, 200, 200),
    "goal": (200, 200, 200),
}

TEAM_COLORS = {
    "team_a": (255, 120, 50),     # Blue
    "team_b": (60, 60, 255),      # Red
}

ZONE_COLORS = {
    "offensive": (0, 100, 255),   # Orange
    "neutral": (200, 200, 200),   # Gray
    "defensive": (255, 100, 100), # Blue
}

# Lane quality colors (BGR)
LANE_COLORS = {
    "clear": (0, 200, 0),        # Green
    "open": (0, 200, 0),         # Green
    "partial": (0, 200, 200),    # Yellow
    "blocked": (0, 0, 200),      # Red
    "covered": (0, 0, 200),      # Red
}

# Ambient connection colors (subtle)
AMBIENT_COLORS = {
    "open": (0, 120, 0),         # Dark green
    "partial": (0, 120, 120),    # Dark yellow
    "covered": (0, 0, 120),      # Dark red
}


class GameAnnotator:
    """Render game analysis overlays on video frames.

    Standard mode: bounding boxes, zones, possession, event banners.
    Coaching mode: adds passing/shooting lanes, probabilities, freeze frames.
    Ambient mode: subtle persistent connections from carrier to teammates.
    """

    def __init__(
        self,
        show_boxes: bool = True,
        show_zone: bool = True,
        show_ids: bool = True,
        show_rink_landmarks: bool = False,
        box_thickness: int = 2,
        coaching_mode: bool = True,
    ):
        self.show_boxes = show_boxes
        self.show_zone = show_zone
        self.show_ids = show_ids
        self.show_rink_landmarks = show_rink_landmarks
        self.box_thickness = box_thickness
        self.coaching_mode = coaching_mode
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self._team_assignments = {}
        self.lane_calculator = LaneCalculator()
        self.space_detector = SpaceDetector()

    def set_team_assignments(self, assignments: dict):
        """Update the current team assignment map."""
        self._team_assignments = assignments

    def render(
        self,
        frame: np.ndarray,
        context: FrameContext,
        events: Optional[list] = None,
        ambient_connections: Optional[list] = None,
        open_spaces: Optional[list] = None,
    ) -> np.ndarray:
        """Render standard overlays (no coaching lanes).

        Args:
            frame: BGR video frame.
            context: FrameContext with tracked objects.
            events: Active GameEvents at this frame.
            ambient_connections: Subtle teammate connections (always-on).
            open_spaces: List of open space dicts from SpaceDetector.
        """
        annotated = frame.copy()

        if not context.is_gameplay:
            self._draw_non_gameplay(annotated)
            return annotated

        if context.is_camera_cut:
            self._draw_camera_cut(annotated)

        # Open space overlay (draw first so it's behind everything)
        if open_spaces:
            self._draw_open_spaces(annotated, open_spaces)

        if self.show_boxes:
            self._draw_objects(annotated, context)

        if self.show_zone and context.zone:
            self._draw_zone_banner(annotated, context.zone)

        if context.possession_player_id is not None:
            self._draw_possession(annotated, context)

        # Always-on ambient connections from carrier to teammates
        if ambient_connections:
            self._draw_ambient_connections(annotated, ambient_connections)

        if events:
            self._draw_events(annotated, events)

        self._draw_frame_info(annotated, context)
        return annotated

    def render_coaching(
        self,
        frame: np.ndarray,
        context: FrameContext,
        events: Optional[list],
        phase_info: dict,
        lane_data: Optional[dict],
    ) -> np.ndarray:
        """Render coaching overlays during decision events.

        Args:
            frame: BGR video frame.
            context: FrameContext.
            events: Active GameEvents.
            phase_info: {"phase": str, "event": GameEvent, "alpha": float}
            lane_data: {"shooting": dict, "passing": list} from LaneCalculator.
        """
        # Start with standard rendering
        annotated = self.render(frame, context, events)

        if lane_data is None:
            return annotated

        phase = phase_info.get("phase", "approach")
        alpha = phase_info.get("alpha", 0.5)
        event = phase_info.get("event")

        # Draw lanes on an overlay for transparency
        overlay = annotated.copy()

        # Passing lanes
        if lane_data.get("passing"):
            self._draw_passing_lanes(overlay, lane_data["passing"], alpha, phase)

        # Shooting lane
        if lane_data.get("shooting"):
            self._draw_shooting_lane(overlay, lane_data["shooting"], alpha, phase)

        # Blend overlay
        cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)

        # Freeze frame special treatment
        if phase == "freeze" and event:
            self._draw_decision_freeze(annotated, event, lane_data)

        # Slowdown indicator
        if phase == "slowdown":
            self._draw_slowdown_indicator(annotated)

        return annotated

    # ── Standard Drawing Methods ──────────────────────────

    def _draw_objects(self, frame, context):
        """Draw bounding boxes with team colors."""
        for obj in context.objects:
            if obj.class_name in ("centroid", "faceoff", "goal"):
                if not self.show_rink_landmarks:
                    continue

            team = self._team_assignments.get(obj.track_id)
            if obj.class_name == "player" and team is not None:
                color = TEAM_COLORS.get(team, (200, 200, 200))
            else:
                color = CLASS_COLORS.get(obj.class_name, (200, 200, 200))

            x1, y1, x2, y2 = [int(v) for v in obj.bbox]
            thickness = 3 if obj.class_name == "puck" else self.box_thickness
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            if self.show_ids and obj.track_id >= 0:
                label = f"#{obj.track_id}"
                if team:
                    label = f"{team[-1].upper()}{label}"  # "A#5" or "B#12"
            else:
                label = obj.class_name

            if obj.class_name == "puck":
                label = f"PUCK {obj.confidence:.0%}"

            sz = cv2.getTextSize(label, self.font, 0.4, 1)[0]
            ly = max(y1 - 4, sz[1] + 2)
            cv2.rectangle(frame, (x1, ly - sz[1] - 2), (x1 + sz[0] + 4, ly + 2), color, -1)
            cv2.putText(frame, label, (x1 + 2, ly), self.font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_zone_banner(self, frame, zone):
        h, w = frame.shape[:2]
        color = ZONE_COLORS.get(zone, (200, 200, 200))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), color, -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        text = f"{zone.upper()} ZONE"
        sz = cv2.getTextSize(text, self.font, 0.7, 2)[0]
        cv2.putText(frame, text, ((w - sz[0]) // 2, 24), self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_possession(self, frame, context):
        for obj in context.players:
            if obj.track_id == context.possession_player_id:
                x1, y1, x2, y2 = [int(v) for v in obj.bbox]
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 0), 3)
                cv2.putText(frame, "PUCK", (x1, y2 + 18), self.font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                break

    def _draw_camera_cut(self, frame):
        h, w = frame.shape[:2]
        cv2.putText(frame, "CAMERA CUT", (w // 2 - 80, h - 20), self.font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    def _draw_non_gameplay(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "NON-GAMEPLAY", (w // 2 - 100, h // 2), self.font, 1.0, (100, 100, 100), 2, cv2.LINE_AA)

    def _draw_events(self, frame, events):
        h, w = frame.shape[:2]
        for i, event in enumerate(events):
            y = 40 + i * 30
            text = f"{event.event_type.upper().replace('_', ' ')}"
            if event.decision_made:
                text += f": {event.decision_made}"
            if event.evaluation:
                rating = event.evaluation.get("rating", "")
                if rating:
                    text += f" [{rating.upper()}]"

            sz = cv2.getTextSize(text, self.font, 0.55, 2)[0]
            cv2.rectangle(frame, (10, y - 18), (20 + sz[0], y + 6), (0, 80, 200), -1)
            cv2.putText(frame, text, (15, y), self.font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_frame_info(self, frame, context):
        h, w = frame.shape[:2]
        info = f"F:{context.frame_idx} P:{len(context.players)} G:{len(context.goalies)} {'PUCK' if context.puck else 'no puck'}"
        cv2.putText(frame, info, (10, h - 10), self.font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Ambient Connections (always-on) ───────────────────

    def _draw_ambient_connections(self, frame, connections):
        """Draw subtle persistent lines from carrier to teammates."""
        overlay = frame.copy()
        for conn in connections:
            color = AMBIENT_COLORS.get(conn["quality"], (80, 80, 80))
            pt1 = tuple(int(v) for v in conn["start"])
            pt2 = tuple(int(v) for v in conn["end"])
            # Thin dashed line
            self._draw_dashed_line(overlay, pt1, pt2, color, thickness=1, dash_length=12)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # ── Open Space Visualization ───────────────────────────

    def _draw_open_spaces(self, frame, spaces):
        """Draw open space regions as subtle colored overlays."""
        overlay = frame.copy()

        for space in spaces:
            value = space["value"]
            adjacent = space["adjacent_to_carrier"]
            dangerous = space["in_dangerous_zone"]

            # Color by value
            if value == "high":
                color = (0, 255, 100)     # Bright green
                alpha = 0.20
            elif value == "medium":
                color = (0, 200, 200)     # Yellow-green
                alpha = 0.12
            else:
                color = (180, 180, 180)   # Light gray
                alpha = 0.06

            # Brighter if adjacent to carrier
            if adjacent:
                alpha = min(alpha + 0.10, 0.30)

            # Draw filled contour
            cv2.drawContours(overlay, [space["contour"]], -1, color, -1)

            # Border for high-value spaces
            if value in ("high", "medium"):
                cv2.drawContours(frame, [space["contour"]], -1, color, 2)

            # Label for high-value spaces
            if value == "high" and space.get("reasons"):
                cx, cy = space["center"]
                label = space["reasons"][0] if space["reasons"] else "Open"
                if adjacent:
                    label = "OPEN ICE"
                elif dangerous:
                    label = "OPEN SLOT"

                sz = cv2.getTextSize(label, self.font, 0.45, 1)[0]
                # Background pill
                cv2.rectangle(
                    frame,
                    (cx - sz[0] // 2 - 4, cy - sz[1] // 2 - 4),
                    (cx + sz[0] // 2 + 4, cy + sz[1] // 2 + 4),
                    color, -1,
                )
                cv2.putText(
                    frame, label,
                    (cx - sz[0] // 2, cy + sz[1] // 2),
                    self.font, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
                )

        # Blend overlay
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # ── Coaching Overlay Methods ──────────────────────────

    def _draw_passing_lanes(self, frame, lanes, alpha, phase):
        """Draw passing lanes with quality colors and success percentages."""
        for lane in lanes:
            color = LANE_COLORS.get(lane["quality"], (200, 200, 200))
            pt1 = tuple(int(v) for v in lane["start"])
            pt2 = tuple(int(v) for v in lane["end"])

            # Dashed line
            self._draw_dashed_line(frame, pt1, pt2, color, thickness=2, dash_length=15)

            # Success % badge at midpoint
            if phase in ("slowdown", "freeze"):
                mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                pct_text = f"{lane['success_pct']:.0%}"
                self._draw_pct_badge(frame, mid, pct_text, color)

            # Blocker indicators
            for blocker in lane.get("blockers_near", []):
                bp = tuple(int(v) for v in blocker["pos"])
                cv2.circle(frame, bp, 8, (0, 0, 200), -1)
                cv2.circle(frame, bp, 8, (0, 0, 150), 2)

    def _draw_shooting_lane(self, frame, shooting, alpha, phase):
        """Draw shooting lane with arrow and blocker indicators."""
        if shooting is None:
            return

        color = LANE_COLORS.get(shooting["quality"], (200, 200, 200))
        pt1 = tuple(int(v) for v in shooting["start"])
        pt2 = tuple(int(v) for v in shooting["end"])

        # Solid arrowed line
        cv2.arrowedLine(frame, pt1, pt2, color, 3, cv2.LINE_AA, tipLength=0.03)

        # Success % badge near the line midpoint
        if phase in ("slowdown", "freeze"):
            mid = ((pt1[0] * 2 + pt2[0]) // 3, (pt1[1] * 2 + pt2[1]) // 3)
            pct_text = f"SHOT {shooting['success_pct']:.0%}"
            self._draw_pct_badge(frame, mid, pct_text, color, radius=24)

        # Blocker indicators
        for blocker in shooting.get("blockers", []):
            bp = tuple(int(v) for v in blocker["pos"])
            cv2.circle(frame, bp, 12, (0, 0, 220), -1)
            # X mark
            cv2.line(frame, (bp[0] - 5, bp[1] - 5), (bp[0] + 5, bp[1] + 5), (255, 255, 255), 2)
            cv2.line(frame, (bp[0] - 5, bp[1] + 5), (bp[0] + 5, bp[1] - 5), (255, 255, 255), 2)

    def _draw_decision_freeze(self, frame, event, lane_data):
        """Draw the full decision breakdown on a freeze frame."""
        h, w = frame.shape[:2]

        # Vignette (darken edges)
        self._draw_vignette(frame, strength=0.5)

        # "DECISION POINT" header
        text = "DECISION POINT"
        sz = cv2.getTextSize(text, self.font, 1.1, 3)[0]
        tx = (w - sz[0]) // 2
        ty = 65
        # Black outline
        cv2.putText(frame, text, (tx, ty), self.font, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, ty), self.font, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

        # What was chosen
        chose_text = f"CHOSE: {event.decision_made.upper()}"
        if event.player_id:
            chose_text += f" (#{event.player_id})"
        sz2 = cv2.getTextSize(chose_text, self.font, 0.7, 2)[0]
        cy = h - 60
        cv2.rectangle(frame, (10, cy - 22), (20 + sz2[0], cy + 6), (255, 255, 0), -1)
        cv2.putText(frame, chose_text, (15, cy), self.font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # Evaluator suggestion (if different from what was chosen)
        if event.evaluation:
            alt = event.evaluation.get("alternative")
            rating = event.evaluation.get("rating", "")
            if alt:
                sug_text = f"SUGGEST: {alt[:60]}"
                sz3 = cv2.getTextSize(sug_text, self.font, 0.55, 2)[0]
                sy = h - 30
                cv2.rectangle(frame, (10, sy - 18), (20 + sz3[0], sy + 6), (200, 200, 0), -1)
                cv2.putText(frame, sug_text, (15, sy), self.font, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

            # Rating badge
            if rating:
                rating_colors = {"good": (0, 180, 0), "warning": (0, 180, 180), "poor": (0, 0, 200)}
                rc = rating_colors.get(rating, (150, 150, 150))
                cv2.rectangle(frame, (w - 120, 45), (w - 10, 80), rc, -1)
                cv2.putText(frame, rating.upper(), (w - 110, 72), self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_slowdown_indicator(self, frame):
        """Subtle indicator that video is in slow motion."""
        h, w = frame.shape[:2]
        cv2.putText(frame, "SLOW", (w - 70, 24), self.font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Drawing Utilities ─────────────────────────────────

    def _draw_dashed_line(self, frame, pt1, pt2, color, thickness=2, dash_length=15):
        """Draw a dashed line between two points."""
        x1, y1 = pt1
        x2, y2 = pt2
        dx = x2 - x1
        dy = y2 - y1
        length = max(1, int(np.sqrt(dx * dx + dy * dy)))
        num_dashes = max(1, length // dash_length)

        for i in range(0, num_dashes, 2):
            t0 = i / num_dashes
            t1 = min((i + 1) / num_dashes, 1.0)
            start = (int(x1 + dx * t0), int(y1 + dy * t0))
            end = (int(x1 + dx * t1), int(y1 + dy * t1))
            cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)

    def _draw_pct_badge(self, frame, center, text, color, radius=18):
        """Draw a filled circle with percentage text inside."""
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(frame, (cx, cy), radius, color, -1)
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 1)
        sz = cv2.getTextSize(text, self.font, 0.35, 1)[0]
        tx = cx - sz[0] // 2
        ty = cy + sz[1] // 2
        cv2.putText(frame, text, (tx, ty), self.font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_vignette(self, frame, strength=0.5):
        """Darken the edges of the frame for focus effect."""
        h, w = frame.shape[:2]
        # Create radial gradient mask
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        # Elliptical distance from center, normalized to [0, 1]
        dist = np.sqrt(((X - cx) / (w / 2)) ** 2 + ((Y - cy) / (h / 2)) ** 2)
        dist = np.clip(dist, 0, 1.5)
        # Only darken the outer portion
        mask = np.clip((dist - 0.6) / 0.8, 0, 1) * strength
        mask = mask.astype(np.float32)
        # Apply darkening
        frame[:] = (frame * (1.0 - mask[:, :, np.newaxis])).astype(np.uint8)
