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
from .goalie_analyzer import GoalieAnalyzer, draw_goalie_analysis
from .rink_calibrator import RINK_LENGTH_FT, RINK_WIDTH_FT, BLUE_LINE_LEFT_X_FT, BLUE_LINE_RIGHT_X_FT, CENTER_X_FT, CENTER_Y_FT, LEFT_GOAL_X_FT, RIGHT_GOAL_X_FT


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
    "team_a": (40, 220, 40),      # Bright green
    "team_b": (40, 40, 220),      # Bright red
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

# Ambient connection colors — binary green (puck can reach) / red (blocked)
AMBIENT_COLORS = {
    "reachable": (0, 220, 0),    # Bright green
    "blocked":   (0, 0, 220),    # Bright red
}


# Per-feature overlay flags. Each key controls one draw path. Quality gates
# (in GameAnnotator._can_show_*) further suppress features that would
# render at low confidence.
FULL_OVERLAYS = {
    "boxes": True,
    "team_colors": True,
    "puck": True,
    "possession_indicator": True,
    "zone_banner": True,
    "ambient_connections": True,
    "open_spaces": True,
    "passing_lanes": True,
    "shooting_lane": True,
    "decision_banner": True,
    "decision_freeze": True,
    "slowdown_indicator": True,
    "goalie_sight_lines": True,
    "frame_info_hud": True,
    "non_gameplay_stamp": True,
    "camera_cut_indicator": True,
    "minimap": True,
}

# Phase A2 default: show reliable structural overlays plus high-confidence
# decision moments. The decision-confidence gate (--decision-conf, default
# 0.7) filters which events actually paint banners/freezes — so this preset
# stays minimal in practice on noisy clips while still surfacing the
# textbook critical moments coaches want to see.
MINIMAL_OVERLAYS = {
    "boxes": True,
    "team_colors": True,
    "puck": True,
    "possession_indicator": False,
    "zone_banner": False,
    "ambient_connections": False,
    "open_spaces": False,
    "passing_lanes": False,
    "shooting_lane": False,
    "decision_banner": True,
    "decision_freeze": True,
    "slowdown_indicator": False,
    "goalie_sight_lines": False,
    "frame_info_hud": True,
    "non_gameplay_stamp": True,
    "camera_cut_indicator": True,
    "minimap": True,
}

OFF_OVERLAYS = {k: False for k in FULL_OVERLAYS}


def resolve_overlay_config(
    preset: str = "minimal",
    show: Optional[list] = None,
    hide: Optional[list] = None,
) -> dict:
    """Build the overlay flag dict from a preset name + show/hide deltas."""
    presets = {"minimal": MINIMAL_OVERLAYS, "full": FULL_OVERLAYS, "off": OFF_OVERLAYS}
    base = dict(presets.get(preset, MINIMAL_OVERLAYS))
    for k in (show or []):
        k = k.strip()
        if k in base:
            base[k] = True
    for k in (hide or []):
        k = k.strip()
        if k in base:
            base[k] = False
    return base


class GameAnnotator:
    """Render game analysis overlays on video frames.

    Standard mode: bounding boxes, zones, possession, event banners.
    Coaching mode: adds passing/shooting lanes, probabilities, freeze frames.
    Ambient mode: subtle persistent connections from carrier to teammates.
    """

    def __init__(
        self,
        overlay_config: Optional[dict] = None,
        show_ids: bool = True,
        show_rink_landmarks: bool = False,
        box_thickness: int = 2,
        coaching_mode: bool = True,
    ):
        # Per-feature overlay flags. Defaults to MINIMAL preset.
        self.overlay = dict(MINIMAL_OVERLAYS)
        if overlay_config:
            self.overlay.update(overlay_config)
        self.show_ids = show_ids
        self.show_rink_landmarks = show_rink_landmarks
        self.box_thickness = box_thickness
        self.coaching_mode = coaching_mode
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self._team_assignments = {}
        self._team_classifier = None  # set via set_classifier(); used by quality gates
        self._calibrator = None  # set via set_calibrator(); used by minimap
        self.lane_calculator = LaneCalculator()
        self.space_detector = SpaceDetector()
        self.goalie_analyzer = GoalieAnalyzer()
        # Cache of the most recently seen goalie object — used so freeze
        # frames still draw sight lines when YOLO drops the goalie for that
        # specific frame.
        self._last_goalie: Optional["object"] = None
        self._last_goalie_age: int = 999

    def set_team_assignments(self, assignments: dict):
        """Update the current team assignment map."""
        self._team_assignments = assignments

    def set_classifier(self, classifier):
        """Pass the TeamClassifier in so quality gates can read its state."""
        self._team_classifier = classifier

    def set_calibrator(self, calibrator):
        """Pass the RinkCalibrator in so the minimap can place objects in ice coords."""
        self._calibrator = calibrator

    # ── Quality gates — return False to suppress an enabled feature ─────
    def _classifier_ready(self) -> bool:
        return self._team_classifier is not None and getattr(self._team_classifier, "is_ready", False)

    def _vote_count(self, track_id: int) -> int:
        if self._team_classifier is None:
            return 0
        return self._team_classifier.get_vote_count(track_id)

    def _can_show_team_colors_for(self, track_id: int) -> bool:
        return self._classifier_ready() and self._vote_count(track_id) >= 5

    def _can_show_ambient(self, teammates: list) -> bool:
        if not self._classifier_ready() or not teammates:
            return False
        return all(self._vote_count(t.track_id) >= 3 for t in teammates)

    def _should_show_ids(self) -> bool:
        """Track-id labels are diagnostic, only render in the full preset
        (we use overlay.zone_banner as a proxy for "diagnostic preset")."""
        return self.show_ids and self.overlay.get("zone_banner", False)

    def _can_show_goalie_sight(self, context, goalie) -> bool:
        if goalie is None:
            return False
        if context.is_shootout_like:
            return True
        return self._team_assignments.get(goalie.track_id) is not None

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
        ov = self.overlay

        if not context.is_gameplay:
            if ov.get("non_gameplay_stamp"):
                self._draw_non_gameplay(annotated)
            return annotated

        if context.is_camera_cut and ov.get("camera_cut_indicator"):
            self._draw_camera_cut(annotated)

        # Update goalie cache (so freeze frames can fall back to it when
        # YOLO drops the goalie for that single frame)
        if context.goalies:
            self._last_goalie = context.goalies[0]
            self._last_goalie_age = 0
        else:
            self._last_goalie_age += 1

        if open_spaces and not context.is_shootout_like and ov.get("open_spaces"):
            self._draw_open_spaces(annotated, open_spaces)

        if ov.get("boxes"):
            self._draw_objects(annotated, context)

        if ov.get("zone_banner") and context.zone and not context.is_shootout_like:
            self._draw_zone_banner(annotated, context.zone)

        if (
            ov.get("possession_indicator")
            and context.possession_player_id is not None
            and self._classifier_ready()
        ):
            self._draw_possession(annotated, context)

        if ov.get("ambient_connections") and ambient_connections:
            # The teammates passed in are reflected in connection target_ids;
            # we re-derive vote-count gate from those.
            connected_ids = {c.get("target_id") for c in ambient_connections}
            teammate_objs = [p for p in context.players if p.track_id in connected_ids]
            if self._can_show_ambient(teammate_objs):
                self._draw_ambient_connections(annotated, ambient_connections)

        if ov.get("goalie_sight_lines") and context.is_shootout_like:
            self._draw_shootout_sight_lines(annotated, context)

        if ov.get("decision_banner") and events:
            self._draw_events(annotated, events)

        if ov.get("frame_info_hud"):
            self._draw_frame_info(annotated, context)

        if ov.get("minimap") and self._calibrator is not None and self._calibrator.is_calibrated:
            self._draw_minimap(annotated, context)
        return annotated

    @staticmethod
    def _find_player_by_id(players, track_id):
        """Find a tracked player by track_id. None if not in the list."""
        if track_id is None:
            return None
        for p in players:
            if p.track_id == track_id:
                return p
        return None

    def _draw_shootout_sight_lines(self, frame, context):
        """Draw always-on goalie sight lines during shootout-style 1-on-1s.

        Picks the lone non-goalie player as the carrier (or falls back to
        possession_player_id) and the most recently seen goalie.
        """
        carrier_obj = self._find_player_by_id(context.players, context.possession_player_id)
        if carrier_obj is None and len(context.players) == 1:
            carrier_obj = context.players[0]
        if carrier_obj is None:
            return

        goalie = context.goalies[0] if context.goalies else self._last_goalie
        if goalie is None or not self._can_show_goalie_sight(context, goalie):
            return

        goal_lm = None
        if context.rink_landmarks.get("goal"):
            goal_lm = context.rink_landmarks["goal"][0]

        h, w = frame.shape[:2]
        analysis = self.goalie_analyzer.analyze_shootout(carrier_obj, goalie, goal_lm, w, h)
        draw_goalie_analysis(frame, analysis)

    def render_coaching(
        self,
        frame: np.ndarray,
        context: FrameContext,
        events: Optional[list],
        phase_info: dict,
        lane_data: Optional[dict],
        ambient_connections: Optional[list] = None,
        open_spaces: Optional[list] = None,
    ) -> np.ndarray:
        """Render coaching overlays during decision events.

        Args:
            frame: BGR video frame.
            context: FrameContext.
            events: Active GameEvents.
            phase_info: {"phase": str, "event": GameEvent, "alpha": float}
            lane_data: {"shooting": dict, "passing": list} from LaneCalculator.
            ambient_connections: Subtle teammate connections (always-on).
            open_spaces: List of open space dicts from SpaceDetector.
        """
        # Start with standard rendering (pass ambient + open spaces through)
        annotated = self.render(frame, context, events,
                                ambient_connections=ambient_connections,
                                open_spaces=open_spaces)

        if lane_data is None:
            return annotated

        phase = phase_info.get("phase", "approach")
        alpha = phase_info.get("alpha", 0.5)
        event = phase_info.get("event")

        # Draw lanes on an overlay for transparency
        overlay = annotated.copy()
        ov = self.overlay
        drew_lane = False

        if (
            ov.get("passing_lanes")
            and lane_data.get("passing")
            and self._classifier_ready()
        ):
            self._draw_passing_lanes(overlay, lane_data["passing"], alpha, phase)
            drew_lane = True

        if (
            ov.get("shooting_lane")
            and lane_data.get("shooting")
            and self._classifier_ready()
        ):
            self._draw_shooting_lane(overlay, lane_data["shooting"], alpha, phase)
            drew_lane = True

        if drew_lane:
            lane_alpha = max(alpha, 0.55)
            cv2.addWeighted(overlay, lane_alpha, annotated, 1.0 - lane_alpha, 0, annotated)

        event_type = event.event_type if event else ""
        is_shot_event = event_type in ("shot_vs_pass", "odd_man_rush", "missed_opportunity")
        if ov.get("goalie_sight_lines") and is_shot_event and phase in ("slowdown", "freeze"):
            goalie_for_analysis = (
                context.goalies[0] if context.goalies
                else (self._last_goalie if self._last_goalie_age <= 30 else None)
            )
            if (
                goalie_for_analysis is not None
                and self._can_show_goalie_sight(context, goalie_for_analysis)
            ):
                eid = event.player_id if event else context.possession_player_id
                carrier_obj = self._find_player_by_id(context.players, eid)
                if carrier_obj:
                    goal_lm = None
                    if context.rink_landmarks.get("goal"):
                        goal_lm = context.rink_landmarks["goal"][0]
                    h, w = annotated.shape[:2]
                    goalie_data = self.goalie_analyzer.analyze_shot(
                        carrier_obj, goalie_for_analysis, goal_lm, w, h
                    )
                    draw_goalie_analysis(annotated, goalie_data)

        if ov.get("decision_freeze") and phase == "freeze" and event:
            self._draw_decision_freeze(annotated, event, lane_data)

        if ov.get("slowdown_indicator") and phase == "slowdown":
            self._draw_slowdown_indicator(annotated)

        return annotated

    # ── Standard Drawing Methods ──────────────────────────

    def _draw_objects(self, frame, context):
        """Draw bounding boxes with team colors and stripe indicators.

        Players get a colored stripe on the left of their box instead of
        painting the whole box in team color. This looks cleaner when
        team classification is noisy. Puck gets a bright highlight circle.
        """
        for obj in context.objects:
            if obj.class_name in ("centroid", "faceoff", "goal"):
                if not self.show_rink_landmarks:
                    continue

            team = self._team_assignments.get(obj.track_id)
            if (
                obj.class_name == "player"
                and team is not None
                and self.overlay.get("team_colors")
                and self._can_show_team_colors_for(obj.track_id)
            ):
                team_color = TEAM_COLORS.get(team, (200, 200, 200))
            else:
                team_color = None

            base_color = CLASS_COLORS.get(obj.class_name, (200, 200, 200))
            x1, y1, x2, y2 = [int(v) for v in obj.bbox]

            # -- Puck: bright green box + highlight circle --
            # Ghost pucks (confidence == 0.0) come from the coast logic in
            # PuckFilter when YOLO drops the puck for a frame or two; render
            # them with a dashed circle and dimmer color so the user can see
            # at a glance which frames are real detections vs coasted.
            if obj.class_name == "puck" and not self.overlay.get("puck", True):
                continue
            if obj.class_name == "puck":
                is_ghost = obj.confidence <= 0.001
                pcx = (x1 + x2) // 2
                pcy = (y1 + y2) // 2
                radius = max(12, (x2 - x1 + y2 - y1) // 2)
                if is_ghost:
                    # Dashed outline, no filled rectangle, dimmer color
                    self._draw_dashed_circle(frame, (pcx, pcy), radius, (0, 180, 0), thickness=2, dash_count=12)
                    label = "PUCK*"
                    sz = cv2.getTextSize(label, self.font, 0.5, 2)[0]
                    ly = max(y1 - 6, sz[1] + 4)
                    cv2.rectangle(frame, (x1, ly - sz[1] - 3), (x1 + sz[0] + 6, ly + 3), (0, 120, 0), -1)
                    cv2.putText(frame, label, (x1 + 3, ly), self.font, 0.5, (220, 220, 220), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 0), 3)
                    cv2.circle(frame, (pcx, pcy), radius, (0, 255, 0), 2)
                    cv2.circle(frame, (pcx, pcy), radius + 4, (0, 200, 0), 1)
                    label = "PUCK"
                    sz = cv2.getTextSize(label, self.font, 0.5, 2)[0]
                    ly = max(y1 - 6, sz[1] + 4)
                    cv2.rectangle(frame, (x1, ly - sz[1] - 3), (x1 + sz[0] + 6, ly + 3), (0, 180, 0), -1)
                    cv2.putText(frame, label, (x1 + 3, ly), self.font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                continue

            # -- Players: bounding-box outline colored by team (gray when
            # the classifier hasn't yet committed to a team for this player).
            # The box is the only team indicator; no separate stripe, no dot,
            # no "A#146" label — those were noise.
            if obj.class_name == "player":
                box_color = team_color if team_color else (220, 220, 220)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, self.box_thickness)
            else:
                # Non-player classes: keep their semantic colors as before
                cv2.rectangle(frame, (x1, y1), (x2, y2), base_color, self.box_thickness)

            # Track-id labels are diagnostic only — render them only when the
            # user explicitly enabled show_ids AND is running a non-minimal
            # preset (the diagnostic / full preset).
            if self._should_show_ids() and obj.track_id >= 0:
                label = f"#{obj.track_id}"
                label_color = team_color if team_color else base_color
                sz = cv2.getTextSize(label, self.font, 0.4, 1)[0]
                ly = max(y1 - 4, sz[1] + 2)
                cv2.rectangle(frame, (x1, ly - sz[1] - 2), (x1 + sz[0] + 4, ly + 2), label_color, -1)
                cv2.putText(frame, label, (x1 + 2, ly), self.font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_minimap(self, frame, context):
        """Top-down rink minimap with player + puck dots in ice coords.

        Sized ~30% of frame width, anchored bottom-right with a small inset.
        Uses the RinkCalibrator to translate detection centers to ice coords;
        clips off-rink dots so noise outside [0,200]x[0,85] doesn't render.
        """
        h, w = frame.shape[:2]
        # Minimap size: 30% of frame width, preserving 200:85 aspect ratio
        mm_w = int(w * 0.30)
        mm_h = int(mm_w * RINK_WIDTH_FT / RINK_LENGTH_FT)
        margin = 12
        x0 = w - mm_w - margin
        y0 = h - mm_h - margin

        # Helper: ice (x_ft, y_ft) -> minimap pixel
        def ice_to_mm(ix: float, iy: float) -> tuple:
            mx = int(x0 + (ix / RINK_LENGTH_FT) * mm_w)
            my = int(y0 + (iy / RINK_WIDTH_FT) * mm_h)
            return mx, my

        # Background panel: semi-transparent white-ice
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0 - 3, y0 - 3), (x0 + mm_w + 3, y0 + mm_h + 3), (40, 40, 40), -1)
        cv2.rectangle(overlay, (x0, y0), (x0 + mm_w, y0 + mm_h), (245, 245, 245), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Boards outline
        cv2.rectangle(frame, (x0, y0), (x0 + mm_w, y0 + mm_h), (60, 60, 60), 2)

        # Blue lines
        for blue_x in (BLUE_LINE_LEFT_X_FT, BLUE_LINE_RIGHT_X_FT):
            bx, _ = ice_to_mm(blue_x, 0)
            cv2.line(frame, (bx, y0), (bx, y0 + mm_h), (220, 90, 30), 2)

        # Center red line
        cx, _ = ice_to_mm(CENTER_X_FT, 0)
        cv2.line(frame, (cx, y0), (cx, y0 + mm_h), (40, 40, 220), 2)

        # Center ice dot + circle
        center_dot = ice_to_mm(CENTER_X_FT, CENTER_Y_FT)
        cv2.circle(frame, center_dot, max(3, mm_w // 80), (40, 40, 220), -1)
        cv2.circle(frame, center_dot, max(8, mm_w // 30), (40, 40, 220), 1)

        # End-zone faceoff dots: 4 corners (NHL geometry: 20 ft from goal line, 22 ft from boards)
        for dot_x_ft in (LEFT_GOAL_X_FT + 20, RIGHT_GOAL_X_FT - 20):
            for dot_y_ft in (22.0, RINK_WIDTH_FT - 22.0):
                pt = ice_to_mm(dot_x_ft, dot_y_ft)
                cv2.circle(frame, pt, max(2, mm_w // 100), (220, 90, 30), -1)
                cv2.circle(frame, pt, max(7, mm_w // 38), (220, 90, 30), 1)

        # Neutral-zone faceoff dots: 4 dots flanking the blue lines
        for dot_x_ft in (BLUE_LINE_LEFT_X_FT + 5, BLUE_LINE_RIGHT_X_FT - 5):
            for dot_y_ft in (22.0, RINK_WIDTH_FT - 22.0):
                pt = ice_to_mm(dot_x_ft, dot_y_ft)
                cv2.circle(frame, pt, max(2, mm_w // 100), (220, 90, 30), -1)

        # Goal creases (filled blue arcs are too fiddly at this scale; use small rect)
        for goal_x_ft in (LEFT_GOAL_X_FT, RIGHT_GOAL_X_FT):
            gx, gy_top = ice_to_mm(goal_x_ft, CENTER_Y_FT - 4)
            _, gy_bot = ice_to_mm(0, CENTER_Y_FT + 4)
            cv2.line(frame, (gx, gy_top), (gx, gy_bot), (40, 40, 220), 2)

        # Plot players (team-colored), goalies (team-colored, larger), and puck
        def in_rink(ix: float, iy: float) -> bool:
            return -2 <= ix <= RINK_LENGTH_FT + 2 and -2 <= iy <= RINK_WIDTH_FT + 2

        for p in context.players:
            if p.ice_xy is None:
                continue
            ix, iy = p.ice_xy
            if not in_rink(ix, iy):
                continue
            team = self._team_assignments.get(p.track_id)
            color = TEAM_COLORS.get(team, (160, 160, 160)) if team else (160, 160, 160)
            cv2.circle(frame, ice_to_mm(ix, iy), max(3, mm_w // 70), color, -1)
            cv2.circle(frame, ice_to_mm(ix, iy), max(3, mm_w // 70), (255, 255, 255), 1)

        for g in context.goalies:
            if g.ice_xy is None:
                continue
            ix, iy = g.ice_xy
            if not in_rink(ix, iy):
                continue
            team = self._team_assignments.get(g.track_id)
            color = TEAM_COLORS.get(team, (200, 200, 200)) if team else (200, 200, 200)
            cv2.circle(frame, ice_to_mm(ix, iy), max(5, mm_w // 50), color, -1)
            cv2.circle(frame, ice_to_mm(ix, iy), max(5, mm_w // 50), (0, 0, 0), 2)

        if context.puck is not None and context.puck.ice_xy is not None:
            ix, iy = context.puck.ice_xy
            if in_rink(ix, iy):
                cv2.circle(frame, ice_to_mm(ix, iy), max(3, mm_w // 90), (0, 220, 220), -1)
                cv2.circle(frame, ice_to_mm(ix, iy), max(3, mm_w // 90), (0, 0, 0), 1)

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
        """Draw a bold possession indicator around the puck carrier.

        Uses a double-border glow effect and a prominent CARRIER label
        below the player box so it stands out from team boxes.
        """
        for obj in context.players:
            if obj.track_id == context.possession_player_id:
                x1, y1, x2, y2 = [int(v) for v in obj.bbox]
                # Outer glow (thicker, darker green)
                cv2.rectangle(frame, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 180, 0), 2)
                # Inner bright highlight
                cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 0), 3)

                # CARRIER label below the box with background
                label = "CARRIER"
                sz = cv2.getTextSize(label, self.font, 0.55, 2)[0]
                lx = x1
                ly = y2 + 20
                cv2.rectangle(frame, (lx - 2, ly - sz[1] - 3), (lx + sz[0] + 4, ly + 4), (0, 180, 0), -1)
                cv2.putText(frame, label, (lx + 1, ly), self.font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
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
        """Draw event banners with rating-colored backgrounds.

        Green = GOOD, Yellow = WARNING, Red = POOR, Orange = default.
        Banners are larger, bolder, with a colored left-edge accent.
        """
        h, w = frame.shape[:2]
        RATING_BANNER_COLORS = {
            "good": (0, 160, 0),       # Green
            "warning": (0, 180, 220),   # Yellow/amber
            "poor": (0, 0, 200),        # Red
        }
        for i, event in enumerate(events):
            y = 44 + i * 38
            text = f"{event.event_type.upper().replace('_', ' ')}"
            if event.decision_made:
                text += f" : {event.decision_made.upper()}"

            rating = ""
            if event.evaluation:
                rating = event.evaluation.get("rating", "")

            banner_color = RATING_BANNER_COLORS.get(rating, (0, 80, 200))

            sz = cv2.getTextSize(text, self.font, 0.6, 2)[0]
            bx2 = 24 + sz[0]
            by1 = y - 22
            by2 = y + 8

            # Dark background
            cv2.rectangle(frame, (10, by1), (bx2, by2), (30, 30, 30), -1)
            # Colored left accent bar
            cv2.rectangle(frame, (10, by1), (18, by2), banner_color, -1)
            # Rating badge on right side
            if rating:
                rating_text = rating.upper()
                rsz = cv2.getTextSize(rating_text, self.font, 0.45, 2)[0]
                rx = bx2 + 6
                cv2.rectangle(frame, (rx, by1), (rx + rsz[0] + 10, by2), banner_color, -1)
                cv2.putText(frame, rating_text, (rx + 5, y - 2), self.font, 0.45, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, text, (22, y), self.font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_frame_info(self, frame, context):
        h, w = frame.shape[:2]
        info = f"F:{context.frame_idx} P:{len(context.players)} G:{len(context.goalies)} {'PUCK' if context.puck else 'no puck'}"
        cv2.putText(frame, info, (10, h - 10), self.font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Ambient Connections (always-on) ───────────────────

    def _draw_ambient_connections(self, frame, connections):
        """Draw subtle persistent lines from carrier to teammates.

        Uses thicker lines with higher alpha so connections are actually
        visible against the ice. Also draws a small target circle at each
        teammate endpoint.
        """
        overlay = frame.copy()
        for conn in connections:
            color = AMBIENT_COLORS.get(conn["quality"], (80, 80, 80))
            # Brighten colors so they show up on ice
            bright_color = tuple(min(255, int(c * 1.8)) for c in color)
            pt1 = tuple(int(v) for v in conn["start"])
            pt2 = tuple(int(v) for v in conn["end"])
            # Thicker dashed line for visibility
            self._draw_dashed_line(overlay, pt1, pt2, bright_color, thickness=2, dash_length=14)
            # Small target circle at teammate end
            cv2.circle(overlay, pt2, 10, bright_color, 2)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # ── Open Space Visualization ───────────────────────────

    def _draw_open_spaces(self, frame, spaces):
        """Draw tactical gaps between defenders as discrete markers.

        Shows the seam/gap between two defenders with:
        - A dashed line connecting the two defenders (showing the gap)
        - A diamond marker at the midpoint
        - A label if it's high value
        No shading of large regions — just the specific exploitable gaps.
        """
        for space in spaces:
            value = space["value"]
            cx, cy = int(space["center"][0]), int(space["center"][1])

            # Color by value
            if value == "high":
                color = (0, 255, 80)
                marker_size = 14
            elif value == "medium":
                color = (0, 200, 180)
                marker_size = 10
            else:
                continue  # Don't show low-value gaps

            # Draw dashed line between the two defenders forming the gap
            if "defender_a" in space and "defender_b" in space:
                da = tuple(int(v) for v in space["defender_a"])
                db = tuple(int(v) for v in space["defender_b"])
                self._draw_dashed_line(frame, da, db, color, thickness=1, dash_length=10)

            # Diamond marker at the gap center
            pts = np.array([
                [cx, cy - marker_size],
                [cx + marker_size, cy],
                [cx, cy + marker_size],
                [cx - marker_size, cy],
            ], dtype=np.int32)
            cv2.fillPoly(frame, [pts], color)
            cv2.polylines(frame, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

            # Label for high-value gaps
            if value == "high" and space.get("reasons"):
                label = space["reasons"][0] if space["reasons"] else "GAP"
                if space.get("adjacent_to_carrier"):
                    label = "OPEN ICE"
                elif space.get("in_dangerous_zone"):
                    label = "SLOT GAP"

                sz = cv2.getTextSize(label, self.font, 0.45, 1)[0]
                tx = cx - sz[0] // 2
                ty = cy - marker_size - 8
                # Background
                cv2.rectangle(frame, (tx - 3, ty - sz[1] - 3), (tx + sz[0] + 3, ty + 3), color, -1)
                cv2.putText(frame, label, (tx, ty), self.font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # ── Coaching Overlay Methods ──────────────────────────

    def _draw_passing_lanes(self, frame, lanes, alpha, phase):
        """Draw passing lanes with quality colors and success percentages.

        Lanes are thicker during slowdown/freeze. Percentages show in
        all phases so the coaching value is always visible.
        """
        for lane in lanes:
            color = LANE_COLORS.get(lane["quality"], (200, 200, 200))
            pt1 = tuple(int(v) for v in lane["start"])
            pt2 = tuple(int(v) for v in lane["end"])

            # Thicker lines during slowdown/freeze
            thickness = 3 if phase in ("slowdown", "freeze") else 2
            self._draw_dashed_line(frame, pt1, pt2, color, thickness=thickness, dash_length=15)

            # Success % badge at midpoint (always show during coaching)
            mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            pct_text = f"{lane['success_pct']:.0%}"
            badge_radius = 26 if phase in ("slowdown", "freeze") else 20
            self._draw_pct_badge(frame, mid, pct_text, color, radius=badge_radius)

            # Blocker indicators
            for blocker in lane.get("blockers_near", []):
                bp = tuple(int(v) for v in blocker["pos"])
                cv2.circle(frame, bp, 8, (0, 0, 200), -1)
                cv2.circle(frame, bp, 8, (0, 0, 150), 2)

    def _draw_shooting_lane(self, frame, shooting, alpha, phase):
        """Draw shooting lane with arrow and blocker indicators.

        Always shows SHOT percentage badge for coaching clarity.
        """
        if shooting is None:
            return

        color = LANE_COLORS.get(shooting["quality"], (200, 200, 200))
        pt1 = tuple(int(v) for v in shooting["start"])
        pt2 = tuple(int(v) for v in shooting["end"])

        # Thicker arrow during key phases
        thickness = 4 if phase in ("slowdown", "freeze") else 3
        cv2.arrowedLine(frame, pt1, pt2, color, thickness, cv2.LINE_AA, tipLength=0.03)

        # Success % badge near the line (always show)
        mid = ((pt1[0] * 2 + pt2[0]) // 3, (pt1[1] * 2 + pt2[1]) // 3)
        pct_text = f"SHOT {shooting['success_pct']:.0%}"
        badge_radius = 30 if phase in ("slowdown", "freeze") else 24
        self._draw_pct_badge(frame, mid, pct_text, color, radius=badge_radius)

        # Blocker indicators
        for blocker in shooting.get("blockers", []):
            bp = tuple(int(v) for v in blocker["pos"])
            cv2.circle(frame, bp, 12, (0, 0, 220), -1)
            # X mark
            cv2.line(frame, (bp[0] - 5, bp[1] - 5), (bp[0] + 5, bp[1] + 5), (255, 255, 255), 2)
            cv2.line(frame, (bp[0] - 5, bp[1] + 5), (bp[0] + 5, bp[1] - 5), (255, 255, 255), 2)

    def _draw_decision_freeze(self, frame, event, lane_data):
        """Draw the full decision breakdown on a freeze frame.

        Creates a dramatic freeze-frame look with:
        - Strong vignette to focus attention
        - Large DECISION POINT header with drop shadow
        - Bottom panel with decision info and suggestion
        - Top-right rating badge
        - Lane summary count
        """
        h, w = frame.shape[:2]

        # Strong vignette (darken edges)
        self._draw_vignette(frame, strength=0.65)

        # ── Top: "DECISION POINT" header ──
        text = "DECISION POINT"
        font_scale = min(1.6, w / 700)  # Scale with frame width
        sz = cv2.getTextSize(text, self.font, font_scale, 4)[0]
        tx = (w - sz[0]) // 2
        ty = 75

        # Drop shadow
        cv2.putText(frame, text, (tx + 2, ty + 2), self.font, font_scale, (0, 0, 0), 6, cv2.LINE_AA)
        # White text with black outline
        cv2.putText(frame, text, (tx, ty), self.font, font_scale, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, ty), self.font, font_scale, (255, 255, 255), 3, cv2.LINE_AA)

        # ── Top-right: Rating badge (big and bold) ──
        rating = ""
        if event.evaluation:
            rating = event.evaluation.get("rating", "")
        if rating:
            rating_colors = {
                "good": (0, 180, 0),
                "warning": (0, 180, 200),
                "poor": (0, 0, 220),
            }
            rc = rating_colors.get(rating, (150, 150, 150))
            badge_text = rating.upper()
            bsz = cv2.getTextSize(badge_text, self.font, 0.9, 2)[0]
            bx = w - bsz[0] - 30
            by1 = 44
            by2 = 82
            cv2.rectangle(frame, (bx - 10, by1), (w - 10, by2), rc, -1)
            cv2.rectangle(frame, (bx - 10, by1), (w - 10, by2), (255, 255, 255), 2)
            cv2.putText(frame, badge_text, (bx, by2 - 8), self.font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # ── Bottom panel: dark translucent bar ──
        panel_h = 90
        panel_y = h - panel_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # What was chosen (large, bright yellow)
        chose_text = f"CHOSE: {event.decision_made.upper()}"
        if event.player_id:
            chose_text += f"  (#{event.player_id})"
        chose_scale = 0.8
        csz = cv2.getTextSize(chose_text, self.font, chose_scale, 2)[0]
        cy = panel_y + 30
        # Cyan/yellow accent bar
        cv2.rectangle(frame, (15, cy - 22), (22, cy + 6), (0, 255, 255), -1)
        cv2.putText(frame, chose_text, (28, cy), self.font, chose_scale, (0, 255, 255), 2, cv2.LINE_AA)

        # Suggestion (if available)
        if event.evaluation:
            alt = event.evaluation.get("alternative")
            if alt:
                sug_text = f"SUGGEST: {alt[:70]}"
                sug_scale = 0.55
                ssz = cv2.getTextSize(sug_text, self.font, sug_scale, 1)[0]
                sy = panel_y + 58
                cv2.rectangle(frame, (15, sy - 16), (22, sy + 4), (180, 180, 0), -1)
                cv2.putText(frame, sug_text, (28, sy), self.font, sug_scale, (180, 200, 200), 1, cv2.LINE_AA)

            # Reasoning snippet
            reasoning = event.evaluation.get("reasoning", "")
            if reasoning:
                # Take first sentence
                first_sentence = reasoning.split("|")[0].strip()[:80]
                ry = panel_y + 78
                cv2.putText(frame, first_sentence, (28, ry), self.font, 0.4, (140, 140, 140), 1, cv2.LINE_AA)

        # Lane count summary (top-left under DECISION POINT)
        if lane_data:
            n_pass = len(lane_data.get("passing", []))
            shot = lane_data.get("shooting")
            shot_text = f"SHOT {shot['quality'].upper()}" if shot else "NO SHOT"
            lane_text = f"{n_pass} passing lanes | {shot_text}"
            cv2.putText(frame, lane_text, (tx, ty + 28), self.font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    def _draw_slowdown_indicator(self, frame):
        """Visible slow-motion indicator in top-right corner."""
        h, w = frame.shape[:2]
        text = "SLOW MOTION"
        sz = cv2.getTextSize(text, self.font, 0.55, 2)[0]
        tx = w - sz[0] - 15
        ty = 24
        # Dark pill background
        cv2.rectangle(frame, (tx - 6, ty - sz[1] - 4), (tx + sz[0] + 6, ty + 4), (40, 40, 40), -1)
        cv2.putText(frame, text, (tx, ty), self.font, 0.55, (200, 200, 255), 2, cv2.LINE_AA)

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

    def _draw_dashed_circle(self, frame, center, radius, color, thickness=2, dash_count=12):
        """Draw a dashed circle (ghost-puck marker)."""
        cx, cy = int(center[0]), int(center[1])
        for i in range(0, dash_count, 2):
            a0 = (i / dash_count) * 2 * np.pi
            a1 = ((i + 1) / dash_count) * 2 * np.pi
            p0 = (int(cx + radius * np.cos(a0)), int(cy + radius * np.sin(a0)))
            p1 = (int(cx + radius * np.cos(a1)), int(cy + radius * np.sin(a1)))
            cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)

    def _draw_pct_badge(self, frame, center, text, color, radius=24):
        """Draw a filled circle with percentage text inside.

        Uses a larger radius and font for readability on broadcast footage.
        """
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(frame, (cx, cy), radius, color, -1)
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2)
        # Scale font to fit radius
        font_scale = 0.45 if len(text) <= 4 else 0.38
        thickness = 2 if len(text) <= 4 else 1
        sz = cv2.getTextSize(text, self.font, font_scale, thickness)[0]
        tx = cx - sz[0] // 2
        ty = cy + sz[1] // 2
        cv2.putText(frame, text, (tx, ty), self.font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

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
