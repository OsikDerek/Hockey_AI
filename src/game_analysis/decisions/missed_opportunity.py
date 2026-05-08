"""Missed opportunity detector.

Detects high-percentage passing or shooting opportunities that the
puck carrier did NOT act on. This catches situations where a great
option opens up but the player holds the puck, makes a worse choice,
or simply doesn't see it.

Unlike other detectors that trigger on actions (puck release, zone
crossing), this detector triggers on INACTION — when a window of
opportunity opens and closes without being exploited.
"""

import numpy as np
from typing import Optional

from ..game_context import FrameContext, GameEvent, TrackedObject
from ..lane_calculator import LaneCalculator


class MissedOpportunityDetector:
    """Detect high-percentage opportunities that were not taken.

    Every frame with possession, calculates passing and shooting lanes.
    When a lane exceeds the opportunity threshold and persists for
    min_window frames, it becomes a "window." If the window closes
    (quality drops or possession changes) without the carrier acting,
    it's flagged as a missed opportunity.

    This ensures the coaching overlay pauses on opportunities that
    OTHER detectors might miss because no explicit action (puck release,
    zone crossing) occurred.
    """

    def __init__(
        self,
        shot_opportunity_threshold: float = 0.18,
        pass_opportunity_threshold: float = 0.75,
        min_window_frames: int = 8,
        cooldown_frames: int = 45,
        event_window: int = 20,
    ):
        """
        Args:
            shot_opportunity_threshold: Minimum shot success % to flag (18% is very high for hockey).
            pass_opportunity_threshold: Minimum pass success % to flag (75% = wide open).
            min_window_frames: Opportunity must persist this many frames to be flagged.
            cooldown_frames: Minimum frames between missed opportunity events.
            event_window: Frames of context around each event for rendering.
        """
        self.shot_threshold = shot_opportunity_threshold
        self.pass_threshold = pass_opportunity_threshold
        self.min_window = min_window_frames
        self.cooldown_frames = cooldown_frames
        self.event_window = event_window

        self.lane_calculator = LaneCalculator()

        self._frames = []
        self._cooldown = 0

        # Track open windows: {type: {"start_frame", "peak_pct", "target_id", "carrier_id"}}
        self._active_shot_window = None
        self._active_pass_windows = {}  # target_id -> window dict

    def add_frame(self, context: FrameContext) -> None:
        """Process one frame."""
        self._frames.append(context)
        if self._cooldown > 0:
            self._cooldown -= 1

    def analyze(
        self,
        fps: float,
        frame_contexts: list = None,
        team_data: list = None,
        frame_data: list = None,
    ) -> list:
        """Analyze all frames for missed opportunities.

        Needs extra data beyond what add_frame provides because we need
        to calculate lanes (which requires the raw frame for team splits).
        If frame_contexts/team_data/frame_data aren't provided, uses
        stored frames with limited lane calculation.

        Returns list of GameEvent with event_type="missed_opportunity".
        """
        contexts = frame_contexts or self._frames
        events = []

        # Track windows across frames
        active_shot = None      # {"start": int, "peak_pct": float, "carrier_id": int}
        active_passes = {}      # target_id -> {"start": int, "peak_pct": float, "carrier_id": int}
        cooldown = 0

        for i, ctx in enumerate(contexts):
            if cooldown > 0:
                cooldown -= 1

            if ctx.possession_player_id is None or not ctx.is_gameplay:
                # No possession — close all windows and check for missed opportunities
                if active_shot and (i - active_shot["start"]) >= self.min_window:
                    event = self._create_event(
                        active_shot, i, fps, "shot",
                        f"Open shot ({active_shot['peak_pct']:.0%}) not taken"
                    )
                    if cooldown <= 0:
                        events.append(event)
                        cooldown = self.cooldown_frames
                for tid, pw in list(active_passes.items()):
                    if (i - pw["start"]) >= self.min_window:
                        event = self._create_event(
                            pw, i, fps, "pass",
                            f"Open pass to #{tid} ({pw['peak_pct']:.0%}) not taken"
                        )
                        if cooldown <= 0:
                            events.append(event)
                            cooldown = self.cooldown_frames
                active_shot = None
                active_passes = {}
                continue

            # Get teammates and opponents for this frame
            carrier_id = ctx.possession_player_id
            carrier = None
            teammates = []
            opponents = []

            for p in ctx.players:
                if p.track_id == carrier_id:
                    carrier = p
                elif team_data and i < len(team_data):
                    carrier_team = team_data[i].get(carrier_id)
                    p_team = team_data[i].get(p.track_id)
                    if carrier_team and p_team == carrier_team:
                        teammates.append(p)
                    else:
                        opponents.append(p)
                else:
                    opponents.append(p)

            if carrier is None:
                continue

            fw = 1280  # Default frame width
            fh = 720
            if frame_data and i < len(frame_data):
                fh, fw = frame_data[i].shape[:2]

            # Calculate shooting lane
            goalie = ctx.goalies[0] if ctx.goalies else None
            shot_lane = self.lane_calculator.calc_shooting_lane(
                carrier, goalie, opponents, fw, fh
            )

            # Calculate passing lanes
            pass_lanes = []
            if teammates:
                pass_lanes = self.lane_calculator.calc_passing_lanes(
                    carrier, teammates, opponents, fw, fh
                )

            # Track shooting opportunity windows
            if shot_lane and shot_lane["success_pct"] >= self.shot_threshold:
                if active_shot is None:
                    active_shot = {
                        "start": i, "peak_pct": shot_lane["success_pct"],
                        "carrier_id": carrier_id, "peak_frame": i,
                    }
                else:
                    if shot_lane["success_pct"] > active_shot["peak_pct"]:
                        active_shot["peak_pct"] = shot_lane["success_pct"]
                        active_shot["peak_frame"] = i
            else:
                # Shot window closed
                if active_shot and (i - active_shot["start"]) >= self.min_window:
                    event = self._create_event(
                        active_shot, i, fps, "shot",
                        f"Open shot ({active_shot['peak_pct']:.0%}) — window closed without shooting"
                    )
                    if cooldown <= 0:
                        events.append(event)
                        cooldown = self.cooldown_frames
                active_shot = None

            # Track passing opportunity windows
            for lane in pass_lanes:
                tid = lane["target_id"]
                if lane["success_pct"] >= self.pass_threshold:
                    if tid not in active_passes:
                        active_passes[tid] = {
                            "start": i, "peak_pct": lane["success_pct"],
                            "carrier_id": carrier_id, "peak_frame": i,
                            "target_id": tid,
                        }
                    else:
                        if lane["success_pct"] > active_passes[tid]["peak_pct"]:
                            active_passes[tid]["peak_pct"] = lane["success_pct"]
                            active_passes[tid]["peak_frame"] = i
                else:
                    # This pass window closed
                    if tid in active_passes:
                        pw = active_passes[tid]
                        if (i - pw["start"]) >= self.min_window:
                            event = self._create_event(
                                pw, i, fps, "pass",
                                f"Open pass to #{tid} ({pw['peak_pct']:.0%}) — window closed"
                            )
                            if cooldown <= 0:
                                events.append(event)
                                cooldown = self.cooldown_frames
                        del active_passes[tid]

        return events

    def _create_event(
        self, window: dict, close_frame: int, fps: float,
        opportunity_type: str, description: str,
    ) -> GameEvent:
        """Create a missed opportunity GameEvent."""
        peak_frame = window.get("peak_frame", window["start"])
        frame_start = max(0, peak_frame - self.event_window)
        frame_end = min(close_frame, peak_frame + self.event_window)

        # Confidence: only flag textbook misses — high peak quality and a
        # sustained window. Borderline opportunities get low confidence and
        # get filtered out of overlays even though the report keeps them.
        peak_pct = window.get("peak_pct", 0.0)
        duration_frames = close_frame - window["start"]
        threshold = (self.shot_threshold if opportunity_type == "shot"
                     else self.pass_threshold)
        # How far above the trigger threshold the peak landed (0..1).
        margin = (peak_pct - threshold) / max(1.0 - threshold, 1e-6)
        confidence = 0.30
        confidence += 0.40 * max(0.0, min(1.0, margin))
        if duration_frames >= 2 * self.min_window:
            confidence += 0.20
        if window.get("carrier_id") is not None:
            confidence += 0.10
        confidence = max(0.0, min(1.0, confidence))

        return GameEvent(
            event_type="missed_opportunity",
            frame_idx=peak_frame,
            frame_start=frame_start,
            frame_end=frame_end,
            timestamp_sec=peak_frame / fps,
            decision_made=f"missed_{opportunity_type}",
            player_id=window.get("carrier_id"),
            confidence=confidence,
            context={
                "opportunity_type": opportunity_type,
                "peak_pct": window["peak_pct"],
                "window_start": window["start"],
                "window_duration_frames": close_frame - window["start"],
                "target_id": window.get("target_id"),
                "description": description,
            },
        )
