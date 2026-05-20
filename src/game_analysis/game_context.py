"""Core data models for game analysis.

All components read from and write to these shared dataclasses.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class TrackedObject:
    """A single tracked entity in one frame."""
    track_id: int
    class_name: str           # "player", "puck", "goalie", "referee", etc.
    class_id: int
    bbox: tuple               # (x1, y1, x2, y2)
    center: tuple             # (cx, cy)
    confidence: float
    ice_xy: Optional[tuple] = None    # (x_ft, y_ft) in NHL rink coords; None when uncalibrated

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def foot_point(self) -> tuple:
        """Bottom-center of the bbox — the point where the player's
        skates contact the ice. This is the geometrically correct point
        to project through an ice-plane homography: the bbox CENTER sits
        at torso height (~3 ft up), and projecting an above-ice point
        through a ground-plane homography places the player offset
        toward the camera. For a side-viewing rink camera that offset
        pushes far-side players toward mid-rink — a direct contributor
        to across-rink position compression.

        Matches the SOTA approach (Multi-Player Tracking in Ice Hockey
        with Homographic Projections, arXiv:2405.13397): project the
        foot keypoint, not the box center.
        """
        return ((self.bbox[0] + self.bbox[2]) / 2.0, self.bbox[3])


@dataclass
class FrameContext:
    """All game state for a single frame."""
    frame_idx: int
    timestamp_sec: float

    # Tracking
    objects: list = field(default_factory=list)       # list[TrackedObject]
    players: list = field(default_factory=list)       # convenience: just players
    puck: Optional[TrackedObject] = None              # highest-conf puck
    goalies: list = field(default_factory=list)       # convenience: goalies
    referees: list = field(default_factory=list)      # convenience: referees

    # Rink landmarks detected this frame
    rink_landmarks: dict = field(default_factory=dict)  # class_name -> list[TrackedObject]

    # Context
    zone: Optional[str] = None                        # "offensive", "neutral", "defensive"
    possession_player_id: Optional[int] = None        # track_id of possessing player
    is_gameplay: bool = True
    is_camera_cut: bool = False
    is_shootout_like: bool = False  # 1-on-1 vs goalie context (shooter + goalie, no defenders)
    ice_calibrated: bool = False  # True when RinkCalibrator has produced a valid pixel<->ice transform
    ice_homography: bool = False  # True when B1 homography is active (else similarity fallback)
    team_assignments: dict = field(default_factory=dict)  # track_id -> "team_a" | "team_b"


@dataclass
class GameEvent:
    """A detected decision point with evaluation."""
    event_type: str                     # "shot_vs_pass", "zone_entry", "breakout"
    frame_idx: int                      # Frame where decision occurs
    frame_start: int                    # Context window start
    frame_end: int                      # Context window end
    timestamp_sec: float

    decision_made: str = ""             # What the player actually did
    player_id: Optional[int] = None     # track_id of decision-maker

    confidence: float = 0.5             # [0,1] — how textbook-clear the trigger was;
                                         # overlays only render when >= --decision-conf
    context: dict = field(default_factory=dict)       # Situation details
    evaluation: Optional[dict] = None                 # Rating + suggested alternative

    def __repr__(self):
        return (
            f"GameEvent({self.event_type} at {self.timestamp_sec:.1f}s, "
            f"decision={self.decision_made}, player={self.player_id}, "
            f"conf={self.confidence:.2f})"
        )


@dataclass
class GameAnalysis:
    """Complete game analysis result."""
    events: list = field(default_factory=list)             # list[GameEvent]
    frame_contexts: list = field(default_factory=list)     # list[FrameContext]

    # Aggregated stats
    total_frames: int = 0
    gameplay_frames: int = 0
    camera_cuts: int = 0
    zone_time: dict = field(default_factory=dict)          # zone -> seconds
    possession_time: dict = field(default_factory=dict)    # track_id -> seconds

    def events_at_frame(self, frame_idx: int, min_confidence: float = 0.0) -> list:
        """Return events active at this frame, optionally filtered by confidence."""
        return [
            e for e in self.events
            if e.frame_start <= frame_idx <= e.frame_end
            and e.confidence >= min_confidence
        ]

    def summarize(self) -> dict:
        """Return a summary dict for reporting."""
        return {
            "total_frames": self.total_frames,
            "gameplay_frames": self.gameplay_frames,
            "gameplay_pct": (self.gameplay_frames / max(self.total_frames, 1)) * 100,
            "camera_cuts": self.camera_cuts,
            "zone_time": dict(self.zone_time),
            "total_events": len(self.events),
            "events_by_type": self._count_by_type(),
        }

    def _count_by_type(self) -> dict:
        counts = {}
        for e in self.events:
            counts[e.event_type] = counts.get(e.event_type, 0) + 1
        return counts
