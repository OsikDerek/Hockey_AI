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

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height


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

    context: dict = field(default_factory=dict)       # Situation details
    evaluation: Optional[dict] = None                 # Rating + suggested alternative

    def __repr__(self):
        return (
            f"GameEvent({self.event_type} at {self.timestamp_sec:.1f}s, "
            f"decision={self.decision_made}, player={self.player_id})"
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

    def events_at_frame(self, frame_idx: int) -> list:
        """Return all events active at this frame."""
        return [
            e for e in self.events
            if e.frame_start <= frame_idx <= e.frame_end
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
