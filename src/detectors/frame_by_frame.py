"""Frame-by-frame detector for simple angle-threshold techniques.

Used for techniques like forward stride where every frame is evaluated
independently against angle thresholds. No temporal event detection needed.

This is the Tier 1 detector — requires zero custom Python code. Just
define angle checks in the technique YAML and this handles the rest.
"""

from typing import Optional


class FrameByFrameDetector:
    """Evaluates per-frame angles. No event detection — every frame is an 'event'."""

    def __init__(self, min_visibility: float = 0.5, analyze_sides: str = "both", **kwargs):
        self.min_visibility = min_visibility
        self.analyze_sides = analyze_sides
        self._frames: list[dict] = []  # List of {angles: dict, landmarks: dict}

    def add_frame(self, landmarks: Optional[dict] = None, angles: Optional[dict] = None) -> None:
        self._frames.append({
            "landmarks": landmarks,
            "angles": angles or {},
        })

    def analyze(self, fps: float) -> list[dict]:
        """Return one event per frame that has valid angle data.

        Each event contains the raw angle values keyed by their
        angle_calculator function name (e.g., left_knee_angle, right_hip_angle).
        """
        events = []
        for i, frame in enumerate(self._frames):
            if not frame["angles"]:
                continue
            event = {
                "frame_idx": i,
                "frame_start": i,
                "frame_end": i,
                "angles": frame["angles"],
                "landmarks": frame["landmarks"],
            }
            events.append(event)
        return events
