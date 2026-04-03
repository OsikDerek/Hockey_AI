"""Base protocol for technique detectors.

All detectors follow the same interface:
  1. add_frame() — accumulate per-frame data during video processing
  2. analyze() — run detection after all frames are collected
  3. Return a list of event dicts with raw metric values

The TechniqueEngine handles classification (good/warning/poor) using
thresholds from the technique YAML. Detectors compute raw metrics only.
"""

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class BaseDetector(Protocol):
    """Protocol that all technique detectors must implement."""

    def add_frame(self, landmarks: Optional[dict], angles: Optional[dict] = None) -> None:
        """Record data for one video frame.

        Args:
            landmarks: Pose landmarks dict (None if no skater detected).
            angles: Computed angles dict (None if not available).
        """
        ...

    def analyze(self, fps: float) -> list[dict]:
        """Analyze all accumulated frames and return detected events.

        Each event is a dict with at minimum:
            - frame_idx: int — the frame where the event occurs
            - frame_start: int — start of annotation window
            - frame_end: int — end of annotation window

        Plus metric fields matching the technique YAML's check metric names.
        Additional context fields (e.g., crossing_leg, stance_leg) are
        passed through to annotation template resolution.

        Args:
            fps: Video frame rate.

        Returns:
            List of event dicts with raw metric values.
        """
        ...
