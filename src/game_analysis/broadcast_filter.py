"""Broadcast film filter for camera cuts and non-gameplay detection.

Detects camera angle changes via HSV histogram comparison and flags
non-gameplay frames (replays, commercials, between-period breaks).
"""

import cv2
import numpy as np
from collections import deque


class BroadcastFilter:
    """Detect camera cuts and non-gameplay in broadcast footage.

    Camera cuts are detected via sudden changes in HSV color histograms
    between consecutive frames. Non-gameplay is flagged when too few
    game objects (players/puck) are detected for several frames.

    Usage:
        bf = BroadcastFilter()
        is_gameplay, is_cut = bf.process_frame(frame, num_game_objects=8)
    """

    def __init__(
        self,
        cut_threshold: float = 0.4,
        min_game_objects: int = 3,
        non_gameplay_window: int = 15,
    ):
        """
        Args:
            cut_threshold: Chi-squared histogram distance threshold for camera cuts.
                Higher = less sensitive. 0.3-0.5 works well for broadcast hockey.
            min_game_objects: Minimum players+puck+goalies for a frame to be "gameplay".
            non_gameplay_window: Consecutive frames below min_game_objects before
                marking as non-gameplay.
        """
        self.cut_threshold = cut_threshold
        self.min_game_objects = min_game_objects
        self.non_gameplay_window = non_gameplay_window

        self._prev_hist = None
        self._low_object_streak = 0
        self._cut_cooldown = 0  # Prevent double-triggering on gradual transitions

    def process_frame(
        self,
        frame: np.ndarray,
        num_game_objects: int = 0,
    ) -> tuple:
        """Process one frame and return broadcast filter results.

        Args:
            frame: BGR video frame.
            num_game_objects: Number of players + puck + goalies detected.

        Returns:
            (is_gameplay, is_camera_cut) tuple of bools.
        """
        is_camera_cut = self._detect_camera_cut(frame)
        is_gameplay = self._detect_gameplay(num_game_objects)

        return is_gameplay, is_camera_cut

    def _detect_camera_cut(self, frame: np.ndarray) -> bool:
        """Detect camera cuts via HSV histogram comparison."""
        # Convert to HSV and compute histogram
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None, [32, 32], [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist)

        is_cut = False

        if self._prev_hist is not None and self._cut_cooldown <= 0:
            # Chi-squared distance between histograms
            dist = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CHISQR)

            if dist > self.cut_threshold:
                is_cut = True
                self._cut_cooldown = 5  # Suppress for 5 frames after a cut

        self._prev_hist = hist

        if self._cut_cooldown > 0:
            self._cut_cooldown -= 1

        return is_cut

    def _detect_gameplay(self, num_game_objects: int) -> bool:
        """Detect whether this frame is actual gameplay."""
        if num_game_objects >= self.min_game_objects:
            self._low_object_streak = 0
            return True

        self._low_object_streak += 1

        # Only flag as non-gameplay after sustained low detection
        if self._low_object_streak >= self.non_gameplay_window:
            return False

        # During the grace period, still consider it gameplay
        return True

    def reset(self):
        """Reset filter state."""
        self._prev_hist = None
        self._low_object_streak = 0
        self._cut_cooldown = 0
