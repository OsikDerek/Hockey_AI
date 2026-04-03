"""Crossover detection — computes raw metrics only.

Detection logic (trajectory analysis, signal processing) lives here.
Thresholds and feedback live in knowledge_base/techniques/crossover.yaml.
The TechniqueEngine handles classification using the YAML config.

Detection strategy:
  - Track left and right ankle x-positions over time
  - A crossover occurs when one ankle's x crosses the other's x
  - For each event, compute raw metric values:
    - knee_drive_score: 0-1, how much the knee led the foot
    - rotation_change: how much internal rotation occurred before crossing
    - step_out_speed: pixels/frame of post-crossover extension
"""

import numpy as np
from typing import Optional
from scipy.signal import savgol_filter


class CrossoverDetector:
    """Detects crossover events and computes raw metrics.

    Thresholds are NOT applied here — raw metric values are returned
    and the TechniqueEngine classifies them using the YAML config.
    """

    def __init__(
        self,
        min_crossing_distance: float = 5.0,
        pre_window: int = 10,
        post_window: int = 10,
        smoothing_window: int = 5,
        dedup_min_frames: int = 5,
        **kwargs,
    ):
        self.min_crossing_distance = min_crossing_distance
        self.pre_window = pre_window
        self.post_window = post_window
        self.smoothing_window = smoothing_window
        self.dedup_min_frames = dedup_min_frames
        self._frames: list[Optional[dict]] = []

    def add_frame(self, landmarks: Optional[dict] = None, angles: Optional[dict] = None) -> None:
        """Record landmarks for one frame."""
        self._frames.append(landmarks)

    def analyze(self, fps: float) -> list[dict]:
        """Analyze all frames and return crossover events with raw metrics."""
        n = len(self._frames)
        if n < self.pre_window + self.post_window + 5:
            return []

        # Extract position arrays
        arrays = self._extract_positions(n)
        left_ankle_x, right_ankle_x = arrays["left_ankle_x"], arrays["right_ankle_x"]
        left_knee_x, right_knee_x = arrays["left_knee_x"], arrays["right_knee_x"]
        left_hip_x, right_hip_x = arrays["left_hip_x"], arrays["right_hip_x"]
        left_knee_y, right_knee_y = arrays["left_knee_y"], arrays["right_knee_y"]

        # Interpolate and smooth
        for arr in arrays.values():
            self._interpolate_inplace(arr)
        for key in ["left_ankle_x", "right_ankle_x", "left_knee_x", "right_knee_x"]:
            self._smooth_inplace(arrays[key])

        # Detect crossover events
        diff = left_ankle_x - right_ankle_x
        events = []

        for i in range(1, n):
            if np.isnan(diff[i]) or np.isnan(diff[i - 1]):
                continue

            if diff[i - 1] * diff[i] < 0 and abs(diff[i - 1]) > self.min_crossing_distance:
                if diff[i - 1] < 0 and diff[i] > 0:
                    crossing_leg, stance_leg = "left", "right"
                else:
                    crossing_leg, stance_leg = "right", "left"

                frame_start = max(0, i - self.pre_window)
                frame_end = min(n - 1, i + self.post_window)

                event = {
                    "frame_idx": i,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "crossing_leg": crossing_leg,
                    "stance_leg": stance_leg,
                }

                # Compute raw metrics
                event["knee_drive_score"] = self._compute_knee_drive(
                    i, crossing_leg, left_ankle_x, right_ankle_x,
                    left_knee_x, right_knee_x,
                )
                event["rotation_change"] = self._compute_rotation(
                    i, crossing_leg, left_ankle_x, right_ankle_x,
                    left_knee_x, right_knee_x, left_hip_x, right_hip_x,
                )
                event["step_out_speed"] = self._compute_step_out(
                    i, stance_leg, left_knee_y, right_knee_y,
                )

                events.append(event)

        return self._deduplicate(events)

    def _extract_positions(self, n: int) -> dict[str, np.ndarray]:
        """Extract landmark position arrays from frame data."""
        names = {
            "left_ankle_x": ("left_ankle", "x"),
            "right_ankle_x": ("right_ankle", "x"),
            "left_knee_x": ("left_knee", "x"),
            "right_knee_x": ("right_knee", "x"),
            "left_hip_x": ("left_hip", "x"),
            "right_hip_x": ("right_hip", "x"),
            "left_knee_y": ("left_knee", "y"),
            "right_knee_y": ("right_knee", "y"),
        }

        arrays = {k: np.full(n, np.nan) for k in names}

        for i, lm in enumerate(self._frames):
            if lm is None:
                continue
            for arr_name, (lm_name, coord) in names.items():
                if lm_name in lm and lm[lm_name]["visibility"] > 0.4:
                    arrays[arr_name][i] = lm[lm_name][coord]

        return arrays

    def _compute_knee_drive(
        self, cross_frame: int, crossing_leg: str,
        left_ankle_x, right_ankle_x, left_knee_x, right_knee_x,
    ) -> Optional[float]:
        """Compute knee drive score (0-1). Higher = knee led more."""
        if crossing_leg == "left":
            cross_knee_x, stance_knee_x = left_knee_x, right_knee_x
            cross_ankle_x, stance_ankle_x = left_ankle_x, right_ankle_x
        else:
            cross_knee_x, stance_knee_x = right_knee_x, left_knee_x
            cross_ankle_x, stance_ankle_x = right_ankle_x, left_ankle_x

        start = max(0, cross_frame - self.pre_window)
        knee_led_count = 0

        for f in range(start, cross_frame):
            if np.isnan(cross_knee_x[f]) or np.isnan(stance_knee_x[f]):
                continue
            knee_diff = cross_knee_x[f] - stance_knee_x[f]
            ankle_diff = cross_ankle_x[f] - stance_ankle_x[f]

            if abs(knee_diff) < abs(ankle_diff) * 0.7:
                knee_led_count += 1

        total_frames = cross_frame - start
        if total_frames > 0:
            return knee_led_count / total_frames
        return None

    def _compute_rotation(
        self, cross_frame: int, crossing_leg: str,
        left_ankle_x, right_ankle_x, left_knee_x, right_knee_x,
        left_hip_x, right_hip_x,
    ) -> Optional[float]:
        """Compute internal rotation change. Higher = more rotation before crossing."""
        if crossing_leg == "left":
            ankle_x, knee_x, hip_x = left_ankle_x, left_knee_x, left_hip_x
        else:
            ankle_x, knee_x, hip_x = right_ankle_x, right_knee_x, right_hip_x

        start = max(0, cross_frame - self.pre_window)
        offsets = []

        for f in range(start, cross_frame):
            if np.isnan(ankle_x[f]) or np.isnan(knee_x[f]) or np.isnan(hip_x[f]):
                continue
            hip_knee_dir = knee_x[f] - hip_x[f]
            ankle_offset = ankle_x[f] - knee_x[f]

            if abs(hip_knee_dir) > 1:
                rotation = ankle_offset / abs(hip_knee_dir)
                offsets.append(rotation)

        if len(offsets) >= 3:
            early = np.mean(offsets[:len(offsets) // 3])
            late = np.mean(offsets[2 * len(offsets) // 3:])
            return float(early - late)
        return None

    def _compute_step_out(
        self, cross_frame: int, stance_leg: str,
        left_knee_y, right_knee_y,
    ) -> Optional[float]:
        """Compute step-out speed (pixels/frame). Higher = more explosive."""
        knee_y = left_knee_y if stance_leg == "left" else right_knee_y
        end = min(len(knee_y) - 1, cross_frame + self.post_window)

        positions = []
        for f in range(cross_frame, end + 1):
            if not np.isnan(knee_y[f]):
                positions.append(knee_y[f])

        if len(positions) >= 3:
            velocities = np.diff(positions)
            return float(np.max(np.abs(velocities)))
        return None

    def _deduplicate(self, events: list[dict]) -> list[dict]:
        if len(events) <= 1:
            return events
        filtered = [events[0]]
        for e in events[1:]:
            if e["frame_idx"] - filtered[-1]["frame_idx"] >= self.dedup_min_frames:
                filtered.append(e)
        return filtered

    def _interpolate_inplace(self, arr: np.ndarray):
        valid = ~np.isnan(arr)
        if valid.sum() < 2 or valid.all():
            return
        indices = np.arange(len(arr))
        arr[:] = np.interp(indices, indices[valid], arr[valid])

    def _smooth_inplace(self, arr: np.ndarray):
        valid = ~np.isnan(arr)
        if valid.sum() < self.smoothing_window + 2:
            return
        window = min(self.smoothing_window, len(arr))
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            arr[:] = savgol_filter(arr, window, polyorder=2)
