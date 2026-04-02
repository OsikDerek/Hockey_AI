"""Stride phase detection from angle time series.

Detects individual skating strides and their phases (push-off, glide, recovery)
by analyzing knee angle patterns over time. Provides per-stride metrics
instead of noisy per-frame readings.

A skating stride cycle:
  1. PUSH-OFF: Driving leg extends (knee angle increases toward ~160-175°)
  2. GLIDE:    Weight on stance leg, knee moderately bent (~100-130°)
  3. RECOVERY: Stride leg returns under body (knee flexes, angle decreases)

We detect strides by finding peaks (max extension = push-off completion)
and valleys (max flexion = deepest knee bend) in the knee angle signal.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.signal import find_peaks, savgol_filter


@dataclass
class StridePhase:
    """A single phase within a stride."""

    phase: str             # 'push_off', 'glide', or 'recovery'
    frame_start: int
    frame_end: int
    peak_angle: float      # Max angle during this phase
    min_angle: float       # Min angle during this phase


@dataclass
class Stride:
    """A complete stride cycle from one push-off to the next."""

    side: str              # 'left' or 'right'
    frame_start: int
    frame_end: int
    push_off_angle: float  # Knee angle at max extension
    glide_angle: float     # Knee angle during glide (deepest bend)
    duration_frames: int
    phases: list[StridePhase] = field(default_factory=list)

    @property
    def extension_range(self) -> float:
        """Range of motion: push-off angle minus glide angle."""
        return self.push_off_angle - self.glide_angle


@dataclass
class StrideAnalysis:
    """Complete stride analysis for a video."""

    left_strides: list[Stride]
    right_strides: list[Stride]
    fps: float

    @property
    def total_strides(self) -> int:
        return len(self.left_strides) + len(self.right_strides)

    @property
    def avg_stride_duration_sec(self) -> Optional[float]:
        all_strides = self.left_strides + self.right_strides
        if not all_strides:
            return None
        return np.mean([s.duration_frames for s in all_strides]) / self.fps

    @property
    def symmetry_ratio(self) -> Optional[float]:
        """L/R symmetry ratio (1.0 = perfect symmetry)."""
        if not self.left_strides or not self.right_strides:
            return None
        left_avg = np.mean([s.push_off_angle for s in self.left_strides])
        right_avg = np.mean([s.push_off_angle for s in self.right_strides])
        return min(left_avg, right_avg) / max(left_avg, right_avg)


class StrideDetector:
    """Detects stride cycles from per-frame angle data.

    Collects knee angle values frame-by-frame, then analyzes the full
    time series to find stride boundaries and phases.
    """

    def __init__(
        self,
        min_stride_frames: int = 8,
        max_stride_frames: int = 90,
        smoothing_window: int = 7,
        peak_prominence: float = 15.0,
    ):
        """Initialize stride detector.

        Args:
            min_stride_frames: Minimum frames for a valid stride (~0.25s at 30fps).
            max_stride_frames: Maximum frames for a valid stride (~3s at 30fps).
            smoothing_window: Savitzky-Golay filter window for noise reduction.
                Must be odd. Larger = smoother but may miss fast strides.
            peak_prominence: Minimum angle change (degrees) to count as a stride.
                Lower = more sensitive, higher = fewer false positives.
        """
        self.min_stride_frames = min_stride_frames
        self.max_stride_frames = max_stride_frames
        self.smoothing_window = smoothing_window
        self.peak_prominence = peak_prominence

        # Accumulate per-frame data
        self._left_knee_angles: list[Optional[float]] = []
        self._right_knee_angles: list[Optional[float]] = []
        self._frame_count = 0

    def add_frame(self, angles: dict) -> None:
        """Record angles for one frame.

        Args:
            angles: Dict from angle_calculator.compute_all_angles().
        """
        self._left_knee_angles.append(angles.get("left_knee_angle"))
        self._right_knee_angles.append(angles.get("right_knee_angle"))
        self._frame_count += 1

    def analyze(self, fps: float) -> StrideAnalysis:
        """Analyze accumulated angle data to detect strides.

        Call this after all frames have been processed via add_frame().

        Args:
            fps: Video frame rate (for timing calculations).

        Returns:
            StrideAnalysis with detected strides for both legs.
        """
        left_strides = self._detect_strides(self._left_knee_angles, "left")
        right_strides = self._detect_strides(self._right_knee_angles, "right")

        return StrideAnalysis(
            left_strides=left_strides,
            right_strides=right_strides,
            fps=fps,
        )

    def _detect_strides(
        self, raw_angles: list[Optional[float]], side: str
    ) -> list[Stride]:
        """Detect strides from a single leg's knee angle time series.

        Strategy:
          1. Interpolate missing values, smooth the signal
          2. Find peaks (max extension = push-off completion)
          3. Find valleys between peaks (max flexion = glide/load)
          4. Each peak-to-peak interval is one stride cycle
        """
        # Convert to array, interpolating None values
        angles = self._interpolate_missing(raw_angles)
        if angles is None or len(angles) < self.smoothing_window + 2:
            return []

        # Smooth to remove jitter
        window = min(self.smoothing_window, len(angles))
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            smoothed = savgol_filter(angles, window, polyorder=2)
        else:
            smoothed = angles

        # Find peaks (max extension = push-off) and valleys (max flexion)
        peaks, peak_props = find_peaks(
            smoothed,
            distance=self.min_stride_frames,
            prominence=self.peak_prominence,
        )

        if len(peaks) < 2:
            return []

        # Find the valley (minimum) between consecutive peaks
        strides = []
        for i in range(len(peaks) - 1):
            start_frame = int(peaks[i])
            end_frame = int(peaks[i + 1])
            duration = end_frame - start_frame

            # Skip if duration is outside valid range
            if duration < self.min_stride_frames or duration > self.max_stride_frames:
                continue

            segment = smoothed[start_frame:end_frame + 1]
            valley_offset = int(np.argmin(segment))
            valley_frame = start_frame + valley_offset

            push_off_angle = float(smoothed[start_frame])
            glide_angle = float(smoothed[valley_frame])

            # Build phases
            phases = [
                StridePhase(
                    phase="recovery",
                    frame_start=start_frame,
                    frame_end=valley_frame,
                    peak_angle=push_off_angle,
                    min_angle=glide_angle,
                ),
                StridePhase(
                    phase="push_off",
                    frame_start=valley_frame,
                    frame_end=end_frame,
                    peak_angle=float(smoothed[end_frame]),
                    min_angle=glide_angle,
                ),
            ]

            strides.append(Stride(
                side=side,
                frame_start=start_frame,
                frame_end=end_frame,
                push_off_angle=push_off_angle,
                glide_angle=glide_angle,
                duration_frames=duration,
                phases=phases,
            ))

        return strides

    def _interpolate_missing(
        self, raw: list[Optional[float]]
    ) -> Optional[np.ndarray]:
        """Interpolate None values in the angle series.

        Returns None if too few valid values (<50% of frames).
        """
        arr = np.array([v if v is not None else np.nan for v in raw], dtype=np.float64)
        valid_mask = ~np.isnan(arr)

        if valid_mask.sum() < len(arr) * 0.5:
            return None

        if valid_mask.all():
            return arr

        # Linear interpolation for gaps
        valid_indices = np.where(valid_mask)[0]
        arr = np.interp(
            np.arange(len(arr)),
            valid_indices,
            arr[valid_indices],
        )
        return arr

    def get_phase_at_frame(
        self, analysis: StrideAnalysis, frame_idx: int
    ) -> Optional[tuple[str, str]]:
        """Get the current stride phase at a given frame.

        Args:
            analysis: Result from analyze().
            frame_idx: Frame number to query.

        Returns:
            Tuple of (side, phase_name) or None if not in a detected stride.
        """
        for stride in analysis.left_strides + analysis.right_strides:
            for phase in stride.phases:
                if phase.frame_start <= frame_idx <= phase.frame_end:
                    return (stride.side, phase.phase)
        return None
