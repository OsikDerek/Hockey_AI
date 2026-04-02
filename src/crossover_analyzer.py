"""Crossover detection and quality analysis.

Detects crossover events from landmark trajectories and evaluates them
against coaching criteria:

  1. KNEE DRIVE: Knee and toe should lead the crossover. The crossing leg's
     knee must move laterally past the stance leg's knee BEFORE the foot
     crosses over.

  2. INTERNAL ROTATION: The crossing leg (including foot) should internally
     rotate slightly before the foot actually crosses over the other foot.
     Detected by tracking ankle/knee lateral displacement relative to hip.

  3. STEP-OUT EXPLOSIVENESS: After the crossover, the step-out should be
     powerful — measured by how quickly the stance leg extends.

Detection strategy:
  - Track left and right ankle x-positions over time
  - A crossover occurs when one ankle's x crosses the other's x
  - Direction of crossing determines which leg is crossing over
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.signal import savgol_filter


@dataclass
class CrossoverEvent:
    """A single crossover event."""

    frame_idx: int            # Frame where the crossover happens (ankles cross)
    crossing_leg: str         # 'left' or 'right' — the leg crossing OVER
    stance_leg: str           # The leg being crossed over

    # Quality metrics (None if not computable)
    knee_drive_score: Optional[float] = None     # 0-1, how much knee led the foot
    knee_led_frames: Optional[int] = None        # How many frames knee was ahead
    internal_rotation: Optional[float] = None    # Degrees of pre-rotation detected
    step_out_speed: Optional[float] = None       # Pixels/frame of post-crossover extension

    # Ratings
    knee_drive_rating: str = "unknown"      # good/warning/poor
    rotation_rating: str = "unknown"
    step_out_rating: str = "unknown"
    overall_rating: str = "unknown"

    # Frame window for annotation
    frame_start: int = 0     # A few frames before crossover
    frame_end: int = 0       # A few frames after crossover

    @property
    def feedback(self) -> list[str]:
        """Generate coaching feedback strings."""
        msgs = []
        if self.knee_drive_rating == "poor":
            msgs.append("Foot is leading the crossover — drive the KNEE first")
        elif self.knee_drive_rating == "warning":
            msgs.append("Knee drive could be more aggressive — lead with the knee")
        elif self.knee_drive_rating == "good":
            msgs.append("Good knee drive — knee leading the crossover")

        if self.rotation_rating == "poor":
            msgs.append("No internal rotation before crossover — rotate the leg inward first")
        elif self.rotation_rating == "warning":
            msgs.append("Slight rotation detected — more internal rotation before crossing")

        if self.step_out_rating == "poor":
            msgs.append("Weak step-out — explode out of the crossover")
        elif self.step_out_rating == "warning":
            msgs.append("Step-out could be more explosive")

        return msgs


@dataclass
class CrossoverAnalysis:
    """Complete crossover analysis for a video."""

    events: list[CrossoverEvent]
    fps: float

    @property
    def total_crossovers(self) -> int:
        return len(self.events)

    @property
    def left_over_right(self) -> list[CrossoverEvent]:
        return [e for e in self.events if e.crossing_leg == "left"]

    @property
    def right_over_left(self) -> list[CrossoverEvent]:
        return [e for e in self.events if e.crossing_leg == "right"]

    def avg_knee_drive_score(self) -> Optional[float]:
        scores = [e.knee_drive_score for e in self.events if e.knee_drive_score is not None]
        return float(np.mean(scores)) if scores else None

    def events_at_frame(self, frame_idx: int) -> list[CrossoverEvent]:
        """Get any crossover events active at this frame."""
        return [e for e in self.events
                if e.frame_start <= frame_idx <= e.frame_end]


class CrossoverDetector:
    """Detects and evaluates crossover events from landmark data.

    Collects per-frame landmark positions, then analyzes the full
    trajectory to find crossover events and evaluate their quality.
    """

    def __init__(
        self,
        min_crossing_distance: float = 5.0,
        pre_window: int = 10,
        post_window: int = 10,
        smoothing_window: int = 5,
    ):
        """Initialize crossover detector.

        Args:
            min_crossing_distance: Minimum x-distance between ankles to
                consider them "separated" (avoids noise at near-overlap).
            pre_window: Frames before crossover to analyze for knee drive/rotation.
            post_window: Frames after crossover to analyze for step-out.
            smoothing_window: Savgol filter window for trajectory smoothing.
        """
        self.min_crossing_distance = min_crossing_distance
        self.pre_window = pre_window
        self.post_window = post_window
        self.smoothing_window = smoothing_window

        # Per-frame landmark positions
        self._frames: list[Optional[dict]] = []

    def add_frame(self, landmarks: Optional[dict]) -> None:
        """Record landmarks for one frame."""
        self._frames.append(landmarks)

    def analyze(self, fps: float) -> CrossoverAnalysis:
        """Analyze all accumulated frames for crossover events.

        Call after all frames have been processed via add_frame().
        """
        n = len(self._frames)
        if n < self.pre_window + self.post_window + 5:
            return CrossoverAnalysis(events=[], fps=fps)

        # Extract ankle x-positions
        left_ankle_x = np.full(n, np.nan)
        right_ankle_x = np.full(n, np.nan)
        left_knee_x = np.full(n, np.nan)
        right_knee_x = np.full(n, np.nan)
        left_hip_x = np.full(n, np.nan)
        right_hip_x = np.full(n, np.nan)
        left_knee_y = np.full(n, np.nan)
        right_knee_y = np.full(n, np.nan)

        for i, lm in enumerate(self._frames):
            if lm is None:
                continue
            for name, arr_x in [
                ("left_ankle", left_ankle_x), ("right_ankle", right_ankle_x),
                ("left_knee", left_knee_x), ("right_knee", right_knee_x),
                ("left_hip", left_hip_x), ("right_hip", right_hip_x),
            ]:
                if name in lm and lm[name]["visibility"] > 0.4:
                    arr_x[i] = lm[name]["x"]

            for name, arr_y in [
                ("left_knee", left_knee_y), ("right_knee", right_knee_y),
            ]:
                if name in lm and lm[name]["visibility"] > 0.4:
                    arr_y[i] = lm[name]["y"]

        # Interpolate gaps
        for arr in [left_ankle_x, right_ankle_x, left_knee_x, right_knee_x,
                    left_hip_x, right_hip_x, left_knee_y, right_knee_y]:
            self._interpolate_inplace(arr)

        # Smooth trajectories
        for arr in [left_ankle_x, right_ankle_x, left_knee_x, right_knee_x]:
            self._smooth_inplace(arr)

        # Detect crossover events: when left and right ankle x-positions swap
        # The difference: left_ankle_x - right_ankle_x
        # When this changes sign, ankles have crossed
        diff = left_ankle_x - right_ankle_x
        events = []

        for i in range(1, n):
            if np.isnan(diff[i]) or np.isnan(diff[i - 1]):
                continue

            # Sign change = crossover
            if diff[i - 1] * diff[i] < 0 and abs(diff[i - 1]) > self.min_crossing_distance:
                # Determine which leg crossed over
                # If left ankle moved from left-of-right to right-of-right → left crossed over
                if diff[i - 1] < 0 and diff[i] > 0:
                    # Left ankle was to the right (lower x in many views),
                    # now to the left → depends on camera angle
                    # Use: if left ankle x increased past right → left leg crossed over right
                    crossing_leg = "left"
                    stance_leg = "right"
                else:
                    crossing_leg = "right"
                    stance_leg = "left"

                frame_start = max(0, i - self.pre_window)
                frame_end = min(n - 1, i + self.post_window)

                event = CrossoverEvent(
                    frame_idx=i,
                    crossing_leg=crossing_leg,
                    stance_leg=stance_leg,
                    frame_start=frame_start,
                    frame_end=frame_end,
                )

                # Evaluate quality
                self._evaluate_knee_drive(
                    event, i,
                    left_ankle_x, right_ankle_x,
                    left_knee_x, right_knee_x,
                )
                self._evaluate_internal_rotation(
                    event, i,
                    left_ankle_x, right_ankle_x,
                    left_knee_x, right_knee_x,
                    left_hip_x, right_hip_x,
                )
                self._evaluate_step_out(
                    event, i,
                    left_knee_y, right_knee_y,
                )
                self._compute_overall_rating(event)

                events.append(event)

        # Filter out duplicate crossovers that are too close together
        events = self._deduplicate(events)

        return CrossoverAnalysis(events=events, fps=fps)

    def _evaluate_knee_drive(
        self, event: CrossoverEvent, cross_frame: int,
        left_ankle_x, right_ankle_x, left_knee_x, right_knee_x,
    ):
        """Evaluate whether the knee led the crossover.

        Good technique: knee of the crossing leg should move laterally
        past the stance leg's knee BEFORE the foot crosses over.
        """
        cl = event.crossing_leg  # 'left' or 'right'

        if cl == "left":
            cross_knee_x = left_knee_x
            stance_knee_x = right_knee_x
            cross_ankle_x = left_ankle_x
            stance_ankle_x = right_ankle_x
        else:
            cross_knee_x = right_knee_x
            stance_knee_x = left_knee_x
            cross_ankle_x = right_ankle_x
            stance_ankle_x = left_ankle_x

        # Check frames before the crossover
        # Count how many frames the crossing knee was past the stance knee
        # BEFORE the crossing ankle passed the stance ankle
        start = max(0, cross_frame - self.pre_window)
        knee_led_count = 0

        for f in range(start, cross_frame):
            if np.isnan(cross_knee_x[f]) or np.isnan(stance_knee_x[f]):
                continue
            knee_diff = cross_knee_x[f] - stance_knee_x[f]
            ankle_diff = cross_ankle_x[f] - stance_ankle_x[f]

            # Knee has crossed but ankle hasn't yet → knee is leading
            # The sign depends on direction, so check if knee is "more crossed" than ankle
            if abs(knee_diff) < abs(ankle_diff) * 0.7:
                # Knee is closer to crossing or past it relative to ankle
                knee_led_count += 1

        total_frames = cross_frame - start
        if total_frames > 0:
            score = knee_led_count / total_frames
            event.knee_drive_score = score
            event.knee_led_frames = knee_led_count

            if score >= 0.4:
                event.knee_drive_rating = "good"
            elif score >= 0.15:
                event.knee_drive_rating = "warning"
            else:
                event.knee_drive_rating = "poor"

    def _evaluate_internal_rotation(
        self, event: CrossoverEvent, cross_frame: int,
        left_ankle_x, right_ankle_x, left_knee_x, right_knee_x,
        left_hip_x, right_hip_x,
    ):
        """Evaluate internal rotation before crossover.

        Good technique: the crossing leg internally rotates (ankle moves
        inward relative to knee-hip line) before the foot crosses over.

        We measure this as the ankle's lateral offset from the knee,
        relative to the hip-knee line, in the frames before crossing.
        """
        cl = event.crossing_leg

        if cl == "left":
            ankle_x = left_ankle_x
            knee_x = left_knee_x
            hip_x = left_hip_x
        else:
            ankle_x = right_ankle_x
            knee_x = right_knee_x
            hip_x = right_hip_x

        start = max(0, cross_frame - self.pre_window)

        # Track ankle offset from knee-hip line over pre-crossover window
        offsets = []
        for f in range(start, cross_frame):
            if np.isnan(ankle_x[f]) or np.isnan(knee_x[f]) or np.isnan(hip_x[f]):
                continue
            # How much is ankle deviating inward from knee position?
            # Internal rotation moves the ankle toward the body midline
            hip_knee_dir = knee_x[f] - hip_x[f]  # Direction of leg
            ankle_offset = ankle_x[f] - knee_x[f]  # Ankle deviation from knee

            # If ankle is moving opposite to the leg direction, it's rotating inward
            if abs(hip_knee_dir) > 1:
                # Normalized rotation: negative = internal rotation
                rotation = ankle_offset / abs(hip_knee_dir)
                offsets.append(rotation)

        if len(offsets) >= 3:
            # Check if rotation increased (became more negative/inward) before crossover
            early = np.mean(offsets[:len(offsets)//3])
            late = np.mean(offsets[2*len(offsets)//3:])
            rotation_change = early - late  # Positive = ankle moved inward

            event.internal_rotation = float(rotation_change * 45)  # Rough degree estimate

            if rotation_change > 0.15:
                event.rotation_rating = "good"
            elif rotation_change > 0.05:
                event.rotation_rating = "warning"
            else:
                event.rotation_rating = "poor"

    def _evaluate_step_out(
        self, event: CrossoverEvent, cross_frame: int,
        left_knee_y, right_knee_y,
    ):
        """Evaluate step-out explosiveness after crossover.

        Good technique: after crossing over, the stance leg pushes off
        powerfully. We measure this by how quickly the knee extends
        (y-position drops = leg straightening in image coordinates).
        """
        sl = event.stance_leg  # Stance leg does the push-out

        knee_y = left_knee_y if sl == "left" else right_knee_y

        end = min(len(knee_y) - 1, cross_frame + self.post_window)

        # Measure knee y-velocity after crossover (speed of extension)
        positions = []
        for f in range(cross_frame, end + 1):
            if not np.isnan(knee_y[f]):
                positions.append(knee_y[f])

        if len(positions) >= 3:
            # Speed = max displacement over the post-crossover window
            velocities = np.diff(positions)
            max_speed = float(np.max(np.abs(velocities)))
            event.step_out_speed = max_speed

            if max_speed > 8.0:
                event.step_out_rating = "good"
            elif max_speed > 4.0:
                event.step_out_rating = "warning"
            else:
                event.step_out_rating = "poor"

    def _compute_overall_rating(self, event: CrossoverEvent):
        """Compute overall crossover rating from component ratings."""
        ratings = [event.knee_drive_rating, event.rotation_rating, event.step_out_rating]
        known = [r for r in ratings if r != "unknown"]

        if not known:
            event.overall_rating = "unknown"
        elif "poor" in known:
            event.overall_rating = "poor"
        elif "warning" in known:
            event.overall_rating = "warning"
        else:
            event.overall_rating = "good"

    def _deduplicate(self, events: list[CrossoverEvent]) -> list[CrossoverEvent]:
        """Remove crossover events that are too close together (likely noise)."""
        if len(events) <= 1:
            return events

        filtered = [events[0]]
        for e in events[1:]:
            if e.frame_idx - filtered[-1].frame_idx >= 5:
                filtered.append(e)
        return filtered

    def _interpolate_inplace(self, arr: np.ndarray):
        """Linearly interpolate NaN values in place."""
        valid = ~np.isnan(arr)
        if valid.sum() < 2 or valid.all():
            return
        indices = np.arange(len(arr))
        arr[:] = np.interp(indices, indices[valid], arr[valid])

    def _smooth_inplace(self, arr: np.ndarray):
        """Apply Savitzky-Golay smoothing in place."""
        valid = ~np.isnan(arr)
        if valid.sum() < self.smoothing_window + 2:
            return
        window = min(self.smoothing_window, len(arr))
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            arr[:] = savgol_filter(arr, window, polyorder=2)
