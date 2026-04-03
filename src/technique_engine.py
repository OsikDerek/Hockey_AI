"""Technique evaluation engine.

Loads a technique YAML file and orchestrates:
  1. Detector creation (from the YAML's detection section)
  2. Metric classification (good/warning/poor using YAML thresholds)
  3. Feedback and drill lookup
  4. Annotation config resolution

This replaces the old MechanicsEngine for new technique-based analysis.
"""

import yaml
from dataclasses import dataclass, field
from typing import Optional

from src.detectors import create_detector


@dataclass
class CheckResult:
    """Result of evaluating one check on one event."""
    check_name: str
    display_name: str
    metric_value: Optional[float]
    rating: str  # good, warning, poor, unknown
    feedback: str
    drills: list[dict] = field(default_factory=list)


@dataclass
class TechniqueEvent:
    """A detected technique event with classified check results."""
    frame_idx: int
    frame_start: int
    frame_end: int
    context: dict           # Extra context (crossing_leg, stance_leg, etc.)
    check_results: list[CheckResult] = field(default_factory=list)
    overall_rating: str = "unknown"

    @property
    def feedback(self) -> list[str]:
        """All non-good feedback messages."""
        return [r.feedback for r in self.check_results if r.rating != "good"]


@dataclass
class TechniqueAnalysis:
    """Complete analysis for a technique across a video."""
    technique_name: str
    display_name: str
    events: list[TechniqueEvent]
    fps: float
    coaching_notes: list[str] = field(default_factory=list)

    def events_at_frame(self, frame_idx: int) -> list[TechniqueEvent]:
        """Get events active at a specific frame."""
        return [e for e in self.events
                if e.frame_start <= frame_idx <= e.frame_end]


class TechniqueEngine:
    """Loads technique YAML and orchestrates detection + evaluation."""

    def __init__(self, technique_path: str):
        with open(technique_path, "r") as f:
            raw = yaml.safe_load(f)

        self.config = raw["technique"]
        self.name = self.config["name"]
        self.display_name = self.config["display_name"]
        self.checks = self.config.get("checks", {})
        self.stride_checks = self.config.get("stride_checks", {})
        self.coaching_notes = self.config.get("coaching_notes", [])
        self.annotation_config = self.config.get("annotation", {})

        # Create detector
        det_config = self.config["detection"]
        self.detector = create_detector(
            det_config["detector"],
            det_config.get("params", {}),
        )
        self.detector_name = det_config["detector"]

    @property
    def is_frame_by_frame(self) -> bool:
        return self.detector_name == "FrameByFrame"

    def add_frame(self, landmarks: Optional[dict] = None, angles: Optional[dict] = None) -> None:
        """Pass frame data to the detector."""
        self.detector.add_frame(landmarks=landmarks, angles=angles)

    def analyze(self, fps: float) -> TechniqueAnalysis:
        """Run detection and classify all events."""
        raw_events = self.detector.analyze(fps=fps)

        technique_events = []
        for raw in raw_events:
            if self.is_frame_by_frame:
                event = self._classify_frame_event(raw)
            else:
                event = self._classify_temporal_event(raw)
            technique_events.append(event)

        return TechniqueAnalysis(
            technique_name=self.name,
            display_name=self.display_name,
            events=technique_events,
            fps=fps,
            coaching_notes=self.coaching_notes,
        )

    def _classify_temporal_event(self, raw_event: dict) -> TechniqueEvent:
        """Classify a temporal event (crossover, shot, etc.) using YAML thresholds."""
        context = {k: v for k, v in raw_event.items()
                   if k not in ("frame_idx", "frame_start", "frame_end")}

        check_results = []
        for check_name, check_config in self.checks.items():
            metric_name = check_config["metric"]
            value = raw_event.get(metric_name)

            if value is not None:
                rating, feedback = self._classify_value(value, check_config)
                drills = self._get_drills(check_config, rating)
            else:
                rating = "unknown"
                feedback = ""
                drills = []

            check_results.append(CheckResult(
                check_name=check_name,
                display_name=check_config["display_name"],
                metric_value=value,
                rating=rating,
                feedback=feedback,
                drills=drills,
            ))

        overall = self._compute_overall(check_results)

        return TechniqueEvent(
            frame_idx=raw_event["frame_idx"],
            frame_start=raw_event["frame_start"],
            frame_end=raw_event["frame_end"],
            context=context,
            check_results=check_results,
            overall_rating=overall,
        )

    def _classify_frame_event(self, raw_event: dict) -> TechniqueEvent:
        """Classify a frame-by-frame event using per-frame angle checks."""
        angles = raw_event.get("angles", {})
        check_results = []

        for check_name, check_config in self.checks.items():
            per_side = check_config.get("per_side", False)
            sides_config = self.config.get("detection", {}).get("params", {}).get("analyze_sides", "both")

            if per_side:
                sides = ["left", "right"] if sides_config == "both" else [sides_config]
                for side in sides:
                    angle_key = f"{side}_{check_name}"
                    value = angles.get(angle_key)

                    if value is not None:
                        rating, feedback = self._classify_value(value, check_config)
                        drills = self._get_drills(check_config, rating)
                    else:
                        continue  # Skip missing angles

                    check_results.append(CheckResult(
                        check_name=f"{check_name}_{side}",
                        display_name=f"{check_config['display_name']} ({side[0].upper()})",
                        metric_value=value,
                        rating=rating,
                        feedback=feedback,
                        drills=drills,
                    ))
            else:
                # Bilateral metric (e.g., trunk_alignment, forward_lean)
                value = angles.get(check_name)
                if value is None:
                    continue

                rating, feedback = self._classify_value(value, check_config)
                drills = self._get_drills(check_config, rating)

                check_results.append(CheckResult(
                    check_name=check_name,
                    display_name=check_config["display_name"],
                    metric_value=value,
                    rating=rating,
                    feedback=feedback,
                    drills=drills,
                ))

        overall = self._compute_overall(check_results)

        return TechniqueEvent(
            frame_idx=raw_event["frame_idx"],
            frame_start=raw_event["frame_start"],
            frame_end=raw_event["frame_end"],
            context={"angles": angles, "landmarks": raw_event.get("landmarks")},
            check_results=check_results,
            overall_rating=overall,
        )

    def _classify_value(self, value: float, check_config: dict) -> tuple[str, str]:
        """Classify a metric value into good/warning/poor using YAML ranges."""
        good_min, good_max = check_config["good_range"]
        warn_min, warn_max = check_config["warning_range"]
        feedback = check_config["feedback"]

        if good_min <= value <= good_max:
            return "good", feedback["good"]
        elif warn_min <= value <= warn_max:
            return "warning", feedback["warning"]
        else:
            return "poor", feedback["poor"]

    def _get_drills(self, check_config: dict, rating: str) -> list[dict]:
        """Get drill recommendations for a given rating."""
        drills_section = check_config.get("drills", {})
        if rating in drills_section:
            return drills_section[rating]
        return []

    def _compute_overall(self, results: list[CheckResult]) -> str:
        """Compute overall rating from individual check results."""
        known = [r.rating for r in results if r.rating != "unknown"]
        if not known:
            return "unknown"
        if "poor" in known:
            return "poor"
        if "warning" in known:
            return "warning"
        return "good"

    def get_check_annotation(self, check_name: str) -> Optional[dict]:
        """Get annotation config for a specific check."""
        base_name = check_name.rsplit("_", 1)[0] if check_name.endswith(("_left", "_right")) else check_name
        check_config = self.checks.get(base_name, {})
        return check_config.get("annotation")

    def get_banner_template(self) -> Optional[str]:
        """Get the banner template string."""
        return self.annotation_config.get("banner_template")
