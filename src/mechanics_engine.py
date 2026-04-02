"""Mechanics evaluation engine.

Loads skating mechanics thresholds from YAML config and classifies
computed angles into good/warning/poor ratings with coaching feedback.

Supports both per-frame evaluation and per-stride evaluation.
"""

import yaml
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MechanicResult:
    """Result of evaluating a single mechanic."""

    name: str            # Config key (e.g., 'knee_angle')
    display_name: str    # Human-readable name (e.g., 'Knee Bend')
    value: float         # Measured angle in degrees
    rating: str          # 'good', 'warning', or 'poor'
    feedback: str        # Coaching feedback text
    side: Optional[str]  # 'left', 'right', or None for bilateral metrics


class MechanicsEngine:
    """Evaluates skating angles against configurable thresholds.

    Loads threshold ranges from a YAML config file so the coach
    can tune without modifying code.
    """

    def __init__(self, config_path: str = "config/skating_mechanics.yaml"):
        """Load mechanics configuration.

        Args:
            config_path: Path to the YAML config file.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.mechanics = self.config["mechanics"]
        self.stride_mechanics = self.config.get("stride_mechanics", {})
        self.min_visibility = self.config.get("min_visibility", 0.5)
        self.analyze_sides = self.config.get("analyze_sides", "both")

    def _classify_angle(
        self, value: float, mechanic_config: dict
    ) -> tuple[str, str]:
        """Classify an angle value into good/warning/poor.

        Args:
            value: Measured angle in degrees.
            mechanic_config: Config dict for this mechanic.

        Returns:
            Tuple of (rating, feedback_text).
        """
        good_min, good_max = mechanic_config["good_range"]
        warn_min, warn_max = mechanic_config["warning_range"]
        feedback = mechanic_config["feedback"]

        if good_min <= value <= good_max:
            return "good", feedback["good"]
        elif warn_min <= value <= warn_max:
            return "warning", feedback["warning"]
        else:
            return "poor", feedback["poor"]

    def evaluate(self, angles: dict) -> list[MechanicResult]:
        """Evaluate all computed angles against thresholds.

        Args:
            angles: Dict from angle_calculator.compute_all_angles().
                Keys like 'left_knee_angle', 'right_hip_angle', 'trunk_alignment'.

        Returns:
            List of MechanicResult objects for all evaluated mechanics.
        """
        results = []

        for mechanic_key, mechanic_config in self.mechanics.items():
            display_name = mechanic_config["display_name"]

            if mechanic_key == "trunk_alignment":
                # Bilateral metric — no side prefix
                value = angles.get("trunk_alignment")
                if value is not None:
                    rating, feedback = self._classify_angle(value, mechanic_config)
                    results.append(
                        MechanicResult(
                            name=mechanic_key,
                            display_name=display_name,
                            value=value,
                            rating=rating,
                            feedback=feedback,
                            side=None,
                        )
                    )
            else:
                # Per-side metrics
                sides = (
                    ["left", "right"]
                    if self.analyze_sides == "both"
                    else [self.analyze_sides]
                )
                for side in sides:
                    angle_key = f"{side}_{mechanic_key}"
                    value = angles.get(angle_key)
                    if value is not None:
                        rating, feedback = self._classify_angle(
                            value, mechanic_config
                        )
                        results.append(
                            MechanicResult(
                                name=mechanic_key,
                                display_name=f"{display_name} ({side[0].upper()})",
                                value=value,
                                rating=rating,
                                feedback=feedback,
                                side=side,
                            )
                        )

        return results

    def evaluate_stride(self, stride) -> list[MechanicResult]:
        """Evaluate a single stride against stride-level thresholds.

        Args:
            stride: A Stride object from stride_detector.

        Returns:
            List of MechanicResult for stride-level metrics.
        """
        results = []

        # Push-off extension
        config = self.stride_mechanics.get("push_off_extension")
        if config:
            rating, feedback = self._classify_angle(stride.push_off_angle, config)
            results.append(MechanicResult(
                name="push_off_extension",
                display_name=f"{config['display_name']} ({stride.side[0].upper()})",
                value=stride.push_off_angle,
                rating=rating,
                feedback=feedback,
                side=stride.side,
            ))

        # Glide knee bend
        config = self.stride_mechanics.get("glide_knee_bend")
        if config:
            rating, feedback = self._classify_angle(stride.glide_angle, config)
            results.append(MechanicResult(
                name="glide_knee_bend",
                display_name=f"{config['display_name']} ({stride.side[0].upper()})",
                value=stride.glide_angle,
                rating=rating,
                feedback=feedback,
                side=stride.side,
            ))

        # Stride range of motion
        config = self.stride_mechanics.get("stride_range_of_motion")
        if config:
            rom = stride.extension_range
            rating, feedback = self._classify_angle(rom, config)
            results.append(MechanicResult(
                name="stride_range_of_motion",
                display_name=f"{config['display_name']} ({stride.side[0].upper()})",
                value=rom,
                rating=rating,
                feedback=feedback,
                side=stride.side,
            ))

        return results

    def evaluate_session(self, stride_analysis) -> list[MechanicResult]:
        """Evaluate an entire session's stride analysis.

        Args:
            stride_analysis: A StrideAnalysis object from stride_detector.

        Returns:
            List of session-level MechanicResult objects.
        """
        results = []

        # Evaluate each stride
        all_strides = stride_analysis.left_strides + stride_analysis.right_strides
        for stride in all_strides:
            results.extend(self.evaluate_stride(stride))

        # Session-level symmetry
        config = self.stride_mechanics.get("symmetry")
        if config and stride_analysis.symmetry_ratio is not None:
            rating, feedback = self._classify_angle(
                stride_analysis.symmetry_ratio, config
            )
            results.append(MechanicResult(
                name="symmetry",
                display_name=config["display_name"],
                value=stride_analysis.symmetry_ratio,
                rating=rating,
                feedback=feedback,
                side=None,
            ))

        return results
