"""Detector registry.

Maps detector names (from technique YAML) to Python classes.
To add a new detector: write the class, import it here, add to REGISTRY.
"""

from src.detectors.frame_by_frame import FrameByFrameDetector
from src.detectors.crossover_detector import CrossoverDetector

REGISTRY = {
    "FrameByFrame": FrameByFrameDetector,
    "CrossoverDetector": CrossoverDetector,
}


def create_detector(name: str, params: dict = None):
    """Create a detector instance by name with optional params.

    Args:
        name: Detector class name (must be in REGISTRY).
        params: Kwargs to pass to the detector's __init__.

    Returns:
        Detector instance.
    """
    if name not in REGISTRY:
        available = ", ".join(REGISTRY.keys())
        raise ValueError(f"Unknown detector '{name}'. Available: {available}")

    cls = REGISTRY[name]
    return cls(**(params or {}))
