"""Biomechanical angle calculations for skating analysis.

All functions take 2D pixel coordinates (numpy arrays or tuples)
and return angles in degrees. Uses arctan2 for numerical stability.
"""

import numpy as np
from typing import Optional


def angle_between_three_points(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> float:
    """Calculate the angle at vertex b formed by segments ba and bc.

    Args:
        a: Point a as (x, y) array.
        b: Vertex point as (x, y) array.
        c: Point c as (x, y) array.

    Returns:
        Angle in degrees [0, 180].
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def _get_point(landmarks: dict, name: str) -> np.ndarray:
    """Extract (x, y) array from landmarks dict."""
    lm = landmarks[name]
    return np.array([lm["x"], lm["y"]], dtype=np.float64)


def _visibility(landmarks: dict, *names: str, threshold: float = 0.5) -> bool:
    """Check if all named landmarks exist and are visible above threshold."""
    return all(
        n in landmarks and landmarks[n]["visibility"] >= threshold
        for n in names
    )


def knee_angle(landmarks: dict, side: str = "left") -> Optional[float]:
    """Knee bend angle: angle at the knee joint.

    Full extension ~170-180°, deep power position ~95-125°.

    Args:
        landmarks: Dict of landmark positions from PoseEstimator.
        side: 'left' or 'right'.

    Returns:
        Angle in degrees, or None if landmarks not visible.
    """
    hip_name = f"{side}_hip"
    knee_name = f"{side}_knee"
    ankle_name = f"{side}_ankle"

    if not _visibility(landmarks, hip_name, knee_name, ankle_name):
        return None

    return angle_between_three_points(
        _get_point(landmarks, hip_name),
        _get_point(landmarks, knee_name),
        _get_point(landmarks, ankle_name),
    )


def hip_angle(landmarks: dict, side: str = "left") -> Optional[float]:
    """Hip hinge angle: angle at the hip joint.

    Indicates depth of hip hinge. Good skating: 70-110°.

    Args:
        landmarks: Dict of landmark positions from PoseEstimator.
        side: 'left' or 'right'.

    Returns:
        Angle in degrees, or None if landmarks not visible.
    """
    shoulder_name = f"{side}_shoulder"
    hip_name = f"{side}_hip"
    knee_name = f"{side}_knee"

    if not _visibility(landmarks, shoulder_name, hip_name, knee_name):
        return None

    return angle_between_three_points(
        _get_point(landmarks, shoulder_name),
        _get_point(landmarks, hip_name),
        _get_point(landmarks, knee_name),
    )


def forward_lean_angle(landmarks: dict, side: str = "left") -> Optional[float]:
    """Forward lean: angle of torso relative to vertical.

    0° = perfectly upright, positive = leaning forward.
    Good skating posture: 30-50° forward lean.

    Args:
        landmarks: Dict of landmark positions from PoseEstimator.
        side: 'left' or 'right'.

    Returns:
        Angle in degrees from vertical, or None if not visible.
    """
    shoulder_name = f"{side}_shoulder"
    hip_name = f"{side}_hip"

    if not _visibility(landmarks, shoulder_name, hip_name):
        return None

    shoulder = _get_point(landmarks, shoulder_name)
    hip = _get_point(landmarks, hip_name)

    # Torso vector (hip to shoulder)
    torso = shoulder - hip

    # Vertical reference (pointing up in image = negative y)
    vertical = np.array([0.0, -1.0])

    # Angle between torso and vertical
    cos_angle = np.dot(torso, vertical) / (np.linalg.norm(torso) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def ankle_dorsiflexion(landmarks: dict, side: str = "left") -> Optional[float]:
    """Ankle dorsiflexion: shin angle indicating edge engagement.

    Angle at the ankle between knee-ankle-toe.
    Good edge engagement: 70-95°.

    Args:
        landmarks: Dict of landmark positions from PoseEstimator.
        side: 'left' or 'right'.

    Returns:
        Angle in degrees, or None if not visible.
    """
    knee_name = f"{side}_knee"
    ankle_name = f"{side}_ankle"
    toe_name = f"{side}_foot_index"

    if not _visibility(landmarks, knee_name, ankle_name, toe_name):
        return None

    return angle_between_three_points(
        _get_point(landmarks, knee_name),
        _get_point(landmarks, ankle_name),
        _get_point(landmarks, toe_name),
    )


def trunk_alignment(landmarks: dict) -> Optional[float]:
    """Trunk lateral alignment: lateral tilt from midline.

    0° = perfectly balanced, positive = tilting right, negative = tilting left.
    Good balance: within ±8°.

    Args:
        landmarks: Dict of landmark positions from PoseEstimator.

    Returns:
        Angle in degrees from vertical midline, or None if not visible.
    """
    if not _visibility(
        landmarks, "left_shoulder", "right_shoulder", "left_hip", "right_hip"
    ):
        return None

    l_shoulder = _get_point(landmarks, "left_shoulder")
    r_shoulder = _get_point(landmarks, "right_shoulder")
    l_hip = _get_point(landmarks, "left_hip")
    r_hip = _get_point(landmarks, "right_hip")

    # Midpoints of shoulders and hips
    mid_shoulder = (l_shoulder + r_shoulder) / 2.0
    mid_hip = (l_hip + r_hip) / 2.0

    # Trunk vector (hip midpoint to shoulder midpoint)
    trunk = mid_shoulder - mid_hip

    # Vertical reference
    vertical = np.array([0.0, -1.0])

    # Signed angle: positive = leaning right in image
    angle = np.degrees(np.arctan2(trunk[0], -trunk[1]))

    return float(angle)


def compute_all_angles(landmarks: dict) -> dict:
    """Compute all skating mechanics angles from landmarks.

    Args:
        landmarks: Dict of landmark positions from PoseEstimator.

    Returns:
        Dict mapping angle name -> value (or None if not computable).
    """
    angles = {}

    for side in ["left", "right"]:
        angles[f"{side}_knee_angle"] = knee_angle(landmarks, side)
        angles[f"{side}_hip_angle"] = hip_angle(landmarks, side)
        angles[f"{side}_forward_lean"] = forward_lean_angle(landmarks, side)
        angles[f"{side}_ankle_dorsiflexion"] = ankle_dorsiflexion(landmarks, side)

    angles["trunk_alignment"] = trunk_alignment(landmarks)

    return angles
