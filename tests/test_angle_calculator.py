"""Unit tests for angle_calculator module."""

import sys
import os
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.angle_calculator import (
    angle_between_three_points,
    knee_angle,
    hip_angle,
    forward_lean_angle,
    ankle_dorsiflexion,
    trunk_alignment,
    compute_all_angles,
)


def _make_landmark(x, y, z=0.0, visibility=1.0):
    """Helper to create a landmark dict."""
    return {"x": x, "y": y, "z": z, "visibility": visibility}


class TestAngleBetweenThreePoints:
    """Tests for the core angle calculation."""

    def test_right_angle(self):
        """90-degree angle at vertex."""
        a = np.array([1, 0])
        b = np.array([0, 0])
        c = np.array([0, 1])
        assert abs(angle_between_three_points(a, b, c) - 90.0) < 0.1

    def test_straight_line(self):
        """180-degree angle (collinear points)."""
        a = np.array([0, 0])
        b = np.array([1, 0])
        c = np.array([2, 0])
        assert abs(angle_between_three_points(a, b, c) - 180.0) < 0.1

    def test_45_degree_angle(self):
        """45-degree angle."""
        a = np.array([1, 0])
        b = np.array([0, 0])
        c = np.array([1, 1])
        assert abs(angle_between_three_points(a, b, c) - 45.0) < 0.1

    def test_60_degree_angle(self):
        """60-degree equilateral triangle angle."""
        a = np.array([1, 0])
        b = np.array([0, 0])
        c = np.array([0.5, np.sqrt(3) / 2])
        assert abs(angle_between_three_points(a, b, c) - 60.0) < 0.1

    def test_zero_angle(self):
        """0-degree angle (overlapping segments)."""
        a = np.array([1, 0])
        b = np.array([0, 0])
        c = np.array([2, 0])  # Same direction
        assert abs(angle_between_three_points(a, b, c) - 0.0) < 0.1

    def test_handles_float_coordinates(self):
        """Works with floating point coordinates."""
        a = np.array([1.5, 2.3])
        b = np.array([0.1, 0.2])
        c = np.array([3.7, 0.5])
        result = angle_between_three_points(a, b, c)
        assert 0 <= result <= 180


class TestKneeAngle:
    """Tests for knee angle calculation."""

    def test_deep_knee_bend(self):
        """Simulated deep knee bend (~90 degrees)."""
        landmarks = {
            "left_hip": _make_landmark(100, 100),
            "left_knee": _make_landmark(100, 200),
            "left_ankle": _make_landmark(200, 200),
        }
        result = knee_angle(landmarks, "left")
        assert result is not None
        assert abs(result - 90.0) < 1.0

    def test_extended_leg(self):
        """Nearly straight leg (~180 degrees)."""
        landmarks = {
            "left_hip": _make_landmark(100, 100),
            "left_knee": _make_landmark(100, 200),
            "left_ankle": _make_landmark(100, 300),
        }
        result = knee_angle(landmarks, "left")
        assert result is not None
        assert abs(result - 180.0) < 1.0

    def test_returns_none_for_low_visibility(self):
        """Returns None when landmarks aren't visible."""
        landmarks = {
            "left_hip": _make_landmark(100, 100, visibility=0.1),
            "left_knee": _make_landmark(100, 200),
            "left_ankle": _make_landmark(100, 300),
        }
        assert knee_angle(landmarks, "left") is None

    def test_right_side(self):
        """Works for right side."""
        landmarks = {
            "right_hip": _make_landmark(100, 100),
            "right_knee": _make_landmark(100, 200),
            "right_ankle": _make_landmark(200, 200),
        }
        result = knee_angle(landmarks, "right")
        assert result is not None
        assert abs(result - 90.0) < 1.0


class TestForwardLean:
    """Tests for forward lean angle."""

    def test_upright(self):
        """Upright posture = ~0 degrees from vertical."""
        landmarks = {
            "left_shoulder": _make_landmark(100, 100),
            "left_hip": _make_landmark(100, 300),
        }
        result = forward_lean_angle(landmarks, "left")
        assert result is not None
        assert abs(result) < 1.0

    def test_45_degree_lean(self):
        """45-degree forward lean."""
        landmarks = {
            "left_shoulder": _make_landmark(200, 100),
            "left_hip": _make_landmark(100, 200),
        }
        result = forward_lean_angle(landmarks, "left")
        assert result is not None
        assert abs(result - 45.0) < 1.0


class TestTrunkAlignment:
    """Tests for trunk lateral alignment."""

    def test_centered(self):
        """Balanced posture = ~0 degrees."""
        landmarks = {
            "left_shoulder": _make_landmark(80, 100),
            "right_shoulder": _make_landmark(120, 100),
            "left_hip": _make_landmark(80, 300),
            "right_hip": _make_landmark(120, 300),
        }
        result = trunk_alignment(landmarks)
        assert result is not None
        assert abs(result) < 1.0

    def test_leaning_right(self):
        """Leaning right should give positive angle."""
        landmarks = {
            "left_shoulder": _make_landmark(120, 100),
            "right_shoulder": _make_landmark(160, 100),
            "left_hip": _make_landmark(80, 300),
            "right_hip": _make_landmark(120, 300),
        }
        result = trunk_alignment(landmarks)
        assert result is not None
        assert result > 0


class TestComputeAllAngles:
    """Tests for the complete angle computation."""

    def test_returns_all_expected_keys(self):
        """Should return all angle keys."""
        landmarks = {
            "left_shoulder": _make_landmark(100, 100),
            "right_shoulder": _make_landmark(200, 100),
            "left_hip": _make_landmark(100, 250),
            "right_hip": _make_landmark(200, 250),
            "left_knee": _make_landmark(100, 400),
            "right_knee": _make_landmark(200, 400),
            "left_ankle": _make_landmark(100, 550),
            "right_ankle": _make_landmark(200, 550),
            "left_foot_index": _make_landmark(120, 570),
            "right_foot_index": _make_landmark(220, 570),
        }
        angles = compute_all_angles(landmarks)

        expected_keys = [
            "left_knee_angle", "right_knee_angle",
            "left_hip_angle", "right_hip_angle",
            "left_forward_lean", "right_forward_lean",
            "left_ankle_dorsiflexion", "right_ankle_dorsiflexion",
            "trunk_alignment",
        ]
        for key in expected_keys:
            assert key in angles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
