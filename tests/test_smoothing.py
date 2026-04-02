"""Unit tests for the Kalman filter smoothing module."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.smoothing import KalmanFilter2D, LandmarkSmoother


class TestKalmanFilter2D:
    """Tests for the per-landmark Kalman filter."""

    def test_first_measurement_returned_exactly(self):
        """First call returns the measurement unchanged."""
        kf = KalmanFilter2D()
        x, y = kf.process(100.0, 200.0)
        assert x == 100.0
        assert y == 200.0

    def test_constant_input_converges(self):
        """Repeated identical measurements converge to that value."""
        kf = KalmanFilter2D(process_noise=0.01, measurement_noise=0.1)
        for _ in range(50):
            x, y = kf.process(100.0, 200.0)

        assert abs(x - 100.0) < 1.0
        assert abs(y - 200.0) < 1.0

    def test_smooths_noisy_input(self):
        """Output variance is lower than input variance."""
        kf = KalmanFilter2D(process_noise=0.01, measurement_noise=0.5)
        np.random.seed(42)

        true_x, true_y = 100.0, 200.0
        noise_std = 10.0

        raw_errors = []
        filtered_errors = []

        for i in range(200):
            noisy_x = true_x + np.random.randn() * noise_std
            noisy_y = true_y + np.random.randn() * noise_std
            raw_errors.append((noisy_x - true_x) ** 2 + (noisy_y - true_y) ** 2)

            fx, fy = kf.process(noisy_x, noisy_y)
            filtered_errors.append((fx - true_x) ** 2 + (fy - true_y) ** 2)

        # After warm-up, filtered should have lower error
        raw_rmse = np.sqrt(np.mean(raw_errors[50:]))
        filtered_rmse = np.sqrt(np.mean(filtered_errors[50:]))
        assert filtered_rmse < raw_rmse

    def test_predict_only_coasts(self):
        """Predict-only uses velocity to coast forward."""
        kf = KalmanFilter2D(process_noise=0.01, measurement_noise=0.1)

        # Establish moving trajectory
        for i in range(20):
            kf.process(float(i * 10), 100.0)

        # Now coast with predict-only
        x1, y1 = kf.predict_only()
        x2, y2 = kf.predict_only()

        # Should continue moving in x direction
        assert x2 > x1

    def test_handles_trajectory_change(self):
        """Filter adapts when trajectory changes direction."""
        kf = KalmanFilter2D(process_noise=0.1, measurement_noise=0.1)

        # Move right
        for i in range(30):
            kf.process(float(i * 5), 100.0)

        # Reverse direction
        for i in range(30):
            x, y = kf.process(float(150 - i * 5), 100.0)

        # After enough frames, should be tracking the new direction
        assert x < 30  # Should be near the reversed position


class TestLandmarkSmoother:
    """Tests for the full landmark smoother."""

    def test_creates_correct_number_of_filters(self):
        """Should create one filter per landmark."""
        smoother = LandmarkSmoother(num_landmarks=33)
        assert len(smoother.filters) == 33

    def test_reset_reinitializes(self):
        """Reset creates fresh filters."""
        smoother = LandmarkSmoother(num_landmarks=5)
        # Use one filter
        smoother.filters[0].process(10.0, 20.0)
        assert smoother.filters[0].initialized

        smoother.reset()
        assert not smoother.filters[0].initialized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
