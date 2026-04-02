"""Temporal keypoint smoothing using Kalman filters.

Adapted from CS6476 PS5 KalmanFilter class. Instantiates one Kalman filter
per MediaPipe landmark (33 total) to smooth jittery detections and handle
brief occlusions via the prediction step.
"""

import numpy as np
from typing import Optional


class KalmanFilter2D:
    """Kalman filter for tracking a 2D point with constant velocity model.

    State: [x, y, vx, vy]
    Adapted from CS6476 PS5 KalmanFilter.
    """

    def __init__(
        self,
        init_x: float = 0.0,
        init_y: float = 0.0,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ):
        """Initialize the Kalman filter.

        Args:
            init_x: Initial x position.
            init_y: Initial y position.
            process_noise: Process noise scalar (lower = smoother output).
            measurement_noise: Measurement noise scalar.
        """
        self.state = np.array(
            [[init_x], [init_y], [0.0], [0.0]], dtype=np.float64
        )
        self.covariance = np.eye(4, dtype=np.float64)

        # State transition: constant velocity model
        self.D = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )

        # Measurement matrix: observe x, y only
        self.M = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64
        )

        self.Q = process_noise * np.eye(4, dtype=np.float64)
        self.R = measurement_noise * np.eye(2, dtype=np.float64)
        self.initialized = False

    def predict(self):
        """Predict next state."""
        self.state = self.D @ self.state
        self.covariance = self.D @ self.covariance @ self.D.T + self.Q

    def correct(self, meas_x: float, meas_y: float):
        """Correct state with measurement."""
        measurement = np.array([[meas_x], [meas_y]], dtype=np.float64)
        S = self.M @ self.covariance @ self.M.T + self.R
        K = self.covariance @ self.M.T @ np.linalg.inv(S)
        innovation = measurement - (self.M @ self.state)
        self.state = self.state + (K @ innovation)
        I = np.eye(4, dtype=np.float64)
        self.covariance = (I - K @ self.M) @ self.covariance

    def process(self, meas_x: float, meas_y: float) -> tuple[float, float]:
        """Run predict + correct cycle.

        Returns:
            Tuple of (smoothed_x, smoothed_y).
        """
        if not self.initialized:
            self.state[0, 0] = meas_x
            self.state[1, 0] = meas_y
            self.initialized = True
            return meas_x, meas_y

        self.predict()
        self.correct(meas_x, meas_y)
        return float(self.state[0, 0]), float(self.state[1, 0])

    def predict_only(self) -> tuple[float, float]:
        """Run prediction without correction (for occlusions).

        Returns:
            Tuple of (predicted_x, predicted_y).
        """
        if not self.initialized:
            return float(self.state[0, 0]), float(self.state[1, 0])

        self.predict()
        return float(self.state[0, 0]), float(self.state[1, 0])


class LandmarkSmoother:
    """Smooths all 33 MediaPipe landmarks using per-landmark Kalman filters.

    Handles low-visibility landmarks by using prediction-only (coasting)
    when confidence drops below threshold.
    """

    def __init__(
        self,
        num_landmarks: int = 33,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        visibility_threshold: float = 0.5,
    ):
        """Initialize smoother with one Kalman filter per landmark.

        Args:
            num_landmarks: Number of landmarks (33 for MediaPipe Pose).
            process_noise: Kalman Q scalar (lower = smoother, more lag).
            measurement_noise: Kalman R scalar.
            visibility_threshold: Below this, use predict-only (coast).
        """
        self.num_landmarks = num_landmarks
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.visibility_threshold = visibility_threshold
        self.filters: list[KalmanFilter2D] = []
        self._init_filters()

    def _init_filters(self):
        """Create fresh Kalman filters for all landmarks."""
        self.filters = [
            KalmanFilter2D(
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
            )
            for _ in range(self.num_landmarks)
        ]

    def update(self, landmarks: dict) -> dict:
        """Smooth all landmarks for one frame.

        Args:
            landmarks: Dict from PoseEstimator.process_frame().

        Returns:
            New dict with smoothed x, y positions (same structure).
        """
        from src.pose_estimator import LANDMARK_NAMES

        smoothed = {}
        for idx, name in enumerate(LANDMARK_NAMES):
            if idx >= self.num_landmarks:
                break

            if name not in landmarks:
                continue

            lm = landmarks[name]
            kf = self.filters[idx]

            if lm["visibility"] >= self.visibility_threshold:
                sx, sy = kf.process(lm["x"], lm["y"])
            else:
                sx, sy = kf.predict_only()

            smoothed[name] = {
                "x": sx,
                "y": sy,
                "z": lm["z"],
                "visibility": lm["visibility"],
            }

        return smoothed

    def reset(self):
        """Reinitialize all filters (e.g., on scene cut)."""
        self._init_filters()
