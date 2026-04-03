"""Object tracking for puck and stick detection.

Runs YOLO object detection alongside pose estimation to track the puck
and compute metrics relative to the player's body. Returns a flat dict
of metrics that gets merged into the angles dict for technique evaluation.

Usage in main.py:
    tracker = ObjectTracker(model_path="models/puck_yolov8n.pt")
    metrics = tracker.process_frame(frame, landmarks)
    angles.update(metrics)  # Merge into angles dict
"""

import numpy as np
from typing import Optional
from pathlib import Path


class ObjectTracker:
    """Tracks puck position and computes metrics relative to player body.

    Runs YOLO detection on each frame (or every Nth frame for speed)
    and returns normalized metrics for technique evaluation.
    """

    def __init__(
        self,
        model_path: str = "models/puck_yolov8n.pt",
        conf: float = 0.25,
        detection_interval: int = 1,
        puck_class_id: int = None,
    ):
        """Initialize object tracker.

        Args:
            model_path: Path to YOLO model weights.
            conf: Minimum detection confidence.
            detection_interval: Run detection every N frames (1=every frame).
            puck_class_id: Class ID for puck in the model. If None, auto-detect
                from model class names.
        """
        self.model = None
        self.model_path = model_path
        self.conf = conf
        self.detection_interval = max(1, detection_interval)
        self.puck_class_id = puck_class_id

        self._frame_count = 0
        self._last_puck_pos = None  # (x, y) of last detected puck
        self._puck_history = []     # Rolling window of positions for width calc

        # Load model if available
        model_file = Path(model_path)
        if model_file.is_file():
            self._load_model()
        else:
            print(f"  ObjectTracker: Model not found at {model_path}")
            print(f"  ObjectTracker: Running without puck detection (landmark metrics only)")

    def _load_model(self):
        """Load the YOLO model and determine puck class ID."""
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)

        # Auto-detect puck class ID from model names
        if self.puck_class_id is None and hasattr(self.model, "names"):
            names = self.model.names
            for class_id, name in names.items():
                if "puck" in name.lower():
                    self.puck_class_id = class_id
                    break

        if self.puck_class_id is not None:
            print(f"  ObjectTracker: Puck class ID = {self.puck_class_id}")
        else:
            print(f"  ObjectTracker: No 'puck' class found in model. Available: {self.model.names}")

    @property
    def has_model(self) -> bool:
        return self.model is not None and self.puck_class_id is not None

    def process_frame(
        self, frame: np.ndarray, landmarks: Optional[dict]
    ) -> dict:
        """Process one frame and return object-tracking metrics.

        Args:
            frame: BGR video frame.
            landmarks: Pose landmarks dict (None if no skater detected).

        Returns:
            Dict of metrics to merge into angles dict. Keys:
                puck_body_distance: float or None
                puck_stick_proximity: float or None
                stickhandling_width: float or None
        """
        metrics = {
            "puck_body_distance": None,
            "puck_stick_proximity": None,
            "stickhandling_width": None,
        }

        self._frame_count += 1

        # Detect puck (skip frames based on interval)
        puck_pos = None
        if self.has_model:
            if self._frame_count % self.detection_interval == 0:
                puck_pos = self._detect_puck(frame)
                if puck_pos is not None:
                    self._last_puck_pos = puck_pos
            else:
                puck_pos = self._last_puck_pos  # Use last known position

        # Compute metrics if we have both puck and player
        if puck_pos is not None and landmarks is not None:
            metrics.update(self._compute_puck_metrics(puck_pos, landmarks))

        return metrics

    def _detect_puck(self, frame: np.ndarray) -> Optional[tuple]:
        """Run YOLO detection and return puck center (x, y) or None."""
        results = self.model(
            frame,
            classes=[self.puck_class_id],
            conf=self.conf,
            verbose=False,
        )

        if len(results) == 0 or len(results[0].boxes) == 0:
            return None

        # Get highest-confidence detection
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax()
        box = boxes.xyxy[best_idx].cpu().numpy()

        # Return center of bounding box
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0

        return (float(cx), float(cy))

    def _compute_puck_metrics(
        self, puck_pos: tuple, landmarks: dict
    ) -> dict:
        """Compute puck metrics relative to player body."""
        metrics = {}
        px, py = puck_pos

        # Get body reference points
        has_hips = (
            "left_hip" in landmarks and "right_hip" in landmarks
            and landmarks["left_hip"]["visibility"] > 0.4
            and landmarks["right_hip"]["visibility"] > 0.4
        )
        has_shoulders = (
            "left_shoulder" in landmarks and "right_shoulder" in landmarks
            and landmarks["left_shoulder"]["visibility"] > 0.4
            and landmarks["right_shoulder"]["visibility"] > 0.4
        )
        has_wrists = (
            "left_wrist" in landmarks and "right_wrist" in landmarks
            and landmarks["left_wrist"]["visibility"] > 0.4
            and landmarks["right_wrist"]["visibility"] > 0.4
        )

        # Shoulder width for normalization
        shoulder_width = None
        if has_shoulders:
            ls = np.array([landmarks["left_shoulder"]["x"], landmarks["left_shoulder"]["y"]])
            rs = np.array([landmarks["right_shoulder"]["x"], landmarks["right_shoulder"]["y"]])
            shoulder_width = np.linalg.norm(rs - ls)
            if shoulder_width < 1:
                shoulder_width = None

        puck = np.array([px, py])

        # Puck-to-body distance (normalized)
        if has_hips and shoulder_width:
            lh = np.array([landmarks["left_hip"]["x"], landmarks["left_hip"]["y"]])
            rh = np.array([landmarks["right_hip"]["x"], landmarks["right_hip"]["y"]])
            mid_hip = (lh + rh) / 2.0
            metrics["puck_body_distance"] = float(np.linalg.norm(puck - mid_hip) / shoulder_width)

        # Puck-to-stick proximity (distance to nearest wrist, normalized)
        if has_wrists and shoulder_width:
            lw = np.array([landmarks["left_wrist"]["x"], landmarks["left_wrist"]["y"]])
            rw = np.array([landmarks["right_wrist"]["x"], landmarks["right_wrist"]["y"]])
            dist_left = np.linalg.norm(puck - lw)
            dist_right = np.linalg.norm(puck - rw)
            metrics["puck_stick_proximity"] = float(min(dist_left, dist_right) / shoulder_width)

        # Stickhandling width (rolling window of puck x-positions)
        if shoulder_width:
            self._puck_history.append(px)
            # Keep last 30 frames (~1 second at 30fps)
            if len(self._puck_history) > 30:
                self._puck_history = self._puck_history[-30:]

            if len(self._puck_history) >= 5:
                x_range = max(self._puck_history) - min(self._puck_history)
                metrics["stickhandling_width"] = float(x_range / shoulder_width)

        return metrics
