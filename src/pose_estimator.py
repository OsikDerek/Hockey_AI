"""Pose estimation backends for skating analysis.

Provides two backends with a common interface:
  - MediaPipePoseEstimator (BlazePose, 33 landmarks, CPU-friendly)
  - YoloPoseEstimator (YOLOv8-Pose, 17 COCO keypoints, GPU-friendly, multi-person)

Both return landmark dicts with pixel coords and visibility scores.
"""

import cv2
import numpy as np
from typing import Optional


# MediaPipe landmark indices — the ones most relevant for skating
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Skating-critical landmark groups
SKATING_LANDMARKS = {
    "left_leg": ["left_hip", "left_knee", "left_ankle", "left_heel", "left_foot_index"],
    "right_leg": ["right_hip", "right_knee", "right_ankle", "right_heel", "right_foot_index"],
    "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "upper_body": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"],
}

# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    # Torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # Left arm
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    # Right arm
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Left leg
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_foot_index"),
    ("left_heel", "left_foot_index"),
    # Right leg
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_foot_index"),
    ("right_heel", "right_foot_index"),
]


class PoseEstimator:
    """Wrapper around MediaPipe PoseLandmarker (Tasks API) for skating analysis.

    Returns landmark positions in pixel coordinates with confidence scores.
    Uses the new Tasks API (mediapipe >= 0.10.14).
    """

    # Model file mapping: complexity -> filename
    MODEL_FILES = {
        0: "pose_landmarker_lite.task",
        1: "pose_landmarker_full.task",
        2: "pose_landmarker_heavy.task",
    }

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_dir: str = "models",
    ):
        """Initialize MediaPipe PoseLandmarker.

        Args:
            model_complexity: 0=lite, 1=full, 2=heavy. Default 2 (best accuracy).
            min_detection_confidence: Minimum confidence for person detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
            model_dir: Directory containing .task model files.
        """
        import os
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )
        from mediapipe.tasks.python import BaseOptions

        model_file = self.MODEL_FILES.get(model_complexity, self.MODEL_FILES[2])
        model_path = os.path.join(model_dir, model_file)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}\n"
                f"Download from: https://storage.googleapis.com/mediapipe-models/"
                f"pose_landmarker/{model_file.replace('.task', '')}/float16/latest/{model_file}"
            )

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_poses=1,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def process_frame(self, bgr_frame: np.ndarray) -> Optional[dict]:
        """Run pose estimation on a single BGR frame.

        Args:
            bgr_frame: OpenCV BGR image (numpy array).

        Returns:
            Dict mapping landmark name -> {x, y, z, visibility}.
            Returns None if no person detected.
        """
        import mediapipe as mp

        h, w = bgr_frame.shape[:2]

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)
        self._frame_timestamp_ms += 33  # ~30fps increment

        if not result.pose_landmarks:
            return None

        pose = result.pose_landmarks[0]  # First person
        landmarks = {}
        for idx, landmark in enumerate(pose):
            if idx < len(LANDMARK_NAMES):
                landmarks[LANDMARK_NAMES[idx]] = {
                    "x": landmark.x * w,
                    "y": landmark.y * h,
                    "z": landmark.z,
                    "visibility": landmark.visibility,
                }

        return landmarks

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# COCO keypoint names (YOLOv8-Pose output order)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class YoloPoseEstimator:
    """YOLOv8-Pose backend for skating analysis.

    Returns the same landmark dict format as PoseEstimator so the rest
    of the pipeline (angle_calculator, smoothing, annotator) works unchanged.

    Advantages over MediaPipe:
      - Better occlusion handling
      - Multi-person detection (returns largest person by default)
      - Faster on GPU

    Limitation: COCO keypoints lack heel/foot_index, so ankle_dorsiflexion
    will be unavailable. All other angles work normally.
    """

    def __init__(
        self,
        model_name: str = "yolov8m-pose.pt",
        conf_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        """Initialize YOLOv8-Pose model.

        Args:
            model_name: YOLO model weight file. Downloads automatically if needed.
                Options: yolov8n-pose.pt (fast), yolov8m-pose.pt (balanced),
                         yolov8l-pose.pt (accurate).
            conf_threshold: Minimum detection confidence.
            device: 'cuda', 'cpu', or None for auto-detect.
        """
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.device = device

    def process_frame(self, bgr_frame: np.ndarray) -> Optional[dict]:
        """Run pose estimation on a single BGR frame.

        Returns landmarks for the largest detected person (by bounding box area).
        Same dict format as PoseEstimator for compatibility.
        """
        results = self.model(
            bgr_frame,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )

        if not results or results[0].keypoints is None:
            return None

        keypoints = results[0].keypoints
        if keypoints.shape[0] == 0:
            return None

        # Select the largest person by bounding box area
        if results[0].boxes is not None and len(results[0].boxes) > 1:
            areas = results[0].boxes.xywh[:, 2] * results[0].boxes.xywh[:, 3]
            best_idx = int(areas.argmax())
        else:
            best_idx = 0

        kps = keypoints.data[best_idx].cpu().numpy()  # shape: (17, 3) = x, y, conf

        landmarks = {}
        for idx, name in enumerate(COCO_KEYPOINT_NAMES):
            x, y, conf = kps[idx]
            landmarks[name] = {
                "x": float(x),
                "y": float(y),
                "z": 0.0,  # YOLO doesn't provide depth
                "visibility": float(conf),
            }

        return landmarks

    def process_frame_multi(self, bgr_frame: np.ndarray) -> list[dict]:
        """Run pose estimation returning all detected persons.

        Returns:
            List of landmark dicts (one per person), sorted by bbox area (largest first).
            Empty list if no detections.
        """
        results = self.model(
            bgr_frame,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )

        if not results or results[0].keypoints is None:
            return []

        keypoints = results[0].keypoints
        if keypoints.shape[0] == 0:
            return []

        # Sort by bbox area (largest first)
        if results[0].boxes is not None and len(results[0].boxes) > 1:
            areas = results[0].boxes.xywh[:, 2] * results[0].boxes.xywh[:, 3]
            order = areas.argsort(descending=True)
        else:
            order = range(keypoints.shape[0])

        all_landmarks = []
        for person_idx in order:
            kps = keypoints.data[person_idx].cpu().numpy()
            landmarks = {}
            for idx, name in enumerate(COCO_KEYPOINT_NAMES):
                x, y, conf = kps[idx]
                landmarks[name] = {
                    "x": float(x),
                    "y": float(y),
                    "z": 0.0,
                    "visibility": float(conf),
                }
            all_landmarks.append(landmarks)

        return all_landmarks

    def close(self):
        """No persistent resources to release for YOLO."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_estimator(
    backend: str = "mediapipe",
    **kwargs,
) -> "PoseEstimator | YoloPoseEstimator":
    """Factory function to create pose estimator by backend name.

    Args:
        backend: 'mediapipe' or 'yolo'.
        **kwargs: Passed to the estimator constructor.

    Returns:
        PoseEstimator or YoloPoseEstimator instance.
    """
    if backend == "mediapipe":
        return PoseEstimator(**kwargs)
    elif backend == "yolo":
        return YoloPoseEstimator(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'mediapipe' or 'yolo'.")
