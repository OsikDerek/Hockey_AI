"""Video preprocessing for skating analysis.

Handles the common case where the skater is small in a wide rink shot
(Instagram clips, game film, rink cameras). Uses YOLO person detection
to find and track the skater, then crops and upscales for better pose
estimation.

Pipeline: detect person bbox → smooth bbox trajectory → crop → upscale
"""

import cv2
import numpy as np
from typing import Optional


class SkaterCropper:
    """Detects and crops around the largest skater in each frame.

    Uses YOLOv8 person detection (not pose) for fast, reliable bounding
    boxes even when the skater is small. Smooths the crop region across
    frames to avoid jitter.
    """

    def __init__(
        self,
        target_height: int = 720,
        padding_ratio: float = 0.4,
        smoothing_alpha: float = 0.15,
        conf_threshold: float = 0.3,
    ):
        """Initialize the cropper.

        Args:
            target_height: Height to upscale the cropped region to.
            padding_ratio: Extra padding around the bounding box (0.4 = 40% each side).
            smoothing_alpha: Exponential smoothing factor for bbox trajectory.
                Lower = smoother but slower to react. Higher = responsive but jittery.
            conf_threshold: Minimum detection confidence.
        """
        from ultralytics import YOLO

        self.detector = YOLO("yolov8n.pt")  # Nano model — fast, just need bbox
        self.target_height = target_height
        self.padding_ratio = padding_ratio
        self.smoothing_alpha = smoothing_alpha
        self.conf_threshold = conf_threshold

        # Smoothed bounding box state
        self._smooth_bbox: Optional[np.ndarray] = None  # [x1, y1, x2, y2]
        self._frames_since_detection = 0
        self._max_coast_frames = 15  # Use last known bbox for this many frames

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """Detect skater and return cropped/upscaled frame.

        Args:
            frame: Full BGR frame.

        Returns:
            Tuple of (cropped_frame, crop_info) where crop_info contains:
                - 'bbox': [x1, y1, x2, y2] in original frame coords
                - 'scale': upscale factor applied
                - 'detected': whether a person was found this frame
                - 'original_size': (h, w) of original frame
        """
        h, w = frame.shape[:2]
        bbox = self._detect_person(frame)

        if bbox is not None:
            self._frames_since_detection = 0
            # Smooth the bbox
            if self._smooth_bbox is None:
                self._smooth_bbox = bbox.astype(np.float64)
            else:
                alpha = self.smoothing_alpha
                self._smooth_bbox = alpha * bbox + (1 - alpha) * self._smooth_bbox
            detected = True
        else:
            self._frames_since_detection += 1
            detected = False

        # Use smoothed bbox if available and not too stale
        if (self._smooth_bbox is not None
                and self._frames_since_detection <= self._max_coast_frames):
            crop_bbox = self._pad_bbox(self._smooth_bbox, w, h)
            cropped = self._crop_and_upscale(frame, crop_bbox)

            return cropped, {
                "bbox": crop_bbox.astype(int).tolist(),
                "scale": self.target_height / (crop_bbox[3] - crop_bbox[1]),
                "detected": detected,
                "original_size": (h, w),
            }

        # No detection and no recent bbox — return original frame
        return frame, {
            "bbox": [0, 0, w, h],
            "scale": 1.0,
            "detected": False,
            "original_size": (h, w),
        }

    def _detect_person(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect the largest person in the frame.

        Returns [x1, y1, x2, y2] ndarray or None.
        """
        results = self.detector(
            frame,
            classes=[0],  # Person class only
            conf=self.conf_threshold,
            verbose=False,
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        # Select largest by area
        xywh = boxes.xywh.cpu().numpy()
        areas = xywh[:, 2] * xywh[:, 3]
        best_idx = int(areas.argmax())

        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        return xyxy  # [x1, y1, x2, y2]

    def _pad_bbox(
        self, bbox: np.ndarray, frame_w: int, frame_h: int
    ) -> np.ndarray:
        """Add padding around the bbox and clamp to frame bounds."""
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        pad_x = bw * self.padding_ratio
        pad_y = bh * self.padding_ratio

        # More padding on top (head room) and bottom (skates)
        x1_pad = x1 - pad_x
        y1_pad = y1 - pad_y * 0.8   # Less top padding
        x2_pad = x2 + pad_x
        y2_pad = y2 + pad_y * 1.2   # More bottom for skates

        # Clamp to frame
        x1_pad = max(0, x1_pad)
        y1_pad = max(0, y1_pad)
        x2_pad = min(frame_w, x2_pad)
        y2_pad = min(frame_h, y2_pad)

        return np.array([x1_pad, y1_pad, x2_pad, y2_pad])

    def _crop_and_upscale(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> np.ndarray:
        """Crop the frame to bbox and upscale to target height."""
        x1, y1, x2, y2 = bbox.astype(int)

        # Ensure minimum size
        if x2 - x1 < 10 or y2 - y1 < 10:
            return frame

        cropped = frame[y1:y2, x1:x2]

        # Upscale maintaining aspect ratio
        crop_h, crop_w = cropped.shape[:2]
        if crop_h <= 0 or crop_w <= 0:
            return frame

        scale = self.target_height / crop_h
        new_w = int(crop_w * scale)
        new_h = self.target_height

        upscaled = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return upscaled

    def reset(self):
        """Reset tracking state (call between videos)."""
        self._smooth_bbox = None
        self._frames_since_detection = 0
