"""Phase 3: inference wrapper — broadcast frame -> ice<->pixel homography.

Loads the trained rink-keypoint YOLOv8-pose model, detects the named
NHL rink keypoints in a frame, and solves the homography directly from
the detected keypoints' KNOWN ice coordinates. No similarity-prior
bootstrap, no dot-disambiguation guesswork — each keypoint channel has
a fixed ice coordinate, so a detection IS a correspondence.

Usage:
    reg = RinkRegistrationModel("models/rink_keypoints.pt")
    result = reg.estimate(frame_bgr)
    if result is not None:
        H_pixel_to_ice, H_ice_to_pixel, info = result
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .keypoints import KEYPOINT_ICE_XY, NUM_KEYPOINTS, UNRELIABLE_KEYPOINTS


class RinkRegistrationModel:
    """Frame -> homography via learned rink-keypoint detection."""

    def __init__(
        self,
        weights_path: str = "models/HockeyRink.pt",
        kp_conf: float = 0.50,          # min per-keypoint confidence to use
        min_correspondences: int = 5,   # min keypoints to attempt a fit
        ransac_reproj_px: float = 8.0,  # cv2.findHomography RANSAC threshold
        max_reproj_err_ft: float = 4.0, # reject fit if median reproj error worse
    ):
        self.weights_path = Path(weights_path)
        self.kp_conf = kp_conf
        self.min_correspondences = min_correspondences
        self.ransac_reproj_px = ransac_reproj_px
        self.max_reproj_err_ft = max_reproj_err_ft
        self._model = None
        self._ice_xy = np.array(KEYPOINT_ICE_XY, dtype=np.float64)
        # Keypoint channels with unreliable template coords -- never solve
        # the homography from these (see keypoints.UNRELIABLE_KEYPOINTS).
        self._reliable = np.ones(NUM_KEYPOINTS, dtype=bool)
        self._reliable[UNRELIABLE_KEYPOINTS] = False
        self.available = False

        if self.weights_path.is_file():
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(self.weights_path))
                self.available = True
            except Exception as e:
                print(f"  RinkRegistrationModel: failed to load "
                      f"{self.weights_path}: {e}")
        else:
            print(f"  RinkRegistrationModel: weights not found at "
                  f"{self.weights_path} — learned registration disabled "
                  f"(train via rink_registration.train).")

    def detect_keypoints(self, frame: np.ndarray):
        """Run the model. Returns (N,3) array [px, py, conf] per keypoint
        channel (conf 0 = not detected), or None on failure."""
        if not self.available:
            return None
        try:
            res = self._model.predict(frame, conf=0.15, verbose=False)
        except Exception:
            return None
        if not res or len(res) == 0:
            return None
        r = res[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return None
        # Single "rink" object expected; if several, take the highest-conf box.
        idx = 0
        if r.boxes is not None and len(r.boxes) > 1:
            idx = int(np.argmax(r.boxes.conf.cpu().numpy()))
        xy = r.keypoints.xy.cpu().numpy()[idx]          # (NUM_KEYPOINTS, 2)
        conf = (r.keypoints.conf.cpu().numpy()[idx]
                if r.keypoints.conf is not None
                else np.ones(len(xy)))
        out = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)
        n = min(NUM_KEYPOINTS, len(xy))
        out[:n, :2] = xy[:n]
        out[:n, 2] = conf[:n]
        return out

    def estimate(self, frame: np.ndarray) -> Optional[tuple]:
        """Detect keypoints and solve the homography.

        Returns (H_pixel_to_ice, H_ice_to_pixel, info) or None.
        info: dict with n_used, median_reproj_err_ft, kp_conf_mean.
        """
        kps = self.detect_keypoints(frame)
        if kps is None:
            return None

        use = (kps[:, 2] >= self.kp_conf) & self._reliable
        n_used = int(use.sum())
        if n_used < self.min_correspondences:
            return None

        pixel_pts = kps[use, :2].astype(np.float64)
        ice_pts = self._ice_xy[use]

        import cv2
        H_p2i, mask = cv2.findHomography(
            pixel_pts, ice_pts, cv2.RANSAC, self.ransac_reproj_px,
        )
        if H_p2i is None:
            return None

        # Reprojection error on the RANSAC inliers, in feet.
        inliers = mask.ravel().astype(bool) if mask is not None else np.ones(n_used, bool)
        if inliers.sum() < self.min_correspondences:
            return None
        proj = cv2.perspectiveTransform(
            pixel_pts[inliers].reshape(-1, 1, 2), H_p2i,
        ).reshape(-1, 2)
        err = np.linalg.norm(proj - ice_pts[inliers], axis=1)
        median_err = float(np.median(err))
        if median_err > self.max_reproj_err_ft:
            return None

        try:
            H_i2p = np.linalg.inv(H_p2i)
        except np.linalg.LinAlgError:
            return None

        info = {
            "n_used": int(inliers.sum()),
            "median_reproj_err_ft": median_err,
            "kp_conf_mean": float(kps[use, 2].mean()),
        }
        return H_p2i, H_i2p, info
