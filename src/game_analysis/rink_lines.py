"""Phase B2: classical CV detection of NHL rink line markings.

The rink has high-contrast painted stripes that don't need a neural net
to find — they're 1 ft wide, span the rink width, and live in narrow
HSV bands:

  - 2 BLUE lines at ice x = 75 ft (left), 125 ft (right)
  - 1 RED center line at x = 100 ft
  - 2 RED goal lines at x = 11 ft (left), 189 ft (right)

YOLO only ever sees the rink's faceoff dots / goal nets / center dot,
giving us 0-3 point correspondences per frame on broadcast follow-cam.
That isn't enough for a real homography (cv2.findHomography needs ≥4
points). Lines fix this — they're visible nearly every frame the rink
is in shot, and they each contribute ≥2 effective correspondences when
sampled.

Pipeline per frame:
  1. HSV mask the blue + red bands (wide ranges to tolerate broadcast
     color grading)
  2. Morphological open/close to defeat jersey + dasher-board noise
  3. Probabilistic Hough on the cleaned masks (ANY orientation —
     broadcast camera puts lines near-horizontal in pixel space)
  4. Cluster collinear segments per color so we emit one representative
     per painted stripe

Caller (RinkCalibrator) takes detected lines + the existing similarity
prior to identify which ice-line each pixel-line corresponds to (by
perpendicular-distance match against predicted ice-line projections),
then samples points along each match for the homography fit.
"""

import cv2
import numpy as np
from typing import Optional


# OpenCV HSV: H in [0, 180], S/V in [0, 255]. Wide ranges chosen to
# survive broadcast color grading variation without false-positiving
# all over jersey colors.
BLUE_HSV_LO = (90, 30, 30)
BLUE_HSV_HI = (135, 255, 255)
# Red wraps the H boundary, so two ranges
RED_HSV_LO_1 = (0, 40, 40)
RED_HSV_HI_1 = (15, 255, 255)
RED_HSV_LO_2 = (165, 40, 40)
RED_HSV_HI_2 = (180, 255, 255)

# Pixel-line acceptance — orientation-agnostic. Broadcast cams put rink
# lines at any angle in pixel space (near-horizontal on flat-angle cams,
# near-vertical on overhead). The homography reproj-error gate and
# similarity-based matching reject false-positive lines downstream.
MIN_LINE_LEN_FRAC = 0.15            # at least 15% of frame's smaller axis
HOUGH_THRESHOLD = 50                # accumulator threshold
HOUGH_MAX_GAP_PX = 30
CLUSTER_DIST_FRAC = 0.04             # cluster lines within 4% of frame size


def detect_rink_lines(frame: np.ndarray) -> dict:
    """Find blue + red rink lines in the frame.

    Returns:
        {
          "blue": [{p1, p2, mid_x, length, angle_rad}, ...],
          "red":  [...],
        }

    Each line entry's coordinates are in pixel space. mid_x is the
    average x of the two endpoints — used by callers for matching to
    ice-line predictions. Lines are sorted by mid_x ascending.
    """
    empty = {"blue": [], "red": []}
    if frame is None or frame.size == 0 or frame.ndim < 3:
        return empty

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, BLUE_HSV_LO, BLUE_HSV_HI)
    red_mask = (cv2.inRange(hsv, RED_HSV_LO_1, RED_HSV_HI_1)
                | cv2.inRange(hsv, RED_HSV_LO_2, RED_HSV_HI_2))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    h, w = frame.shape[:2]
    min_len = max(50, int(min(w, h) * MIN_LINE_LEN_FRAC))

    blue = _hough_any_orientation(blue_mask, min_len)
    red = _hough_any_orientation(red_mask, min_len)

    blue = _cluster_by_midpoint(blue, w, h)
    red = _cluster_by_midpoint(red, w, h)

    return {"blue": blue, "red": red}


def sample_line_points(line: dict, n_samples: int = 7) -> list:
    """Return n evenly-spaced (x, y) pixel points along the segment."""
    p1 = np.array(line["p1"], dtype=np.float64)
    p2 = np.array(line["p2"], dtype=np.float64)
    pts = []
    for i in range(n_samples):
        t = i / max(1, n_samples - 1)
        pts.append(tuple(p1 * (1 - t) + p2 * t))
    return pts


def _hough_any_orientation(mask: np.ndarray, min_len: int) -> list:
    """Probabilistic Hough at any orientation. The downstream matcher uses
    the similarity prior to figure out which ice-line each detected line
    corresponds to, so we don't bias the detector by orientation here."""
    edges = cv2.Canny(mask, 50, 150)
    raw = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=min_len,
        maxLineGap=HOUGH_MAX_GAP_PX,
    )
    if raw is None:
        return []
    out = []
    for ln in raw:
        x1, y1, x2, y2 = ln[0]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        # Compute the line's normal angle (0 = vertical line, π/2 = horizontal).
        # Used by the cluster step to keep lines with similar pose together.
        if length < 1e-6:
            continue
        # Use the line direction as a unit vector
        ux = dx / length
        uy = dy / length
        out.append({
            "p1": (float(x1), float(y1)),
            "p2": (float(x2), float(y2)),
            "mid": ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0),
            "length": length,
            "ux": ux, "uy": uy,
        })
    return out


def _cluster_by_midpoint(lines: list, frame_w: int, frame_h: int) -> list:
    """Greedy cluster: keep the longest line per spatial neighborhood."""
    if not lines:
        return []
    lines = sorted(lines, key=lambda L: -L["length"])  # longest first
    threshold = max(frame_w, frame_h) * CLUSTER_DIST_FRAC
    kept = []
    for ln in lines:
        cx, cy = ln["mid"]
        clobbered = False
        for k in kept:
            kx, ky = k["mid"]
            if (cx - kx) ** 2 + (cy - ky) ** 2 < threshold ** 2:
                # Same neighborhood — but if the orientations differ
                # significantly, keep both (could be perpendicular lines
                # crossing).
                cos_angle = abs(ln["ux"] * k["ux"] + ln["uy"] * k["uy"])
                if cos_angle > 0.85:  # within ~30° = same line
                    clobbered = True
                    break
        if not clobbered:
            kept.append(ln)
    return kept
