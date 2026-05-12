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


def detect_rink_boards(frame: np.ndarray) -> dict:
    """Find the top + bottom edges of the visible ice surface.

    Where ice (bright + low saturation) meets the boards (colored ads,
    brown wood-tone, kick plate, players) is a high-contrast boundary.
    For each column we record the topmost + bottommost ice pixel, then
    fit a robust line to each (RANSAC via cv2.fitLine).

    Returns:
        {
          "top": {"a": float, "b": float, "c": float} | None,    # ax+by+c=0
          "bottom": {...} | None,
        }
    where each line is in normalized form (a²+b² = 1) and the segment-form
    fields p1, p2 are also included for downstream sampling. None if too
    few ice pixels were found to fit reliably.
    """
    empty = {"top": None, "bottom": None}
    if frame is None or frame.size == 0 or frame.ndim < 3:
        return empty

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Ice = high-V, low-S (white-ish). Player jerseys / boards / dasher
    # boards are colored or dark; this naturally filters them out.
    ice_mask = ((hsv[..., 1] < 60) & (hsv[..., 2] > 170)).astype(np.uint8) * 255

    # Morphological open to drop small ice-colored noise (jersey numbers,
    # whitecaps from ice ruts) and close to bridge gaps under players.
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_OPEN, k_open)
    ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_CLOSE, k_close)

    # Keep only the LARGEST connected component as "the rink." Otherwise
    # bright noise outside the rink (overhead lights, bench tunnels) gets
    # mistaken for ice and pulls the boundary to y=0.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ice_mask, connectivity=8)
    if n <= 1:
        return empty
    # Component 0 is background; pick the largest non-background by area
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    rink_mask = (labels == largest_idx).astype(np.uint8) * 255

    # Sample columns at coarse stride; record top + bottom rink pixel per col.
    # Skip columns near the frame x-edges (rink corners curve, not lines)
    # and reject y-values within 15px of the frame top/bottom (these are
    # almost always broadcast graphics or stray bright regions that the
    # close-operation merged into the rink blob).
    stride = max(4, w // 200)
    edge_margin_x = max(20, w // 30)
    edge_margin_y = 15
    top_pts = []
    bot_pts = []
    for x in range(edge_margin_x, w - edge_margin_x, stride):
        col = rink_mask[:, x]
        nonzero = np.where(col > 0)[0]
        if nonzero.size < 30:  # require ≥30 rink pixels in this column
            continue
        top_y = int(nonzero[0])
        bot_y = int(nonzero[-1])
        if top_y > edge_margin_y:
            top_pts.append((x, top_y))
        if bot_y < h - edge_margin_y:
            bot_pts.append((x, bot_y))

    return {
        "top": _fit_line_robust(top_pts),
        "bottom": _fit_line_robust(bot_pts),
    }


def detect_rink_edges_via_row_brightness(frame: np.ndarray) -> dict:
    """Alternate board detector for wide-pano fixed-cam clips.

    Key insight: the rink ICE is the brightest horizontal band in the
    frame. Find it via a row-wise low-saturation+high-brightness
    pixel count, locate the band's top/bottom, then per-column scan
    OUT from that band to find the exact rink edges. Anchoring the
    per-column search around a known mid-ice row defeats the player-
    occluded-column problem that breaks naïve longest-run detectors.

    Returns {"top": <line>, "bottom": <line>} where each is a line dict
    (a,b,c form + p1/p2 segment), or None.
    """
    empty = {"top": None, "bottom": None}
    if frame is None or frame.size == 0 or frame.ndim < 3:
        return empty
    h, w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    s = hsv[..., 1]
    # Stricter ice mask: ice is desaturated AND bright. Most other bright
    # stuff (advertising, jersey numbers, lights with color cast) has
    # some saturation.
    ice_mask = (s < 50) & (v > 160)

    # Row-wise count of ice pixels — collapses x to find the ice band's
    # vertical extent regardless of horizontal players/gaps.
    row_counts = ice_mask.sum(axis=1).astype(np.float32)
    if row_counts.max() < w * 0.05:  # essentially no ice detected
        return empty

    # Smooth so single-row noise doesn't dominate
    kernel = np.ones(15, dtype=np.float32) / 15
    row_counts_s = np.convolve(row_counts, kernel, mode="same")

    peak_row = int(row_counts_s.argmax())
    peak_val = float(row_counts_s[peak_row])
    if peak_val < w * 0.10:
        return empty

    # Find the OUTER edges of the ice band: topmost AND bottommost rows
    # whose smoothed count exceeds 15% of peak. Lower threshold + outer-
    # most match means painted rink lines (blue/red — they cause dips in
    # the ice mask) don't prematurely terminate the search. The result
    # bounds the per-column scan below.
    thr = peak_val * 0.15
    above = np.where(row_counts_s >= thr)[0]
    if len(above) < 10:
        return empty
    top_y = int(above[0])
    bot_y = int(above[-1])
    # Sanity: top and bottom should sandwich a real ice band
    if bot_y - top_y < h * 0.15:
        return empty

    # Per-column: within the [top_y, bot_y] band found globally, locate
    # the topmost and bottommost ice pixels.
    stride = max(2, w // 300)
    edge_margin_x = max(20, w // 30)
    pad = max(5, int(h * 0.02))
    band_top = max(0, top_y - pad)
    band_bot = min(h, bot_y + pad)
    top_pts = []
    bot_pts = []
    for x in range(edge_margin_x, w - edge_margin_x, stride):
        col = ice_mask[band_top:band_bot, x]
        nz = np.where(col)[0]
        if nz.size < 10:
            continue
        top_pts.append((x, int(nz[0]) + band_top))
        bot_pts.append((x, int(nz[-1]) + band_top))

    # Players + advertising on the boards obscure the rink edges
    # inconsistently per column. The TRUE far-side edge is the topmost
    # row consistently bright across MANY columns — i.e., the points
    # with the LOWEST y values are most likely to lie on the actual far
    # boards. Symmetrically, the true near-side edge has the HIGHEST y
    # values. Filter to those bands before fitting.
    top_filtered = _filter_extremum(top_pts, prefer_low=True)
    bot_filtered = _filter_extremum(bot_pts, prefer_low=False)

    return {
        "top": _fit_line_robust(top_filtered),
        "bottom": _fit_line_robust(bot_filtered),
    }


def _filter_extremum(points: list, prefer_low: bool, percentile: float = 0.4) -> list:
    """Keep only the points whose y is in the desired-extreme percentile.

    For the rink TOP edge, the "true" edge points are those with the
    SMALLEST y (closer to the top of the frame). Mid-rink players /
    occlusions push some per-column tops DOWN; those are outliers.
    By keeping only the lower 40% of y-values, we eliminate occluded
    columns without losing the true edge.
    """
    if not points:
        return points
    ys = sorted(p[1] for p in points)
    cut_idx = max(1, int(len(ys) * percentile))
    if prefer_low:
        threshold = ys[cut_idx]
        return [p for p in points if p[1] <= threshold]
    else:
        threshold = ys[-cut_idx]
        return [p for p in points if p[1] >= threshold]


def detect_rink_edges_via_hough(
    frame: np.ndarray,
    expected_orientation: Optional[float] = None,
) -> dict:
    """Alternate board detector for wide-pano fixed-cam clips (e.g. LiveBarn).

    `detect_rink_boards` above uses ice-color-blob segmentation, which
    works on tight broadcast crops but fails on wide-pano clips where
    overhead lights / rafters get merged into the "ice" blob and skew
    the fitted edges.

    This function instead looks for the two longest near-horizontal lines
    in the lower 70% of the frame via Canny + probabilistic Hough. The
    longest line below the frame midline is "bottom boards" (near side
    of the rink, ice y=0 from camera POV); the longest line above the
    midline but below the top 30% is "top boards" (far side, ice y=85).

    `expected_orientation` is the rink-line slope in radians from the
    similarity prior (if available); we filter detected lines to within
    ±15° of it. When None, accepts any orientation within ±25° of
    horizontal.

    Returns {"top": <line>, "bottom": <line>} where each value is a
    line dict (a,b,c form + p1/p2) or None.
    """
    empty = {"top": None, "bottom": None}
    if frame is None or frame.size == 0 or frame.ndim < 3:
        return empty
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Canny on the lower part of the frame — the rink is typically in
    # the bottom 70%, and excluding the top reduces false positives from
    # roof structure / scoreboard / crowd.
    roi_top = int(h * 0.05)  # keep a tiny strip above for far boards
    roi = gray[roi_top:, :]
    edges = cv2.Canny(roi, 50, 150)
    # Dilate a hair so short collinear segments merge into one Hough vote
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # Probabilistic Hough — require long segments (40% of width minimum)
    min_len = int(w * 0.30)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=80,
        minLineLength=min_len, maxLineGap=30,
    )
    if lines is None or len(lines) == 0:
        return empty

    # Filter by orientation
    orient_tol_rad = (15 if expected_orientation is not None else 25) * np.pi / 180
    base_angle = expected_orientation if expected_orientation is not None else 0.0

    candidates = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        # Coordinates in original-frame space (add roi_top to y)
        y1 += roi_top
        y2 += roi_top
        dx = x2 - x1
        dy = y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length < min_len:
            continue
        angle = np.arctan2(dy, dx)  # [-pi, pi]
        # Normalize to ±pi/2 (we care about line orientation, not direction)
        while angle > np.pi / 2:
            angle -= np.pi
        while angle < -np.pi / 2:
            angle += np.pi
        # Compare against expected orientation (also normalized)
        diff = angle - base_angle
        while diff > np.pi / 2:
            diff -= np.pi
        while diff < -np.pi / 2:
            diff += np.pi
        if abs(diff) > orient_tol_rad:
            continue
        mid_y = (y1 + y2) / 2.0
        candidates.append((length, mid_y, (x1, y1), (x2, y2)))

    if not candidates:
        return empty

    # Group candidates by mid-y into clusters (lines from the same physical
    # board form a thick stripe of similar mid-y values). Then pick the
    # longest line per cluster.
    candidates.sort(key=lambda c: c[1])  # by mid_y ascending
    clusters: list = []
    cluster_tol = max(15, h // 40)
    for c in candidates:
        if clusters and abs(c[1] - clusters[-1][-1][1]) <= cluster_tol:
            clusters[-1].append(c)
        else:
            clusters.append([c])

    # For each cluster, keep the longest line
    representatives = []
    for cluster in clusters:
        rep = max(cluster, key=lambda c: c[0])
        representatives.append(rep)

    if not representatives:
        return empty

    # Pick top + bottom — by mid-y
    representatives.sort(key=lambda r: r[1])

    def _line_from_endpoints(p1, p2):
        return line_from_segment(p1, p2)

    top_line = _line_from_endpoints(representatives[0][2], representatives[0][3])
    bottom_line = _line_from_endpoints(representatives[-1][2], representatives[-1][3])

    # Sanity: top and bottom should be separated by a reasonable fraction
    # of the frame height. If both are basically the same line, this isn't
    # a useful rink detection.
    if representatives[-1][1] - representatives[0][1] < h * 0.15:
        # Only one band found — assign by position. If it's in the upper
        # half, call it top; bottom otherwise.
        if representatives[0][1] < h * 0.5:
            return {"top": top_line, "bottom": None}
        return {"top": None, "bottom": top_line}

    return {"top": top_line, "bottom": bottom_line}


def _fit_line_robust(points: list) -> Optional[dict]:
    """RANSAC-ish line fit via cv2.fitLine + outlier rejection.

    Returns dict {a, b, c, p1, p2} or None if too few points.
    Line equation form: a*x + b*y + c = 0 with a²+b² = 1.
    """
    if len(points) < 8:
        return None
    arr = np.array(points, dtype=np.float32)

    # cv2.fitLine returns (vx, vy, x0, y0) — direction unit vector + a
    # point on the line. Use DIST_L2 as a baseline; could swap for L1 to
    # be more robust to outliers if real-world tests warrant it.
    vx, vy, x0, y0 = cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    # Convert to ax+by+c=0 form. Normal is perpendicular to direction.
    nx, ny = -vy, vx
    norm = (nx * nx + ny * ny) ** 0.5
    if norm < 1e-6:
        return None
    a = nx / norm
    b = ny / norm
    c = -(a * x0 + b * y0)

    # Reject obvious bad fits — too few inliers (within 10 px of the fitted line)
    residuals = np.abs(arr[:, 0] * a + arr[:, 1] * b + c)
    inliers = residuals < 10.0
    if inliers.sum() < max(8, len(points) // 3):
        return None

    # Get segment endpoints by clipping to the inlier x-range
    inlier_pts = arr[inliers]
    x_min = float(inlier_pts[:, 0].min())
    x_max = float(inlier_pts[:, 0].max())
    # Use the line equation to compute y at x_min and x_max
    if abs(b) > 1e-6:
        p1 = (x_min, -(a * x_min + c) / b)
        p2 = (x_max, -(a * x_max + c) / b)
    else:
        p1 = (-c / a, float(inlier_pts[:, 1].min()))
        p2 = (-c / a, float(inlier_pts[:, 1].max()))

    return {"a": float(a), "b": float(b), "c": float(c), "p1": p1, "p2": p2}


def line_intersection(line_a: dict, line_b: dict) -> Optional[tuple]:
    """Intersect two lines given in ax+by+c=0 form. Returns (x, y) or None
    if parallel."""
    if line_a is None or line_b is None:
        return None
    a1, b1, c1 = line_a["a"], line_a["b"], line_a["c"]
    a2, b2, c2 = line_b["a"], line_b["b"], line_b["c"]
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return (float(x), float(y))


def line_from_segment(p1: tuple, p2: tuple) -> dict:
    """Convert a (p1, p2) segment to ax+by+c=0 normalized form.

    Useful for converting the rink-line detector's segment-form output
    into a form line_intersection() can consume.
    """
    x1, y1 = p1
    x2, y2 = p2
    a = float(y2 - y1)
    b = float(x1 - x2)
    c = float(x2 * y1 - x1 * y2)
    norm = (a * a + b * b) ** 0.5
    if norm < 1e-6:
        return {"a": 1.0, "b": 0.0, "c": 0.0}
    return {"a": a / norm, "b": b / norm, "c": c / norm,
            "p1": (float(x1), float(y1)), "p2": (float(x2), float(y2))}


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
