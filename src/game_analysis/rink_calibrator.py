"""Pixel <-> ice coordinate calibration from detected rink landmarks.

Phase B0 used a 2D similarity transform (translation + rotation + uniform
scale) anchored on the goal positions and centroid. Phase B0.5 stabilized
that transform across frames so player positions on the minimap (and the
3D viewer in Phase C) move smoothly instead of teleporting.

Phase B1 adds a full 8-DoF homography on top of the similarity. The
similarity acts as a coarse prior: once it converges, we use it to predict
where each NHL faceoff dot should appear in pixels, then snap the YOLO
"faceoff" detections (which are all one class — undifferentiated) to the
nearest predicted dot. With ≥4 confident correspondences plus the goals
and centroid we run `cv2.findHomography` (RANSAC) and EMA-blend the
resulting matrix across frames. Homography wins for pixel<->ice transforms
when valid; we fall back to similarity when not.

Stability strategies (preserved from B0.5):
1. **EMA smoothing** of both the similarity transform AND the homography
   matrix elementwise.
2. **Outlier rejection**: drop fits whose scale changes >50% or origin
   jumps >200 px; drop homographies whose reprojection error exceeds a
   sanity threshold or whose condition number is huge (degenerate).
3. **Side disambiguation by prediction**: when a single goal is visible
   and we already have a previous transform, pick the goal-side guess
   whose predicted pixel position is closest to the observed goal.
4. **Hard reset on camera cuts**: caller invokes .reset() when
   `is_camera_cut` is true so we don't blend across angle changes.

Ice coordinates use the standard NHL convention: x in [0, 200] running
goal-line to goal-line, y in [0, 85] running board-to-board, units feet.
"""

from typing import Optional

import cv2
import numpy as np

from .rink_lines import (
    detect_rink_lines, detect_rink_boards, detect_rink_edges_via_row_brightness,
    line_from_segment, line_intersection,
)


# NHL rink geometry (feet)
RINK_LENGTH_FT = 200.0
RINK_WIDTH_FT = 85.0
GOAL_LINE_FROM_END_FT = 11.0
NHL_GOAL_WIDTH_FT = 6.0
LEFT_GOAL_X_FT = GOAL_LINE_FROM_END_FT
RIGHT_GOAL_X_FT = RINK_LENGTH_FT - GOAL_LINE_FROM_END_FT  # 189
CENTER_X_FT = RINK_LENGTH_FT / 2.0     # 100
CENTER_Y_FT = RINK_WIDTH_FT / 2.0      # 42.5
BLUE_LINE_LEFT_X_FT = 75.0
BLUE_LINE_RIGHT_X_FT = 125.0

# NHL faceoff dot ice coordinates (8 dots — all class "faceoff" in YOLO).
# Dots are 22 ft apart laterally (centered on y=42.5, so y=20.5 / y=64.5)
# and 20 ft from the goal line for D-zone dots, 5 ft from blue lines for
# neutral-zone dots.
NHL_FACEOFF_DOTS_FT = (
    (20.0, 20.5),    # left-D bottom
    (20.0, 64.5),    # left-D top
    (75.0, 20.5),    # left-N bottom
    (75.0, 64.5),    # left-N top
    (125.0, 20.5),   # right-N bottom
    (125.0, 64.5),   # right-N top
    (180.0, 20.5),   # right-D bottom
    (180.0, 64.5),   # right-D top
)


class RinkCalibrator:
    """Maintain a smoothed pixel<->ice transform across frames.

    Holds two fits: a similarity (B0/B0.5) used as a coarse prior and a
    homography (B1) used as the high-fidelity transform when enough
    landmarks are visible. Public transforms prefer the homography when
    valid and fall back to similarity otherwise.
    """

    def __init__(
        self,
        ema_alpha: float = 0.15,                 # weight of new fit (0=hold, 1=replace)
        max_scale_change: float = 0.5,           # reject sim fit if scale changes by >50%
        max_origin_jump_px: float = 200.0,       # reject sim fit if origin moves >200 px
        min_scale_px_per_ft: float = 2.0,        # plausibility floor (rink 4x frame width)
        max_scale_px_per_ft: float = 25.0,       # plausibility ceiling (rink 0.4x frame width)
        # B1 homography params
        homography_ema_alpha: float = 0.25,      # blend weight for new H matrix
        homography_max_reproj_px: float = 30.0,  # reject H if any correspondence reprojects worse
        homography_min_correspondences: int = 4, # minimum point pairs to attempt H
        faceoff_match_max_err_px: float = 120.0, # snap-to-nearest-dot tolerance — loose
                                                  # because the similarity prior is often
                                                  # only approximately right on broadcast cams
        landmark_conf_floor: float = 0.25,       # accept lower-conf rink-landmark detections
                                                  # to feed B1; quality is checked downstream
                                                  # via the homography's reproj-error gate
        # B2: classical CV line + board detection
        line_snap_tolerance_px: float = 80.0,    # max perpendicular distance from a
                                                  # detected pixel-line to the predicted
                                                  # projection of an ice-line for them to
                                                  # be paired
    ):
        self.ema_alpha = ema_alpha
        self.max_scale_change = max_scale_change
        self.max_origin_jump_px = max_origin_jump_px
        self.min_scale_px_per_ft = min_scale_px_per_ft
        self.max_scale_px_per_ft = max_scale_px_per_ft
        self.homography_ema_alpha = homography_ema_alpha
        self.homography_max_reproj_px = homography_max_reproj_px
        self.homography_min_correspondences = homography_min_correspondences
        self.faceoff_match_max_err_px = faceoff_match_max_err_px
        self.landmark_conf_floor = landmark_conf_floor
        self.line_snap_tolerance_px = line_snap_tolerance_px

        # Similarity transform (B0): ice (x_ft, y_ft) -> pixel (px, py)
        # px = origin_px[0] + scale * (cos*x_ft - sin*y_ft)
        # py = origin_px[1] + scale * (sin*x_ft + cos*y_ft)
        self._origin_px: Optional[np.ndarray] = None
        self._scale_px_per_ft: Optional[float] = None
        self._cos: float = 1.0
        self._sin: float = 0.0
        self._calibrated: bool = False

        # Homography transform (B1): 3x3 matrix mapping ice (x_ft, y_ft, 1)
        # to pixel (px, py, 1). Both forward and inverse are kept hot.
        self._homography: Optional[np.ndarray] = None
        self._homography_inv: Optional[np.ndarray] = None
        self._homography_correspondences: int = 0  # count from most recent fit

        self._frame_count: int = 0

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def has_homography(self) -> bool:
        return self._homography is not None

    def reset(self) -> None:
        """Clear all calibration state — call on camera cut."""
        self._origin_px = None
        self._scale_px_per_ft = None
        self._cos = 1.0
        self._sin = 0.0
        self._calibrated = False
        self._homography = None
        self._homography_inv = None
        self._homography_correspondences = 0

    def update(self, rink_landmarks: dict, frame_width: int, frame_height: int,
               frame: Optional[np.ndarray] = None) -> None:
        """Per-frame calibration update with smoothing + outlier rejection.

        When `frame` is provided AND the similarity transform has converged,
        we additionally run B2 line detection and feed sampled points along
        each matched ice-line into the homography fit. This is the path
        that takes broadcast follow-cam from "rare 2-3 YOLO landmarks per
        frame" to "5+ correspondences every frame the rink is in shot."
        """
        self._frame_count += 1

        goals = [g for g in rink_landmarks.get("goal", []) if g.confidence >= 0.4]
        centroids = [c for c in rink_landmarks.get("centroid", []) if c.confidence >= 0.4]
        # Faceoff dots feed the homography only; the reprojection-error gate
        # downstream filters bad matches, so we accept lower-confidence
        # detections here to maximize the chance of ≥4 correspondences.
        faceoffs = [f for f in rink_landmarks.get("faceoff", [])
                    if f.confidence >= self.landmark_conf_floor]

        # Reject obviously bad goal detections (the bbox aspect ratio of a
        # real net is roughly square; far-zoom misdetections are often
        # vertical slivers or thin strips).
        goals = [g for g in goals if self._goal_bbox_plausible(g)]

        # ── Step 1: similarity update (B0/B0.5) ─────────────────────────
        # Side-bin goals using the previous transform if we have one;
        # otherwise fall back to frame-half.
        left_goals, right_goals = self._bin_goals_by_side(goals, frame_width)

        candidate = None
        if left_goals and right_goals:
            candidate = self._fit_from_two_goals(left_goals[0], right_goals[0], centroids)
        elif goals and centroids:
            candidate = self._fit_from_goal_and_centroid(goals[0], centroids[0])
        elif goals:
            candidate = self._fit_from_single_goal(goals[0], frame_width)

        if candidate is not None:
            cand_scale = candidate[1]
            if not (self.min_scale_px_per_ft <= cand_scale <= self.max_scale_px_per_ft):
                pass  # Reject implausible-size fits
            elif self._calibrated and not self._fit_is_plausible(candidate):
                pass  # Outlier — keep previous
            else:
                if not self._calibrated:
                    self._apply_fit(candidate, alpha=1.0)
                else:
                    self._apply_fit(candidate, alpha=self.ema_alpha)
                self._calibrated = True

        # ── Step 2: homography update (B1 + B2) ─────────────────────────
        # The similarity must be valid first — we use it to predict where
        # each NHL faceoff dot AND each rink line should appear, then snap
        # detections to the nearest predicted feature to build correspondences.
        if self._calibrated:
            self._try_homography_fit(
                goals, centroids, faceoffs, left_goals, right_goals,
                frame=frame,
            )

    # ── B1 + B2: homography fit ─────────────────────────────────────────
    def _try_homography_fit(
        self, goals: list, centroids: list, faceoffs: list,
        left_goals: list, right_goals: list,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """Build correspondences from THIS frame's landmarks, fit H, blend.

        Earlier versions accumulated correspondences across a 15-frame
        rolling buffer to scrape together ≥4 points. That backfired badly:
        on a panning broadcast camera, stale pixel coords from older frames
        formed a self-consistent inlier set that RANSAC happily fit, but the
        resulting homography mapped the current pixel space to a tiny patch
        of ice — players visibly stacked on top of each other near the
        boards. We now require ≥4 correspondences in a SINGLE frame; the
        homography will engage less often, but when it does it will be
        spatially honest. Phase B2 (CV line detection) is the right way to
        increase per-frame correspondence count.
        """
        ice_pts = []
        pixel_pts = []

        if left_goals:
            ice_pts.append((LEFT_GOAL_X_FT, CENTER_Y_FT))
            pixel_pts.append(left_goals[0].center)
        if right_goals:
            ice_pts.append((RIGHT_GOAL_X_FT, CENTER_Y_FT))
            pixel_pts.append(right_goals[0].center)
        if centroids:
            ice_pts.append((CENTER_X_FT, CENTER_Y_FT))
            pixel_pts.append(centroids[0].center)
        for ice_pt, pix_pt in self._disambiguate_faceoffs(faceoffs):
            ice_pts.append(ice_pt)
            pixel_pts.append(pix_pt)

        # B2: detect rink lines + boards and use line×board intersections
        # as point correspondences. Each intersection has a TRUE ice
        # coordinate (e.g., left blue × top boards = (75, 0)), so these
        # are higher-quality than the similarity-derived line samples
        # used in the earlier B2 attempt. The pixel-spread + rink-quad
        # geometric checks downstream filter bad fits regardless of
        # source.
        if frame is not None:
            for ice_pt, pix_pt in self._extract_line_correspondences(frame):
                ice_pts.append(ice_pt)
                pixel_pts.append(pix_pt)

        if len(ice_pts) < self.homography_min_correspondences:
            return  # Not enough this frame — keep prior homography (or stay similarity)

        ice_arr = np.array(ice_pts, dtype=np.float64)
        pixel_arr = np.array(pixel_pts, dtype=np.float64)

        # Reject collinear / degenerate sets — needs 2D spread on both axes
        # in BOTH ice-coord AND pixel-coord space. Without the pixel-spread
        # check, we accept correspondence sets where all samples cluster
        # along a single line in pixel space → the resulting homography is
        # grossly underconstrained.
        if (np.ptp(ice_arr[:, 0]) < 20.0) or (np.ptp(ice_arr[:, 1]) < 5.0):
            return
        if (np.ptp(pixel_arr[:, 0]) < 100.0) or (np.ptp(pixel_arr[:, 1]) < 60.0):
            return

        H, mask = cv2.findHomography(
            ice_arr, pixel_arr,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.homography_max_reproj_px,
        )
        if H is None or H.shape != (3, 3):
            return

        # Validate: reproject every (not just inliers) correspondence and
        # measure max error. Bail on any obviously broken fit.
        reproj = self._apply_homography(H, ice_arr)
        errors = np.linalg.norm(reproj - pixel_arr, axis=1)
        if float(errors.max()) > self.homography_max_reproj_px * 2:
            return

        # Geometric sanity check: project the rink corners and verify the
        # resulting quadrilateral isn't degenerate. Without this, line-only
        # correspondences (highly correlated samples along a single line)
        # often produce near-singular Hs that collapse the entire rink to
        # a tiny patch — visible as "all players stacked on top of each
        # other" in the viewer.
        corners_ice = np.array([
            [0.0, 0.0],
            [RINK_LENGTH_FT, 0.0],
            [RINK_LENGTH_FT, RINK_WIDTH_FT],
            [0.0, RINK_WIDTH_FT],
        ], dtype=np.float64)
        corners_px = self._apply_homography(H, corners_ice)
        if not self._rink_quad_is_plausible(corners_px):
            return

        # Sanity: forward-backward stability (H @ inv(H) should approximate I)
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return
        if not np.all(np.isfinite(H_inv)):
            return

        # Normalize so H[2,2] == 1 for consistent EMA blending
        if abs(H[2, 2]) < 1e-8:
            return
        H_norm = H / H[2, 2]

        # EMA-blend with prior homography (or replace on first fit)
        if self._homography is None:
            self._homography = H_norm
        else:
            a = self.homography_ema_alpha
            self._homography = (1.0 - a) * self._homography + a * H_norm
            # Re-normalize after blend
            if abs(self._homography[2, 2]) > 1e-8:
                self._homography = self._homography / self._homography[2, 2]

        try:
            self._homography_inv = np.linalg.inv(self._homography)
        except np.linalg.LinAlgError:
            self._homography_inv = None
            self._homography = None
            return
        self._homography_correspondences = int(len(ice_pts))

    @staticmethod
    def _board_y_band(boards: dict) -> Optional[tuple]:
        """Return (top_y, bottom_y) bracket of the detected ice in pixel
        space, padding outward by 6 px so the band edges don't clip the
        actual painted lines (which sit ~2 px wide near the boards in
        these clips). None when either board edge wasn't detected."""
        if boards["top"] is None or boards["bottom"] is None:
            return None
        top_y = min(boards["top"]["p1"][1], boards["top"]["p2"][1])
        bot_y = max(boards["bottom"]["p1"][1], boards["bottom"]["p2"][1])
        pad = 6
        return (max(0, top_y - pad), bot_y + pad)

    def _detect_boards_cached(self, frame: np.ndarray) -> dict:
        """Detect rink top/bottom edges. Cached for the lifetime of one
        frame so we don't run the detector twice in case downstream call
        sites need it again.

        Combines two detectors:
        - `detect_rink_boards`: the original ice-blob approach. Works on
          tight broadcast crops but returns nothing on wide-pano fixed-cam
          clips (overhead rafters / lights get merged into the "ice" blob).
        - `detect_rink_edges_via_row_brightness`: row-density fallback
          that finds the brightest horizontal band of low-saturation
          pixels. Much more reliable on LiveBarn-style wide-pano views.

        We always run the blob detector first, then fill in any missing
        edge from the row-brightness detector. Never overrides a
        successful blob detection.
        """
        if getattr(self, "_boards_cache_key", None) == id(frame):
            return self._boards_cache_val
        boards = detect_rink_boards(frame)
        if boards["top"] is None or boards["bottom"] is None:
            alt = detect_rink_edges_via_row_brightness(frame)
            if boards["top"] is None and alt["top"] is not None:
                boards["top"] = alt["top"]
            if boards["bottom"] is None and alt["bottom"] is not None:
                boards["bottom"] = alt["bottom"]
        self._boards_cache_key = id(frame)
        self._boards_cache_val = boards
        return boards

    def _extract_line_correspondences(self, frame: np.ndarray) -> list:
        """B2: detect rink lines + boards, intersect them for point correspondences.

        The key insight: each detected rink line (blue/red) intersected with
        a detected board edge gives a TRUE point correspondence at a known
        ice coordinate — e.g., left blue × top boards = (75, 0). Two
        intersections per rink line, both anchored, well-distributed in
        pixel space. This is much better than sampling along the line
        (correlated samples) because each pair has independent ice y info.

        Falls back to line-only sampling when boards aren't detected; both
        paths feed the same homography fit downstream with safety guards.
        """
        if self._scale_px_per_ft is None or self._origin_px is None:
            return []

        # Detect boards FIRST so we can constrain line detection to the
        # ice region. Inside-ice line detection uses a much lower HSV
        # saturation threshold (junior LiveBarn paint is faded), which
        # would false-positive all over jersey/bleacher noise without
        # the band mask.
        boards = self._detect_boards_cached(frame)
        ice_band = self._board_y_band(boards)

        rink_lines = detect_rink_lines(frame, ice_band=ice_band)
        if not rink_lines["blue"] and not rink_lines["red"]:
            return []
        # Determine which detected board is "ice y=0" vs "y=85" using the
        # similarity prior. Project the predicted top + bottom board
        # midpoints, then assign whichever detected board is closer to each.
        board_at_y0 = None  # ice y = 0
        board_at_y85 = None  # ice y = 85
        if boards["top"] is not None or boards["bottom"] is not None:
            pred_y0 = self._similarity_ice_to_pixel((CENTER_X_FT, 0.0))
            pred_y85 = self._similarity_ice_to_pixel((CENTER_X_FT, RINK_WIDTH_FT))
            if pred_y0 is not None and pred_y85 is not None:
                detected = []
                if boards["top"] is not None:
                    detected.append(("top", boards["top"]))
                if boards["bottom"] is not None:
                    detected.append(("bottom", boards["bottom"]))
                # Score each detected board against each ice-y assignment
                # by perpendicular distance from the predicted midpoint.
                for label, board in detected:
                    mid_px = ((board["p1"][0] + board["p2"][0]) / 2,
                              (board["p1"][1] + board["p2"][1]) / 2)
                    d_y0 = abs(board["a"] * pred_y0[0] + board["b"] * pred_y0[1] + board["c"])
                    d_y85 = abs(board["a"] * pred_y85[0] + board["b"] * pred_y85[1] + board["c"])
                    if d_y0 < d_y85:
                        if board_at_y0 is None:
                            board_at_y0 = board
                    else:
                        if board_at_y85 is None:
                            board_at_y85 = board

        # Predict each rink line as a pixel segment via similarity, then
        # match each detected line to its closest prediction (orientation +
        # distance agnostic — see _match_lines).
        blue_x_options = [BLUE_LINE_LEFT_X_FT, BLUE_LINE_RIGHT_X_FT]
        red_x_options = [LEFT_GOAL_X_FT, CENTER_X_FT, RIGHT_GOAL_X_FT]

        def predict_segment(ice_x):
            top = self._similarity_ice_to_pixel((ice_x, 0.0))
            bot = self._similarity_ice_to_pixel((ice_x, RINK_WIDTH_FT))
            if top is None or bot is None:
                return None
            return {"ice_x": ice_x, "top": top, "bot": bot}

        blue_preds = [s for s in (predict_segment(x) for x in blue_x_options) if s]
        red_preds = [s for s in (predict_segment(x) for x in red_x_options) if s]

        matched = []  # list of (ice_x, detected_pixel_line)
        matched.extend(self._match_lines(rink_lines["blue"], blue_preds))
        matched.extend(self._match_lines(rink_lines["red"], red_preds))

        pairs = []
        for ice_x, det_line in matched:
            det_eq = line_from_segment(det_line["p1"], det_line["p2"])
            # PRIMARY: intersect with detected boards (true correspondences)
            if board_at_y0 is not None:
                isect = line_intersection(det_eq, board_at_y0)
                if isect is not None:
                    pairs.append(((float(ice_x), 0.0), (float(isect[0]), float(isect[1]))))
            if board_at_y85 is not None:
                isect = line_intersection(det_eq, board_at_y85)
                if isect is not None:
                    pairs.append(((float(ice_x), float(RINK_WIDTH_FT)),
                                  (float(isect[0]), float(isect[1]))))
            # FALLBACK: if no boards found, sample 3 points along the line
            # with similarity-based y (less reliable but better than nothing)
            if board_at_y0 is None and board_at_y85 is None:
                for px, py in [det_line["p1"], det_line["mid"], det_line["p2"]]:
                    tent = self._similarity_pixel_to_ice((px, py))
                    if tent is None:
                        continue
                    y_clamped = max(-5.0, min(RINK_WIDTH_FT + 5.0, tent[1]))
                    pairs.append(((float(ice_x), float(y_clamped)),
                                  (float(px), float(py))))
        return pairs

    def _match_lines(self, detected_lines: list, predictions: list) -> list:
        """Match each detected pixel-line to its closest predicted ice-line.

        Returns list of (ice_x, detected_line_dict). Greedy 1-to-1; closest
        pair (perpendicular distance) wins each round. Direction agreement
        is required to prevent matching e.g. a horizontal blue line to a
        vertical-projection prediction.
        """
        if not detected_lines or not predictions:
            return []

        def perp_distance(midpoint, seg):
            mx, my = midpoint
            ax, ay = seg["top"]
            bx, by = seg["bot"]
            dx = bx - ax
            dy = by - ay
            n = (dx * dx + dy * dy) ** 0.5
            if n < 1e-6:
                return abs(mx - ax) + abs(my - ay)
            return abs(dy * mx - dx * my + bx * ay - by * ax) / n

        def direction_agreement(line, seg):
            sx = seg["bot"][0] - seg["top"][0]
            sy = seg["bot"][1] - seg["top"][1]
            sn = (sx * sx + sy * sy) ** 0.5
            if sn < 1e-6:
                return 0.0
            return abs((line["ux"] * sx + line["uy"] * sy) / sn)

        scored = []
        for li, ln in enumerate(detected_lines):
            for pi, pred in enumerate(predictions):
                d = perp_distance(ln["mid"], pred)
                if direction_agreement(ln, pred) < 0.5:
                    continue
                scored.append((d, li, pi, ln, pred))
        scored.sort(key=lambda t: t[0])

        used_dets = set()
        used_preds = set()
        out = []
        for d, li, pi, ln, pred in scored:
            if d > self.line_snap_tolerance_px:
                break
            if li in used_dets or pi in used_preds:
                continue
            used_dets.add(li)
            used_preds.add(pi)
            out.append((pred["ice_x"], ln))
        return out

    def _disambiguate_faceoffs(self, faceoffs: list) -> list:
        """Match each detected faceoff dot to its NHL-spec position.

        Strategy: predict where each of the 8 NHL dots SHOULD appear in
        pixel space using the current similarity transform, then for each
        detection greedily pair it with its nearest predicted dot. Each
        NHL dot is consumed at most once. Drop pairs whose pixel residual
        exceeds the snap tolerance.

        Returns list of (ice_pt, pixel_pt) tuples.
        """
        if not faceoffs or self._scale_px_per_ft is None:
            return []

        # Predict pixel position of each NHL dot.
        predictions = []
        for ice_pt in NHL_FACEOFF_DOTS_FT:
            pred = self._similarity_ice_to_pixel(ice_pt)
            if pred is not None:
                predictions.append((ice_pt, pred))
        if not predictions:
            return []

        # Greedy nearest-neighbor matching. Sort detections by their best
        # match distance so the most-confident pairs are made first.
        unmatched_dets = list(faceoffs)
        unmatched_preds = list(predictions)
        pairs = []

        while unmatched_dets and unmatched_preds:
            best = None  # (det_idx, pred_idx, distance)
            for di, det in enumerate(unmatched_dets):
                dx, dy = det.center
                for pi, (_ice, (px, py)) in enumerate(unmatched_preds):
                    d = (dx - px) ** 2 + (dy - py) ** 2
                    if best is None or d < best[2]:
                        best = (di, pi, d)
            if best is None:
                break
            di, pi, d2 = best
            d = d2 ** 0.5
            if d > self.faceoff_match_max_err_px:
                break  # Even the best remaining match is too far
            ice_pt, pix_pt = unmatched_preds[pi][0], unmatched_dets[di].center
            pairs.append((ice_pt, pix_pt))
            unmatched_dets.pop(di)
            unmatched_preds.pop(pi)

        return pairs

    @staticmethod
    def _rink_quad_is_plausible(corners_px: np.ndarray) -> bool:
        """Reject homographies whose rink-corner projection is degenerate.

        A real rink projects to a quadrilateral with non-zero area, finite
        coords, and reasonable aspect — even the most zoomed-in broadcast
        view, when extrapolated to the full rink corners, projects to a
        large quadrilateral (typically 1M+ px² since the visible portion
        is a small piece of the full rink). The most common degenerate
        failure mode (line-correlated correspondences) maps everything to
        a tiny strip; the most extreme failure produces NaN/inf.
        """
        if corners_px.shape != (4, 2):
            return False
        if not np.all(np.isfinite(corners_px)):
            return False
        # Shoelace area in pixel space
        x = corners_px[:, 0]
        y = corners_px[:, 1]
        area = 0.5 * abs(
            x[0] * (y[1] - y[3]) + x[1] * (y[2] - y[0]) +
            x[2] * (y[3] - y[1]) + x[3] * (y[0] - y[2])
        )
        # Reject obvious collapses. The "stacked players" failure mode
        # produces extremely small areas (the whole rink mapped to a
        # patch of a few thousand px²). Any plausible calibration has
        # >> 100k px² of rink in pixel space.
        if area < 100_000.0:
            return False
        # Aspect ratio sanity: longest side / shortest side. A real rink
        # quad's aspect (in pixel space) shouldn't be wildly extreme.
        sides = [
            float(np.linalg.norm(corners_px[i] - corners_px[(i + 1) % 4]))
            for i in range(4)
        ]
        if min(sides) < 1.0 or (max(sides) / max(min(sides), 1.0)) > 20.0:
            return False
        return True

    @staticmethod
    def _apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply 3x3 H to Nx2 pts; return Nx2."""
        if pts.size == 0:
            return pts
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        homo = np.hstack([pts, ones])
        out = (H @ homo.T).T
        w = out[:, 2:3]
        w[np.abs(w) < 1e-12] = 1e-12
        return out[:, :2] / w

    # ── Similarity fit candidates: each returns (origin_px, scale, cos, sin) ──
    def _fit_from_two_goals(self, left_goal, right_goal, centroids: list):
        left_x, left_y = left_goal.center
        right_x, right_y = right_goal.center
        goal_axis_px = right_x - left_x
        if goal_axis_px <= 1.0:
            return None
        scale = goal_axis_px / (RIGHT_GOAL_X_FT - LEFT_GOAL_X_FT)
        line_y_px = float(centroids[0].center[1]) if centroids else (left_y + right_y) / 2.0
        origin_x = left_x - LEFT_GOAL_X_FT * scale
        origin_y = line_y_px - CENTER_Y_FT * scale
        return (np.array([origin_x, origin_y], dtype=np.float64), float(scale), 1.0, 0.0)

    def _fit_from_goal_and_centroid(self, goal, centroid):
        gx, gy = goal.center
        cx, cy = centroid.center
        goal_ice_x = LEFT_GOAL_X_FT if gx < cx else RIGHT_GOAL_X_FT
        ice_axis_ft = abs(CENTER_X_FT - goal_ice_x)
        goal_axis_px = abs(cx - gx)
        if goal_axis_px <= 1.0:
            return None
        scale = goal_axis_px / ice_axis_ft
        origin_x = gx - goal_ice_x * scale
        origin_y = float(cy) - CENTER_Y_FT * scale
        return (np.array([origin_x, origin_y], dtype=np.float64), float(scale), 1.0, 0.0)

    def _fit_from_single_goal(self, goal, frame_width: int):
        """Single-goal observations only refine the origin, never scale."""
        if self._scale_px_per_ft is None or not self._calibrated:
            return None
        gx, gy = goal.center
        scale = self._scale_px_per_ft
        goal_ice_x = self._best_goal_side_guess(gx, frame_width)
        origin_x = gx - goal_ice_x * scale
        origin_y = gy - CENTER_Y_FT * scale
        return (np.array([origin_x, origin_y], dtype=np.float64), float(scale), 1.0, 0.0)

    # ── Helpers ──
    def _bin_goals_by_side(self, goals: list, frame_width: int):
        """Split detected goals into left-side / right-side groups."""
        if not self._calibrated or self._origin_px is None or self._scale_px_per_ft is None:
            half = frame_width / 2.0
            left = [g for g in goals if g.center[0] < half]
            right = [g for g in goals if g.center[0] >= half]
            return left, right
        left_pred = self._similarity_ice_to_pixel((LEFT_GOAL_X_FT, CENTER_Y_FT))
        right_pred = self._similarity_ice_to_pixel((RIGHT_GOAL_X_FT, CENTER_Y_FT))
        left, right = [], []
        for g in goals:
            gx = g.center[0]
            d_left = abs(gx - left_pred[0]) if left_pred else float("inf")
            d_right = abs(gx - right_pred[0]) if right_pred else float("inf")
            if d_left <= d_right:
                left.append(g)
            else:
                right.append(g)
        return left, right

    def _best_goal_side_guess(self, gx: float, frame_width: int) -> float:
        if self._calibrated and self._origin_px is not None and self._scale_px_per_ft is not None:
            left_pred = self._similarity_ice_to_pixel((LEFT_GOAL_X_FT, CENTER_Y_FT))
            right_pred = self._similarity_ice_to_pixel((RIGHT_GOAL_X_FT, CENTER_Y_FT))
            if left_pred is not None and right_pred is not None:
                if abs(gx - left_pred[0]) <= abs(gx - right_pred[0]):
                    return LEFT_GOAL_X_FT
                return RIGHT_GOAL_X_FT
        return LEFT_GOAL_X_FT if gx < frame_width / 2.0 else RIGHT_GOAL_X_FT

    @staticmethod
    def _goal_bbox_plausible(goal) -> bool:
        x1, y1, x2, y2 = goal.bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        ratio = w / h
        return 0.6 <= ratio <= 3.0

    def _fit_is_plausible(self, candidate) -> bool:
        new_origin, new_scale, _, _ = candidate
        if self._scale_px_per_ft is None or self._origin_px is None:
            return True
        scale_change = abs(new_scale - self._scale_px_per_ft) / max(1e-6, self._scale_px_per_ft)
        if scale_change > self.max_scale_change:
            return False
        origin_jump = float(np.linalg.norm(new_origin - self._origin_px))
        if origin_jump > self.max_origin_jump_px:
            return False
        return True

    def _apply_fit(self, candidate, alpha: float) -> None:
        new_origin, new_scale, new_cos, new_sin = candidate
        if self._origin_px is None or self._scale_px_per_ft is None or alpha >= 1.0:
            self._origin_px = new_origin.copy()
            self._scale_px_per_ft = new_scale
            self._cos = new_cos
            self._sin = new_sin
            return
        self._origin_px = (1.0 - alpha) * self._origin_px + alpha * new_origin
        self._scale_px_per_ft = (1.0 - alpha) * self._scale_px_per_ft + alpha * new_scale
        self._cos = (1.0 - alpha) * self._cos + alpha * new_cos
        self._sin = (1.0 - alpha) * self._sin + alpha * new_sin

    # ── Coordinate transforms ──
    def pixel_to_ice(self, pt) -> Optional[tuple]:
        """Map a pixel point to ice coords. Prefers homography when valid."""
        if self._homography_inv is not None:
            arr = np.array([[float(pt[0]), float(pt[1])]], dtype=np.float64)
            ice = self._apply_homography(self._homography_inv, arr)
            if np.all(np.isfinite(ice)):
                return (float(ice[0, 0]), float(ice[0, 1]))
        return self._similarity_pixel_to_ice(pt)

    def ice_to_pixel(self, pt) -> Optional[tuple]:
        """Map an ice-coord point to pixels. Prefers homography when valid."""
        if self._homography is not None:
            arr = np.array([[float(pt[0]), float(pt[1])]], dtype=np.float64)
            pix = self._apply_homography(self._homography, arr)
            if np.all(np.isfinite(pix)):
                return (float(pix[0, 0]), float(pix[0, 1]))
        return self._similarity_ice_to_pixel(pt)

    # Similarity-only transforms (used internally for predictions even when
    # the homography has taken over the public-facing transform).
    def _similarity_pixel_to_ice(self, pt) -> Optional[tuple]:
        if not self._calibrated or self._origin_px is None or self._scale_px_per_ft is None:
            return None
        px, py = float(pt[0]), float(pt[1])
        dx = px - self._origin_px[0]
        dy = py - self._origin_px[1]
        x_ft = (self._cos * dx + self._sin * dy) / self._scale_px_per_ft
        y_ft = (-self._sin * dx + self._cos * dy) / self._scale_px_per_ft
        return (float(x_ft), float(y_ft))

    def _similarity_ice_to_pixel(self, pt) -> Optional[tuple]:
        if not self._calibrated or self._origin_px is None or self._scale_px_per_ft is None:
            return None
        x_ft, y_ft = float(pt[0]), float(pt[1])
        rx = self._cos * x_ft - self._sin * y_ft
        ry = self._sin * x_ft + self._cos * y_ft
        px = self._origin_px[0] + self._scale_px_per_ft * rx
        py = self._origin_px[1] + self._scale_px_per_ft * ry
        return (float(px), float(py))
