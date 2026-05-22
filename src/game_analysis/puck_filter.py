"""Puck filtering and single-puck enforcement.

The HockeyAI YOLO model fires on anything that looks puck-shaped, including
glass-board reflections, tarp logos, and dark spots above the rink. There is
only ever ONE puck in play at a time, and it is always on the ice sheet.
This module enforces both constraints, plus a short coast when YOLO drops
the puck for a frame or two.

Pipeline per frame:
1. Build (or reuse) an ice-region mask for the frame.
2. Drop puck candidates whose center is not on ice.
3. Drop puck candidates below `min_conf`.
4. Score remaining candidates: conf + small bonus for proximity to last
   known position (decays as frames-since-seen grows).
5. Keep the single highest-scoring candidate. Warn (rate-limited) if
   multiple candidates passed all filters — typically indicates the
   model genuinely cannot disambiguate.
6. If none survived but the puck was seen within the last `coast_frames`,
   synthesize a "ghost" puck at the last known position
   (confidence=0.0 so consumers can detect it).

Returned objects list has all non-puck objects unchanged plus at most one
puck object.
"""

import cv2
import numpy as np
from typing import Optional

from .game_context import TrackedObject


class PuckFilter:
    """Enforce single-puck + on-ice constraints with short coast logic."""

    def __init__(
        self,
        min_conf: float = 0.18,
        coast_frames: int = 5,
        ice_min_v: int = 180,
        ice_max_s: int = 60,
        ice_min_coverage: float = 0.10,
        proximity_decay_px: float = 200.0,
        proximity_weight: float = 0.3,
    ):
        self.min_conf = min_conf
        self.coast_frames = coast_frames
        self.ice_min_v = ice_min_v
        self.ice_max_s = ice_max_s
        self.ice_min_coverage = ice_min_coverage
        self.proximity_decay_px = proximity_decay_px
        self.proximity_weight = proximity_weight

        # Coast state
        self._last_pos: Optional[tuple] = None
        self._last_track_id: int = -1
        self._frames_since_seen: int = 9999

        # Frame-cache for the ice mask (built once per frame index)
        self._cached_mask: Optional[np.ndarray] = None
        self._cached_frame_idx: int = -1

        # Diagnostics
        self._multi_puck_warnings: int = 0
        self._frame_idx: int = 0

    # ------------------------------------------------------------------ public
    def filter(
        self,
        objects: list,
        frame: np.ndarray,
    ) -> list:
        """Filter the puck class in `objects` to a single survivor (or ghost).

        Returns the same list with all puck-class entries replaced by either
        zero or one chosen puck object.
        """
        self._frame_idx += 1
        h, w = frame.shape[:2]

        # Partition
        non_pucks = [o for o in objects if o.class_name != "puck"]
        puck_candidates = [o for o in objects if o.class_name == "puck"]

        if not puck_candidates:
            # Maybe coast a ghost
            ghost = self._maybe_ghost()
            self._frames_since_seen += 1
            return non_pucks + ([ghost] if ghost is not None else [])

        # Build / reuse ice mask
        ice_mask = self._get_ice_mask(frame, non_pucks)

        # Apply on-ice filter (if mask exists) and conf cutoff
        survivors = []
        for p in puck_candidates:
            if p.confidence < self.min_conf:
                continue
            if ice_mask is not None and not self._is_on_ice(p, ice_mask, w, h):
                continue
            survivors.append(p)

        if not survivors:
            # All candidates rejected; coast if possible
            ghost = self._maybe_ghost()
            self._frames_since_seen += 1
            return non_pucks + ([ghost] if ghost is not None else [])

        # Score and pick best
        chosen = self._pick_best(survivors)

        if len(survivors) > 1:
            # Soft single-puck: warn but keep going. Rate-limit to a few.
            if self._multi_puck_warnings < 5:
                others = [s for s in survivors if s is not chosen]
                self._multi_puck_warnings += 1
                print(
                    f"  PuckFilter: frame {self._frame_idx}: "
                    f"{len(survivors)} puck candidates passed filters "
                    f"(kept conf={chosen.confidence:.2f} at {chosen.center}, "
                    f"dropped={[round(o.confidence, 2) for o in others]})"
                )

        # Update coast state
        self._last_pos = chosen.center
        self._last_track_id = chosen.track_id
        self._frames_since_seen = 0

        return non_pucks + [chosen]

    # --------------------------------------------------------------- internals
    def _is_on_ice(
        self, puck: TrackedObject, ice_mask: np.ndarray, w: int, h: int
    ) -> bool:
        """Check whether the puck sits on or beside ice pixels.

        Samples a window around the puck's bounding box, expanded outward by
        a margin -- NOT the puck's centre. The puck is a small dark object;
        at native broadcast resolution its centre pixels read as non-ice, so
        a centre-only check rejects genuine on-ice pucks. A real puck always
        has ice pixels immediately surrounding it; a false positive in the
        crowd or on the scoreboard does not.
        """
        x1, y1, x2, y2 = puck.bbox
        bw = max(x2 - x1, 4.0)
        bh = max(y2 - y1, 4.0)
        # Expand the bbox outward by ~1x its size (at least 16 px) so the
        # sampled window straddles the puck and the ice around it.
        mx = max(bw, 16.0)
        my = max(bh, 16.0)
        x0 = int(np.clip(x1 - mx, 0, w - 1))
        xe = int(np.clip(x2 + mx, 0, w))
        y0 = int(np.clip(y1 - my, 0, h - 1))
        ye = int(np.clip(y2 + my, 0, h))
        patch = ice_mask[y0:ye, x0:xe]
        if patch.size == 0:
            return False
        # Pass if any pixel in the surrounding window is ice.
        return bool(patch.any())

    def _pick_best(self, survivors: list) -> TrackedObject:
        """Score = conf + proximity_weight * exp(-dist_to_last / decay)."""
        best = None
        best_score = -1.0
        for p in survivors:
            score = p.confidence
            if self._last_pos is not None and self._frames_since_seen <= self.coast_frames:
                dx = p.center[0] - self._last_pos[0]
                dy = p.center[1] - self._last_pos[1]
                dist = float(np.hypot(dx, dy))
                score += self.proximity_weight * float(np.exp(-dist / self.proximity_decay_px))
            if score > best_score:
                best_score = score
                best = p
        return best

    def _maybe_ghost(self) -> Optional[TrackedObject]:
        """Synthesize a ghost puck at last known position if coast is active."""
        if self._last_pos is None:
            return None
        if self._frames_since_seen >= self.coast_frames:
            return None
        cx, cy = self._last_pos
        # 8x8 box around last center -- approximate puck size
        bbox = (cx - 4.0, cy - 4.0, cx + 4.0, cy + 4.0)
        return TrackedObject(
            track_id=self._last_track_id,
            class_name="puck",
            class_id=5,                  # HockeyAI puck class id
            bbox=bbox,
            center=(float(cx), float(cy)),
            confidence=0.0,              # marker for "this is a ghost"
        )

    def _get_ice_mask(
        self, frame: np.ndarray, non_puck_objects: list
    ) -> Optional[np.ndarray]:
        """Build a boolean mask of ice pixels.

        Color path: HSV with high V + low S (ice is bright and desaturated).
        Geometric fallback: bounding rect of all players/goalies/landmarks
        padded outward, used when color mask coverage is below threshold.
        Returns None when neither method has enough signal — in that case the
        on-ice filter is skipped (we accept the puck unfiltered).
        """
        if self._cached_frame_idx == self._frame_idx and self._cached_mask is not None:
            return self._cached_mask

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Ice: bright (V high) + low saturation (gray/white)
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        ice_color = (v >= self.ice_min_v) & (s <= self.ice_max_s)

        coverage = float(ice_color.sum()) / float(h * w)
        mask = ice_color

        if coverage < self.ice_min_coverage:
            # Geometric fallback: hull around all non-puck game objects
            geom = self._geometric_ice_mask(non_puck_objects, w, h)
            if geom is not None:
                mask = geom
            else:
                mask = None

        self._cached_mask = mask
        self._cached_frame_idx = self._frame_idx
        return mask

    @staticmethod
    def _geometric_ice_mask(non_puck_objects: list, w: int, h: int) -> Optional[np.ndarray]:
        """Build an approximate ice mask from bounding rect of game objects.

        Players, goalies, and rink landmarks are all on the ice. Their
        collective bounding rect (padded outward) is a conservative ice
        region. Returns None if too few objects are present to be meaningful.
        """
        usable = [
            o for o in non_puck_objects
            if o.class_name in ("player", "goalie", "centroid", "faceoff", "goal")
        ]
        if len(usable) < 3:
            return None
        xs = []
        ys = []
        for o in usable:
            x1, y1, x2, y2 = o.bbox
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        x_min = max(0, int(min(xs) - 60))
        x_max = min(w, int(max(xs) + 60))
        # Pad bottom more than top: players' feet are near the ice line
        y_min = max(0, int(min(ys) - 40))
        y_max = min(h, int(max(ys) + 80))
        mask = np.zeros((h, w), dtype=bool)
        mask[y_min:y_max, x_min:x_max] = True
        return mask
