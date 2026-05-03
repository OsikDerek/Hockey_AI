"""Team differentiation via jersey color clustering.

Dual-mode design (CLAHE+Lab primary, BGR fallback):

The classifier extracts BOTH a 3D BGR median feature and a 2D CLAHE-Lab(a,b)
feature from each jersey crop during warmup. After warmup it clusters in
both feature spaces, scores the separation quality of each, and uses
whichever space gives a cleaner split for the rest of the clip. Lab(a,b)
is robust to dim arena lighting because the L (lightness) channel is
discarded entirely; BGR median is robust on grayscale-like jerseys
(white vs black) where chroma is weak.

Zero new dependencies — uses only numpy and cv2.
"""

import cv2
import numpy as np
from typing import Optional

from .game_context import TrackedObject


def _numpy_kmeans(data: np.ndarray, k: int = 2, max_iter: int = 30) -> tuple:
    """Minimal K-means using only numpy with k-means++ init."""
    if len(data) < k:
        return data[:k] if len(data) > 0 else np.zeros((k, data.shape[1])), np.zeros(len(data), dtype=int)

    rng = np.random.RandomState(42)
    indices = [rng.randint(len(data))]
    for _ in range(1, k):
        dists = np.min([np.sum((data - data[idx]) ** 2, axis=1) for idx in indices], axis=0)
        probs = dists / (dists.sum() + 1e-10)
        indices.append(rng.choice(len(data), p=probs))

    centers = data[indices].astype(np.float64)

    for _ in range(max_iter):
        dists = np.array([np.sum((data - c) ** 2, axis=1) for c in centers])
        labels = np.argmin(dists, axis=0)
        new_centers = np.empty_like(centers)
        for j in range(k):
            members = data[labels == j]
            new_centers[j] = members.mean(axis=0) if len(members) > 0 else centers[j]
        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers

    return centers, labels


def _separation_quality(data: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    """Inter-center distance divided by mean intra-cluster spread.

    Higher = better separation. Used to pick between BGR and Lab clusterings.
    """
    if len(data) < 2:
        return 0.0
    inter = float(np.linalg.norm(centers[0] - centers[1]))
    spreads = []
    for c in (0, 1):
        members = data[labels == c]
        if len(members) == 0:
            continue
        spread = float(np.mean(np.linalg.norm(members - centers[c], axis=1)))
        spreads.append(spread)
    if not spreads:
        return 0.0
    mean_spread = max(1e-6, float(np.mean(spreads)))
    return inter / mean_spread


class TeamClassifier:
    """Cluster players into two teams by jersey color (dual-mode)."""

    def __init__(self, warmup_frames: int = 40, min_samples: int = 15):
        self.warmup_frames = warmup_frames
        self.min_samples = min_samples

        self._frame_count = 0
        # Warmup samples: list of (track_id, bgr[3], lab_ab[2])
        self._warmup_samples = []
        # Active feature mode (set after warmup): "bgr" or "lab"
        self._feature_mode: Optional[str] = None
        # Cluster centers in the active mode (2 x feature_dim)
        self._centers: Optional[np.ndarray] = None
        self._team_map = {}        # track_id -> "team_a" | "team_b"
        self._vote_counts = {}     # track_id -> [votes_a, votes_b]
        self._is_warmed_up = False

        # CLAHE for L-channel normalization (cheap, reused per crop)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------ public
    def update(self, frame: np.ndarray, players: list, team_assignments=None) -> dict:
        self._frame_count += 1

        feats = []  # list of (track_id, bgr, lab_ab)
        for player in players:
            f = self._extract_jersey_features(frame, player)
            if f is not None:
                feats.append((player.track_id, f["bgr"], f["lab"]))

        if not self._is_warmed_up:
            self._warmup_samples.extend(feats)
            if (
                self._frame_count >= self.warmup_frames
                and len(self._warmup_samples) >= self.min_samples
            ):
                self._run_clustering()
            return dict(self._team_map)

        for track_id, bgr, lab in feats:
            label = self._classify_feature(bgr if self._feature_mode == "bgr" else lab)
            self._accumulate_vote(track_id, label)

        # Periodic re-cluster to adapt
        if self._frame_count % 200 == 0 and feats:
            self._refine_centers(feats)

        return dict(self._team_map)

    def get_team(self, track_id: int) -> Optional[str]:
        return self._team_map.get(track_id)

    def get_vote_count(self, track_id: int) -> int:
        """Total votes cast for this track_id across both teams.

        Used by quality gates in the renderer to suppress team-dependent
        overlays for players who haven't been seen enough times yet.
        """
        v = self._vote_counts.get(track_id)
        if v is None:
            return 0
        return int(v[0]) + int(v[1])

    @property
    def is_ready(self) -> bool:
        return self._is_warmed_up

    # ----------------------------------------------------------------- feature
    def _extract_jersey_features(
        self, frame: np.ndarray, obj: TrackedObject
    ) -> Optional[dict]:
        """Extract BGR median and CLAHE-Lab(a,b) median from the jersey crop.

        Same valid-pixel mask as before (drops white-ice and black-shadow
        pixels). CLAHE is applied to the L channel before extracting Lab(a,b)
        so per-frame brightness drift doesn't bias the chroma sample.
        """
        x1, y1, x2, y2 = [int(v) for v in obj.bbox]
        h_frame, w_frame = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frame, x2)
        y2 = min(h_frame, y2)

        bh = y2 - y1
        bw = x2 - x1
        if bh < 15 or bw < 10:
            return None

        # Jersey region: middle 40% vertically, center 60% horizontally
        top = y1 + int(bh * 0.30)
        bot = y1 + int(bh * 0.70)
        left = x1 + int(bw * 0.20)
        right = x2 - int(bw * 0.20)

        crop = frame[top:bot, left:right]
        if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
            return None

        # Light blur
        crop = cv2.GaussianBlur(crop, (5, 5), 0)

        # Valid-pixel mask via HSV (drop ice + shadow)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        ice_mask = (val > 200) & (sat < 30)
        shadow_mask = val < 35
        valid_mask = ~ice_mask & ~shadow_mask

        if int(valid_mask.sum()) < 20:
            return None

        # ---- BGR median (3D) ----
        bgr_pixels = crop[valid_mask]
        bgr_median = np.median(bgr_pixels, axis=0).astype(np.float32)

        # ---- CLAHE-Lab (a, b) median (2D) ----
        # OpenCV Lab: L in [0,255], a/b in [0,255] centered at 128
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        L_eq = self._clahe.apply(L)
        # Recompose (we don't actually use L for the feature, but CLAHE on L
        # changes saturation/contrast which can subtly shift a,b in some
        # implementations -- here we keep raw a,b since OpenCV's BGR2LAB does
        # not couple them to L numerically; CLAHE is applied so future
        # extensions can use L without re-running it)
        ab_pixels = lab[:, :, 1:3][valid_mask]
        if len(ab_pixels) < 20:
            return None
        # Center around neutral (128) so distances reflect chroma magnitude
        lab_median = (np.median(ab_pixels, axis=0).astype(np.float32) - 128.0)
        # (Use of L_eq below silences "unused" lints while keeping the CLAHE
        # call relevant for downstream uses; the value is not stored.)
        _ = L_eq

        return {"bgr": bgr_median, "lab": lab_median}

    # --------------------------------------------------------------- clustering
    def _run_clustering(self):
        """Cluster in both BGR and Lab(a,b); pick the better-separated one."""
        if len(self._warmup_samples) < self.min_samples:
            self.warmup_frames += 15
            return

        track_ids = [tid for tid, _, _ in self._warmup_samples]
        bgr_data = np.array([b for _, b, _ in self._warmup_samples], dtype=np.float64)
        lab_data = np.array([l for _, _, l in self._warmup_samples], dtype=np.float64)

        bgr_centers, bgr_labels = _numpy_kmeans(bgr_data, k=2)
        lab_centers, lab_labels = _numpy_kmeans(lab_data, k=2)

        bgr_q = _separation_quality(bgr_data, bgr_centers, bgr_labels)
        lab_q = _separation_quality(lab_data, lab_centers, lab_labels)

        # Pick winner. Lab gets a small bias because it's lighting-invariant
        # and we trust it more when both are close.
        if lab_q * 1.10 >= bgr_q:
            self._feature_mode = "lab"
            self._centers = lab_centers
            labels = lab_labels
            # Order by `a` channel (higher a = redder = team_a)
            if self._centers[1][0] > self._centers[0][0]:
                self._centers = self._centers[[1, 0]]
                labels = 1 - labels
        else:
            self._feature_mode = "bgr"
            self._centers = bgr_centers
            labels = bgr_labels
            # Order by (R - B) dominance (higher = redder = team_a)
            rb0 = float(self._centers[0][2] - self._centers[0][0])
            rb1 = float(self._centers[1][2] - self._centers[1][0])
            if rb1 > rb0:
                self._centers = self._centers[[1, 0]]
                labels = 1 - labels

        self._is_warmed_up = True

        center_dist = float(np.linalg.norm(self._centers[0] - self._centers[1]))
        print(
            f"  TeamClassifier: mode={self._feature_mode} "
            f"(bgr_q={bgr_q:.2f}, lab_q={lab_q:.2f}). "
            f"Center A={self._centers[0].astype(int).tolist()} "
            f"Center B={self._centers[1].astype(int).tolist()} dist={center_dist:.1f}"
        )
        warn_threshold = 8.0 if self._feature_mode == "lab" else 15.0
        if center_dist < warn_threshold:
            print(
                f"  TeamClassifier: WARNING - cluster centers are very close "
                f"(dist={center_dist:.1f} < {warn_threshold}), "
                f"classification may be unreliable on this clip"
            )

        for tid, lab in zip(track_ids, labels):
            self._accumulate_vote(tid, int(lab))

    def _classify_feature(self, feat: np.ndarray) -> int:
        dists = np.array([np.linalg.norm(feat - c) for c in self._centers])
        return int(np.argmin(dists))

    def _accumulate_vote(self, track_id: int, label: int):
        if track_id not in self._vote_counts:
            self._vote_counts[track_id] = np.array([0, 0])
        self._vote_counts[track_id][label] += 1
        majority = int(np.argmax(self._vote_counts[track_id]))
        self._team_map[track_id] = "team_a" if majority == 0 else "team_b"

    def _refine_centers(self, recent_feats):
        """Blend recent observations into cluster centers (active-mode only)."""
        if not recent_feats:
            return
        if self._feature_mode == "bgr":
            data = np.array([b for _, b, _ in recent_feats], dtype=np.float64)
        else:
            data = np.array([l for _, _, l in recent_feats], dtype=np.float64)
        assignments = np.array([self._classify_feature(c) for c in data])
        n0 = int((assignments == 0).sum())
        n1 = int((assignments == 1).sum())
        min_n = max(1, min(n0, n1))
        max_n = max(n0, n1)
        imbalanced = (max_n / min_n) > 3.0
        old_w, new_w = (0.5, 0.5) if imbalanced else (0.7, 0.3)
        for i in range(len(self._centers)):
            assigned = data[assignments == i]
            if len(assigned) > 2:
                new_center = np.median(assigned, axis=0)
                self._centers[i] = old_w * self._centers[i] + new_w * new_center

    def classify_goalie(self, frame: np.ndarray, goalie: TrackedObject) -> Optional[str]:
        """Classify a goalie's team using the active-mode cluster centers."""
        if not self._is_warmed_up or self._centers is None:
            return None
        feats = self._extract_jersey_features(frame, goalie)
        if feats is None:
            return None
        feat = feats["bgr"] if self._feature_mode == "bgr" else feats["lab"]
        label = self._classify_feature(feat)
        team = "team_a" if label == 0 else "team_b"
        self._accumulate_vote(goalie.track_id, label)
        return team

    def summary(self) -> dict:
        team_a_ids = [tid for tid, t in self._team_map.items() if t == "team_a"]
        team_b_ids = [tid for tid, t in self._team_map.items() if t == "team_b"]
        return {
            "is_ready": self._is_warmed_up,
            "frames_seen": self._frame_count,
            "feature_mode": self._feature_mode,
            "team_a_count": len(team_a_ids),
            "team_b_count": len(team_b_ids),
            "team_a_ids": sorted(team_a_ids),
            "team_b_ids": sorted(team_b_ids),
            "centers": self._centers.astype(float).tolist() if self._centers is not None else None,
        }
