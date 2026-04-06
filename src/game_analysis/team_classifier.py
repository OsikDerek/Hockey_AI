"""Team differentiation via jersey color clustering.

Extracts dominant jersey colors from player bounding boxes and uses
K-means (k=2) to cluster players into two teams. After a warmup period
of color sample collection, new players are assigned to the nearest
cluster center.

Zero new dependencies -- uses only numpy and cv2.
"""

import cv2
import numpy as np
from typing import Optional

from .game_context import TrackedObject


def _numpy_kmeans(data: np.ndarray, k: int = 2, max_iter: int = 20) -> tuple:
    """Minimal K-means using only numpy.

    Args:
        data: (N, D) feature matrix.
        k: Number of clusters.
        max_iter: Maximum iterations.

    Returns:
        (centers, labels) where centers is (k, D) and labels is (N,).
    """
    if len(data) < k:
        # Not enough data -- assign all to cluster 0
        return data[:k] if len(data) > 0 else np.zeros((k, data.shape[1])), np.zeros(len(data), dtype=int)

    # Initialize with k-means++ style: pick first center randomly, rest by distance
    rng = np.random.RandomState(42)
    indices = [rng.randint(len(data))]
    for _ in range(1, k):
        dists = np.min([np.sum((data - data[idx]) ** 2, axis=1) for idx in indices], axis=0)
        probs = dists / (dists.sum() + 1e-10)
        indices.append(rng.choice(len(data), p=probs))

    centers = data[indices].astype(np.float64)

    for _ in range(max_iter):
        # Assign
        dists = np.array([np.sum((data - c) ** 2, axis=1) for c in centers])  # (k, N)
        labels = np.argmin(dists, axis=0)

        # Update
        new_centers = np.empty_like(centers)
        for j in range(k):
            members = data[labels == j]
            if len(members) > 0:
                new_centers[j] = members.mean(axis=0)
            else:
                new_centers[j] = centers[j]

        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers

    return centers, labels


class TeamClassifier:
    """Cluster players into two teams by jersey color.

    During the warmup period, color features are collected from player
    bounding boxes.  After ``warmup_frames`` frames, K-means (k=2) is
    run to establish team cluster centers.  Subsequent players are
    classified by nearest center, and per-track-ID assignments are
    smoothed to avoid flickering.

    Parameters
    ----------
    warmup_frames : int
        Number of frames to collect samples before running the initial
        clustering.  Default 30.
    hue_bins : int
        Number of bins for the Hue channel histogram.  Default 30.
    sat_bins : int
        Number of bins for the Saturation channel histogram.  Default 32.
    min_jersey_pixels : int
        Minimum number of valid pixels in the jersey crop to accept a
        sample.  Default 100.
    """

    def __init__(
        self,
        warmup_frames: int = 30,
        hue_bins: int = 30,
        sat_bins: int = 32,
        min_jersey_pixels: int = 100,
    ):
        self.warmup_frames = warmup_frames
        self.hue_bins = hue_bins
        self.sat_bins = sat_bins
        self.min_jersey_pixels = min_jersey_pixels

        # State
        self._frame_count = 0
        self._warmup_samples: list[tuple[int, np.ndarray]] = []  # (track_id, feature)
        self._centers: Optional[np.ndarray] = None                # (2, D) cluster centers
        self._team_map: dict[int, str] = {}                       # track_id -> "team_a" | "team_b"
        self._vote_counts: dict[int, np.ndarray] = {}             # track_id -> [votes_a, votes_b]
        self._is_warmed_up = False

    # ── public API ────────────────────────────────────────────

    def update(self, frame: np.ndarray, players: list, team_assignments: Optional[dict] = None) -> dict:
        """Process one frame of players and return team assignments.

        Args:
            frame: BGR video frame.
            players: List of ``TrackedObject`` with ``class_name == "player"``.
            team_assignments: Ignored (for future use with manual overrides).

        Returns:
            Dict mapping ``track_id`` -> ``"team_a"`` or ``"team_b"``.
            During warmup, returns ``{}`` (empty).
        """
        self._frame_count += 1

        # Extract color features for every player in this frame
        features: list[tuple[int, np.ndarray]] = []
        for player in players:
            feat = self._extract_jersey_feature(frame, player)
            if feat is not None:
                features.append((player.track_id, feat))

        if not self._is_warmed_up:
            # Accumulate samples
            self._warmup_samples.extend(features)
            if self._frame_count >= self.warmup_frames and len(self._warmup_samples) >= 8:
                self._run_initial_clustering()
            return dict(self._team_map)

        # Post-warmup: classify each new observation and update votes
        for track_id, feat in features:
            label = self._classify(feat)
            self._accumulate_vote(track_id, label)

        # Periodically re-cluster to adapt to camera/lighting changes
        # (every 300 frames = ~10 seconds at 30fps)
        if self._frame_count % 300 == 0 and len(self._warmup_samples) > 20:
            self._recluster(features)

        return dict(self._team_map)

    def get_team(self, track_id: int) -> Optional[str]:
        """Return team assignment for a single track, or None if unknown."""
        return self._team_map.get(track_id)

    @property
    def is_ready(self) -> bool:
        """True once initial clustering is complete."""
        return self._is_warmed_up

    # ── feature extraction ────────────────────────────────────

    def _extract_jersey_feature(self, frame: np.ndarray, obj: TrackedObject) -> Optional[np.ndarray]:
        """Extract a normalised HS histogram from the jersey region.

        Improved extraction:
        - Uses the middle 50% vertically (top 25% = helmet, bottom 25% = skates/ice)
        - Uses the center 70% horizontally (avoids arms at edges)
        - Better ice/shadow masking with adaptive thresholds
        - Applies Gaussian blur to reduce noise from broadcast compression
        """
        x1, y1, x2, y2 = [int(v) for v in obj.bbox]
        h_frame, w_frame = frame.shape[:2]

        # Clamp to frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frame, x2)
        y2 = min(h_frame, y2)

        bh = y2 - y1
        bw = x2 - x1
        if bh < 12 or bw < 8:
            return None

        # Center 50% vertically (skip helmet and skates)
        top = y1 + int(bh * 0.25)
        bot = y1 + int(bh * 0.75)
        # Center 70% horizontally (skip arm edges)
        left = x1 + int(bw * 0.15)
        right = x2 - int(bw * 0.15)
        crop = frame[top:bot, left:right]

        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return None

        # Light blur to reduce compression artifacts
        crop = cv2.GaussianBlur(crop, (3, 3), 0)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Better masking:
        # - Exclude very dark (shadows, black equipment): V > 50
        # - Exclude very bright/white (ice glare): V < 230
        # - Require some saturation (avoid gray/white): S > 25
        # - Also exclude very low saturation + high value (ice itself): not (S < 40 and V > 180)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = (sat > 25) & (val > 50) & (val < 230)
        # Extra: exclude ice-like pixels (low sat, high val)
        ice_mask = (sat < 40) & (val > 180)
        mask = mask & ~ice_mask

        if mask.sum() < self.min_jersey_pixels:
            return None

        # Compute HS histogram (ignore Value to be illumination-robust)
        hist = cv2.calcHist(
            [hsv], [0, 1], mask.astype(np.uint8) * 255,
            [self.hue_bins, self.sat_bins],
            [0, 180, 0, 256],
        )
        hist = hist.flatten().astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    # ── clustering ────────────────────────────────────────────

    def _run_initial_clustering(self):
        """Run K-means on collected warmup samples and assign teams."""
        if len(self._warmup_samples) < 4:
            # Not enough data yet -- extend warmup
            self.warmup_frames += 10
            return

        features = np.array([f for _, f in self._warmup_samples])
        track_ids = [tid for tid, _ in self._warmup_samples]

        self._centers, labels = _numpy_kmeans(features, k=2)
        self._is_warmed_up = True

        # Determine which cluster is "team_a" vs "team_b".  Convention:
        # the cluster whose center has higher mean saturation is team_a
        # (home jerseys tend to be more saturated / darker).  This is
        # arbitrary but keeps assignment stable across runs.
        # If both have similar saturation, just use cluster index order.
        sat_a = self._centers[0, self.hue_bins:].sum()
        sat_b = self._centers[1, self.hue_bins:].sum()
        if sat_b > sat_a:
            # Swap so cluster 0 = team_a (higher saturation)
            self._centers = self._centers[[1, 0]]
            labels = 1 - labels

        # Build initial votes from warmup samples
        for tid, lab in zip(track_ids, labels):
            self._accumulate_vote(tid, int(lab))

    def _recluster(self, recent_features):
        """Periodically refine cluster centers with recent data."""
        if not recent_features:
            return
        recent = np.array([f for _, f in recent_features])
        # Blend recent data with existing centers (80% old, 20% new)
        for i in range(len(self._centers)):
            assigned = recent[np.array([self._classify(f) for f in recent]) == i]
            if len(assigned) > 0:
                new_center = assigned.mean(axis=0)
                self._centers[i] = 0.8 * self._centers[i] + 0.2 * new_center

    def _classify(self, feature: np.ndarray) -> int:
        """Return cluster index (0 = team_a, 1 = team_b) for a feature."""
        dists = np.array([np.sum((feature - c) ** 2) for c in self._centers])
        return int(np.argmin(dists))

    def _accumulate_vote(self, track_id: int, label: int):
        """Add a vote for a track and update the team map by majority."""
        if track_id not in self._vote_counts:
            self._vote_counts[track_id] = np.array([0, 0])
        self._vote_counts[track_id][label] += 1

        majority = int(np.argmax(self._vote_counts[track_id]))
        self._team_map[track_id] = "team_a" if majority == 0 else "team_b"

    # ── diagnostics ───────────────────────────────────────────

    def summary(self) -> dict:
        """Return a diagnostic summary."""
        team_a_ids = [tid for tid, t in self._team_map.items() if t == "team_a"]
        team_b_ids = [tid for tid, t in self._team_map.items() if t == "team_b"]
        return {
            "is_ready": self._is_warmed_up,
            "frames_seen": self._frame_count,
            "team_a_count": len(team_a_ids),
            "team_b_count": len(team_b_ids),
            "team_a_ids": sorted(team_a_ids),
            "team_b_ids": sorted(team_b_ids),
        }
