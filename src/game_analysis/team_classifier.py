"""Team differentiation via jersey color clustering.

Revised approach: Uses mean BGR color of the jersey region as the feature
(simple, low-dimensional, robust) instead of HS histograms. K-means on
3D color space with better masking that keeps more jersey pixels.

Zero new dependencies -- uses only numpy and cv2.
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


class TeamClassifier:
    """Cluster players into two teams by jersey color.

    Uses mean BGR color of the jersey region as features. Much simpler
    and more robust than histogram-based approaches for broadcast footage.
    """

    def __init__(self, warmup_frames: int = 40, min_samples: int = 15):
        """
        Args:
            warmup_frames: Frames to collect samples before clustering.
            min_samples: Minimum color samples needed to attempt clustering.
        """
        self.warmup_frames = warmup_frames
        self.min_samples = min_samples

        self._frame_count = 0
        self._warmup_samples = []       # list of (track_id, [B, G, R])
        self._centers = None             # (2, 3) BGR cluster centers
        self._team_map = {}              # track_id -> "team_a" | "team_b"
        self._vote_counts = {}           # track_id -> [votes_a, votes_b]
        self._is_warmed_up = False

    def update(self, frame: np.ndarray, players: list, team_assignments=None) -> dict:
        """Process one frame and return team assignments."""
        self._frame_count += 1

        features = []
        for player in players:
            color = self._extract_jersey_color(frame, player)
            if color is not None:
                features.append((player.track_id, color))

        if not self._is_warmed_up:
            self._warmup_samples.extend(features)
            if self._frame_count >= self.warmup_frames and len(self._warmup_samples) >= self.min_samples:
                self._run_clustering()
            return dict(self._team_map)

        for track_id, color in features:
            label = self._classify(color)
            self._accumulate_vote(track_id, label)

        # Periodic re-cluster to adapt
        if self._frame_count % 200 == 0 and features:
            self._refine_centers(features)

        return dict(self._team_map)

    def get_team(self, track_id: int) -> Optional[str]:
        return self._team_map.get(track_id)

    @property
    def is_ready(self) -> bool:
        return self._is_warmed_up

    def _extract_jersey_color(self, frame: np.ndarray, obj: TrackedObject) -> Optional[np.ndarray]:
        """Extract mean BGR color from the jersey region.

        Key improvements over histogram approach:
        - Returns simple 3D BGR vector (not 960D histogram)
        - Relaxed masking to keep more jersey pixels
        - Uses median instead of mean (more robust to outliers)
        - Wider crop region for more data
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

        # Jersey region: middle 40% vertically (sweet spot for torso)
        # Center 60% horizontally
        top = y1 + int(bh * 0.30)
        bot = y1 + int(bh * 0.70)
        left = x1 + int(bw * 0.20)
        right = x2 - int(bw * 0.20)

        crop = frame[top:bot, left:right]
        if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
            return None

        # Light blur
        crop = cv2.GaussianBlur(crop, (5, 5), 0)

        # Convert to HSV for masking, but use BGR for features
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # RELAXED mask: keep anything that isn't pure white (ice) or pure black (shadow)
        # Ice: high value + low saturation
        # Shadow: very low value
        ice_mask = (val > 200) & (sat < 30)
        shadow_mask = val < 35
        valid_mask = ~ice_mask & ~shadow_mask

        valid_pixels = crop[valid_mask]
        if len(valid_pixels) < 20:
            return None

        # Use median BGR (robust to outliers like equipment, numbers, logos)
        median_color = np.median(valid_pixels, axis=0).astype(np.float32)
        return median_color

    def _run_clustering(self):
        """Run K-means on warmup samples."""
        if len(self._warmup_samples) < self.min_samples:
            self.warmup_frames += 15
            return

        colors = np.array([c for _, c in self._warmup_samples], dtype=np.float64)
        track_ids = [tid for tid, _ in self._warmup_samples]

        self._centers, labels = _numpy_kmeans(colors, k=2)
        self._is_warmed_up = True

        # Check cluster quality: centers should be meaningfully different
        center_dist = np.linalg.norm(self._centers[0] - self._centers[1])
        if center_dist < 15:
            # Clusters too similar — classification won't be reliable
            print(f"  TeamClassifier: WARNING - cluster centers very similar "
                  f"(dist={center_dist:.1f}), classification may be unreliable")

        # Convention: redder cluster = team_a, bluer cluster = team_b.
        # Use (R - B) dominance instead of brightness because it's stable under
        # lighting changes (brightness swings with arena lighting; the color
        # axis between two jerseys stays constant).
        # BGR indexing: [0]=B, [2]=R.
        rb0 = float(self._centers[0][2] - self._centers[0][0])
        rb1 = float(self._centers[1][2] - self._centers[1][0])
        if rb1 > rb0:
            self._centers = self._centers[[1, 0]]
            labels = 1 - labels

        print(f"  TeamClassifier: Clustered. Center A={self._centers[0].astype(int)} "
              f"Center B={self._centers[1].astype(int)} dist={center_dist:.1f}")

        for tid, lab in zip(track_ids, labels):
            self._accumulate_vote(tid, int(lab))

    def _classify(self, color: np.ndarray) -> int:
        dists = np.array([np.linalg.norm(color - c) for c in self._centers])
        return int(np.argmin(dists))

    def _accumulate_vote(self, track_id: int, label: int):
        if track_id not in self._vote_counts:
            self._vote_counts[track_id] = np.array([0, 0])
        self._vote_counts[track_id][label] += 1
        majority = int(np.argmax(self._vote_counts[track_id]))
        self._team_map[track_id] = "team_a" if majority == 0 else "team_b"

    def _refine_centers(self, recent_features):
        """Blend recent observations into cluster centers.

        Uses a 70/30 (old/new) blend for stability, but jumps to 50/50 if the
        cluster counts are heavily imbalanced (>3:1) — indicates one cluster is
        absorbing outliers and we need to re-center faster.
        """
        if not recent_features:
            return
        colors = np.array([c for _, c in recent_features], dtype=np.float64)
        assignments = np.array([self._classify(c) for c in colors])
        n0 = int((assignments == 0).sum())
        n1 = int((assignments == 1).sum())
        min_n = max(1, min(n0, n1))
        max_n = max(n0, n1)
        imbalanced = (max_n / min_n) > 3.0
        old_w, new_w = (0.5, 0.5) if imbalanced else (0.7, 0.3)
        for i in range(len(self._centers)):
            assigned = colors[assignments == i]
            if len(assigned) > 2:
                new_center = np.median(assigned, axis=0)
                self._centers[i] = old_w * self._centers[i] + new_w * new_center

    def classify_goalie(self, frame: np.ndarray, goalie: TrackedObject) -> Optional[str]:
        """Classify a goalie's team using the same jersey-color model.

        Goalies aren't part of the warmup clustering (their bulky equipment
        alters the color histogram), but once the clusters are established we
        can match a single goalie color sample against them.
        """
        if not self._is_warmed_up or self._centers is None:
            return None
        color = self._extract_jersey_color(frame, goalie)
        if color is None:
            return None
        label = self._classify(color)
        team = "team_a" if label == 0 else "team_b"
        # Track goalie votes the same way we track players so repeated
        # classifications converge.
        self._accumulate_vote(goalie.track_id, label)
        return team

    def summary(self) -> dict:
        team_a_ids = [tid for tid, t in self._team_map.items() if t == "team_a"]
        team_b_ids = [tid for tid, t in self._team_map.items() if t == "team_b"]
        return {
            "is_ready": self._is_warmed_up,
            "frames_seen": self._frame_count,
            "team_a_count": len(team_a_ids),
            "team_b_count": len(team_b_ids),
            "team_a_ids": sorted(team_a_ids),
            "team_b_ids": sorted(team_b_ids),
            "centers": self._centers.astype(int).tolist() if self._centers is not None else None,
        }
