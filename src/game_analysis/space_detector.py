"""Open space detection on the ice surface.

Identifies areas of open ice by computing an influence map based on
player positions. Regions far from all players are "open space."
Particularly valuable space is highlighted when it's:
- Adjacent to the puck carrier (immediately usable)
- Reachable by a simple pass from the carrier
- In a dangerous area (slot, between defenders)

Uses a grid-based approach for efficiency — divides the frame into
cells and computes the minimum distance from each cell to any player.
"""

import cv2
import numpy as np
from typing import Optional

from .game_context import TrackedObject


class SpaceDetector:
    """Detect and classify open space on the ice.

    Computes an influence map where each pixel's value represents
    how "open" that area is (distance from nearest player).
    Then identifies significant open regions and classifies them
    by tactical value.
    """

    def __init__(
        self,
        grid_size: int = 20,
        min_space_radius: float = 40,
        carrier_adjacent_radius: float = 150,
        dangerous_zone_x_range: tuple = (0.30, 0.75),
    ):
        """
        Args:
            grid_size: Pixel size of each grid cell for the influence map.
            min_space_radius: Minimum distance from all players to count as "open" (px).
            carrier_adjacent_radius: Distance from carrier to count as "adjacent" (px).
            dangerous_zone_x_range: Normalized x-range for "dangerous" ice (slot area).
        """
        self.grid_size = grid_size
        self.min_space_radius = min_space_radius
        self.carrier_adjacent_radius = carrier_adjacent_radius
        self.dangerous_zone_x_range = dangerous_zone_x_range

    def find_open_spaces(
        self,
        players: list,
        carrier: Optional[TrackedObject],
        goalies: list,
        frame_width: int,
        frame_height: int,
        teammates: list = None,
        opponents: list = None,
    ) -> list:
        """Find significant open space regions.

        Returns list of OpenSpace dicts sorted by tactical value.
        """
        # Build list of all player positions (both teams + goalies)
        all_positions = []
        for p in players:
            all_positions.append(np.array(p.center))
        for g in goalies:
            all_positions.append(np.array(g.center))

        if not all_positions:
            return []

        all_positions = np.array(all_positions)

        # Compute influence map on a grid
        grid_h = frame_height // self.grid_size
        grid_w = frame_width // self.grid_size

        # Grid cell centers
        gy = np.arange(grid_h) * self.grid_size + self.grid_size // 2
        gx = np.arange(grid_w) * self.grid_size + self.grid_size // 2
        grid_x, grid_y = np.meshgrid(gx, gy)
        grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (N, 2)

        # Min distance from each grid cell to any player
        min_dists = np.full(len(grid_points), np.inf)
        for pos in all_positions:
            dists = np.linalg.norm(grid_points - pos, axis=1)
            min_dists = np.minimum(min_dists, dists)

        # Reshape to grid
        dist_map = min_dists.reshape(grid_h, grid_w)

        # Find open cells (above threshold)
        open_mask = dist_map > self.min_space_radius

        if not open_mask.any():
            return []

        # Find contiguous open regions using connected components
        open_uint8 = open_mask.astype(np.uint8) * 255
        # Upscale for better contour detection
        open_upscaled = cv2.resize(open_uint8, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(open_upscaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        spaces = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Skip tiny regions
                continue

            # Bounding rect and centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(contour)

            # Classify tactical value
            value = self._classify_space(
                cx, cy, area, carrier, frame_width, frame_height,
                teammates, opponents,
            )

            spaces.append({
                "center": (cx, cy),
                "contour": contour,
                "area": area,
                "bbox": (x, y, w, h),
                "value": value["rating"],
                "value_score": value["score"],
                "reasons": value["reasons"],
                "adjacent_to_carrier": value["adjacent"],
                "in_dangerous_zone": value["dangerous"],
                "passable": value["passable"],
            })

        # Sort by value score (highest first)
        spaces.sort(key=lambda s: s["value_score"], reverse=True)

        # Return top 5 most valuable spaces
        return spaces[:5]

    def _classify_space(
        self, cx, cy, area, carrier, frame_width, frame_height,
        teammates, opponents,
    ) -> dict:
        """Classify the tactical value of an open space region."""
        score = 0.0
        reasons = []
        adjacent = False
        dangerous = False
        passable = False

        # Size bonus (bigger space = more valuable)
        norm_area = area / (frame_width * frame_height)
        if norm_area > 0.02:
            score += 0.2
            reasons.append("Large open area")
        elif norm_area > 0.005:
            score += 0.1

        # Adjacent to carrier (immediately usable)
        if carrier is not None:
            dist_to_carrier = np.linalg.norm(
                np.array([cx, cy]) - np.array(carrier.center)
            )
            if dist_to_carrier < self.carrier_adjacent_radius:
                adjacent = True
                score += 0.35
                reasons.append("Adjacent to puck carrier")
            elif dist_to_carrier < self.carrier_adjacent_radius * 2:
                score += 0.15
                reasons.append("One pass away")
                passable = True

        # In dangerous zone (between blue line and goal)
        norm_x = cx / max(frame_width, 1)
        if self.dangerous_zone_x_range[0] <= norm_x <= self.dangerous_zone_x_range[1]:
            # Center of ice vertically is more dangerous
            norm_y = cy / max(frame_height, 1)
            if 0.25 <= norm_y <= 0.75:
                dangerous = True
                score += 0.25
                reasons.append("Dangerous ice (slot area)")

        # Has a teammate nearby who could receive a pass into the space
        if teammates:
            for t in teammates:
                dist = np.linalg.norm(np.array([cx, cy]) - np.array(t.center))
                if dist < self.carrier_adjacent_radius * 1.5:
                    passable = True
                    score += 0.15
                    reasons.append(f"Teammate #{t.track_id} can attack this space")
                    break

        # Between opponents (seam space)
        if opponents and len(opponents) >= 2:
            opp_positions = [np.array(o.center) for o in opponents]
            space_pos = np.array([cx, cy])
            # Check if space is between two opponents
            for i in range(len(opp_positions)):
                for j in range(i + 1, len(opp_positions)):
                    mid = (opp_positions[i] + opp_positions[j]) / 2
                    dist_to_mid = np.linalg.norm(space_pos - mid)
                    opp_gap = np.linalg.norm(opp_positions[i] - opp_positions[j])
                    if dist_to_mid < opp_gap * 0.4 and opp_gap > 80:
                        score += 0.2
                        reasons.append("Seam between defenders")
                        break
                else:
                    continue
                break

        # Rating
        if score >= 0.5:
            rating = "high"
        elif score >= 0.25:
            rating = "medium"
        else:
            rating = "low"

        return {
            "score": score,
            "rating": rating,
            "reasons": reasons,
            "adjacent": adjacent,
            "dangerous": dangerous,
            "passable": passable,
        }

    def compute_influence_heatmap(
        self, players: list, goalies: list,
        frame_width: int, frame_height: int,
    ) -> np.ndarray:
        """Compute a full-resolution influence heatmap for visualization.

        Returns a single-channel float32 image (0=crowded, 1=open) at
        frame resolution. Used for overlay rendering.
        """
        all_positions = []
        for p in players:
            all_positions.append(np.array(p.center))
        for g in goalies:
            all_positions.append(np.array(g.center))

        if not all_positions:
            return np.ones((frame_height, frame_width), dtype=np.float32)

        all_positions = np.array(all_positions)

        # Compute on grid, then upscale
        grid_h = frame_height // self.grid_size
        grid_w = frame_width // self.grid_size

        gy = np.arange(grid_h) * self.grid_size + self.grid_size // 2
        gx = np.arange(grid_w) * self.grid_size + self.grid_size // 2
        grid_x, grid_y = np.meshgrid(gx, gy)
        grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        min_dists = np.full(len(grid_points), np.inf)
        for pos in all_positions:
            dists = np.linalg.norm(grid_points - pos, axis=1)
            min_dists = np.minimum(min_dists, dists)

        dist_map = min_dists.reshape(grid_h, grid_w)

        # Normalize: 0 = player standing here, 1 = very open
        max_dist = 200.0  # Cap at 200px
        heatmap = np.clip(dist_map / max_dist, 0, 1).astype(np.float32)

        # Upscale to frame resolution with smooth interpolation
        heatmap = cv2.resize(heatmap, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

        return heatmap
