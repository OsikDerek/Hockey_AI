"""Lane calculation for passing and shooting lane visualization.

Computes passing lanes, shooting lanes, and success probability
estimates from tracked object positions. Pure math — no drawing.
"""

import numpy as np
from typing import Optional

from .game_context import TrackedObject


# Tunable constants — Derek can adjust these
SHOT_BASE_PCT = 0.08         # NHL average shooting percentage
PASS_BASE_PCT = 0.70         # Base pass completion rate
LANE_BLOCK_RADIUS = 50       # Pixels: defender must be this close to "block" a lane
SLOT_X_RANGE = (0.30, 0.70)  # Normalized x-range considered "the slot"
CROSS_ICE_THRESHOLD = 0.35   # Normalized y-distance for cross-ice pass


def _point_to_line_distance(point, line_start, line_end):
    """Compute perpendicular distance from point to a line segment.

    Returns:
        (perp_distance, projection_fraction) where fraction is 0.0 at
        line_start and 1.0 at line_end. Values outside [0,1] mean the
        point projects beyond the segment endpoints.
    """
    p = np.array(point, dtype=np.float64)
    a = np.array(line_start, dtype=np.float64)
    b = np.array(line_end, dtype=np.float64)

    ab = b - a
    ap = p - a
    ab_len_sq = np.dot(ab, ab)

    if ab_len_sq < 1e-8:
        return float(np.linalg.norm(ap)), 0.0

    t = np.dot(ap, ab) / ab_len_sq
    projection = a + t * ab
    perp_dist = float(np.linalg.norm(p - projection))

    return perp_dist, float(t)


class LaneCalculator:
    """Calculate passing and shooting lanes with success estimates."""

    def __init__(
        self,
        lane_block_radius: float = LANE_BLOCK_RADIUS,
        shot_base_pct: float = SHOT_BASE_PCT,
        pass_base_pct: float = PASS_BASE_PCT,
    ):
        self.lane_block_radius = lane_block_radius
        self.shot_base_pct = shot_base_pct
        self.pass_base_pct = pass_base_pct

    def calc_shooting_lane(
        self,
        carrier: TrackedObject,
        goalie: Optional[TrackedObject],
        defenders: list,
        frame_width: int,
        frame_height: int,
    ) -> Optional[dict]:
        """Calculate shooting lane from carrier to goal.

        Returns dict with: start, end, blockers, quality, success_pct, in_slot.
        Returns None if no target (no goalie and can't estimate goal position).
        """
        start = carrier.center

        # Goal target: goalie position or frame edge estimate
        if goalie is not None:
            end = goalie.center
        else:
            # Rough estimate: goal is at the right edge, middle height
            # (assumes offensive zone is to the right in broadcast view)
            end = (frame_width * 0.95, frame_height * 0.5)

        # Find blockers in the lane
        blockers = []
        for d in defenders:
            perp_dist, t = _point_to_line_distance(d.center, start, end)
            if 0.0 < t < 1.0 and perp_dist < self.lane_block_radius:
                blockers.append({
                    "pos": d.center,
                    "perp_dist": perp_dist,
                    "track_id": d.track_id,
                })

        # Quality
        n_blockers = len(blockers)
        if n_blockers == 0:
            quality = "clear"
        elif n_blockers == 1:
            quality = "partial"
        else:
            quality = "blocked"

        # In slot?
        norm_x = carrier.center[0] / max(frame_width, 1)
        in_slot = SLOT_X_RANGE[0] <= norm_x <= SLOT_X_RANGE[1]

        # Distance
        distance = float(np.linalg.norm(np.array(end) - np.array(start)))

        # Success estimate
        success_pct = self.estimate_shot_success(distance, n_blockers, in_slot, frame_width)

        return {
            "start": start,
            "end": end,
            "blockers": blockers,
            "quality": quality,
            "success_pct": success_pct,
            "distance": distance,
            "in_slot": in_slot,
            "num_blockers": n_blockers,
        }

    def calc_passing_lanes(
        self,
        carrier: TrackedObject,
        teammates: list,
        opponents: list,
        frame_width: int,
        frame_height: int,
    ) -> list:
        """Calculate passing lanes from carrier to each teammate.

        Returns list of dicts with: target_id, start, end, blockers_near,
        quality, success_pct, is_cross_ice, distance.
        """
        lanes = []
        start = carrier.center

        for teammate in teammates:
            if teammate.track_id < 0:
                continue

            end = teammate.center
            distance = float(np.linalg.norm(np.array(end) - np.array(start)))

            if distance < 10:  # Skip if basically on top of carrier
                continue

            # Find opponents near the lane
            blockers_near = []
            for opp in opponents:
                perp_dist, t = _point_to_line_distance(opp.center, start, end)
                if 0.0 < t < 1.0 and perp_dist < self.lane_block_radius:
                    blockers_near.append({
                        "pos": opp.center,
                        "perp_dist": perp_dist,
                        "track_id": opp.track_id,
                    })

            # Cross-ice check
            y_diff = abs(carrier.center[1] - teammate.center[1]) / max(frame_height, 1)
            is_cross_ice = y_diff > CROSS_ICE_THRESHOLD

            # Quality
            n_blockers = len(blockers_near)
            if n_blockers == 0:
                quality = "open"
            elif n_blockers == 1:
                quality = "partial"
            else:
                quality = "covered"

            # Success estimate
            success_pct = self.estimate_pass_success(distance, n_blockers, is_cross_ice, frame_width)

            lanes.append({
                "target_id": teammate.track_id,
                "start": start,
                "end": end,
                "blockers_near": blockers_near,
                "quality": quality,
                "success_pct": success_pct,
                "is_cross_ice": is_cross_ice,
                "distance": distance,
                "num_blockers": n_blockers,
            })

        # Sort by success_pct descending (best option first)
        lanes.sort(key=lambda l: l["success_pct"], reverse=True)
        return lanes

    def calc_ambient_connections(
        self,
        carrier: TrackedObject,
        teammates: list,
        opponents: list,
        frame_width: int,
        frame_height: int,
        open_spaces: list = None,
        open_space_radius: float = 110.0,
    ) -> list:
        """Always-on connections from carrier to every teammate.

        Binary outcome: "reachable" (green) or "blocked" (red). A teammate
        is REACHABLE if ANY of these is true:
          1. Direct passing lane has zero opponents in the lane corridor.
          2. Nearest opponent to the teammate is farther than
             `open_space_radius` (teammate is alone — sauce/flip/bank pass
             can still get them the puck even if the direct lane is blocked).
          3. Teammate's position falls inside a nav-mesh open pocket (from
             space_detector.find_open_spaces).

        Otherwise BLOCKED. Returns one entry per teammate (no distance
        filtering — every teammate stays connected at all times).
        """
        connections = []
        start = carrier.center

        # Pre-extract open-pocket cells for fast point-in-pocket checks.
        # Each pocket dict has "center" (cx, cy), "roominess" (peak distance
        # to nearest node). We approximate "inside pocket" as "within
        # roominess radius of pocket center" -- close enough for receiver
        # placement.
        pockets = []
        if open_spaces:
            for sp in open_spaces:
                center = sp.get("center")
                roominess = float(sp.get("roominess", 0.0)) or float(sp.get("gap_width", 0.0)) / 2.0
                if center is not None and roominess > 0:
                    pockets.append((np.array(center, dtype=np.float32), roominess))

        for teammate in teammates:
            if teammate.track_id < 0:
                continue

            end = teammate.center
            distance = float(np.linalg.norm(np.array(end) - np.array(start)))
            tpos = np.array(teammate.center, dtype=np.float32)

            # Single pass over opponents: count direct-lane blockers AND
            # find the nearest opponent to the teammate. Hot path — avoid
            # iterating opponents twice per teammate.
            n_blockers = 0
            nearest_opp_dist = float("inf")
            for opp in opponents:
                opp_center = opp.center
                perp_dist, t = _point_to_line_distance(opp_center, start, end)
                if 0.0 < t < 1.0 and perp_dist < self.lane_block_radius:
                    n_blockers += 1
                d = float(np.linalg.norm(tpos - np.array(opp_center, dtype=np.float32)))
                if d < nearest_opp_dist:
                    nearest_opp_dist = d

            # Inside a nav-mesh open pocket?
            in_pocket = False
            for pcenter, roominess in pockets:
                if float(np.linalg.norm(tpos - pcenter)) < roominess:
                    in_pocket = True
                    break

            reachable = (
                n_blockers == 0
                or nearest_opp_dist > open_space_radius
                or in_pocket
            )
            quality = "reachable" if reachable else "blocked"

            # Reason tag (helps debugging / tooltips)
            if n_blockers == 0:
                reason = "direct lane"
            elif nearest_opp_dist > open_space_radius:
                reason = f"alone ({nearest_opp_dist:.0f}px)"
            elif in_pocket:
                reason = "in open pocket"
            else:
                reason = f"{n_blockers} in lane"

            connections.append({
                "target_id": teammate.track_id,
                "start": start,
                "end": end,
                "quality": quality,
                "distance": distance,
                "num_blockers": n_blockers,
                "nearest_opp_dist": nearest_opp_dist,
                "reason": reason,
            })

        return connections

    def estimate_shot_success(
        self, distance: float, num_blockers: int, in_slot: bool, frame_width: int
    ) -> float:
        """Estimate shot success probability (0.0 - 1.0)."""
        pct = self.shot_base_pct
        pct += 0.03 * max(0, 3 - num_blockers)  # Bonus for clear lanes
        if in_slot:
            pct += 0.05                           # Slot bonus
        pct -= 0.02 * num_blockers               # Penalty per blocker

        # Distance penalty (further = harder)
        norm_dist = distance / max(frame_width, 1)
        pct -= 0.03 * max(0, norm_dist - 0.3)    # Penalty beyond 30% of frame width

        return float(np.clip(pct, 0.01, 0.50))

    def estimate_pass_success(
        self, distance: float, num_blockers: int, is_cross_ice: bool, frame_width: int
    ) -> float:
        """Estimate pass success probability (0.0 - 1.0)."""
        pct = self.pass_base_pct
        pct -= 0.15 * num_blockers               # Penalty per blocker near lane
        if is_cross_ice:
            pct -= 0.10                           # Cross-ice penalty

        # Distance penalty
        norm_dist = distance / max(frame_width, 1)
        pct -= 0.05 * max(0, norm_dist - 0.4)    # Penalty for long passes

        return float(np.clip(pct, 0.05, 0.95))
