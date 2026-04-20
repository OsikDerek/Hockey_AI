"""Open ice detection via a nav-mesh-style distance field.

Old approach (pairwise gaps between opponents) missed two things: pockets of
space along the boards (since it only considered midpoints between two
players), and contiguous open regions spanning multiple defenders. This
rewrite uses a distance-field / nav-mesh approach:

1. Treat every player + goalie (both teams) as a node.
2. Treat the frame edges as implicit walls (boards) — distance naturally
   drops off there, so board-pockets are scored correctly.
3. Rasterize a coarse grid; for each cell, compute distance to nearest node.
4. Threshold cells with distance > min_clear_distance = "open."
5. Connected-component the open cells; keep components with area above a
   minimum.
6. Score each component by: zone weighting (offensive > neutral > defensive),
   distance to carrier, "roominess" (peak distance inside the component),
   teammate presence within the component, and a stretch-space bump when a
   teammate has already passed the deepest defender on the attack side.

Output schema is preserved so game_annotator doesn't break.
"""

import numpy as np
from typing import Optional

from scipy import ndimage

from .game_context import TrackedObject


class SpaceDetector:
    """Find tactical open-ice pockets using a distance-field nav-mesh."""

    def __init__(
        self,
        cell_size: int = 40,                 # pixels per grid cell
        min_clear_distance: float = 80.0,    # cells farther than this from any player are "open"
        min_component_cells: int = 20,       # discard components smaller than this
        carrier_relevant_radius: float = 400.0,
        max_gaps: int = 3,
    ):
        self.cell_size = cell_size
        self.min_clear_distance = min_clear_distance
        self.min_component_cells = min_component_cells
        self.carrier_relevant_radius = carrier_relevant_radius
        self.max_gaps = max_gaps

    def find_open_spaces(
        self,
        players: list,
        carrier: Optional[TrackedObject],
        goalies: list,
        frame_width: int,
        frame_height: int,
        teammates: list = None,
        opponents: list = None,
        zone: Optional[str] = None,
    ) -> list:
        """Find nav-mesh open-space pockets.

        Returns list of gap dicts sorted by value_score, capped at max_gaps.
        Output schema matches the legacy pairwise-gap detector so
        game_annotator keeps working unchanged.
        """
        # Collect all node positions (both teams + goalies). Boards handled
        # implicitly via frame edges dominating distance-to-nearest.
        nodes = [np.array(p.center, dtype=np.float32) for p in players]
        for g in goalies:
            nodes.append(np.array(g.center, dtype=np.float32))

        if len(nodes) < 2:
            return []

        nodes = np.stack(nodes, axis=0)

        # Build grid of cell centers
        gw = max(1, frame_width // self.cell_size)
        gh = max(1, frame_height // self.cell_size)
        xs = (np.arange(gw) + 0.5) * self.cell_size
        ys = (np.arange(gh) + 0.5) * self.cell_size
        gx, gy = np.meshgrid(xs, ys)
        cell_centers = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (gw*gh, 2)

        # Distance from each cell to each node, take the min → distance field
        # (gw*gh, N) via broadcasting
        diffs = cell_centers[:, None, :] - nodes[None, :, :]
        sq_dist = np.sum(diffs * diffs, axis=2)
        nearest_sq = np.min(sq_dist, axis=1)
        distance_field = np.sqrt(nearest_sq).reshape(gh, gw)

        # Threshold: cells where the nearest node is far enough away = "open"
        open_mask = distance_field > self.min_clear_distance
        if not open_mask.any():
            return []

        # Connected-component the open cells
        labeled, n_components = ndimage.label(open_mask)
        if n_components == 0:
            return []

        # Opponent-side geometry for stretch-space check
        deepest_def_x = None
        attack_sign = 0  # +1 = carrier attacks rightward, -1 = leftward
        if carrier is not None and opponents:
            cx = float(carrier.center[0])
            # Rough heuristic: carrier's attacking direction is toward the
            # side with the most opponents (they're defending their own end).
            opp_xs = np.array([o.center[0] for o in opponents], dtype=np.float32)
            opp_center = float(opp_xs.mean())
            if opp_center > cx:
                attack_sign = 1
                deepest_def_x = float(opp_xs.max())   # rightmost defender
            elif opp_center < cx:
                attack_sign = -1
                deepest_def_x = float(opp_xs.min())   # leftmost defender

        zone_weight = {"offensive": 1.25, "neutral": 1.0, "defensive": 0.6}.get(zone or "neutral", 1.0)

        gaps = []
        carrier_pos = np.array(carrier.center, dtype=np.float32) if carrier is not None else None

        for comp_id in range(1, n_components + 1):
            mask = labeled == comp_id
            n_cells = int(mask.sum())
            if n_cells < self.min_component_cells:
                continue

            # Component centroid (in pixel coords)
            ys_idx, xs_idx = np.where(mask)
            cx = float(((xs_idx + 0.5) * self.cell_size).mean())
            cy = float(((ys_idx + 0.5) * self.cell_size).mean())
            midpoint = np.array([cx, cy], dtype=np.float32)

            # Roominess = peak distance-to-nearest-node inside the component
            comp_distances = distance_field[mask]
            roominess = float(comp_distances.max())
            area_px = n_cells * self.cell_size * self.cell_size

            # Pick two "framing" defenders (closest opponents) to preserve the
            # defender_a/defender_b fields in the output schema. When fewer
            # than two opponents are available (e.g., shootout, 1v1), fall
            # back to points offset horizontally from the centroid by the
            # roominess, so the renderer still gets a valid 2-tuple and draws
            # a visible "seam" line across the pocket.
            defender_a = None
            defender_b = None
            if opponents:
                opp_dists = [(float(np.linalg.norm(midpoint - np.array(o.center))), o) for o in opponents]
                opp_dists.sort(key=lambda t: t[0])
                if len(opp_dists) >= 1:
                    defender_a = tuple(opp_dists[0][1].center)
                if len(opp_dists) >= 2:
                    defender_b = tuple(opp_dists[1][1].center)

            if defender_a is None:
                defender_a = (float(cx - roominess), float(cy))
            if defender_b is None:
                defender_b = (float(cx + roominess), float(cy))

            # Score the pocket
            score = 0.0
            reasons = []
            adjacent = False
            dangerous = False
            passable = False

            # Base: roominess as a fraction of min_clear_distance above threshold
            score += 0.15 + min(0.25, (roominess - self.min_clear_distance) / 300.0)
            if roominess > self.min_clear_distance * 1.8:
                reasons.append(f"Open pocket ({roominess:.0f}px roominess)")

            # Zone weighting
            score *= zone_weight
            if zone_weight != 1.0:
                reasons.append(f"{zone or 'neutral'} zone ({zone_weight:.2f}x)")

            # Carrier distance
            if carrier_pos is not None:
                dist_to_carrier = float(np.linalg.norm(midpoint - carrier_pos))
                if dist_to_carrier < self.carrier_relevant_radius * 0.5:
                    adjacent = True
                    score += 0.35
                    reasons.append("Pocket next to carrier")
                elif dist_to_carrier < self.carrier_relevant_radius:
                    score += 0.2
                    passable = True
                    reasons.append("Pocket within reach")
                else:
                    score -= 0.1

            # Slot-area bonus (rough dangerous zone for shot)
            norm_x = cx / max(frame_width, 1)
            norm_y = cy / max(frame_height, 1)
            if 0.30 <= norm_x <= 0.75 and 0.25 <= norm_y <= 0.75:
                dangerous = True
                score += 0.15
                reasons.append("Slot area")

            # Teammate inside the pocket = passable
            if teammates:
                for t in teammates:
                    tpos = np.array(t.center, dtype=np.float32)
                    # Is teammate inside this component?
                    tcell_x = int(tpos[0] // self.cell_size)
                    tcell_y = int(tpos[1] // self.cell_size)
                    if 0 <= tcell_x < gw and 0 <= tcell_y < gh and labeled[tcell_y, tcell_x] == comp_id:
                        passable = True
                        score += 0.2
                        reasons.append(f"#{t.track_id} inside pocket")
                        break

            # Stretch space: teammate strictly past the deepest defender on
            # the attack side, pocket in that same beyond-defender region
            if (
                teammates
                and deepest_def_x is not None
                and attack_sign != 0
            ):
                def past_line(x_val: float) -> bool:
                    return (
                        (attack_sign > 0 and x_val > deepest_def_x)
                        or (attack_sign < 0 and x_val < deepest_def_x)
                    )
                has_stretch_teammate = any(past_line(float(t.center[0])) for t in teammates)
                pocket_past_line = past_line(cx)
                if has_stretch_teammate and pocket_past_line:
                    score += 0.3
                    passable = True
                    reasons.append("Stretch space past defense")

            if score < 0.2:
                continue

            if score >= 0.6:
                rating = "high"
            elif score >= 0.35:
                rating = "medium"
            else:
                rating = "low"

            gaps.append({
                "center": (float(cx), float(cy)),
                "gap_width": float(roominess * 2.0),   # diameter-ish proxy
                "defender_a": defender_a,
                "defender_b": defender_b,
                "value": rating,
                "value_score": float(score),
                "reasons": reasons,
                "adjacent_to_carrier": adjacent,
                "in_dangerous_zone": dangerous,
                "passable": passable,
                "area_px": float(area_px),
                "roominess": float(roominess),
            })

        gaps.sort(key=lambda g: g["value_score"], reverse=True)
        return gaps[:self.max_gaps]
