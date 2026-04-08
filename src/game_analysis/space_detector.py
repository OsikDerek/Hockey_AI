"""Open ice detection — finds tactical gaps between defenders.

NOT "everywhere without a player" — that's the whole rink.
Instead, finds the specific GAPS and SEAMS between opponent players
that a carrier could skate through or pass through. These are the
pockets of space that matter tactically.

Approach:
1. Look at pairs of opponents and find the midpoints between them
2. Check if those midpoints are actually open (no other player nearby)
3. Classify by tactical value (near carrier, in slot, passable)
4. Return only 1-3 most valuable gaps as discrete markers, not shaded regions
"""

import numpy as np
from typing import Optional

from .game_context import TrackedObject


class SpaceDetector:
    """Find tactical gaps between defenders.

    Instead of shading huge regions of empty ice, identifies specific
    exploitable seams and pockets between opponent players.
    """

    def __init__(
        self,
        min_gap_distance: float = 100,
        max_gap_distance: float = 400,
        gap_clear_radius: float = 80,
        carrier_relevant_radius: float = 350,
        max_gaps: int = 3,
    ):
        """
        Args:
            min_gap_distance: Min distance between two opponents to consider a gap (px).
            max_gap_distance: Max distance — beyond this they're too far apart to be a "gap."
            gap_clear_radius: The midpoint must be this far from any other player to be "open."
            carrier_relevant_radius: Only show gaps within this distance of the carrier.
            max_gaps: Maximum number of gaps to return.
        """
        self.min_gap_distance = min_gap_distance
        self.max_gap_distance = max_gap_distance
        self.gap_clear_radius = gap_clear_radius
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
    ) -> list:
        """Find tactical gaps between opponents.

        Returns list of gap dicts sorted by tactical value.
        Each gap has a center point, the two defenders forming it,
        and a tactical classification.
        """
        # We need opponents to find gaps between them
        # If no opponent list, use all non-carrier players as opponents
        if opponents is None or len(opponents) < 2:
            opp_list = [p for p in players if carrier is None or p.track_id != carrier.track_id]
        else:
            opp_list = opponents

        if len(opp_list) < 2:
            return []

        opp_positions = [(o, np.array(o.center)) for o in opp_list]
        all_positions = [np.array(p.center) for p in players]
        for g in goalies:
            all_positions.append(np.array(g.center))

        gaps = []

        # Check every pair of opponents for gaps
        for i in range(len(opp_positions)):
            for j in range(i + 1, len(opp_positions)):
                obj_a, pos_a = opp_positions[i]
                obj_b, pos_b = opp_positions[j]

                dist = float(np.linalg.norm(pos_a - pos_b))

                # Skip if too close (no real gap) or too far (not a defensive pair)
                if dist < self.min_gap_distance or dist > self.max_gap_distance:
                    continue

                # Midpoint of the gap
                midpoint = (pos_a + pos_b) / 2.0

                # Check if the midpoint is actually clear
                # (no other player standing in the gap)
                is_clear = True
                for pos in all_positions:
                    d = np.linalg.norm(midpoint - pos)
                    if d < self.gap_clear_radius:
                        is_clear = False
                        break

                if not is_clear:
                    continue

                # Classify the gap
                value = self._classify_gap(
                    midpoint, dist, carrier, frame_width, frame_height,
                    teammates,
                )

                if value["score"] < 0.2:
                    continue  # Not tactically useful

                gaps.append({
                    "center": (float(midpoint[0]), float(midpoint[1])),
                    "gap_width": dist,
                    "defender_a": obj_a.center,
                    "defender_b": obj_b.center,
                    "value": value["rating"],
                    "value_score": value["score"],
                    "reasons": value["reasons"],
                    "adjacent_to_carrier": value["adjacent"],
                    "in_dangerous_zone": value["dangerous"],
                    "passable": value["passable"],
                })

        # Sort by value
        gaps.sort(key=lambda g: g["value_score"], reverse=True)
        return gaps[:self.max_gaps]

    def _classify_gap(
        self, midpoint, gap_width, carrier, frame_width, frame_height,
        teammates,
    ) -> dict:
        """Classify the tactical value of a gap between defenders."""
        score = 0.0
        reasons = []
        adjacent = False
        dangerous = False
        passable = False

        mx, my = midpoint

        # Gap width bonus (wider = easier to exploit)
        if gap_width > 250:
            score += 0.15
            reasons.append(f"Wide seam ({gap_width:.0f}px)")
        elif gap_width > 150:
            score += 0.1
            reasons.append("Exploitable gap")

        # Adjacent to carrier
        if carrier is not None:
            dist_to_carrier = float(np.linalg.norm(midpoint - np.array(carrier.center)))
            if dist_to_carrier < self.carrier_relevant_radius * 0.5:
                adjacent = True
                score += 0.4
                reasons.append("Gap right next to carrier")
            elif dist_to_carrier < self.carrier_relevant_radius:
                score += 0.2
                reasons.append("Gap within reach")
                passable = True
            else:
                # Too far from carrier to be relevant right now
                score -= 0.1

        # In dangerous zone (slot area)
        norm_x = mx / max(frame_width, 1)
        norm_y = my / max(frame_height, 1)
        if 0.30 <= norm_x <= 0.75 and 0.25 <= norm_y <= 0.75:
            dangerous = True
            score += 0.25
            reasons.append("Slot area")

        # Teammate near the gap (could receive a pass through it)
        if teammates:
            for t in teammates:
                d = float(np.linalg.norm(midpoint - np.array(t.center)))
                if d < self.carrier_relevant_radius:
                    passable = True
                    score += 0.15
                    reasons.append(f"#{t.track_id} can receive through gap")
                    break

        # Rating
        if score >= 0.5:
            rating = "high"
        elif score >= 0.3:
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
