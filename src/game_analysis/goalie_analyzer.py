"""Goalie analysis — shot angles and open net visualization.

Computes the shooting angle from the puck carrier to the goal,
identifies which parts of the net the goalie is NOT covering,
and visualizes open scoring spots.

Works in two modes:
1. Game mode: Shows shot angle + open net spots during gameplay
2. Shootout mode: Detailed analysis of goalie positioning for
   penalty shots / shootouts / breakaways
"""

import cv2
import numpy as np
from typing import Optional

from .game_context import TrackedObject


class GoalieAnalyzer:
    """Analyze goalie positioning and find open net spots.

    Uses the goalie's bounding box as a proxy for their coverage area.
    The net area is estimated from the goal detection or goalie position.
    Open spots are areas of the net not covered by the goalie's body.
    """

    def __init__(
        self,
        net_width_ratio: float = 0.08,
        net_height_ratio: float = 0.06,
    ):
        """
        Args:
            net_width_ratio: Estimated net width as fraction of frame width.
            net_height_ratio: Estimated net height as fraction of frame height.
        """
        self.net_width_ratio = net_width_ratio
        self.net_height_ratio = net_height_ratio

    def analyze_shot(
        self,
        carrier: TrackedObject,
        goalie: Optional[TrackedObject],
        goal_landmark: Optional[TrackedObject],
        frame_width: int,
        frame_height: int,
    ) -> Optional[dict]:
        """Analyze the current shot opportunity.

        Returns dict with:
            - shot_angle: Angle from carrier to goal in degrees
            - goal_center: (x, y) estimated center of the net
            - net_rect: (x1, y1, x2, y2) estimated net rectangle
            - goalie_coverage: 0.0-1.0 how much of the net the goalie covers
            - open_spots: list of {"position": str, "center": (x,y), "size": str}
            - best_spot: the highest-value open spot
        """
        # Estimate goal/net position
        if goal_landmark is not None:
            goal_center = goal_landmark.center
        elif goalie is not None:
            # Estimate goal behind the goalie
            goal_center = goalie.center
        else:
            return None

        gx, gy = goal_center

        # Estimate net rectangle
        net_w = frame_width * self.net_width_ratio
        net_h = frame_height * self.net_height_ratio
        net_rect = (
            gx - net_w / 2, gy - net_h / 2,
            gx + net_w / 2, gy + net_h / 2,
        )

        # Shot angle from carrier to goal
        cx, cy = carrier.center
        dx = gx - cx
        dy = gy - cy
        distance = np.sqrt(dx * dx + dy * dy)
        shot_angle = np.degrees(np.arctan2(abs(dy), abs(dx)))

        # Goalie coverage analysis
        open_spots = []
        goalie_coverage = 0.0
        best_spot = None

        if goalie is not None:
            gx1, gy1, gx2, gy2 = goalie.bbox
            goalie_cx = (gx1 + gx2) / 2
            goalie_cy = (gy1 + gy2) / 2
            goalie_w = gx2 - gx1
            goalie_h = gy2 - gy1

            # Coverage: how much of the net width does the goalie cover?
            if net_w > 0:
                goalie_coverage = min(1.0, goalie_w / net_w)

            # Find open spots by checking 5 target zones on the net
            # (glove high, glove low, blocker high, blocker low, five-hole)
            net_x1, net_y1, net_x2, net_y2 = net_rect

            targets = [
                ("glove high", (net_x1 + net_w * 0.2, net_y1 + net_h * 0.2)),
                ("glove low", (net_x1 + net_w * 0.2, net_y1 + net_h * 0.8)),
                ("blocker high", (net_x1 + net_w * 0.8, net_y1 + net_h * 0.2)),
                ("blocker low", (net_x1 + net_w * 0.8, net_y1 + net_h * 0.8)),
                ("five hole", (goalie_cx, goalie_cy + goalie_h * 0.3)),
            ]

            for name, (tx, ty) in targets:
                # Check if this spot is covered by the goalie bbox
                inside_goalie = (gx1 <= tx <= gx2) and (gy1 <= ty <= gy2)

                if not inside_goalie:
                    # Calculate how far this spot is from the goalie center
                    spot_dist = np.sqrt((tx - goalie_cx) ** 2 + (ty - goalie_cy) ** 2)
                    # Normalize by goalie size
                    openness = min(1.0, spot_dist / max(goalie_w, goalie_h, 1))

                    size = "large" if openness > 0.6 else "medium" if openness > 0.3 else "small"

                    spot = {
                        "position": name,
                        "center": (float(tx), float(ty)),
                        "openness": openness,
                        "size": size,
                    }
                    open_spots.append(spot)

            # Five hole special case: check if goalie legs are apart
            # (goalie bbox wider at bottom = legs spread = five hole open)
            # This is a heuristic — in broadcast view we can't see legs clearly
            # but a wider bottom-half of the goalie bbox suggests butterfly/spread

            # Best spot = most open
            if open_spots:
                open_spots.sort(key=lambda s: s["openness"], reverse=True)
                best_spot = open_spots[0]

        return {
            "shot_angle": float(shot_angle),
            "distance": float(distance),
            "goal_center": (float(gx), float(gy)),
            "net_rect": tuple(float(v) for v in net_rect),
            "goalie_coverage": float(goalie_coverage),
            "open_spots": open_spots,
            "best_spot": best_spot,
            "carrier_pos": carrier.center,
        }

    def analyze_shootout(
        self,
        carrier: TrackedObject,
        goalie: Optional[TrackedObject],
        goal_landmark: Optional[TrackedObject],
        frame_width: int,
        frame_height: int,
    ) -> Optional[dict]:
        """Detailed shootout/penalty shot analysis.

        Same as analyze_shot but with additional context for 1-on-1:
        - Deke vs shot recommendation
        - Goalie depth (how far out they are)
        - Movement direction (which way the shooter is skating)
        """
        base = self.analyze_shot(carrier, goalie, goal_landmark, frame_width, frame_height)
        if base is None:
            return None

        # Additional shootout analysis
        recommendation = "shot"
        reasoning = []

        if goalie is not None:
            gx1, gy1, gx2, gy2 = goalie.bbox
            goalie_cx = (gx1 + gx2) / 2
            goalie_cy = (gy1 + gy2) / 2

            # Goalie depth: how far from the goal line
            # In broadcast view, if goalie is further from goal center = challenging more
            goal_cx, goal_cy = base["goal_center"]
            depth = np.sqrt((goalie_cx - goal_cx) ** 2 + (goalie_cy - goal_cy) ** 2)
            norm_depth = depth / max(frame_height, 1)

            base["goalie_depth"] = float(norm_depth)

            if norm_depth > 0.05:
                reasoning.append("Goalie is challenging — cuts down angle")
                if base["goalie_coverage"] > 0.7:
                    recommendation = "deke"
                    reasoning.append("High coverage — deke to create opening")
                else:
                    recommendation = "shot"
                    reasoning.append("But there are open spots — shoot to the opening")
            else:
                reasoning.append("Goalie is deep in crease — more net visible")
                recommendation = "shot"
                reasoning.append("Shoot to the open corner")

            # If best spot is five hole and goalie is in butterfly
            if base["best_spot"] and "five hole" in base["best_spot"]["position"]:
                if base["best_spot"]["openness"] > 0.4:
                    reasoning.append("Five hole looks open")

        base["recommendation"] = recommendation
        base["reasoning"] = reasoning
        base["is_shootout"] = True

        return base


def draw_goalie_analysis(frame, analysis, alpha=0.7):
    """Draw goalie shot analysis overlay on a frame.

    Args:
        frame: BGR video frame (modified in-place).
        analysis: Dict from GoalieAnalyzer.analyze_shot().
        alpha: Overlay transparency.
    """
    if analysis is None:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]

    carrier_pos = tuple(int(v) for v in analysis["carrier_pos"])
    goal_center = tuple(int(v) for v in analysis["goal_center"])

    # Draw shot angle line
    cv2.line(frame, carrier_pos, goal_center, (0, 200, 255), 2, cv2.LINE_AA)

    # Draw angle arc
    angle = analysis["shot_angle"]
    angle_text = f"{angle:.0f} deg"
    mid_x = (carrier_pos[0] + goal_center[0]) // 2
    mid_y = (carrier_pos[1] + goal_center[1]) // 2
    cv2.putText(frame, angle_text, (mid_x - 20, mid_y - 10),
                font, 0.5, (0, 200, 255), 2, cv2.LINE_AA)

    # Draw net rectangle
    net = analysis["net_rect"]
    nx1, ny1, nx2, ny2 = [int(v) for v in net]
    cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (255, 255, 255), 2)

    # Draw open spots
    overlay = frame.copy()
    for spot in analysis.get("open_spots", []):
        sx, sy = int(spot["center"][0]), int(spot["center"][1])
        openness = spot["openness"]

        # Color by openness: green = very open, yellow = moderate
        if openness > 0.5:
            color = (0, 255, 100)
            radius = 12
        elif openness > 0.3:
            color = (0, 220, 220)
            radius = 9
        else:
            color = (0, 180, 255)
            radius = 6

        # Filled circle for open spot
        cv2.circle(overlay, (sx, sy), radius, color, -1)
        cv2.circle(overlay, (sx, sy), radius, (255, 255, 255), 1)

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    # Label best spot
    best = analysis.get("best_spot")
    if best:
        bx, by = int(best["center"][0]), int(best["center"][1])
        label = best["position"].upper()
        sz = cv2.getTextSize(label, font, 0.4, 1)[0]
        cv2.rectangle(frame, (bx - sz[0] // 2 - 3, by + 14),
                      (bx + sz[0] // 2 + 3, by + 14 + sz[1] + 6), (0, 200, 0), -1)
        cv2.putText(frame, label, (bx - sz[0] // 2, by + 14 + sz[1] + 2),
                    font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Coverage indicator
    coverage = analysis["goalie_coverage"]
    cov_text = f"Coverage: {coverage:.0%}"
    cv2.putText(frame, cov_text, (nx1, ny1 - 8), font, 0.45,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Shootout recommendation
    if analysis.get("is_shootout"):
        rec = analysis.get("recommendation", "")
        reasons = analysis.get("reasoning", [])
        rec_text = f"RECOMMEND: {rec.upper()}"
        cv2.putText(frame, rec_text, (10, h - 40), font, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)
        if reasons:
            cv2.putText(frame, reasons[0], (10, h - 15), font, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)
