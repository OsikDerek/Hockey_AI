"""Goalie analysis — sight lines from puck to net openings.

Draws rays from the puck position fanning out to the net edges/corners.
Each ray is color-coded: green = clear path to net, red = goalie blocking.
Shows what scoring options the shooter actually has.

Works for both game situations and shootout/penalty shot analysis.
"""

import cv2
import numpy as np
from typing import Optional

from .game_context import TrackedObject


class GoalieAnalyzer:
    """Analyze goalie positioning via sight lines from puck to net.

    Casts rays from the puck to points along the net opening.
    Each ray is checked against the goalie's bounding box to determine
    if that path to the net is open or blocked.
    """

    def __init__(
        self,
        net_width_ratio: float = 0.065,
        net_height_ratio: float = 0.055,
        num_rays: int = 9,
    ):
        """
        Args:
            net_width_ratio: Estimated net width as fraction of frame width.
            net_height_ratio: Estimated net height as fraction of frame height.
            num_rays: Number of sight lines to cast across the net opening.
        """
        self.net_width_ratio = net_width_ratio
        self.net_height_ratio = net_height_ratio
        self.num_rays = num_rays

    def analyze_shot(
        self,
        carrier: TrackedObject,
        goalie: Optional[TrackedObject],
        goal_landmark: Optional[TrackedObject],
        frame_width: int,
        frame_height: int,
    ) -> Optional[dict]:
        """Analyze scoring options via sight lines.

        Returns dict with:
            - rays: list of {target, blocked, label} for each sight line
            - open_pct: percentage of net that's open
            - best_target: the most open scoring spot
            - shot_angle: angle from carrier to goal center
            - goalie_coverage: fraction of rays blocked
        """
        # Determine goal center
        if goal_landmark is not None:
            goal_center = np.array(goal_landmark.center)
        elif goalie is not None:
            goal_center = np.array(goalie.center)
        else:
            return None

        gx, gy = goal_center
        puck_pos = np.array(carrier.center)

        # Estimate net rectangle
        net_w = frame_width * self.net_width_ratio
        net_h = frame_height * self.net_height_ratio

        # Generate target points along the net opening
        # Fan across the net width and height to create a grid of targets
        targets = []
        labels = ["far high", "far mid", "far low",
                  "center high", "center mid", "center low",
                  "near high", "near mid", "near low"]

        for col, x_frac in enumerate([0.15, 0.5, 0.85]):  # across net width
            for row, y_frac in enumerate([0.15, 0.5, 0.85]):  # across net height
                tx = gx - net_w / 2 + net_w * x_frac
                ty = gy - net_h / 2 + net_h * y_frac
                idx = col * 3 + row
                label = labels[idx] if idx < len(labels) else f"spot_{idx}"
                targets.append({"pos": (float(tx), float(ty)), "label": label})

        # Check each ray against the goalie
        rays = []
        open_count = 0

        for target in targets:
            target_pos = np.array(target["pos"])
            blocked = False

            if goalie is not None:
                blocked = self._ray_hits_goalie(puck_pos, target_pos, goalie)

            rays.append({
                "start": (float(puck_pos[0]), float(puck_pos[1])),
                "end": target["pos"],
                "blocked": blocked,
                "label": target["label"],
            })

            if not blocked:
                open_count += 1

        # Stats
        total_rays = len(rays)
        open_pct = open_count / max(total_rays, 1)
        goalie_coverage = 1.0 - open_pct

        # Best target = first open ray (they're ordered: far high, far mid, etc.)
        best_target = None
        for r in rays:
            if not r["blocked"]:
                best_target = r
                break

        # Shot angle
        dx = float(gx - puck_pos[0])
        dy = float(gy - puck_pos[1])
        distance = np.sqrt(dx * dx + dy * dy)
        shot_angle = float(np.degrees(np.arctan2(abs(dy), abs(dx))))

        return {
            "rays": rays,
            "open_pct": float(open_pct),
            "goalie_coverage": float(goalie_coverage),
            "best_target": best_target,
            "shot_angle": shot_angle,
            "distance": float(distance),
            "goal_center": (float(gx), float(gy)),
            "net_rect": (float(gx - net_w / 2), float(gy - net_h / 2),
                         float(gx + net_w / 2), float(gy + net_h / 2)),
            "carrier_pos": (float(puck_pos[0]), float(puck_pos[1])),
        }

    def _ray_hits_goalie(
        self, start: np.ndarray, end: np.ndarray, goalie: TrackedObject
    ) -> bool:
        """Check if a ray from start to end passes through the goalie bbox."""
        gx1, gy1, gx2, gy2 = goalie.bbox

        # Parametric line: P(t) = start + t * (end - start), t in [0, 1]
        d = end - start
        length = np.linalg.norm(d)
        if length < 1:
            return False

        # Check multiple points along the ray
        for t in np.linspace(0.1, 0.9, 15):
            px = start[0] + t * d[0]
            py = start[1] + t * d[1]
            if gx1 <= px <= gx2 and gy1 <= py <= gy2:
                return True

        return False

    def analyze_shootout(
        self,
        carrier: TrackedObject,
        goalie: Optional[TrackedObject],
        goal_landmark: Optional[TrackedObject],
        frame_width: int,
        frame_height: int,
    ) -> Optional[dict]:
        """Shootout analysis with recommendation."""
        base = self.analyze_shot(carrier, goalie, goal_landmark, frame_width, frame_height)
        if base is None:
            return None

        recommendation = "shot"
        reasoning = []

        if base["open_pct"] > 0.4:
            recommendation = "shot"
            reasoning.append(f"{base['open_pct']:.0%} of net is open — shoot to the opening")
        elif base["open_pct"] > 0.15:
            recommendation = "shot"
            reasoning.append(f"Some net visible ({base['open_pct']:.0%}) — quick release to open spot")
        else:
            recommendation = "deke"
            reasoning.append(f"Only {base['open_pct']:.0%} of net visible — deke to create opening")

        if base["best_target"]:
            reasoning.append(f"Best spot: {base['best_target']['label']}")

        base["recommendation"] = recommendation
        base["reasoning"] = reasoning
        base["is_shootout"] = True
        return base


def draw_goalie_analysis(frame, analysis, alpha=0.7):
    """Draw sight lines from puck to net on the frame.

    Green rays = open path to net
    Red rays = goalie blocking
    """
    if analysis is None:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]

    # Draw net rectangle
    net = analysis["net_rect"]
    nx1, ny1, nx2, ny2 = [int(v) for v in net]
    cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (255, 255, 255), 2)

    # Draw each sight line ray
    puck = tuple(int(v) for v in analysis["carrier_pos"])

    for ray in analysis["rays"]:
        end = tuple(int(v) for v in ray["end"])

        if ray["blocked"]:
            color = (0, 0, 180)  # Red = blocked
            thickness = 1
        else:
            color = (0, 220, 0)  # Green = open
            thickness = 2

        cv2.line(frame, puck, end, color, thickness, cv2.LINE_AA)

        # Small circle at net target point
        if not ray["blocked"]:
            cv2.circle(frame, end, 5, (0, 255, 0), -1)
        else:
            cv2.circle(frame, end, 3, (0, 0, 200), -1)

    # Label best target
    best = analysis.get("best_target")
    if best:
        bx, by = int(best["end"][0]), int(best["end"][1])
        cv2.circle(frame, (bx, by), 8, (0, 255, 100), 2)
        label = best["label"].upper()
        sz = cv2.getTextSize(label, font, 0.4, 1)[0]
        cv2.rectangle(frame, (bx - sz[0] // 2 - 2, by - 18 - sz[1]),
                      (bx + sz[0] // 2 + 2, by - 14), (0, 200, 0), -1)
        cv2.putText(frame, label, (bx - sz[0] // 2, by - 16),
                    font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Open net percentage
    open_pct = analysis["open_pct"]
    pct_color = (0, 200, 0) if open_pct > 0.3 else (0, 180, 200) if open_pct > 0.15 else (0, 0, 200)
    pct_text = f"Net open: {open_pct:.0%}"
    cv2.putText(frame, pct_text, (nx1, ny1 - 10), font, 0.5, pct_color, 2, cv2.LINE_AA)

    # Shot angle
    angle_text = f"Angle: {analysis['shot_angle']:.0f} deg"
    mid_x = (puck[0] + int(analysis["goal_center"][0])) // 2
    mid_y = (puck[1] + int(analysis["goal_center"][1])) // 2
    cv2.putText(frame, angle_text, (mid_x - 30, mid_y - 10),
                font, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

    # Shootout recommendation
    if analysis.get("is_shootout"):
        rec = analysis.get("recommendation", "")
        reasons = analysis.get("reasoning", [])
        rec_text = f"RECOMMEND: {rec.upper()}"
        cv2.putText(frame, rec_text, (10, h - 40), font, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)
        if reasons:
            cv2.putText(frame, reasons[0][:70], (10, h - 15), font, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)
