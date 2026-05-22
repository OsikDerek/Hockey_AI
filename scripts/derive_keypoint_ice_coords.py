"""Derive ice coordinates for all 56 HockeyRink keypoints.

Inputs:
  1. Derek's identification of 38 keypoints (rink features -> known NHL
     ice coordinates), encoded below. Most come in symmetric PAIRS whose
     top/bottom (or left/right) order is ambiguous.
  2. The reconstructed template (output/_rink_template/template_raw.json)
     — all 55 reconstructible keypoints rigidly tied together in one
     plane at ~2px precision.

Method:
  - Pair midpoints map unambiguously (midpoint of recon pair <-> midpoint
    of the two ice candidates). Fit a bootstrap homography recon->ice
    from those midpoints + the lone center-ice dot.
  - Resolve each pair/group's order: project recon points through the
    bootstrap homography, assign each to its nearest ice candidate.
  - Re-fit the final recon->ice homography on all resolved anchors.
  - Apply it to ALL reconstructed keypoints -> ice coords for the 18
    un-identified ones (hashmarks, circle edges, bench). Identified
    keypoints keep their exact NHL coordinates.

Output:
  src/game_analysis/rink_registration/keypoints.py   (KEYPOINT_ICE_XY)
  output/_rink_template/template_topdown.png         (spot-check render)

NHL rink: 200 x 85 ft. x = end board (0) to end board (200);
y = side board (0) to side board (85); center (100, 42.5).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECON = PROJECT_ROOT / "output/_rink_template/template_raw.json"
OUT_KP = PROJECT_ROOT / "src/game_analysis/rink_registration/keypoints.py"
OUT_RENDER = PROJECT_ROOT / "output/_rink_template/template_topdown.png"

NUM_KP = 56
RINK_L, RINK_W = 200.0, 85.0
CY = 42.5
GOAL_L_X, GOAL_R_X = 11.0, 189.0          # goal lines
BLUE_L_X, BLUE_R_X = 75.0, 125.0          # blue lines
CENTER_X = 100.0
DOT_Y_LO, DOT_Y_HI = 20.5, 64.5           # faceoff dot y (end + neutral)
ENDDOT_L_X, ENDDOT_R_X = 31.0, 169.0      # end-zone faceoff dot x
NEUTDOT_L_X, NEUTDOT_R_X = 80.0, 120.0    # neutral-zone faceoff dot x
GOAL_HALF_W = 3.0                          # goal posts +/- 3 ft
CENTER_CIRCLE_R = 15.0
REF_CIRCLE_R = 10.0
TRAP_GL_HALF = 11.0                        # trapezoid half-width at goal line
TRAP_EB_HALF = 14.0                        # trapezoid half-width at end boards
CREASE_DEPTH = 4.5
CREASE_HALF = 4.0

# Derek's identifications. Each entry: (label, [kp indices], [ice coords]).
# The order within a group is UNKNOWN — resolved programmatically.
GROUPS = [
    # --- faceoff dots ---
    ("end-zone dots LEFT", [14, 15],
        [(ENDDOT_L_X, DOT_Y_LO), (ENDDOT_L_X, DOT_Y_HI)]),
    ("end-zone dots RIGHT", [40, 41],
        [(ENDDOT_R_X, DOT_Y_LO), (ENDDOT_R_X, DOT_Y_HI)]),
    ("neutral-zone dots (4)", [22, 23, 32, 33],
        [(NEUTDOT_L_X, DOT_Y_LO), (NEUTDOT_L_X, DOT_Y_HI),
         (NEUTDOT_R_X, DOT_Y_LO), (NEUTDOT_R_X, DOT_Y_HI)]),
    ("center ice dot", [26], [(CENTER_X, CY)]),
    # --- LEFT-end trapezoid + goal line ---
    ("trapezoid x end-boards LEFT", [0, 1],
        [(0.0, CY - TRAP_EB_HALF), (0.0, CY + TRAP_EB_HALF)]),
    ("trapezoid x goal-line LEFT", [3, 6],
        [(GOAL_L_X, CY - TRAP_GL_HALF), (GOAL_L_X, CY + TRAP_GL_HALF)]),
    ("goal-line x boards LEFT", [2, 7],
        [(GOAL_L_X, 0.0), (GOAL_L_X, RINK_W)]),
    # --- blue lines x boards ---
    ("blue-line x boards LEFT", [20, 21],
        [(BLUE_L_X, 0.0), (BLUE_L_X, RINK_W)]),
    ("blue-line x boards RIGHT", [34, 35],
        [(BLUE_R_X, 0.0), (BLUE_R_X, RINK_W)]),
    # --- red center line ---
    ("red-line x boards", [24, 30],
        [(CENTER_X, 0.0), (CENTER_X, RINK_W)]),
    ("red-line x center circle", [25, 27],
        [(CENTER_X, CY - CENTER_CIRCLE_R), (CENTER_X, CY + CENTER_CIRCLE_R)]),
    # ref-circle apex: one point, board ambiguous -> two candidates
    ("red-line x ref circle", [29],
        [(CENTER_X, REF_CIRCLE_R), (CENTER_X, RINK_W - REF_CIRCLE_R)]),
    # --- RIGHT-end goal line + trapezoid ---
    ("goal-line x boards RIGHT", [48, 53],
        [(GOAL_R_X, 0.0), (GOAL_R_X, RINK_W)]),
    ("trapezoid x end-boards RIGHT", [54, 55],
        [(RINK_L, CY - TRAP_EB_HALF), (RINK_L, CY + TRAP_EB_HALF)]),
    ("trapezoid x goal-line RIGHT", [49, 52],
        [(GOAL_R_X, CY - TRAP_GL_HALF), (GOAL_R_X, CY + TRAP_GL_HALF)]),
    # --- nets + creases ---
    ("goal posts RIGHT", [50, 51],
        [(GOAL_R_X, CY - GOAL_HALF_W), (GOAL_R_X, CY + GOAL_HALF_W)]),
    ("crease top RIGHT", [46, 47],
        [(GOAL_R_X - CREASE_DEPTH, CY - CREASE_HALF),
         (GOAL_R_X - CREASE_DEPTH, CY + CREASE_HALF)]),
    ("goal posts LEFT", [4, 5],
        [(GOAL_L_X, CY - GOAL_HALF_W), (GOAL_L_X, CY + GOAL_HALF_W)]),
    ("crease top LEFT", [8, 9],
        [(GOAL_L_X + CREASE_DEPTH, CY - CREASE_HALF),
         (GOAL_L_X + CREASE_DEPTH, CY + CREASE_HALF)]),
]

# Human-readable feature names for the identified keypoints (for keypoints.py).
FEATURE_LABELS = {g[1][k]: g[0] for g in GROUPS for k in range(len(g[1]))}


def _hungarian(cost):
    """Minimal assignment (greedy is fine for <=4 items)."""
    cost = np.array(cost, dtype=float)
    n, m = cost.shape
    assigned = {}
    used_cols = set()
    order = sorted(range(n), key=lambda r: cost[r].min())
    for r in order:
        cols = sorted(range(m), key=lambda c: cost[r, c])
        for c in cols:
            if c not in used_cols:
                assigned[r] = c
                used_cols.add(c)
                break
    return assigned


def main():
    recon = json.load(open(RECON))
    recon = {int(k): np.array(v, dtype=float) for k, v in recon.items()}
    print(f"reconstruction: {len(recon)}/56 keypoints")

    # Group midpoints: unambiguous correspondences, BUT every pair is
    # symmetric about the rink centerline so every midpoint lands on
    # y=42.5 — they're all collinear and can't fix a homography alone.
    # We break the symmetry by forking on one well-separated pair.
    mids_src, mids_dst = [], []
    for label, idxs, ices in GROUPS:
        present = [i for i in idxs if i in recon]
        if not present:
            continue
        mids_src.append(np.mean([recon[i] for i in present], axis=0))
        mids_dst.append(np.mean(ices, axis=0))
    mids_src = np.array(mids_src)
    mids_dst = np.array(mids_dst)

    def resolve_with(H):
        """Assign every group's keypoints to ice candidates via H."""
        out = {}
        for label, idxs, ices in GROUPS:
            present = [i for i in idxs if i in recon]
            if not present:
                continue
            proj = cv2.perspectiveTransform(
                np.array([recon[i] for i in present], float).reshape(-1, 1, 2),
                H).reshape(-1, 2)
            cost = [[np.linalg.norm(p - np.array(ic)) for ic in ices]
                    for p in proj]
            for r, c in _hungarian(cost).items():
                out[present[r]] = tuple(ices[c])
        return out

    # Fork on pair [20,21] (blue-line x boards LEFT) — far off-centerline.
    fork_kps = [20, 21]
    fork_ice = [(BLUE_L_X, 0.0), (BLUE_L_X, RINK_W)]
    best = None
    for flip in (False, True):
        ice_a, ice_b = (fork_ice[::-1] if flip else fork_ice)
        src = np.vstack([mids_src,
                         recon[fork_kps[0]], recon[fork_kps[1]]])
        dst = np.vstack([mids_dst, ice_a, ice_b])
        H0, _ = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
        if H0 is None:
            continue
        res = resolve_with(H0)
        # global residual after a refit on the resolution
        a = [(i, res[i]) for i in res if i in recon]
        s = np.array([recon[i] for i, _ in a], float)
        d = np.array([xy for _, xy in a], float)
        Hr, _ = cv2.findHomography(s, d, cv2.RANSAC, 4.0)
        pj = cv2.perspectiveTransform(s.reshape(-1, 1, 2), Hr).reshape(-1, 2)
        med = float(np.median(np.linalg.norm(pj - d, axis=1)))
        print(f"  fork flip={flip}: median residual {med:.2f} ft")
        if best is None or med < best[0]:
            best = (med, res)

    resolved = best[1]
    for label, idxs, ices in GROUPS:
        present = [i for i in idxs if i in recon and i in resolved]
        if present:
            print(f"  {label:32s}: " +
                  ", ".join(f"{i}->{resolved[i]}" for i in present))

    # keypoint 53: identified but absent from recon. It pairs with 48 in
    # "goal-line x boards RIGHT" -> it's whichever board 48 is NOT.
    if 53 not in resolved and 48 in resolved:
        right_gl = [(GOAL_R_X, 0.0), (GOAL_R_X, RINK_W)]
        other = [c for c in right_gl if tuple(c) != resolved[48]]
        if other:
            resolved[53] = tuple(other[0])
            print(f"  kp 53 set to {resolved[53]} (opposite board from kp 48)")

    # The reconstruction is internally consistent within each rink end
    # (~0.5 ft) but NOT globally metric (the left+center and right
    # clusters were stitched and sit in slightly different projective
    # frames). So a single global homography can't place everything.
    #
    # Instead: the 38 identified keypoints keep Derek's EXACT NHL
    # coordinates. The 18 unidentified keypoints are each placed by a
    # LOCAL homography fit on their nearest identified keypoints in
    # reconstruction space — local fits stay within one consistent
    # cluster, sidestepping the global inconsistency.
    ice_xy = [None] * NUM_KP
    for i, xy in resolved.items():
        ice_xy[i] = (round(xy[0], 2), round(xy[1], 2))   # exact, identified

    ident_in_recon = [i for i in resolved if i in recon]
    propagated = 0
    for i in range(NUM_KP):
        if ice_xy[i] is not None or i not in recon:
            continue
        # nearest identified keypoints in recon space
        dists = sorted(ident_in_recon,
                       key=lambda j: np.linalg.norm(recon[j] - recon[i]))
        local = dists[:12]
        src = np.array([recon[j] for j in local], float)
        dst = np.array([resolved[j] for j in local], float)
        Hl, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if Hl is None:
            continue
        p = cv2.perspectiveTransform(recon[i].reshape(1, 1, 2), Hl).reshape(2)
        ice_xy[i] = (round(float(p[0]), 2), round(float(p[1]), 2))
        propagated += 1

    # Quality readout: per-end homography residual on the identified set.
    for label, ids in [("LEFT/center", [k for k in resolved
                                         if resolved[k][0] <= 100 and k in recon]),
                        ("RIGHT", [k for k in resolved
                                   if resolved[k][0] > 100 and k in recon])]:
        if len(ids) < 4:
            continue
        s = np.array([recon[k] for k in ids], float)
        d = np.array([resolved[k] for k in ids], float)
        Hq, mq = cv2.findHomography(s, d, cv2.RANSAC, 3.0)
        pq = cv2.perspectiveTransform(s.reshape(-1, 1, 2), Hq).reshape(-1, 2)
        eq = np.linalg.norm(pq - d, axis=1)
        print(f"  {label:12s}: {int(mq.sum())}/{len(ids)} inliers, "
              f"median residual {np.median(eq):.2f} ft")
    print(f"identified (exact): {len(resolved)}, "
          f"propagated (local): {propagated}, "
          f"unfilled: {sum(1 for x in ice_xy if x is None)}")

    _write_keypoints_py(ice_xy, resolved)
    _render_topdown(ice_xy, resolved)
    return 0


def _write_keypoints_py(ice_xy, resolved):
    lines = [
        '"""HockeyRink 56-keypoint rink template - ice coordinates (NHL feet).',
        '',
        'Auto-derived by scripts/derive_keypoint_ice_coords.py:',
        "  - 38 keypoints identified by Derek (faceoff dots, line/board",
        "    intersections, trapezoid, nets, creases) -> exact NHL coords.",
        "  - The remaining 18 (hashmarks, circle edges, bench) propagated",
        "    via the reconstructed-template homography.",
        '',
        'Ice frame: 200 x 85 ft. x = end board to end board; y = side board',
        'to side board; center (100, 42.5). Keypoint index = HockeyRink',
        'pose-model channel.',
        '"""',
        '',
        'NUM_KEYPOINTS = 56',
        '',
        '# (ice_x_ft, ice_y_ft) per keypoint channel.',
        'KEYPOINT_ICE_XY = [',
    ]
    for i, xy in enumerate(ice_xy):
        tag = "identified" if i in resolved else "propagated"
        lbl = FEATURE_LABELS.get(xy, "") if i in resolved else ""
        comment = f"  # {i:2d} {tag}" + (f" - {lbl}" if lbl else "")
        if xy is None:
            lines.append(f"    None,{comment} (UNRESOLVED)")
        else:
            lines.append(f"    ({xy[0]}, {xy[1]}),{comment}")
    lines.append("]")
    lines.append("")
    lines.append("# Indices Derek identified directly (exact coords, highest trust).")
    lines.append(f"IDENTIFIED_KEYPOINTS = {sorted(resolved.keys())}")
    lines.append("")
    OUT_KP.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_KP.relative_to(PROJECT_ROOT)}")


def _render_topdown(ice_xy, resolved):
    """Top-down rink render with all 56 keypoints for spot-checking."""
    scale = 7
    pad = 40
    W = int(RINK_L * scale) + 2 * pad
    H = int(RINK_W * scale) + 2 * pad
    img = np.full((H, W, 3), 235, np.uint8)

    def px(x, y):
        return int(pad + x * scale), int(pad + y * scale)

    # rink outline + key lines
    cv2.rectangle(img, px(0, 0), px(RINK_L, RINK_W), (180, 180, 180), 2)
    for lx, col in [(GOAL_L_X, (0, 0, 200)), (GOAL_R_X, (0, 0, 200)),
                    (BLUE_L_X, (200, 0, 0)), (BLUE_R_X, (200, 0, 0)),
                    (CENTER_X, (0, 0, 200))]:
        cv2.line(img, px(lx, 0), px(lx, RINK_W), col, 1)
    cv2.circle(img, px(CENTER_X, CY), int(CENTER_CIRCLE_R * scale),
               (0, 0, 200), 1)
    for i, xy in enumerate(ice_xy):
        if xy is None:
            continue
        p = px(xy[0], xy[1])
        identified = i in resolved
        color = (0, 150, 0) if identified else (0, 110, 220)
        cv2.circle(img, p, 6, color, -1)
        cv2.circle(img, p, 6, (0, 0, 0), 1)
        cv2.putText(img, str(i), (p[0] + 7, p[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(img, "green=identified  orange=propagated", (pad, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imwrite(str(OUT_RENDER), img)
    print(f"wrote {OUT_RENDER.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
