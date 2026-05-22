"""Per-frame tracking verification — no sampling, every frame audited.

Two outputs:

1. AUDIT CSV + summary (objective, every frame): for each frame records
   calibrated?, n_players, n_goalies, has_puck, how many positions fall
   inside the rink vs outside, and the max frame-to-frame player speed
   (a physical-plausibility check). Flags every frame that fails a
   check. This answers "which frames have gaps / bad data" exactly.

2. SIDE-BY-SIDE verification video: each source frame on top, the
   reconstructed top-down sim for that exact frame below it, frame-
   synced. Scrub the whole thing to see, per frame, whether the sim
   matches the play. Gaps show as empty sim frames.

Run:
  .venv/Scripts/python.exe scripts/verify_tracking.py --clip caufield_b3
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
RINK_L, RINK_W = 200.0, 85.0
CY = 42.5                    # rink centerline (ice-y), feet
MAX_PLAYER_FT_PER_S = 48.0   # hockey top speed ~40 ft/s; >48 = implausible
EXPECTED_SKATERS = 10        # 5v5; +2 goalies = 12 on ice


def _source_video_for(clip: str) -> Path | None:
    """Map a positions-JSON basename to its source video."""
    raw = PROJECT_ROOT / "data/raw_videos"
    base = clip.replace("_positions", "")
    # strip pipeline tags (_b3, _v7, ...)
    import re
    stem = re.sub(r"_(v|b)\d+$", "", base)
    for cand in (f"{base}.web.mp4", f"{base}.mp4", f"{stem}.web.mp4",
                 f"{stem}.mp4", f"{stem}_goal.web.mp4", f"{stem}_goal.mp4",
                 f"{stem}_60sec.mp4"):
        p = raw / cand
        if p.exists():
            return p
    return None


def _detect_orientation(video_path):
    """Determine how the broadcast camera maps ice axes to screen axes,
    so the top-down panel is drawn in the SAME orientation as the
    footage (not mirrored). Returns (flip_x, flip_y).

    Runs the rink-registration model on a spread of frames and, for each
    that fits, projects known ice points to pixels: if ice-x increases
    leftward on screen we flip x; if ice-y increases upward on screen we
    flip y. Decisions are MAJORITY-VOTED across all fitting frames so a
    single degenerate homography can't dictate the orientation.

    Note: the rink markings are mirror-symmetric, so a calibration
    overlay cannot reveal a left/right or near/far flip -- only the
    asymmetric camera->ice homography can, which is why this is needed.
    """
    try:
        from src.game_analysis.rink_registration.registration_model import (
            RinkRegistrationModel)
    except Exception as e:
        print(f"  orientation: registration import failed ({e}) - no flip")
        return False, False
    reg = RinkRegistrationModel(str(PROJECT_ROOT / "models" / "HockeyRink.pt"))
    if not reg.available:
        print("  orientation: registration model unavailable - no flip")
        return False, False
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 600
    votes_x, votes_y = [], []
    for fi in range(0, n, max(1, n // 60)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        r = reg.estimate(frame)
        if r is None:
            continue
        _H_p2i, H_i2p, _info = r

        def proj(pt):
            return cv2.perspectiveTransform(
                np.array([[pt]], np.float64), H_i2p).reshape(2)

        pL, pR = proj((10.0, CY)), proj((190.0, CY))
        pT, pB = proj((100.0, 5.0)), proj((100.0, 80.0))
        # top-down draws ice-x rightward, ice-y downward; flip to match
        votes_x.append(pL[0] > pR[0])
        votes_y.append(pT[1] > pB[1])
    cap.release()
    if not votes_x:
        print("  orientation: no frames fit - no flip")
        return False, False
    flip_x = sum(votes_x) > len(votes_x) / 2
    flip_y = sum(votes_y) > len(votes_y) / 2
    print(f"  orientation: {len(votes_x)} frames voted "
          f"(x:{sum(votes_x)} y:{sum(votes_y)})")
    return bool(flip_x), bool(flip_y)


def _draw_topdown(frame_data, w, h, flip_x=False, flip_y=False):
    """Render one top-down rink frame from a positions-JSON frame dict,
    oriented to match the broadcast camera (flip_x / flip_y)."""
    img = np.full((h, w, 3), 60, np.uint8)
    pad = 24
    sx = (w - 2 * pad) / RINK_L
    sy = (h - 2 * pad) / RINK_W
    s = min(sx, sy)

    def px(x, y):
        if flip_x:
            x = RINK_L - x
        if flip_y:
            y = RINK_W - y
        return int(pad + x * s), int(pad + y * s)

    cv2.rectangle(img, px(0, 0), px(RINK_L, RINK_W), (150, 150, 150), 2)
    for lx, c in [(11, (0, 0, 180)), (189, (0, 0, 180)), (100, (0, 0, 200)),
                  (75, (180, 60, 0)), (125, (180, 60, 0))]:
        cv2.line(img, px(lx, 0), px(lx, RINK_W), c, 1)
    cv2.circle(img, px(100, 42.5), int(15 * s), (0, 0, 200), 1)

    if not frame_data.get("calibrated"):
        cv2.putText(img, "NO CALIBRATION", (pad, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 90, 200), 2)
        return img

    for p in frame_data.get("players", []):
        team = p.get("team", "")
        col = ((40, 200, 40) if team == "team_a"
               else (200, 80, 40) if team == "team_b" else (160, 160, 160))
        cv2.circle(img, px(p["ice_x"], p["ice_y"]), 7, col, -1)
    for g in frame_data.get("goalies", []):
        cv2.circle(img, px(g["ice_x"], g["ice_y"]), 8, (0, 200, 200), -1)
    pk = frame_data.get("puck")
    if pk:
        cv2.circle(img, px(pk["ice_x"], pk["ice_y"]), 5, (0, 165, 255), -1)
        cv2.circle(img, px(pk["ice_x"], pk["ice_y"]), 8, (255, 255, 255), 1)
    return img


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="caufield_trim_b3",
                    help="positions JSON basename (without _positions.json)")
    args = ap.parse_args(argv)

    json_path = PROJECT_ROOT / "output" / f"{args.clip}_positions.json"
    if not json_path.exists():
        print(f"missing {json_path}")
        return 2
    data = json.loads(json_path.read_text())
    frames = data["frames"]
    fps = data.get("fps", 30.0)
    n = len(frames)

    src = _source_video_for(args.clip)
    cap = cv2.VideoCapture(str(src)) if src else None
    print(f"clip: {args.clip}  ({n} frames)   source: {src.name if src else 'NOT FOUND'}")

    # Orient the top-down panel to match the broadcast camera (the
    # rink-registration homography tells us which way the axes run).
    flip_x = flip_y = False
    if src is not None:
        flip_x, flip_y = _detect_orientation(src)
        print(f"camera orientation: flip_x={flip_x} flip_y={flip_y}")

    out_dir = PROJECT_ROOT / "output" / "_verify"
    out_dir.mkdir(parents=True, exist_ok=True)

    # video writer (lazy — sized to first source frame)
    vw = None
    panel_h = 360

    # per-track last position for speed check
    last_pos = {}   # track_id -> (frame_idx, x, y)
    audit = []

    for fi, fr in enumerate(frames):
        players = fr.get("players", []) or []
        goalies = fr.get("goalies", []) or []
        puck = fr.get("puck")
        calibrated = bool(fr.get("calibrated"))

        # in-rink check
        in_rink = out_rink = 0
        for p in players + goalies:
            if 0 <= p["ice_x"] <= RINK_L and 0 <= p["ice_y"] <= RINK_W:
                in_rink += 1
            else:
                out_rink += 1

        # speed check
        max_speed = 0.0
        for p in players:
            tid = p.get("track_id")
            if tid is None:
                continue
            if tid in last_pos:
                pf, px_, py_ = last_pos[tid]
                df = fi - pf
                if 0 < df <= 5:
                    d = math.hypot(p["ice_x"] - px_, p["ice_y"] - py_)
                    spd = d / (df / fps)
                    max_speed = max(max_speed, spd)
            last_pos[tid] = (fi, p["ice_x"], p["ice_y"])

        audit.append({
            "frame": fi,
            "calibrated": int(calibrated),
            "n_players": len(players),
            "n_goalies": len(goalies),
            "has_puck": int(puck is not None),
            "n_out_of_rink": out_rink,
            "max_speed_fps": round(max_speed, 1),
        })

        # composite frame for the verification video
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, sframe = cap.read()
            if ok:
                sw = 960
                sh = int(sframe.shape[0] * sw / sframe.shape[1])
                sframe = cv2.resize(sframe, (sw, sh))
                top = _draw_topdown(fr, sw, panel_h, flip_x, flip_y)
                comp = np.vstack([sframe, top])
                cv2.putText(comp, f"frame {fi}  players={len(players)} "
                            f"goalies={len(goalies)} puck={'Y' if puck else 'N'} "
                            f"{'CAL' if calibrated else 'NO-CAL'}",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)
                if vw is None:
                    vw = cv2.VideoWriter(
                        str(out_dir / f"{args.clip}_verify.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), fps,
                        (comp.shape[1], comp.shape[0]))
                vw.write(comp)
    if vw is not None:
        vw.release()
    if cap is not None:
        cap.release()

    # write audit CSV
    csv_path = out_dir / f"{args.clip}_audit.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(audit[0].keys()))
        wr.writeheader()
        wr.writerows(audit)

    # ---- objective summary, every frame ----
    cal = sum(a["calibrated"] for a in audit)
    with_puck = sum(a["has_puck"] for a in audit)
    gap_frames = [a["frame"] for a in audit if a["n_players"] + a["n_goalies"] == 0]
    sparse = [a["frame"] for a in audit
              if 0 < a["n_players"] < EXPECTED_SKATERS - 3]
    out_rink_frames = [a["frame"] for a in audit if a["n_out_of_rink"] > 0]
    fast_frames = [a["frame"] for a in audit
                   if a["max_speed_fps"] > MAX_PLAYER_FT_PER_S]
    counts = [a["n_players"] for a in audit if a["calibrated"]]

    print(f"\n=== PER-FRAME AUDIT ({n} frames) ===")
    print(f"calibrated:            {cal}/{n} ({100*cal/n:.0f}%)")
    print(f"puck present:          {with_puck}/{n} ({100*with_puck/n:.0f}%)")
    print(f"EMPTY frames (0 chars): {len(gap_frames)} ({100*len(gap_frames)/n:.0f}%)")
    print(f"sparse (<{EXPECTED_SKATERS-3} players): {len(sparse)} "
          f"({100*len(sparse)/n:.0f}%)")
    print(f"frames w/ out-of-rink positions: {len(out_rink_frames)}")
    print(f"frames w/ implausible speed (>{MAX_PLAYER_FT_PER_S:.0f}ft/s): "
          f"{len(fast_frames)}")
    if counts:
        cs = sorted(counts)
        print(f"player count on calibrated frames: "
              f"min={cs[0]} p50={cs[len(cs)//2]} max={cs[-1]} "
              f"(expect ~{EXPECTED_SKATERS})")
    print(f"\naudit CSV:  {csv_path.relative_to(PROJECT_ROOT)}")
    if vw is not None or (out_dir / f'{args.clip}_verify.mp4').exists():
        print(f"verify video: output/_verify/{args.clip}_verify.mp4")
    # verdict
    good = (100 * cal / n >= 95 and len(gap_frames) == 0
            and len(sparse) < n * 0.1 and len(fast_frames) == 0)
    print(f"\nVERDICT: {'every-frame checks PASS' if good else 'NOT READY - see flagged frames above'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
