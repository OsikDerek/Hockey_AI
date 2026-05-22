"""Per-frame calibration ACCURACY verification.

"Calibrated" only means a homography exists. This checks whether the
homography is actually CORRECT, every frame, two ways:

1. OVERLAY VIDEO: for each source frame, project the known NHL rink
   markings (blue/red/goal lines, center + faceoff circles, faceoff
   dots) through that frame's homography onto the broadcast image. If
   the projected lines sit on the real painted lines, the calibration
   is accurate for that frame. If they're offset, it's wrong — and you
   can see exactly which frames and by how much.

2. PER-FRAME REPROJECTION ERROR: the registration model fits the
   homography to detected rink keypoints; the median residual (feet) is
   an objective per-frame accuracy number. Reported as a distribution
   over every frame — no sampling.

Run:  .venv/Scripts/python.exe scripts/verify_calibration_overlay.py \
          --video data/raw_videos/caufield_goal.mp4
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.game_analysis.rink_registration.registration_model import RinkRegistrationModel

RINK_L, RINK_W, CY = 200.0, 85.0, 42.5
OUT_DIR = PROJECT_ROOT / "output" / "_verify"


def _rink_ice_geometry():
    """Return rink markings as lists of ice-coordinate polylines."""
    polylines = []
    # vertical lines (goal / blue / red), each spanning the rink width
    for x, _name in [(11, "goalL"), (75, "blueL"), (100, "red"),
                     (125, "blueR"), (189, "goalR")]:
        polylines.append([(x, 0.0), (x, RINK_W)])
    # rink outline
    polylines.append([(0, 0), (RINK_L, 0), (RINK_L, RINK_W),
                      (0, RINK_W), (0, 0)])
    # circles: center + 4 end-zone faceoff circles (radius 15)
    for cx, cy in [(100, CY), (31, 20.5), (31, 64.5), (169, 20.5), (169, 64.5)]:
        circ = [(cx + 15 * np.cos(t), cy + 15 * np.sin(t))
                for t in np.linspace(0, 2 * np.pi, 40)]
        polylines.append(circ)
    # faceoff dots (just markers, drawn separately)
    dots = [(100, CY), (31, 20.5), (31, 64.5), (169, 20.5), (169, 64.5),
            (80, 20.5), (80, 64.5), (120, 20.5), (120, 64.5)]
    return polylines, dots


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)

    video = Path(args.video)
    name = args.name or video.stem
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reg = RinkRegistrationModel("models/HockeyRink.pt")
    if not reg.available:
        print("registration model unavailable")
        return 2

    polylines, dots = _rink_ice_geometry()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{name}: {n} frames @ {fps:.0f}fps")

    vw = None
    rows = []
    fi = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        result = reg.estimate(frame)
        out = frame.copy()
        if result is None:
            err = None
            n_kp = 0
            cv2.putText(out, f"frame {fi}  NO FIT", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 90, 255), 2)
        else:
            H_p2i, H_i2p, info = result
            err = info["median_reproj_err_ft"]
            n_kp = info["n_used"]
            # project rink markings ice -> pixel
            for poly in polylines:
                pts = cv2.perspectiveTransform(
                    np.array(poly, np.float64).reshape(-1, 1, 2), H_i2p
                ).reshape(-1, 2)
                pts_i = pts.astype(np.int32)
                cv2.polylines(out, [pts_i], False, (0, 255, 255), 2)
            for d in dots:
                p = cv2.perspectiveTransform(
                    np.array([[d]], np.float64), H_i2p).reshape(2)
                cv2.circle(out, (int(p[0]), int(p[1])), 5, (0, 165, 255), -1)
            color = ((0, 220, 0) if err < 1.5
                     else (0, 200, 255) if err < 3.0 else (0, 90, 255))
            cv2.putText(out, f"frame {fi}  reproj={err:.2f}ft  kp={n_kp}",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        rows.append({"frame": fi, "fit": int(result is not None),
                     "reproj_err_ft": "" if err is None else round(err, 3),
                     "n_keypoints": n_kp})

        out = cv2.resize(out, (1280, 720))
        if vw is None:
            vw = cv2.VideoWriter(str(OUT_DIR / f"{name}_calib_overlay.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (1280, 720))
        vw.write(out)
        if fi % 100 == 0:
            print(f"  frame {fi}/{n}")
    cap.release()
    if vw:
        vw.release()

    with open(OUT_DIR / f"{name}_calib_audit.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # objective per-frame accuracy distribution
    errs = [r["reproj_err_ft"] for r in rows if r["reproj_err_ft"] != ""]
    fits = sum(r["fit"] for r in rows)
    print(f"\n=== CALIBRATION ACCURACY, every frame ({len(rows)}) ===")
    print(f"homography fitted: {fits}/{len(rows)}")
    if errs:
        e = sorted(errs)
        m = len(e)
        print(f"reprojection error (ft) on fitted frames:")
        print(f"  p50={e[m//2]:.2f}  p90={e[9*m//10]:.2f}  max={e[-1]:.2f}")
        for lo, hi, lbl in [(0, 1.0, "excellent <1ft"),
                            (1.0, 2.0, "good 1-2ft"),
                            (2.0, 4.0, "marginal 2-4ft"),
                            (4.0, 1e9, "BAD >4ft")]:
            c = sum(1 for x in errs if lo <= x < hi)
            print(f"  {lbl:18s}: {c:4d} ({100*c/len(errs):.0f}%)")
    print(f"\noverlay video: output/_verify/{name}_calib_overlay.mp4")
    print(f"audit CSV:     output/_verify/{name}_calib_audit.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
