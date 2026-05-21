"""Render HockeyRink dataset frames with their ground-truth 56-keypoint
annotations, big index labels, so the keypoint->rink-feature mapping
can be identified by eye on real rink images.

Picks the frames that collectively cover the most keypoints and renders
each at full resolution with numbered markers.

Run:  .venv/Scripts/python.exe scripts/render_annotated_frames.py
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "data/hockeyrink_meta/Dataset/SHL"
ANN_DIR = BASE / "annotations"
FRM_DIR = BASE / "frames"
OUT_DIR = PROJECT_ROOT / "output/_rink_template/annotated"
NUM_KP = 56
FRAME_W, FRAME_H = 1920, 1080


def load_kp(ann_path):
    with open(ann_path) as f:
        vals = [float(v) for v in f.readline().split()]
    if len(vals) < 5 + NUM_KP * 3:
        return None
    kp = np.zeros((NUM_KP, 3))
    for i in range(NUM_KP):
        x, y, v = vals[5 + i * 3: 5 + i * 3 + 3]
        kp[i] = [x * FRAME_W, y * FRAME_H, v]
    return kp


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    anns = []
    for p in sorted(glob.glob(str(ANN_DIR / "*.txt"))):
        kp = load_kp(p)
        if kp is not None:
            anns.append((Path(p).stem, kp))
    print(f"loaded {len(anns)} annotations")

    # Greedy set-cover: pick frames that maximise NEW keypoint coverage.
    covered = set()
    picks = []
    remaining = list(anns)
    while len(covered) < NUM_KP and remaining and len(picks) < 8:
        def newcov(item):
            kp = item[1]
            return len({i for i in range(NUM_KP) if kp[i, 2] >= 2} - covered)
        best = max(remaining, key=newcov)
        if newcov(best) == 0:
            break
        picks.append(best)
        for i in range(NUM_KP):
            if best[1][i, 2] >= 2:
                covered.add(i)
        remaining.remove(best)

    print(f"selected {len(picks)} frames covering {len(covered)}/{NUM_KP} "
          f"keypoints (visible flag)")

    for n, (name, kp) in enumerate(picks):
        fp = FRM_DIR / f"{name}.jpg"
        frame = cv2.imread(str(fp))
        if frame is None:
            print(f"  missing frame {name}")
            continue
        out = frame.copy()
        nvis = 0
        for i, (x, y, v) in enumerate(kp):
            if v < 1:
                continue
            nvis += 1
            color = (0, 255, 0) if v >= 2 else (0, 165, 255)
            cv2.circle(out, (int(x), int(y)), 9, (0, 0, 0), -1)
            cv2.circle(out, (int(x), int(y)), 7, color, -1)
            label = str(i)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            lx, ly = int(x) + 10, int(y) - 8
            cv2.rectangle(out, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 4), (0, 0, 0), -1)
            cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 0), 2)
        op = OUT_DIR / f"frame_{n+1}_{name[:8]}.png"
        cv2.imwrite(str(op), out)
        print(f"  {op.name}: {nvis} keypoints marked")
    print(f"\nrendered to {OUT_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
