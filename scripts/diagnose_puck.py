"""Diagnose raw puck detection by the HockeyAI YOLO model.

Runs the model on every frame of a clip, puck-class only, at a very low
confidence floor, and reports how many frames have a recoverable puck
detection at each candidate threshold. The point is to separate two
failure modes:

  A. YOLO never fires on the puck   -> need interpolation / a better model
  B. YOLO fires but weak / filtered -> need a lower threshold

Usage:
  .venv/Scripts/python.exe scripts/diagnose_puck.py --video data/raw_videos/caufield_goal.mp4
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2

THRESHOLDS = [0.03, 0.05, 0.08, 0.10, 0.15, 0.18, 0.25]
FLOOR = 0.03  # run the model this low; everything above is recoverable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/raw_videos/caufield_goal.mp4")
    ap.add_argument("--model", default="models/HockeyAI_model_weight.pt")
    ap.add_argument("--out", default="output/_verify/puck_diag.csv")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="inference image size; puck is tiny, larger helps")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    puck_id = None
    for cid, name in model.names.items():
        if "puck" in str(name).lower():
            puck_id = int(cid)
    if puck_id is None:
        raise SystemExit("model has no puck class")
    print(f"puck class id = {puck_id}, imgsz = {args.imgsz}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{args.video}: {n_frames} frames")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    counts = {t: 0 for t in THRESHOLDS}
    multi = Counter()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, classes=[puck_id], conf=FLOOR,
                            imgsz=args.imgsz, verbose=False)
        confs = []
        if res and len(res) > 0 and res[0].boxes is not None:
            confs = sorted((float(c) for c in res[0].boxes.conf), reverse=True)
        max_conf = confs[0] if confs else 0.0
        for t in THRESHOLDS:
            n_at = sum(1 for c in confs if c >= t)
            if n_at >= 1:
                counts[t] += 1
            if t == 0.10:
                multi[min(n_at, 5)] += 1
        rows.append({
            "frame": frame_idx,
            "n_det": len(confs),
            "max_conf": round(max_conf, 4),
            "confs": " ".join(f"{c:.3f}" for c in confs[:6]),
        })
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{n_frames}")

    cap.release()

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "n_det", "max_conf", "confs"])
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    print(f"\n=== puck detection headroom ({total} frames) ===")
    for t in THRESHOLDS:
        pct = 100.0 * counts[t] / total if total else 0.0
        print(f"  conf >= {t:.2f}:  {counts[t]:4d}/{total}  ({pct:5.1f}%)")
    print(f"\n=== candidate count per frame at conf>=0.10 ===")
    for k in sorted(multi):
        label = f"{k}+" if k == 5 else str(k)
        print(f"  {label} puck candidates: {multi[k]:4d} frames")
    print(f"\nCSV: {args.out}")


if __name__ == "__main__":
    main()
