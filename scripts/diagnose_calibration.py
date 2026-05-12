"""Diagnose why the rink calibrator's homography is producing compressed
across-rink coordinates on livebarn_cropped.

The hypothesis: YOLO isn't seeing enough faceoff dots to constrain the
across-rink axis, so the homography falls back to a near-degenerate fit
using only line × near-boards intersections (all at one y value).

What this prints, per sampled frame:
- HockeyAI YOLO detections by class (player/puck/goalie/referee/faceoff/...)
- Landmark-specialist YOLO detections (the secondary model)
- Combined faceoff detections (the only thing the dot-disambiguation cares about)
- Bounding-box sizes (small/medium/large) for faceoff dets to see if some are detected but tiny

Run from project root:
    .venv/Scripts/python.exe scripts/diagnose_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO = PROJECT_ROOT / "data" / "raw_videos" / "livebarn_60sec_cropped.mp4"
MAIN_MODEL = PROJECT_ROOT / "models" / "HockeyAI_model_weight.pt"
LM_MODEL = PROJECT_ROOT / "models" / "landmarks_yolov8n.pt"

SAMPLE_FRAMES = [60, 180, 300, 420, 540, 660, 780, 900, 1020, 1140, 1260, 1500, 1700]


def main():
    if not VIDEO.exists():
        print(f"missing: {VIDEO}")
        return 2
    if not MAIN_MODEL.exists():
        print(f"missing: {MAIN_MODEL}")
        return 2

    cap = cv2.VideoCapture(str(VIDEO))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total} frames")

    print(f"Loading main YOLO model: {MAIN_MODEL.name}")
    main_model = YOLO(str(MAIN_MODEL))
    print(f"  classes: {main_model.names}")

    lm_model = None
    if LM_MODEL.exists():
        print(f"Loading landmark specialist: {LM_MODEL.name}")
        lm_model = YOLO(str(LM_MODEL))
        print(f"  classes: {lm_model.names}")
    else:
        print(f"  (no landmark specialist at {LM_MODEL})")

    print()
    print(f"Sampling {len(SAMPLE_FRAMES)} frames…")
    print()
    print(f"{'frame':>6} | {'main det counts':<48} | {'lm det counts':<32} | "
          f"{'#faceoff':>9} | {'faceoff bbox-areas px²':<40}")
    print("-" * 150)

    grand_faceoff_count = 0
    grand_frames_with_4plus = 0

    for fi in SAMPLE_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            print(f"{fi:>6}: read failed")
            continue

        # Main model
        main_res = main_model.predict(frame, conf=0.25, verbose=False)
        main_counts = Counter()
        main_faceoffs = []
        for r in main_res:
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                cname = main_model.names[int(c)]
                main_counts[cname] += 1
                if "faceoff" in cname.lower() or "centroid" in cname.lower():
                    x1, y1, x2, y2 = b
                    main_faceoffs.append(((x2 - x1) * (y2 - y1)))

        # Landmark specialist
        lm_counts = Counter()
        lm_faceoffs = []
        if lm_model is not None:
            lm_res = lm_model.predict(frame, conf=0.20, verbose=False)
            for r in lm_res:
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                    cname = lm_model.names[int(c)]
                    lm_counts[cname] += 1
                    if "faceoff" in cname.lower() or "centroid" in cname.lower():
                        x1, y1, x2, y2 = b
                        lm_faceoffs.append(((x2 - x1) * (y2 - y1)))

        all_faceoffs = main_faceoffs + lm_faceoffs
        grand_faceoff_count += len(all_faceoffs)
        if len(all_faceoffs) >= 4:
            grand_frames_with_4plus += 1

        # Format
        main_str = ", ".join(f"{c}:{n}" for c, n in main_counts.most_common())
        lm_str = ", ".join(f"{c}:{n}" for c, n in lm_counts.most_common()) if lm_counts else "(none)"
        areas_str = ", ".join(f"{a:.0f}" for a in sorted(all_faceoffs, reverse=True)[:6])
        print(f"{fi:>6} | {main_str[:48]:<48} | {lm_str[:32]:<32} | "
              f"{len(all_faceoffs):>9} | {areas_str[:40]:<40}")

    cap.release()

    print()
    print(f"Total faceoff detections across {len(SAMPLE_FRAMES)} frames: {grand_faceoff_count}")
    print(f"Frames with ≥4 faceoff dets (homography eligible): {grand_frames_with_4plus}/{len(SAMPLE_FRAMES)}")
    print()
    if grand_frames_with_4plus == 0:
        print("CONCLUSION: faceoff dot detection is failing on this clip. "
              "Fixing the homography requires better dot detection — "
              "either fine-tune the model on junior LiveBarn frames "
              "or add a CV-based dot detector as a fallback.")
    elif grand_frames_with_4plus < len(SAMPLE_FRAMES) / 4:
        print("CONCLUSION: faceoff dots detected sparsely. Some frames have "
              "enough dots; others don't. The intermittent fits won't "
              "stabilize the homography. Need either denser dot detection "
              "or to supplement with far-boards detection.")
    else:
        print("CONCLUSION: dot detection looks adequate. Y-compression is "
              "likely caused by disambiguation mismatches or a too-loose "
              "fit gate. Look at the similarity-prior accuracy and the "
              "snap-to-nearest matching threshold.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
