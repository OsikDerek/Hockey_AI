"""Auto-label a new clip's puck detections for YOLO fine-tuning.

Runs the game-analysis pipeline on a new clip and turns its highest-quality
puck detections into YOLO training labels for free -- the model itself
generates ~70-90% of the labels, so the human only has to review the hard
frames (a fraction of the work of annotating from scratch).

Per frame the positions JSON says one of three things:
  - high-confidence real detection (conf >= AUTOLABEL_CONF) -> trust as a
    YOLO label. Save the frame + write a labels .txt.
  - low-confidence or interpolated -> flag the frame as REVIEW. Save the
    frame; review_autolabels.py will surface it for human correction.
  - no puck at all -> skip (probably correctly: occluded or off-screen).

Output layout (so all clips merge cleanly into one YOLO dataset):
    data/training/puck_nhl/
        images/<clip>_f<N>.jpg
        labels/<clip>_f<N>.txt          # only for AUTO frames
        manifests/<clip>.json           # per-frame status (auto/review/none)

Usage:
    .venv/Scripts/python.exe scripts/autolabel_puck.py \\
        --video data/raw_videos/<new_clip>.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Pipeline puck confidence to trust as a training label without human review.
# From the caufield_trim distribution real pucks cluster at 0.45-0.65 (p50
# 0.53), false-positive clusters peak at 0.36. 0.50 keeps the safe top half
# and pushes everything ambiguous to the review tool.
AUTOLABEL_CONF = 0.50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="path to the new clip")
    ap.add_argument("--out", default="data/training/puck_nhl")
    ap.add_argument("--clip-name", default=None,
                    help="override label prefix (default: video filename stem)")
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="reuse an existing positions JSON instead of re-running")
    ap.add_argument("--autolabel-conf", type=float, default=AUTOLABEL_CONF)
    args = ap.parse_args()

    video = (PROJECT_ROOT / args.video).resolve()
    if not video.exists():
        raise SystemExit(f"not found: {video}")
    clip = args.clip_name or video.stem
    out_root = PROJECT_ROOT / args.out
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)
    (out_root / "manifests").mkdir(parents=True, exist_ok=True)

    # --- 1. ensure a fresh positions JSON for this clip --------------------
    # Use the canonical b3 naming so autolabel reads the same positions JSON
    # the rest of the toolchain (scorecard, verify_*) produces and consumes.
    pipeline_out = PROJECT_ROOT / "output" / f"{clip}_b3.mp4"
    pos_json = pipeline_out.with_name(f"{pipeline_out.stem}_positions.json")
    if not args.skip_pipeline:
        print(f"running pipeline on {video.name} ...")
        r = subprocess.run(
            [sys.executable, "main.py", "--game-analysis", "-i", str(video),
             "-o", str(pipeline_out.relative_to(PROJECT_ROOT))],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print((r.stdout or "")[-2000:])
            print((r.stderr or "")[-2000:])
            raise SystemExit("pipeline run failed")
    if not pos_json.exists():
        raise SystemExit(f"positions JSON missing: {pos_json}")

    data = json.loads(pos_json.read_text())
    frames = data["frames"]

    # --- 2. classify each frame + emit images + labels ---------------------
    cap = cv2.VideoCapture(str(video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    manifest = {"clip": clip, "video": str(video.relative_to(PROJECT_ROOT)),
                "frame_w": w, "frame_h": h,
                "autolabel_conf": args.autolabel_conf,
                "frames": []}
    n_auto = n_review = n_none = 0

    def write_frame(fi, frame):
        path = out_root / "images" / f"{clip}_f{fi:05d}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return path

    def write_label(fi, bbox):
        # YOLO format: <class> <cx> <cy> <w> <h>, all normalised to [0,1].
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h
        bw, bh = abs(x2 - x1) / w, abs(y2 - y1) / h
        # Floor the puck box to a small minimum -- the broadcast puck can be
        # a handful of pixels and an over-tight box hurts YOLO training.
        bw = max(bw, 12.0 / w)
        bh = max(bh, 12.0 / h)
        path = out_root / "labels" / f"{clip}_f{fi:05d}.txt"
        path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        return path

    for fr in frames:
        fi = fr["frame_idx"]
        puck = fr.get("puck")
        if puck and puck.get("confidence", 0) >= args.autolabel_conf \
                and puck.get("bbox_px"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            write_frame(fi, frame)
            write_label(fi, puck["bbox_px"])
            manifest["frames"].append({"frame_idx": fi, "status": "auto",
                                       "confidence": puck["confidence"],
                                       "bbox_px": puck["bbox_px"]})
            n_auto += 1
        elif puck:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            write_frame(fi, frame)
            manifest["frames"].append({"frame_idx": fi, "status": "review",
                                       "confidence": puck.get("confidence", 0),
                                       "hint_bbox_px": puck.get("bbox_px")})
            n_review += 1
        else:
            manifest["frames"].append({"frame_idx": fi, "status": "none"})
            n_none += 1

    cap.release()
    (out_root / "manifests" / f"{clip}.json").write_text(
        json.dumps(manifest, indent=1))

    print(f"\n=== {clip}: {len(frames)} frames ===")
    print(f"  auto-labelled (free): {n_auto}")
    print(f"  flagged for review:   {n_review}")
    print(f"  no puck (skipped):    {n_none}")
    print(f"\ndataset: {out_root}")
    print(f"next: .venv/Scripts/python.exe scripts/review_autolabels.py "
          f"--clip {clip}")


if __name__ == "__main__":
    main()
