"""Phase 2: train the rink-keypoint detection model.

Trains a YOLOv8-pose model to detect the 29 named NHL rink keypoints
(see keypoints.py) as a single "rink" object. The pretrained COCO-pose
backbone is reused; the pose head is re-initialized for our keypoint
count.

Run from project root:
    .venv/Scripts/python.exe -m src.game_analysis.rink_registration.train \
        --data data/synth_rink/data.yaml --epochs 100

Notes
-----
- Horizontal/vertical flip augmentation is DISABLED: flipping the image
  would swap left/right (and lo/hi) keypoint identities, corrupting
  labels. The synthetic generator already covers camera-angle variety
  (incl. the opposite-side camera), so flips aren't needed.
- imgsz 960 balances accuracy vs the RTX 2080 Super's 8 GB VRAM.
- Output weights land in runs/pose/<name>/weights/best.pt — copy the
  chosen weights to models/rink_keypoints.pt for the inference wrapper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synth_rink/data.yaml",
                   help="YOLO-pose data.yaml")
    p.add_argument("--base", default="yolov8s-pose.pt",
                   help="pretrained base model (downloaded on first run)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="rink_kp_synth")
    p.add_argument("--project", default=None,
                   help="output parent dir for runs. Default keeps runs in "
                        "the repo; pass a non-OneDrive path to avoid sync "
                        "churn during training.")
    p.add_argument("--device", default="0", help="GPU id, or 'cpu'")
    p.add_argument("--workers", type=int, default=8,
                   help="dataloader workers. Lower (4) reduces system-memory "
                        "commit / pagefile pressure during training.")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)

    data_path = (PROJECT_ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    if not data_path.is_file():
        print(f"ERROR: data yaml not found: {data_path}")
        print("Generate the dataset first:  python -m "
              "src.game_analysis.rink_registration.synth_data --out data/synth_rink")
        return 2

    print(f"Training rink-keypoint model")
    print(f"  base:   {args.base}")
    print(f"  data:   {data_path}")
    print(f"  epochs: {args.epochs}  imgsz: {args.imgsz}  batch: {args.batch}")

    model = YOLO(args.base)
    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        resume=args.resume,
        cache=False,          # don't pre-cache the dataset to disk/RAM
        save_period=-1,       # only save best.pt + last.pt, not per-epoch
        workers=args.workers,
    )
    if args.project:
        train_kwargs["project"] = args.project
    model.train(
        **train_kwargs,
        # Flip augmentation OFF — would scramble left/right keypoint identity.
        fliplr=0.0,
        flipud=0.0,
        # Geometric aug stays modest; the synthetic generator already
        # provides the broad camera-angle distribution.
        degrees=4.0,
        translate=0.06,
        scale=0.25,
        shear=2.0,
        perspective=0.0,      # camera perspective is already baked into samples
        mosaic=0.0,           # mosaic would fuse 4 rinks into one image — nonsense here
        hsv_h=0.012,
        hsv_s=0.5,
        hsv_v=0.4,
        patience=30,
    )

    best = PROJECT_ROOT / "runs" / "pose" / args.name / "weights" / "best.pt"
    print(f"\nDone. Best weights: {best}")
    print(f"Copy to models/ for inference:")
    print(f"  cp \"{best}\" models/rink_keypoints.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
