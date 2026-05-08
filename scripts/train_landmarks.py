"""Train a rink-landmarks-only YOLO model on the SHL annotated corpus.

The current general-purpose HockeyAI model (7 classes incl. players,
pucks, refs) routinely under-detects rink landmarks (centroid, faceoff
dots, goal nets) on broadcast follow-cam clips. A focused, smaller-class
model trained on the same data tends to recover meaningfully better
recall on the specific classes we care about (no representation budget
wasted on the majority "player" class).

This script:
  1. Filters SHL annotations to keep only classes 0 (centroid),
     1 (faceoff), 2 (goal).
  2. Splits into train/val (90/10) with a deterministic seed.
  3. Writes a dataset YAML pointing at the prepared dirs.
  4. Trains yolov8n with imgsz=640 for 80 epochs.
  5. Saves the best weights to models/landmarks_yolov8n.pt.

Run from project root:
    .venv/Scripts/python.exe scripts/train_landmarks.py
"""

from __future__ import annotations

import random
import shutil
import sys
import yaml
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHL_FRAMES = PROJECT_ROOT / "data" / "HockeyAI_raw" / "SHL" / "frames"
SHL_ANNOTS = PROJECT_ROOT / "data" / "HockeyAI_raw" / "SHL" / "annotations"
PREPPED_ROOT = PROJECT_ROOT / "data" / "landmarks_yolo"
TRAIN_DIR = PREPPED_ROOT / "train"
VAL_DIR = PREPPED_ROOT / "val"
DATA_YAML = PREPPED_ROOT / "data.yaml"
RUNS_PROJECT = PROJECT_ROOT / "runs" / "detect"
EXPERIMENT_NAME = "landmarks_yolov8n"
FINAL_WEIGHTS = PROJECT_ROOT / "models" / "landmarks_yolov8n.pt"

# Classes we keep. Indices preserved (0/1/2) so trained model is
# drop-in compatible with the existing pipeline's class_name lookups.
KEEP_CLASSES = {0, 1, 2}
CLASS_NAMES = ["centroid", "faceoff", "goal"]

VAL_FRACTION = 0.10
RANDOM_SEED = 42


def filter_annotation(src_path: Path) -> str | None:
    """Return filtered annotation text (only classes in KEEP_CLASSES) or
    None if the resulting file would be empty."""
    kept = []
    for line in src_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cid = int(line.split()[0])
        except (ValueError, IndexError):
            continue
        if cid in KEEP_CLASSES:
            kept.append(line)
    return "\n".join(kept) if kept else None


def prep_dataset() -> tuple[int, int]:
    """Build train/val image+label dirs from SHL. Returns (train_n, val_n)."""
    if not SHL_FRAMES.exists() or not SHL_ANNOTS.exists():
        sys.exit(f"Missing source dirs:\n  {SHL_FRAMES}\n  {SHL_ANNOTS}")

    if PREPPED_ROOT.exists():
        print(f"Removing stale {PREPPED_ROOT} ...")
        shutil.rmtree(PREPPED_ROOT)

    for sub in (TRAIN_DIR, VAL_DIR):
        (sub / "images").mkdir(parents=True, exist_ok=True)
        (sub / "labels").mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)
    annot_files = sorted(SHL_ANNOTS.glob("*.txt"))
    print(f"Source annotations: {len(annot_files)}")

    train_n = 0
    val_n = 0
    skipped = 0
    for annot_path in annot_files:
        filtered = filter_annotation(annot_path)
        if filtered is None:
            skipped += 1
            continue
        stem = annot_path.stem
        img_path = SHL_FRAMES / f"{stem}.jpg"
        if not img_path.exists():
            # Try other extensions
            alt = next(SHL_FRAMES.glob(f"{stem}.*"), None)
            if alt is None:
                skipped += 1
                continue
            img_path = alt

        target_dir = VAL_DIR if rng.random() < VAL_FRACTION else TRAIN_DIR
        shutil.copy(img_path, target_dir / "images" / img_path.name)
        (target_dir / "labels" / f"{stem}.txt").write_text(filtered)
        if target_dir is TRAIN_DIR:
            train_n += 1
        else:
            val_n += 1

    print(f"Train: {train_n}  Val: {val_n}  Skipped (no landmark labels or missing image): {skipped}")
    return train_n, val_n


def write_yaml() -> None:
    payload = {
        "path": str(PREPPED_ROOT.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    DATA_YAML.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"Wrote {DATA_YAML}")


def train() -> None:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # cold-start from COCO-pretrained weights
    print("Starting training (yolov8n, imgsz=640, 80 epochs)...")
    results = model.train(
        data=str(DATA_YAML),
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,  # first CUDA device
        project=str(RUNS_PROJECT),
        name=EXPERIMENT_NAME,
        exist_ok=True,
        patience=20,            # early stop if no val mAP improvement for 20 epochs
        save_period=10,
        verbose=True,
        deterministic=True,
    )
    # Copy the best weights into models/ for easy pipeline integration
    best = RUNS_PROJECT / EXPERIMENT_NAME / "weights" / "best.pt"
    if best.exists():
        FINAL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, FINAL_WEIGHTS)
        print(f"Copied best weights -> {FINAL_WEIGHTS}")
    else:
        print(f"WARNING: best weights not found at {best}")
    print("Done.")


def main() -> int:
    train_n, val_n = prep_dataset()
    if train_n == 0:
        print("No training data — aborting.")
        return 1
    write_yaml()
    train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
