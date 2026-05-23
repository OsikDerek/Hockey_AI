"""Fine-tune a dedicated NHL puck detector from the assembled training set.

Aggregates labels from every manifest under `data/training/puck_nhl/`, holds
one clip out for validation (clip-level split, so near-duplicate consecutive
frames don't leak between train/val), and fine-tunes YOLOv8 on the result.

Defaults to fine-tuning the Forzasys SHL puck model (single-class, already
hockey-specific features -- our A/B test showed it loses to HockeyAI on SHL
features alone, but those features are a good starting point for NHL
fine-tuning). The HockeyAI 7-class model can't be fine-tuned directly --
class-count mismatch -- so we'd have to head-surgery it. Forzasys is the
clean base.

Output: models/puck_nhl.pt . The game pipeline auto-picks this up when it
exists (see GameTracker._init_puck_model), so the scorecard immediately
reflects the new detector.

    .venv/Scripts/python.exe scripts/finetune_puck.py
    .venv/Scripts/python.exe scripts/finetune_puck.py --epochs 50 --val-clip clip3
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/training/puck_nhl")
    ap.add_argument("--base", default="models/forzasys_puck_v2.pt",
                    help="weights to fine-tune from (single-class)")
    ap.add_argument("--out", default="models/puck_nhl.pt",
                    help="where to write the fine-tuned weights")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1920,
                    help="match the inference imgsz (game_tracker uses 1920)")
    ap.add_argument("--batch", type=int, default=4,
                    help="lowered for 1920 imgsz on 8 GB VRAM")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--val-clip", default=None,
                    help="clip name to hold out for validation; default = the "
                         "last manifest alphabetically")
    ap.add_argument("--dry-run", action="store_true",
                    help="build dataset.yaml + train/val splits and exit "
                         "(no training -- useful for verifying the setup)")
    args = ap.parse_args()

    root = PROJECT_ROOT / args.dataset
    manifests_dir = root / "manifests"
    manifests = sorted(manifests_dir.glob("*.json"))
    if not manifests:
        raise SystemExit(f"no manifests in {manifests_dir}; run autolabel_puck.py first")

    # --- 1. clip-level train/val split + per-frame label inclusion ---------
    val_clip = args.val_clip or manifests[-1].stem
    train_imgs, val_imgs = [], []
    for mp in manifests:
        m = json.loads(mp.read_text())
        clip = m["clip"]
        is_val = clip == val_clip
        for fr in m["frames"]:
            # Include any frame with a usable label: auto, human-corrected,
            # or human-marked negative (empty label = "no puck here").
            if fr["status"] not in ("auto", "human", "negative"):
                continue
            img = root / "images" / f"{clip}_f{fr['frame_idx']:05d}.jpg"
            if not img.exists():
                continue
            (val_imgs if is_val else train_imgs).append(img)

    if not train_imgs:
        raise SystemExit("no training images — review autolabels first")
    if not val_imgs:
        # Fall back: random 10% of train.
        n_val = max(20, len(train_imgs) // 10)
        val_imgs = train_imgs[-n_val:]
        train_imgs = train_imgs[:-n_val]
        print(f"WARN: val_clip '{val_clip}' had no frames; fell back to last "
              f"{n_val} train frames as val")

    train_txt = root / "train.txt"
    val_txt = root / "val.txt"
    train_txt.write_text("\n".join(str(p) for p in train_imgs) + "\n")
    val_txt.write_text("\n".join(str(p) for p in val_imgs) + "\n")

    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {root}\n"
        f"train: {train_txt.name}\n"
        f"val: {val_txt.name}\n"
        f"nc: 1\n"
        f"names: ['puck']\n"
    )

    print(f"=== dataset ready ===")
    print(f"  train: {len(train_imgs)} images")
    print(f"  val:   {len(val_imgs)} images  (clip='{val_clip}')")
    print(f"  yaml:  {yaml_path}")
    if args.dry_run:
        print("\n--dry-run: skipping training. Re-run without it to fine-tune.")
        return

    # --- 2. fine-tune -----------------------------------------------------
    from ultralytics import YOLO
    base = PROJECT_ROOT / args.base
    if not base.exists():
        raise SystemExit(f"base weights missing: {base}")
    print(f"\nfine-tuning {base.name} on {len(train_imgs)} frames "
          f"({args.epochs} epochs, imgsz {args.imgsz}, batch {args.batch}) ...")
    model = YOLO(str(base))
    results = model.train(
        data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, patience=args.patience,
        project=str(PROJECT_ROOT / "runs" / "detect"), name="puck_nhl",
        exist_ok=True, verbose=True,
    )

    # ultralytics writes runs/detect/puck_nhl/weights/best.pt
    best = PROJECT_ROOT / "runs" / "detect" / "puck_nhl" / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"best.pt not produced at {best}")

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"\nfine-tuned puck detector -> {out}")
    print(f"the game pipeline auto-picks this up on the next run.")
    print(f"verify: .venv/Scripts/python.exe scripts/scorecard.py")


if __name__ == "__main__":
    main()
