"""Validate the trained rink-keypoint model on REAL broadcast frames.

The model trained on synthetic rink renders; this checks the
synthetic-to-real gap. For sampled frames of each broadcast clip it
reports keypoints detected, homography fit success, and reprojection
error — and saves an annotated frame with the detected keypoints drawn.

Run from project root:
    .venv/Scripts/python.exe scripts/validate_rink_registration.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.game_analysis.rink_registration.registration_model import RinkRegistrationModel
from src.game_analysis.rink_registration.keypoints import KEYPOINT_NAMES

OUT_DIR = PROJECT_ROOT / "output" / "_rink_reg_val"
CLIPS = {
    "wpg_pp": "data/raw_videos/wpg_pp_60sec.mp4",
    "rush": "data/raw_videos/rush_30sec_clip.mp4",
}
SAMPLE_FRAMES = [120, 360, 600, 900, 1200]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reg = RinkRegistrationModel("models/rink_keypoints.pt")
    if not reg.available:
        print("model not available")
        return 2

    total_fits = 0
    total_attempts = 0

    for clip_name, rel in CLIPS.items():
        path = PROJECT_ROOT / rel
        if not path.exists():
            print(f"skip {clip_name}: missing {path}")
            continue
        cap = cv2.VideoCapture(str(path))
        print(f"\n=== {clip_name} ({rel}) ===")
        for fi in SAMPLE_FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            total_attempts += 1
            kps = reg.detect_keypoints(frame)
            n_conf = 0 if kps is None else int((kps[:, 2] >= reg.kp_conf).sum())
            result = reg.estimate(frame)

            if result is None:
                print(f"  frame {fi:>4}: {n_conf:>2} keypoints "
                      f">= conf{reg.kp_conf}  -> NO FIT")
            else:
                _, _, info = result
                total_fits += 1
                print(f"  frame {fi:>4}: {n_conf:>2} keypoints  -> FIT  "
                      f"n_used={info['n_used']:>2}  "
                      f"reproj_err={info['median_reproj_err_ft']:.2f}ft  "
                      f"kp_conf={info['kp_conf_mean']:.2f}")

            # Annotate detected keypoints
            out = frame.copy()
            if kps is not None:
                for i, (px, py, c) in enumerate(kps):
                    if c < 0.15:
                        continue
                    color = (0, 255, 0) if c >= reg.kp_conf else (0, 165, 255)
                    cv2.circle(out, (int(px), int(py)), 5, color, -1)
                    cv2.putText(out, KEYPOINT_NAMES[i][:10], (int(px) + 6, int(py)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            cv2.imwrite(str(OUT_DIR / f"{clip_name}_{fi:04d}.png"), out)
        cap.release()

    print(f"\nfits: {total_fits}/{total_attempts} sampled frames")
    print(f"annotated frames -> {OUT_DIR.relative_to(PROJECT_ROOT)}")
    if total_fits == 0:
        print("\nVERDICT: synthetic-to-real gap is severe — model doesn't "
              "transfer. Need real-frame fine-tuning before integration.")
    elif total_fits < total_attempts // 2:
        print("\nVERDICT: partial transfer. Fine-tuning on real frames "
              "recommended before integration.")
    else:
        print("\nVERDICT: model transfers to real broadcast footage. "
              "Proceed to integrate into RinkCalibrator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
