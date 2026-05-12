"""Visualize whether detect_rink_boards finds the FAR (top) edge of
the ice on livebarn_cropped frames. If only the near (bottom) edge is
found, every line × board intersection collapses to y=0 — homography
has no across-rink y constraint and y-compression is the inevitable
result.
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.game_analysis.rink_lines import (
    detect_rink_boards, detect_rink_lines,
    detect_rink_edges_via_hough, detect_rink_edges_via_row_brightness,
)

VIDEO = PROJECT_ROOT / "data" / "raw_videos" / "livebarn_60sec_cropped.mp4"
SAMPLE_FRAMES = [60, 300, 600, 900, 1200, 1500]
OUT_DIR = PROJECT_ROOT / "output" / "_calib_diag"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    print(f"{'frame':>6}  {'top':<14}  {'bottom':<14}  {'#blue':>5}  {'#red':>4}")
    print("-" * 60)
    for fi in SAMPLE_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            print(f"{fi}: read failed")
            continue

        boards = detect_rink_boards(frame)
        hough_boards = detect_rink_edges_via_hough(frame)
        bright_boards = detect_rink_edges_via_row_brightness(frame)
        lines = detect_rink_lines(frame)

        def fmt(b):
            if b is None:
                return "(none)"
            return f"y=({b['p1'][1]:.0f},{b['p2'][1]:.0f})"

        n_blue = len(lines.get("blue", []))
        n_red = len(lines.get("red", []))
        print(f"{fi:>6} | blob:t={fmt(boards['top']):<14} b={fmt(boards['bottom']):<14} | "
              f"bright:t={fmt(bright_boards['top']):<14} b={fmt(bright_boards['bottom']):<14}")

        # Annotate + save
        out = frame.copy()
        for label, board, color in [
            ("TOP", bright_boards["top"], (0, 255, 0)),
            ("BOTTOM", bright_boards["bottom"], (0, 200, 0)),
        ]:
            if board is None:
                continue
            p1, p2 = board["p1"], board["p2"]
            cv2.line(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 3)
            cv2.putText(out, label,
                        (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        for color_name, segs in [("blue", lines.get("blue", [])),
                                  ("red", lines.get("red", []))]:
            col = (255, 0, 0) if color_name == "blue" else (0, 0, 255)
            for s in segs:
                p1, p2 = s["p1"], s["p2"]
                cv2.line(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), col, 2)

        out_path = OUT_DIR / f"boards_{fi:04d}.png"
        cv2.imwrite(str(out_path), out)
    cap.release()
    print()
    print(f"Annotated frames written to: {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
