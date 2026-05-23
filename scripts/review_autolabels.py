"""Review-and-correct UI for auto-labelled puck training frames.

autolabel_puck.py flags every uncertain or missing puck frame as REVIEW;
this tool walks those frames in an OpenCV window so the human can click
the true puck (or press 'x' for "no puck visible"). Each click writes a
YOLO label and promotes the frame's manifest status to "human".

Far faster than annotating from scratch: most frames are already auto-
labelled (no review needed), and the review frames pre-fill any hint
bbox the pipeline tried but wasn't confident in -- you typically just
nudge or skip.

    .venv/Scripts/python.exe scripts/review_autolabels.py --clip <name>

Controls (also shown across the top of the window):
    left-click   set the puck position (overrides the hint)
    y            ACCEPT the pipeline hint as the label + advance
    x            mark "no puck visible" + advance (writes an empty label)
    space / n    next frame (no decision)
    b / p        previous frame
    u            clear this frame's label
    s            save now         q / Esc   save + quit

Workflow: glance at the hint, hit y to accept, click to correct, x for
no-puck. Most frames are 1-2 seconds.

Auto-saves on every keypress; resumable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PUCK_COLOR = (0, 215, 255)
HINT_COLOR = (0, 140, 200)
LOUPE_SRC = 150
LOUPE_VIEW = 420
PUCK_BBOX_PX = 14   # synthetic bbox size for a human click (a puck is tiny)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="basename used by autolabel_puck")
    ap.add_argument("--dataset", default="data/training/puck_nhl")
    ap.add_argument("--max-width", type=int, default=1600)
    args = ap.parse_args()

    root = PROJECT_ROOT / args.dataset
    mpath = root / "manifests" / f"{args.clip}.json"
    if not mpath.exists():
        raise SystemExit(f"manifest not found: {mpath}  (run autolabel_puck.py first)")
    manifest = json.loads(mpath.read_text())

    review_frames = [f for f in manifest["frames"] if f["status"] == "review"]
    if not review_frames:
        print(f"{args.clip}: nothing to review (all frames auto-labelled or empty)")
        return

    w_full, h_full = manifest["frame_w"], manifest["frame_h"]
    scale = min(1.0, args.max_width / w_full)
    disp_w, disp_h = int(w_full * scale), int(h_full * scale)

    state = {"mx": 0, "my": 0}

    def write_label(fi, cx, cy):
        cxn, cyn = cx / w_full, cy / h_full
        wn = hn = PUCK_BBOX_PX / max(w_full, h_full)
        path = root / "labels" / f"{args.clip}_f{fi:05d}.txt"
        path.write_text(f"0 {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f}\n")

    def empty_label(fi):
        path = root / "labels" / f"{args.clip}_f{fi:05d}.txt"
        path.write_text("")   # negative example: no puck this frame

    def clear_label(fi):
        path = root / "labels" / f"{args.clip}_f{fi:05d}.txt"
        if path.exists():
            path.unlink()

    def save():
        mpath.write_text(json.dumps(manifest, indent=1))

    idx = [0]   # boxed so the mouse callback can read it
    cur_frame = None  # numpy array of the current displayed frame

    def on_mouse(event, x, y, flags, _):
        state["mx"], state["my"] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            nx, ny = int(x / scale), int(y / scale)
            fr = review_frames[idx[0]]
            fi = fr["frame_idx"]
            write_label(fi, nx, ny)
            fr["status"] = "human"
            fr["click_px"] = [nx, ny]
            save()

    win = "review puck auto-labels — see console for controls"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    def load_frame():
        nonlocal cur_frame
        fi = review_frames[idx[0]]["frame_idx"]
        img = root / "images" / f"{args.clip}_f{fi:05d}.jpg"
        cur_frame = cv2.imread(str(img))
        if cur_frame is None:
            cur_frame = np.zeros((h_full, w_full, 3), np.uint8)

    load_frame()
    running = True
    while running:
        fr = review_frames[idx[0]]
        fi = fr["frame_idx"]
        disp = cv2.resize(cur_frame, (disp_w, disp_h))

        # Hint bbox the pipeline tried but wasn't confident enough in.
        hb = fr.get("hint_bbox_px")
        if hb:
            x1, y1, x2, y2 = [int(v * scale) for v in hb]
            cv2.rectangle(disp, (x1, y1), (x2, y2), HINT_COLOR, 1)
            cv2.putText(disp, "pipeline hint", (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, HINT_COLOR, 1, cv2.LINE_AA)
        # User's click (if any).
        cl = fr.get("click_px")
        if cl:
            cx, cy = int(cl[0] * scale), int(cl[1] * scale)
            cv2.drawMarker(disp, (cx, cy), PUCK_COLOR, cv2.MARKER_CROSS, 22, 2)
            cv2.circle(disp, (cx, cy), 13, PUCK_COLOR, 1)

        # Magnifier loupe.
        cx, cy = int(state["mx"] / scale), int(state["my"] / scale)
        x0, y0 = (np.clip(cx - LOUPE_SRC // 2, 0, w_full - LOUPE_SRC),
                  np.clip(cy - LOUPE_SRC // 2, 0, h_full - LOUPE_SRC))
        loupe = cur_frame[y0:y0 + LOUPE_SRC, x0:x0 + LOUPE_SRC]
        if loupe.size:
            loupe = cv2.resize(loupe, (LOUPE_VIEW, LOUPE_VIEW),
                               interpolation=cv2.INTER_NEAREST)
            ctr = LOUPE_VIEW // 2
            cv2.drawMarker(loupe, (ctr, ctr), (0, 0, 255), cv2.MARKER_CROSS, 30, 1)
            cv2.rectangle(loupe, (0, 0), (LOUPE_VIEW - 1, LOUPE_VIEW - 1),
                          (255, 255, 255), 1)
            disp[0:LOUPE_VIEW, disp_w - LOUPE_VIEW:disp_w] = loupe

        # HUD.
        for i, line in enumerate([
            f"review {idx[0]+1}/{len(review_frames)}  "
            f"clip frame {fi}   status: {fr['status']}",
            f"y = ACCEPT hint | L-click = correct | x = no puck | "
            f"space = skip | b = back | q = save+quit",
        ]):
            cv2.putText(disp, line, (12, 26 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(disp, line, (12, 26 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1,
                        cv2.LINE_AA)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF

        if k in (ord(" "), ord("n")):
            save()
            idx[0] = min(len(review_frames) - 1, idx[0] + 1)
            load_frame()
        elif k in (ord("b"), ord("p")):
            save()
            idx[0] = max(0, idx[0] - 1)
            load_frame()
        elif k == ord("y"):
            # Accept the pipeline's hint bbox as the label and advance.
            hb = fr.get("hint_bbox_px")
            if hb:
                cx, cy = (hb[0] + hb[2]) / 2.0, (hb[1] + hb[3]) / 2.0
                write_label(fi, cx, cy)
                fr["status"] = "human"
                fr["click_px"] = [int(cx), int(cy)]
                save()
                idx[0] = min(len(review_frames) - 1, idx[0] + 1)
                load_frame()
        elif k == ord("x"):
            empty_label(fi)
            fr["status"] = "negative"
            fr.pop("click_px", None)
            save()
            idx[0] = min(len(review_frames) - 1, idx[0] + 1)
            load_frame()
        elif k == ord("u"):
            clear_label(fi)
            fr["status"] = "review"
            fr.pop("click_px", None)
            save()
        elif k == ord("s"):
            save()
            print(f"saved -> {mpath}")
        elif k in (ord("q"), 27):
            running = False

    save()
    cv2.destroyAllWindows()
    done = sum(1 for f in review_frames if f["status"] in ("human", "negative"))
    print(f"\nreview complete: {done}/{len(review_frames)} review frames resolved")


if __name__ == "__main__":
    main()
