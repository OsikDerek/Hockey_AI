"""Ground-truth annotation tool for the accuracy oracle.

The motion/calibration/viewer probes all measure smoothness and internal
consistency -- but a track can be perfectly smooth and confidently WRONG
(the puck tracker locking onto a stick blade, say). To measure real
accuracy we need ground truth: a human marking where the puck (and a few
players) ACTUALLY are, independent of what the pipeline detected.

This is that tool. It steps through evenly-spaced keyframes of a clip and
lets you click the true position of the puck on each. The result is a JSON
the scorecard harness scores every pipeline change against -- "mean error
in feet vs the truth".

Run it (a desktop OpenCV window opens):
    .venv/Scripts/python.exe scripts/annotate_ground_truth.py

Controls (shown in the window):
    left-click   place the puck (puck mode) / add a player (player mode)
    Tab          toggle puck <-> player mode
    space / n    next keyframe        b / p   previous keyframe
    x            mark the puck as NOT visible this frame
    u            undo the last click on this frame
    c            clear this frame
    s            save now            q / Esc  save and quit

Progress auto-saves on every keyframe change, so quitting is always safe;
re-running resumes where you left off.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PUCK_COLOR = (0, 215, 255)     # amber
PLAYER_COLOR = (80, 220, 80)   # green
LOUPE_SRC = 150                # native px sampled into the magnifier
LOUPE_VIEW = 420               # on-screen size of the magnifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/raw_videos/caufield_trim.mp4")
    ap.add_argument("--n", type=int, default=40, help="number of keyframes")
    ap.add_argument("--out", default="output/_gt/caufield_trim_gt.json")
    ap.add_argument("--max-width", type=int, default=1600,
                    help="window width cap; the frame is scaled to fit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the keyframe plan and exit (no GUI)")
    args = ap.parse_args()

    video = PROJECT_ROOT / args.video
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Evenly-spaced keyframes across the clip.
    keyframes = [int(round(i * (total - 1) / max(1, args.n - 1)))
                 for i in range(args.n)]
    keyframes = sorted(set(keyframes))

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from any existing annotations.
    ann: dict = {}
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        ann = {int(k): v for k, v in prev.get("annotations", {}).items()}

    print(f"{video.name}: {total} frames {w}x{h}")
    print(f"keyframes ({len(keyframes)}): {keyframes}")
    print(f"output: {out_path}   already annotated: {len(ann)}")
    if args.dry_run:
        return

    scale = min(1.0, args.max_width / w)
    disp_w, disp_h = int(w * scale), int(h * scale)

    state = {"mx": 0, "my": 0, "mode": "puck"}  # cursor in display coords

    def on_mouse(event, x, y, flags, _):
        state["mx"], state["my"] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            nx, ny = int(x / scale), int(y / scale)   # -> native pixels
            rec = ann.setdefault(keyframes[idx[0]], {"puck": "unset", "players": []})
            if state["mode"] == "puck":
                rec["puck"] = [nx, ny]
            else:
                rec["players"].append([nx, ny])

    win = "ground truth — see console for controls"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    def save():
        out_path.write_text(json.dumps({
            "video": args.video, "frame_w": w, "frame_h": h,
            "keyframes": keyframes,
            "annotations": {str(k): v for k, v in sorted(ann.items())},
        }, indent=1))

    idx = [0]  # current keyframe position (boxed so the callback can read it)

    def load_frame():
        cap.set(cv2.CAP_PROP_POS_FRAMES, keyframes[idx[0]])
        ok, fr = cap.read()
        return fr if ok else np.zeros((h, w, 3), np.uint8)

    frame = load_frame()
    running = True
    while running:
        kf = keyframes[idx[0]]
        rec = ann.get(kf, {"puck": "unset", "players": []})
        disp = cv2.resize(frame, (disp_w, disp_h))

        # Drawn annotations.
        if isinstance(rec.get("puck"), list):
            px, py = int(rec["puck"][0] * scale), int(rec["puck"][1] * scale)
            cv2.drawMarker(disp, (px, py), PUCK_COLOR, cv2.MARKER_CROSS, 22, 2)
            cv2.circle(disp, (px, py), 13, PUCK_COLOR, 1)
        for pl in rec.get("players", []):
            px, py = int(pl[0] * scale), int(pl[1] * scale)
            cv2.drawMarker(disp, (px, py), PLAYER_COLOR, cv2.MARKER_CROSS, 18, 2)

        # Magnifier loupe around the cursor (native-resolution zoom).
        cx, cy = int(state["mx"] / scale), int(state["my"] / scale)
        x0, y0 = np.clip(cx - LOUPE_SRC // 2, 0, w - LOUPE_SRC), \
            np.clip(cy - LOUPE_SRC // 2, 0, h - LOUPE_SRC)
        loupe = frame[y0:y0 + LOUPE_SRC, x0:x0 + LOUPE_SRC]
        if loupe.size:
            loupe = cv2.resize(loupe, (LOUPE_VIEW, LOUPE_VIEW),
                               interpolation=cv2.INTER_NEAREST)
            ctr = LOUPE_VIEW // 2
            cv2.drawMarker(loupe, (ctr, ctr), (0, 0, 255), cv2.MARKER_CROSS, 30, 1)
            cv2.rectangle(loupe, (0, 0), (LOUPE_VIEW - 1, LOUPE_VIEW - 1),
                          (255, 255, 255), 1)
            disp[0:LOUPE_VIEW, disp_w - LOUPE_VIEW:disp_w] = loupe

        # HUD.
        done = sum(1 for v in ann.values() if v.get("puck") != "unset")
        puck_state = ("SET" if isinstance(rec.get("puck"), list)
                      else "NOT VISIBLE" if rec.get("puck") is None else "unset")
        for i, line in enumerate([
            f"keyframe {idx[0]+1}/{len(keyframes)}  (clip frame {kf})   "
            f"annotated {done}/{len(keyframes)}",
            f"MODE: {state['mode'].upper()}   puck: {puck_state}   "
            f"players: {len(rec.get('players', []))}",
            "L-click set  |  Tab mode  |  space/b next/prev  |  x no-puck  "
            "|  u undo  |  c clear  |  q save+quit",
        ]):
            cv2.putText(disp, line, (12, 26 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(disp, line, (12, 26 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1,
                        cv2.LINE_AA)

        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF

        if k in (ord(" "), ord("n")):
            ann.setdefault(kf, {"puck": "unset", "players": []})
            save()
            idx[0] = min(len(keyframes) - 1, idx[0] + 1)
            frame = load_frame()
        elif k in (ord("b"), ord("p")):
            save()
            idx[0] = max(0, idx[0] - 1)
            frame = load_frame()
        elif k == 9:  # Tab
            state["mode"] = "player" if state["mode"] == "puck" else "puck"
        elif k == ord("x"):
            ann.setdefault(kf, {"puck": "unset", "players": []})["puck"] = None
        elif k == ord("u"):
            r = ann.get(kf)
            if r and r.get("players"):
                r["players"].pop()
            elif r:
                r["puck"] = "unset"
        elif k == ord("c"):
            ann[kf] = {"puck": "unset", "players": []}
        elif k == ord("s"):
            save()
            print(f"saved -> {out_path}")
        elif k in (ord("q"), 27):
            running = False

    save()
    cap.release()
    cv2.destroyAllWindows()
    done = sum(1 for v in ann.values() if v.get("puck") != "unset")
    print(f"\nsaved {out_path}  ({done}/{len(keyframes)} keyframes annotated)")


if __name__ == "__main__":
    main()
