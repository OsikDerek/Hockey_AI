"""Unified tracking-quality scorecard.

One command that re-runs the game-analysis pipeline on the canonical clip,
scores the result against three things, and prints a single scorecard with
green/red deltas vs a saved baseline:

  ACCURACY  -- mean placement error in feet vs the hand-annotated ground
               truth (scripts/annotate_ground_truth.py). This is the oracle
               a smoothness metric cannot give you: it catches a track that
               is smooth but confidently wrong.
  SMOOTHNESS -- per-object motion plausibility from the positions JSON
               (puck + player tracks): implausible transitions, smoothness.
  CALIBRATION -- homography reprojection error over the annotated keyframes.

Iteration loop: change code -> run this -> read the deltas -> keep or revert.

    .venv/Scripts/python.exe scripts/scorecard.py                # run + score
    .venv/Scripts/python.exe scripts/scorecard.py --skip-pipeline # score only
    .venv/Scripts/python.exe scripts/scorecard.py --save-baseline # set baseline
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))                       # for `src.*`
sys.path.insert(0, str(Path(__file__).resolve().parent))    # for verify_motion
import verify_motion as vm  # noqa: E402  (sibling module: analyze_track + caps)

# Player recall radius (feet). A ground-truth player mark with a tracked
# player within this distance counts as detected. Deliberately generous:
# the puck sits on the ice so its error is exact, but a player mark is
# clicked on the body while the pipeline projects the skates, a 5-9 ft
# offset through the homography -- so the player metric is recall ("did we
# find this skater"), not a precise placement error.
PLAYER_RECALL_FT = 16.0


def _project(pts_px, H):
    """Project (N,2) pixel points to ice feet via a pixel->ice homography."""
    if not pts_px:
        return []
    arr = np.array(pts_px, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(arr, H).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in out]


def accuracy_metrics(gt, frames_by_idx, video_path):
    """Score the positions JSON against the hand-annotated ground truth."""
    from src.game_analysis.rink_registration.registration_model import (
        RinkRegistrationModel)
    reg = RinkRegistrationModel()

    cap = cv2.VideoCapture(str(video_path))
    ann = {int(k): v for k, v in gt["annotations"].items()}

    puck_err = []
    puck_missed = 0           # GT has a puck, pipeline placed none
    player_found = 0          # GT player mark with a tracked player nearby
    player_total = 0          # GT player marks scored
    reproj = []
    n_uncalibrated = 0

    for kf in sorted(ann):
        rec = ann[kf]
        cap.set(cv2.CAP_PROP_POS_FRAMES, kf)
        ok, frame = cap.read()
        if not ok:
            continue
        est = reg.estimate(frame)
        if est is None:
            n_uncalibrated += 1
            continue
        H_p2i, _, info = est
        reproj.append(info["median_reproj_err_ft"])

        fr = frames_by_idx.get(kf)
        if fr is None:
            continue

        # --- puck ---
        if isinstance(rec.get("puck"), list):
            gt_ice = _project([rec["puck"]], H_p2i)[0]
            if fr.get("puck"):
                p = fr["puck"]
                puck_err.append(float(np.hypot(p["ice_x"] - gt_ice[0],
                                                p["ice_y"] - gt_ice[1])))
            else:
                puck_missed += 1

        # --- players: recall -- each GT mark greedily claims the nearest
        #     still-unused tracked player within the recall radius ---
        gt_players = _project(rec.get("players", []), H_p2i)
        pipe = [(pl["ice_x"], pl["ice_y"]) for pl in fr.get("players", [])]
        used = set()
        for gx, gy in gt_players:
            player_total += 1
            best_d, best_j = 1e9, -1
            for j, (px, py) in enumerate(pipe):
                if j in used:
                    continue
                d = float(np.hypot(px - gx, py - gy))
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_d <= PLAYER_RECALL_FT:
                used.add(best_j)
                player_found += 1

    cap.release()

    def pct(vals, p):
        if not vals:
            return 0.0
        s = sorted(vals)
        return s[min(len(s) - 1, int(p * len(s)))]

    return {
        "puck_err_p50_ft": round(pct(puck_err, 0.50), 2),
        "puck_err_p90_ft": round(pct(puck_err, 0.90), 2),
        "puck_err_max_ft": round(max(puck_err), 2) if puck_err else 0.0,
        "puck_scored": len(puck_err),
        "puck_missed": puck_missed,
        "player_recall_pct": round(100.0 * player_found / max(1, player_total), 1),
        "player_marks_gt": player_total,
        "calib_reproj_p50_ft": round(pct(reproj, 0.50), 2),
        "calib_reproj_p90_ft": round(pct(reproj, 0.90), 2),
        "keyframes_uncalibrated": n_uncalibrated,
    }


def smoothness_metrics(data):
    """Puck + aggregate player motion plausibility from the positions JSON."""
    fps = float(data.get("fps", 30.0))
    frames = data["frames"]

    puck_s = [(f["frame_idx"], f["puck"]["ice_x"], f["puck"]["ice_y"])
              for f in frames if f.get("puck")]
    puck = vm.analyze_track(puck_s, fps, vm.PUCK_MAX_FT_S, vm.PUCK_SPIKE_FT, "puck")

    tracks: dict = {}
    for f in frames:
        for p in f.get("players", []) or []:
            tracks.setdefault(p["track_id"], []).append(
                (f["frame_idx"], p["ice_x"], p["ice_y"]))
    pms = []
    for s in tracks.values():
        if len(s) >= 8:
            s.sort()
            pms.append(vm.analyze_track(s, fps, vm.PLAYER_MAX_FT_S,
                                        vm.PLAYER_SPIKE_FT, "p"))
    return {
        "puck_smoothness": puck["smoothness"],
        "puck_implausible": puck["n_implausible"],
        "puck_present": len(puck_s),
        "player_smoothness": round(
            sum(m["smoothness"] for m in pms) / max(1, len(pms)), 3),
        "player_implausible": sum(m["n_implausible"] for m in pms),
    }


# Metrics where a SMALLER value is better (for delta arrows). Everything
# else (smoothness, recall, present/scored counts) is better when larger.
LOWER_BETTER = {
    "puck_err_p50_ft", "puck_err_p90_ft", "puck_err_max_ft", "puck_missed",
    "calib_reproj_p50_ft", "calib_reproj_p90_ft", "keyframes_uncalibrated",
    "puck_implausible", "player_implausible",
}


def print_scorecard(metrics, baseline):
    groups = [
        ("ACCURACY vs ground truth", [
            "puck_err_p50_ft", "puck_err_p90_ft", "puck_err_max_ft",
            "puck_scored", "puck_missed",
            "player_recall_pct", "player_marks_gt"]),
        ("SMOOTHNESS", [
            "puck_smoothness", "puck_implausible", "puck_present",
            "player_smoothness", "player_implausible"]),
        ("CALIBRATION", [
            "calib_reproj_p50_ft", "calib_reproj_p90_ft",
            "keyframes_uncalibrated"]),
    ]
    print("\n" + "=" * 62)
    print("  TRACKING SCORECARD" + ("  (vs baseline)" if baseline else ""))
    print("=" * 62)
    for title, keys in groups:
        print(f"\n{title}")
        for k in keys:
            v = metrics.get(k)
            if v is None:
                continue
            line = f"  {k:<26} {v:>9}"
            if baseline and k in baseline:
                b = baseline[k]
                dv = round(v - b, 2)
                if abs(dv) < 1e-9:
                    line += "    (no change)"
                else:
                    better = (dv < 0) == (k in LOWER_BETTER)
                    mark = "improved" if better else "REGRESSED"
                    line += f"   {dv:+.2f}  {mark}"
            print(line)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="caufield_trim_b3")
    ap.add_argument("--video", default="data/raw_videos/caufield_trim.mp4")
    ap.add_argument("--gt", default="output/_gt/caufield_trim_gt.json")
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="score the existing positions JSON without re-running")
    ap.add_argument("--save-baseline", action="store_true",
                    help="store this run as the baseline for future deltas")
    args = ap.parse_args()

    video = PROJECT_ROOT / args.video
    json_path = PROJECT_ROOT / "output" / f"{args.clip}_positions.json"
    gt_path = PROJECT_ROOT / args.gt

    if not args.skip_pipeline:
        print(f"running pipeline on {video.name} ...")
        r = subprocess.run(
            [sys.executable, "main.py", "--game-analysis", "-i", str(video),
             "-o", f"output/{args.clip}.mp4"],
            cwd=PROJECT_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            raise SystemExit("pipeline run failed")

    if not json_path.exists():
        raise SystemExit(f"not found: {json_path}")
    if not gt_path.exists():
        raise SystemExit(f"ground truth not found: {gt_path}")
    data = json.loads(json_path.read_text())
    gt = json.loads(gt_path.read_text())
    frames_by_idx = {f["frame_idx"]: f for f in data["frames"]}

    metrics = {}
    metrics.update(smoothness_metrics(data))
    print("scoring against ground truth ...")
    metrics.update(accuracy_metrics(gt, frames_by_idx, video))

    out = PROJECT_ROOT / "output" / "_scorecard.json"
    out.write_text(json.dumps(metrics, indent=1))
    baseline_path = PROJECT_ROOT / "output" / "_scorecard_baseline.json"
    baseline = (json.loads(baseline_path.read_text())
                if baseline_path.exists() else None)

    print_scorecard(metrics, baseline)
    print(f"scorecard: {out}")
    if args.save_baseline:
        baseline_path.write_text(json.dumps(metrics, indent=1))
        print(f"baseline saved: {baseline_path}")
    elif baseline is None:
        print("no baseline yet — run with --save-baseline to set one")


if __name__ == "__main__":
    main()
