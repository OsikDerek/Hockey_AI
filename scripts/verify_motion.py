"""Motion-quality audit for a positions JSON.

A detection being *present* in a frame is not the same as it being *correct*.
A false-positive puck or a mis-assigned player jumps frame-to-frame in ways
no real skater or puck physically can. This tool measures how smooth and
physically plausible each tracked object's motion is, in ice coordinates:

  - per-transition speed (ft/s) between consecutive present frames
  - implausible-speed transitions  (speed above a physical ceiling)
  - return-spike outliers          (jumps out one frame and back the next --
                                    the classic false-positive signature)
  - a smoothness score per object  (fraction of clean transitions)

Run it before and after any tracking change to see whether motion got
cleaner, not just whether more frames have a detection.

Usage:
  .venv/Scripts/python.exe scripts/verify_motion.py --clip caufield_trim_b3
"""

import argparse
import csv
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Physical ceilings in ft/s. A transition faster than this cannot be the
# same real object moving -- it is a tracking error.
PUCK_MAX_FT_S = 165.0     # ~112 mph; the hardest NHL shots sit near 108 mph
PLAYER_MAX_FT_S = 48.0    # ~33 mph; NHL top skating speed bursts to ~30 mph
GOALIE_MAX_FT_S = 40.0

# A single-frame jump beyond this many feet is "suspicious" -- used only to
# pair with the return test, so a real fast shot (consistent direction) is
# NOT flagged but an out-and-back spike is.
PUCK_SPIKE_FT = 6.0
PLAYER_SPIKE_FT = 3.0

# Transitions across a gap longer than this are not scored for speed -- the
# object could legitimately be anywhere after a long absence.
MAX_GAP = 6


def analyze_track(samples, fps, max_ft_s, spike_ft, label):
    """samples: sorted list of (frame_idx, x, y). Returns a metrics dict."""
    transitions = []          # (frame, speed_ft_s, gap)
    implausible = []          # frames whose incoming speed exceeds the ceiling
    for (f0, x0, y0), (f1, x1, y1) in zip(samples, samples[1:]):
        gap = f1 - f0
        if gap <= 0 or gap > MAX_GAP:
            continue
        dist = math.hypot(x1 - x0, y1 - y0)
        speed = dist / (gap / fps)
        transitions.append((f1, speed, gap))
        if speed > max_ft_s:
            implausible.append(f1)

    # Return-spike: a present frame i whose neighbours i-1 and i+1 are both
    # present and consecutive, where i jumped away from and back to a
    # consistent i-1 / i+1 pair. i is the outlier.
    by_frame = {f: (x, y) for f, x, y in samples}
    spikes = []
    for f, x, y in samples:
        a = by_frame.get(f - 1)
        b = by_frame.get(f + 1)
        if a is None or b is None:
            continue
        d_prev = math.hypot(x - a[0], y - a[1])
        d_next = math.hypot(x - b[0], y - b[1])
        d_skip = math.hypot(b[0] - a[0], b[1] - a[1])
        if d_prev > spike_ft and d_next > spike_ft and d_skip < spike_ft:
            spikes.append(f)

    n_trans = len(transitions)
    clean = sum(1 for _, s, _ in transitions if s <= max_ft_s)
    clean -= sum(1 for f, _, _ in transitions if f in set(spikes))
    smoothness = (clean / n_trans) if n_trans else 1.0
    speeds = sorted(s for _, s, _ in transitions)

    def pct(p):
        if not speeds:
            return 0.0
        return speeds[min(len(speeds) - 1, int(p * len(speeds)))]

    return {
        "label": label,
        "n_samples": len(samples),
        "n_transitions": n_trans,
        "speed_p50": round(pct(0.50), 1),
        "speed_p90": round(pct(0.90), 1),
        "speed_max": round(speeds[-1], 1) if speeds else 0.0,
        "n_implausible": len(implausible),
        "n_spikes": len(spikes),
        "implausible_frames": implausible,
        "spike_frames": spikes,
        "smoothness": round(smoothness, 3),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="caufield_trim_b3",
                    help="positions JSON basename (without _positions.json)")
    ap.add_argument("--min-track-len", type=int, default=8,
                    help="ignore player/goalie tracks shorter than this")
    args = ap.parse_args(argv)

    json_path = PROJECT_ROOT / "output" / f"{args.clip}_positions.json"
    if not json_path.exists():
        raise SystemExit(f"not found: {json_path}")
    data = json.loads(json_path.read_text())
    fps = float(data.get("fps", 30.0))
    frames = data["frames"]
    n = len(frames)

    # ---- puck: one implicit track --------------------------------------
    puck_samples = []
    for fr in frames:
        p = fr.get("puck")
        if p is not None:
            puck_samples.append((fr["frame_idx"], p["ice_x"], p["ice_y"]))
    puck = analyze_track(puck_samples, fps, PUCK_MAX_FT_S, PUCK_SPIKE_FT, "puck")

    # ---- players + goalies: keyed by track_id --------------------------
    def collect(key, max_ft_s, spike_ft):
        tracks = {}
        for fr in frames:
            for o in fr.get(key, []) or []:
                tid = o.get("track_id")
                if tid is None:
                    continue
                tracks.setdefault(tid, []).append(
                    (fr["frame_idx"], o["ice_x"], o["ice_y"]))
        out = []
        for tid, s in tracks.items():
            if len(s) < args.min_track_len:
                continue
            s.sort()
            out.append(analyze_track(s, fps, max_ft_s, spike_ft,
                                     f"{key[:-1]}#{tid}"))
        return out

    players = collect("players", PLAYER_MAX_FT_S, PLAYER_SPIKE_FT)
    goalies = collect("goalies", GOALIE_MAX_FT_S, PLAYER_SPIKE_FT)

    # ---- report --------------------------------------------------------
    print(f"\n=== MOTION AUDIT: {args.clip} ({n} frames @ {fps:.1f} fps) ===\n")

    def line(m):
        print(f"  {m['label']:<14} samples {m['n_samples']:4d}  "
              f"speed p50/p90/max {m['speed_p50']:6.1f}/{m['speed_p90']:6.1f}/"
              f"{m['speed_max']:7.1f} ft/s  "
              f"implausible {m['n_implausible']:3d}  spikes {m['n_spikes']:3d}  "
              f"smoothness {m['smoothness']:.3f}")

    print("PUCK")
    line(puck)
    if puck["implausible_frames"]:
        print(f"    implausible-speed frames: {puck['implausible_frames']}")
    if puck["spike_frames"]:
        print(f"    return-spike frames:      {puck['spike_frames']}")

    for grp_name, grp in (("PLAYERS", players), ("GOALIES", goalies)):
        if not grp:
            continue
        print(f"\n{grp_name} ({len(grp)} tracks >= {args.min_track_len} frames)")
        for m in sorted(grp, key=lambda d: d["smoothness"]):
            line(m)
        tot_imp = sum(m["n_implausible"] for m in grp)
        tot_spk = sum(m["n_spikes"] for m in grp)
        avg_smooth = sum(m["smoothness"] for m in grp) / len(grp)
        print(f"  -> total implausible {tot_imp}, total spikes {tot_spk}, "
              f"mean smoothness {avg_smooth:.3f}")

    # ---- CSV: per-frame puck speed + flags -----------------------------
    out_dir = PROJECT_ROOT / "output" / "_verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.clip}_motion.csv"
    puck_by_frame = {f: (x, y) for f, x, y in puck_samples}
    imp = set(puck["implausible_frames"])
    spk = set(puck["spike_frames"])
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "puck_present", "puck_x", "puck_y",
                    "puck_implausible", "puck_spike"])
        for fr in frames:
            fi = fr["frame_idx"]
            pos = puck_by_frame.get(fi)
            w.writerow([fi, int(pos is not None),
                        round(pos[0], 2) if pos else "",
                        round(pos[1], 2) if pos else "",
                        int(fi in imp), int(fi in spk)])
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
