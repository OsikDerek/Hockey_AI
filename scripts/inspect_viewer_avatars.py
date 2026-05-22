"""Frame-by-frame fidelity probe for the 3D viewer's avatars.

The side-by-side verification video draws the positions JSON directly, so
it shows each object exactly where the (motion-filtered) data places it.
The 3D viewer re-processes that same data -- an exponential-decay lerp plus
its own outlier rejection -- before drawing. This probe measures how far
the viewer's rendered avatars drift from the underlying data, so the
viewer can be made to match the verification view.

It plays the viewer through the whole clip, samples every RAF tick via
window.__hockeyAI.snapshot(), and for each sample compares, per track:
  - rendered position vs the data position at that frame  (total drift:
    lerp lag + outlier-rejection stall -- this is the gap vs the verify view)
  - target position  vs the data position at that frame  (outlier-rejection
    stall alone: how often the viewer refused a data sample)

Run from the project root with the viewer server already up:
    .venv/Scripts/python.exe scripts/serve_viewer.py        # (background)
    .venv/Scripts/python.exe scripts/inspect_viewer_avatars.py --clip caufield_trim_b3
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="caufield_trim_b3")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    json_path = PROJECT_ROOT / "output" / f"{args.clip}_positions.json"
    if not json_path.exists():
        print(f"not found: {json_path}")
        return 1
    data = json.loads(json_path.read_text())
    frames = data["frames"]
    # Per data frame: track_id -> (ice_x, ice_y) for players + goalies.
    frame_pos = []
    for fr in frames:
        d = {}
        for o in (fr.get("players") or []) + (fr.get("goalies") or []):
            d[o["track_id"]] = (o["ice_x"], o["ice_y"])
        frame_pos.append(d)

    url = (f"http://localhost:{args.port}/viewer/index.html"
           f"?data=/output/{args.clip}_positions.json")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_context(
            viewport={"width": 1400, "height": 800}).new_page()
        page.on("pageerror", lambda e: print(f"[pageerror] {e}"))
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        page.wait_for_function(
            "document.getElementById('data-status').textContent.includes('frames')",
            timeout=8000)
        page.wait_for_function("typeof window.__hockeyAI === 'object'",
                               timeout=5000)

        # RAF probe: record a snapshot every render tick.
        page.evaluate("""
          () => {
            window.__samples = [];
            const origRAF = window.requestAnimationFrame.bind(window);
            window.requestAnimationFrame = function(cb) {
              return origRAF(function(t) {
                cb(t);
                if (window.__hockeyAI) window.__samples.push(window.__hockeyAI.snapshot());
              });
            };
          }
        """)

        page.select_option("#speed", "1")
        page.click("#play-btn")
        # caufield_trim is ~12 s at 1x; sample a little longer.
        page.wait_for_timeout(14000)
        samples = page.evaluate("() => window.__samples")
        browser.close()

    print(f"\n=== VIEWER AVATAR FIDELITY: {args.clip} ===")
    print(f"render samples: {len(samples)}\n")
    if len(samples) < 30:
        print("too few samples — render rate too low")
        return 1

    drift = []        # |rendered - data[frame]|  per (sample, track)
    stall = []        # |target   - data[frame]|  per (sample, track)
    step = []         # |rendered_t - rendered_{t-1}|  per track per tick
    worst = []        # (drift, fi, tid, rendered, target, data)
    prev_rendered: dict = {}
    n_frames = len(frame_pos)

    for s in samples:
        fidx = float(s.get("frameIdx", 0))
        fi = int(fidx)
        frac = fidx - fi
        if fi < 0 or fi >= n_frames:
            continue
        dpos = frame_pos[fi]
        dnext = frame_pos[min(fi + 1, n_frames - 1)]
        seen = set()
        for a in s["avatars"]:
            tid = a["tid"]
            seen.add(tid)
            if tid in dpos:
                dx, dy = dpos[tid]
                # Expected rendered position = data interpolated to the
                # fractional frame (what frame-bracket interpolation targets).
                ex, ey = dx, dy
                if tid in dnext:
                    nx, ny = dnext[tid]
                    if math.hypot(nx - dx, ny - dy) < 12:  # not an id-reuse jump
                        ex, ey = dx + (nx - dx) * frac, dy + (ny - dy) * frac
                dr = math.hypot(a["x"] - ex, a["z"] - ey)
                drift.append(dr)
                worst.append((dr, fi, tid, (round(a["x"], 1), round(a["z"], 1)),
                              (round(a.get("tx") or 0, 1), round(a.get("ty") or 0, 1)),
                              (round(dx, 1), round(dy, 1))))
                if a.get("tx") is not None:
                    stall.append(math.hypot(a["tx"] - dx, a["ty"] - dy))
            if tid in prev_rendered:
                px, pz = prev_rendered[tid]
                step.append(math.hypot(a["x"] - px, a["z"] - pz))
            prev_rendered[tid] = (a["x"], a["z"])
        for tid in list(prev_rendered):
            if tid not in seen:
                prev_rendered.pop(tid, None)

    def report(label, vals, unit):
        if not vals:
            print(f"  {label}: no data")
            return
        v = sorted(vals)
        p50 = v[len(v) // 2]
        p90 = v[int(len(v) * 0.9)]
        p99 = v[min(len(v) - 1, int(len(v) * 0.99))]
        print(f"  {label}: n={len(v):5d}  p50={p50:.2f}  p90={p90:.2f}  "
              f"p99={p99:.2f}  max={v[-1]:.2f}  {unit}")

    print("rendered avatar vs data position (the gap vs the verify view):")
    report("drift", drift, "ft")
    print("\ntarget vs data position (outlier-rejection stalls):")
    report("stall", stall, "ft")
    n_stall = sum(1 for x in stall if x > 1.0)
    print(f"  target >1 ft off the data: {n_stall}/{len(stall)} "
          f"({100*n_stall/max(1,len(stall)):.1f}%)")
    print("\nper-render-tick rendered step (smoothness; lower = smoother):")
    report("step", step, "ft/tick")

    print("\nworst drift samples (drift ft | frame | track | rendered | target | data):")
    for dr, fi, tid, r, t, d in sorted(worst, reverse=True)[:12]:
        print(f"  {dr:6.1f} | f{fi:3d} | t{tid:<4d} | rend {r} | tgt {t} | data {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
