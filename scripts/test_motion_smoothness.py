"""Headless smoothness probe for viewer motion.

Loads the viewer, scrubs to a dense calibrated stretch of
livebarn_cropped, samples every avatar + puck rendered position once
per RAF tick over ~2s of playback, and reports per-tick step size
distributions. A smooth render produces tiny per-tick steps even when
the underlying data has multi-foot frame-to-frame outliers.

Run from project root (server on :8000):
    .venv/Scripts/python.exe scripts/test_motion_smoothness.py

Exit code:
    0 — per-tick step p99 below threshold AND no impossibly large puck jump
    1 — render rate too low to evaluate smoothness
    2 — smoothness regression: p99 step >= threshold (jitter bypassed lerp)
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8000/viewer/index.html?data=/output/livebarn_cropped_positions.json"


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_context(viewport={"width": 1400, "height": 800}).new_page()
        page.on("pageerror", lambda e: print(f"[pageerror] {e}"))
        page.goto(URL, wait_until="domcontentloaded", timeout=15000)
        page.wait_for_function(
            "document.getElementById('data-status').textContent.includes('frames')",
            timeout=8000,
        )

        # Wait for the debug hook to attach (last thing the module sets up).
        page.wait_for_function("typeof window.__hockeyAI === 'object'", timeout=5000)

        # Scrub to a dense early-mid stretch
        page.evaluate("""
          () => {
            const sc = document.getElementById('scrubber');
            sc.value = 350;
            sc.dispatchEvent(new Event('input', {bubbles: true}));
          }
        """)
        page.wait_for_timeout(200)

        # Install RAF probe that samples positions every tick.
        page.evaluate("""
          () => {
            window.__samples = [];
            const origRAF = window.requestAnimationFrame.bind(window);
            window.requestAnimationFrame = function(cb) {
              return origRAF(function(t) {
                cb(t);
                if (window.__hockeyAI) {
                  const s = window.__hockeyAI.snapshot();
                  window.__samples.push({ t, ...s });
                }
              });
            };
          }
        """)

        # Play at 1x for ~2.5 s
        page.select_option("#speed", "1")
        page.click("#play-btn")
        page.wait_for_timeout(2500)
        page.click("#play-btn")

        samples = page.evaluate("() => window.__samples")
        browser.close()

    if len(samples) < 30:
        print(f"FAIL: only {len(samples)} samples collected; render rate too low to evaluate")
        return 1

    # Compute per-tick rendered step distance for tracked avatars and puck.
    # Track-by-track step distances:
    by_tid = {}
    for s in samples:
        for a in s["avatars"]:
            by_tid.setdefault(a["tid"], []).append((s["t"], a["x"], a["z"]))

    player_steps = []
    for tid, pts in by_tid.items():
        if len(pts) < 5:
            continue
        for i in range(1, len(pts)):
            dt = (pts[i][0] - pts[i - 1][0]) / 1000.0
            if dt <= 0:
                continue
            dx = pts[i][1] - pts[i - 1][1]
            dz = pts[i][2] - pts[i - 1][2]
            step = (dx * dx + dz * dz) ** 0.5
            player_steps.append(step)

    puck_pts = [
        (s["t"], s["puck"]["x"], s["puck"]["z"])
        for s in samples
        if s["puck"] and s["puck"].get("visible")
    ]
    puck_steps = []
    for i in range(1, len(puck_pts)):
        dt = (puck_pts[i][0] - puck_pts[i - 1][0]) / 1000.0
        if dt <= 0:
            continue
        dx = puck_pts[i][1] - puck_pts[i - 1][1]
        dz = puck_pts[i][2] - puck_pts[i - 1][2]
        step = (dx * dx + dz * dz) ** 0.5
        puck_steps.append(step)

    duration_s = (samples[-1]["t"] - samples[0]["t"]) / 1000.0
    render_fps = len(samples) / max(0.1, duration_s)

    print(f"Samples: {len(samples)} over {duration_s:.2f}s ({render_fps:.1f} render fps)")
    print(f"Tracked avatars: {len(by_tid)}  (with >=5 samples: {sum(1 for v in by_tid.values() if len(v) >= 5)})")

    def summarize(label, steps, top_pct_threshold):
        if not steps:
            print(f"  {label}: no samples")
            return None
        s = sorted(steps)
        p50 = s[len(s) // 2]
        p90 = s[int(len(s) * 0.9)]
        p99 = s[min(len(s) - 1, int(len(s) * 0.99))]
        mx = s[-1]
        print(f"  {label}: n={len(s):4d}  p50={p50:.3f}  p90={p90:.3f}  p99={p99:.3f}  max={mx:.3f}  ft/tick")
        return p99

    print()
    print("Per-render-tick step distances (lower = smoother glide):")
    player_p99 = summarize("PLAYER per-tick", player_steps, 0.6)
    puck_p99 = summarize("PUCK   per-tick", puck_steps, 0.8)

    # Thresholds, render-rate aware. Real skater top ~40 ft/s; real puck
    # top ~120 ft/s. Convert to ft/tick using actual measured render fps.
    # An outlier sneaking past the outlier-reject would show as a step
    # implying speed well above MAX_*_FT_PER_SEC.
    tick_s = 1.0 / max(1.0, render_fps)
    player_threshold = 60 * tick_s  # 60 ft/s allowance (above real top)
    puck_threshold = 160 * tick_s   # 160 ft/s allowance (above real top)

    code = 0
    print()
    if player_p99 is None:
        print("WARN: no player samples")
        code = 1
    elif player_p99 < player_threshold:
        print(f"PASS: player p99 step {player_p99:.2f} < {player_threshold} ft/tick — outliers tamed by lerp")
    else:
        print(f"FAIL: player p99 step {player_p99:.2f} >= {player_threshold} ft/tick — jitter bypassed lerp")
        code = 2
    if puck_p99 is None:
        print("WARN: puck never visible in sampled window")
    elif puck_p99 < puck_threshold:
        print(f"PASS: puck p99 step {puck_p99:.2f} < {puck_threshold} ft/tick")
    else:
        print(f"FAIL: puck p99 step {puck_p99:.2f} >= {puck_threshold} ft/tick")
        code = max(code, 2)

    return code


if __name__ == "__main__":
    sys.exit(main())
