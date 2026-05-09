"""Headless-browser test for the Phase D Quiz UI.

Drives a real Chromium via Playwright, loads the viewer with a known
positions JSON, exercises Quiz Mode end-to-end, and screenshots each
phase. Lets me self-verify quiz behavior without needing a human in the
loop.

Run from project root (the static server should ALSO be running on :8000):
    .venv/Scripts/python.exe scripts/test_quiz_browser.py [--clip rush_b1]

Outputs:
    output/_quiz_browser/
      00_loaded.png            — initial state after JSON loads
      01_quiz_active.png       — after clicking Quiz Mode toggle
      02_paused_overlay.png    — first time the choice overlay appears
      03_after_commit.png      — after committing a choice (reveal panel)
      04_score_widget.png      — final score after a few questions
      console.log              — full browser console output
      result.json              — pass/fail summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "output" / "_quiz_browser"
SERVER_BASE = "http://localhost:8000"


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--clip", default="rush_b1",
                   help="positions JSON basename (without _positions.json)")
    p.add_argument("--max-questions", type=int, default=3,
                   help="how many quiz questions to commit before stopping")
    p.add_argument("--headed", action="store_true",
                   help="run with a visible browser window")
    args = p.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{SERVER_BASE}/viewer/index.html?data=/output/{args.clip}_positions.json"
    console_log_path = OUT_DIR / "console.log"
    result_path = OUT_DIR / "result.json"

    log_lines = []
    summary = {"clip": args.clip, "url": url, "phases": {}, "errors": []}

    def screenshot(page, name):
        path = OUT_DIR / f"{name}.png"
        page.screenshot(path=str(path), full_page=False)
        print(f"  screenshot {path.relative_to(PROJECT_ROOT)}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        context = browser.new_context(viewport={"width": 1400, "height": 800})
        page = context.new_page()

        page.on("console", lambda m: log_lines.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: log_lines.append(f"[pageerror] {e}"))

        print(f"Loading {url} ...")
        try:
            page.goto(url, wait_until="networkidle", timeout=15000)
        except Exception as e:
            summary["errors"].append(f"goto: {e}")
            console_log_path.write_text("\n".join(log_lines))
            result_path.write_text(json.dumps(summary, indent=2))
            return 1

        # 1. Initial load — wait for data-status to update beyond placeholder
        try:
            page.wait_for_function(
                "document.getElementById('data-status').textContent.includes('frames')",
                timeout=8000,
            )
        except Exception as e:
            summary["errors"].append(f"data-status never updated: {e}")
        status = page.text_content("#data-status")
        summary["phases"]["loaded"] = {"data_status": status}
        screenshot(page, "00_loaded")

        # Snapshot quiz-eligible event count visible to the quiz module
        eligible = page.evaluate("""
          () => {
            // Re-import the QUIZ_OPTIONS map by reading window-scope is awkward;
            // count via the global data object instead.
            const opts = ['shot_vs_pass','odd_man_rush','zone_entry','breakout','missed_opportunity'];
            try {
              // The viewer doesn't expose `data` on window; pull events count from #event-markers DOM
              const markers = document.querySelectorAll('.event-marker');
              return { markers: markers.length };
            } catch(e) { return { error: String(e) }; }
          }
        """)
        summary["phases"]["loaded"]["dom_event_markers"] = eligible

        # 2. Click Quiz Mode toggle
        page.click("#quiz-toggle")
        page.wait_for_timeout(300)
        score_visible = page.evaluate(
            "() => !document.getElementById('quiz-score').classList.contains('hidden')"
        )
        summary["phases"]["quiz_active"] = {"score_visible": score_visible}
        screenshot(page, "01_quiz_active")
        if not score_visible:
            summary["errors"].append("Quiz Mode toggle didn't reveal score widget")

        # 3. Hit Play; wait up to 30s for the first quiz overlay to appear.
        page.click("#play-btn")
        first_overlay = False
        try:
            page.wait_for_selector("#quiz-overlay:not(.hidden)", timeout=30_000)
            first_overlay = True
            screenshot(page, "02_paused_overlay")
        except Exception as e:
            summary["errors"].append(f"Quiz overlay never appeared in 30s: {e}")
            screenshot(page, "02_no_overlay_timeout")
            # Capture what state the playback ended up in for diagnostics
            playback_idx = page.evaluate(
                "() => document.getElementById('frame-counter').textContent"
            )
            summary["phases"]["timeout_state"] = {"frame_counter": playback_idx}

        if first_overlay:
            # Read the question text + available choices
            question = page.text_content("#quiz-question")
            choices = page.eval_on_selector_all(
                ".quiz-choice-btn",
                "els => els.map(e => e.dataset.choice)",
            )
            summary["phases"]["first_overlay"] = {
                "question": question, "choices": choices,
            }

            # 4. Commit choices for up to N questions
            committed = 0
            for q in range(args.max_questions):
                # commit the FIRST option each time (deterministic)
                btns = page.query_selector_all(".quiz-choice-btn")
                if not btns:
                    summary["errors"].append("no choice buttons present at commit time")
                    break
                btn = btns[0]
                choice = btn.get_attribute("data-choice")
                btn.click()
                committed += 1
                # Reveal panel should appear
                try:
                    page.wait_for_selector("#quiz-reveal:not(.hidden)", timeout=5000)
                except Exception as e:
                    summary["errors"].append(f"Reveal didn't appear after commit {q}: {e}")
                if q == 0:
                    screenshot(page, "03_after_commit")
                # Wait for reveal to clear + next overlay (or timeout)
                try:
                    page.wait_for_selector("#quiz-reveal.hidden", timeout=8000)
                except Exception:
                    pass
                if q < args.max_questions - 1:
                    try:
                        page.wait_for_selector("#quiz-overlay:not(.hidden)", timeout=30_000)
                    except Exception as e:
                        summary["errors"].append(f"No subsequent overlay after q{q}: {e}")
                        break
            summary["phases"]["committed"] = committed

            # 5. Final score
            score_text = page.text_content("#quiz-score")
            summary["phases"]["final_score"] = score_text
            screenshot(page, "04_score_widget")

        browser.close()

    console_log_path.write_text("\n".join(log_lines))
    result_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary: {result_path.relative_to(PROJECT_ROOT)}")
    print(f"Console log: {console_log_path.relative_to(PROJECT_ROOT)}")
    print(json.dumps(summary, indent=2))
    return 0 if not summary["errors"] else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
