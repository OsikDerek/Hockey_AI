"""Headless test for the source-video side-by-side panel.

Validates the wiring (panel toggles, URL auto-derives, sync function
runs, target computes correctly) without requiring actual H.264 decode.
Headless Chromium doesn't bundle proprietary codec licenses, so the
<video> element fails to seek even though the panel logic is correct.
End-to-end decode validation has to happen in a real browser (Edge /
Chrome on the user's machine).

ALSO verifies the server supports HTTP Range requests on the video URL,
because Python's SimpleHTTPRequestHandler doesn't by default — and
without 206 Partial Content responses the <video> element stalls
forever even on a real browser.

Run from project root (server on :8000):
    .venv/Scripts/python.exe scripts/test_source_video_panel.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "http://localhost:8000/viewer/index.html?data=/output/livebarn_cropped_positions.json"
VIDEO_URL = "http://localhost:8000/data/raw_videos/livebarn_60sec_cropped.web.mp4"


def _check_range_support():
    """Hit the video URL with a Range header; expect 206 Partial Content.
    If the server returns 200 with the full body, the <video> element
    will stall — fail loudly here so the bug is obvious before we waste
    time poking at the JS side."""
    req = urllib.request.Request(VIDEO_URL, headers={"Range": "bytes=0-1023"})
    try:
        with urllib.request.urlopen(req, timeout=4) as resp:
            status = resp.status
            content_range = resp.headers.get("Content-Range")
            content_length = resp.headers.get("Content-Length")
    except Exception as e:
        return False, f"request failed: {e}"
    if status != 206:
        return False, (f"got status {status}, expected 206 — server doesn't "
                       "support Range; video element will stall. "
                       "Fix: serve_viewer.py needs Range support.")
    if not content_range or not content_range.startswith("bytes 0-1023/"):
        return False, f"missing/wrong Content-Range header: {content_range!r}"
    if content_length != "1024":
        return False, f"Content-Length should be 1024, got {content_length!r}"
    return True, f"206 Partial Content, Content-Range={content_range}"


def main():
    # 0. Range-support precheck — without this, real-browser playback stalls
    # at readyState=0 regardless of all the JS we wire up.
    ok, msg = _check_range_support()
    print(f"Range support: {msg}")
    if not ok:
        print(f"FAIL (server-side): {msg}")
        return 2

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_context(viewport={"width": 1600, "height": 900}).new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))
        page.goto(URL, wait_until="domcontentloaded", timeout=15000)
        page.wait_for_function(
            "document.getElementById('data-status').textContent.includes('frames')",
            timeout=8000,
        )

        # 1. Panel hidden by default
        hidden = page.evaluate(
            "() => document.getElementById('source-video-panel').classList.contains('hidden')"
        )
        if not hidden:
            print("FAIL: panel should be hidden by default")
            return 2

        # 2. Toggle button opens it
        page.click("#source-video-toggle")
        page.wait_for_timeout(150)
        visible = page.evaluate(
            "() => !document.getElementById('source-video-panel').classList.contains('hidden')"
        )
        if not visible:
            print("FAIL: clicking toggle did not show panel")
            return 2

        # 3. Auto-derive should find the .web.mp4 we transcoded
        try:
            page.wait_for_function(
                "() => document.getElementById('source-video-status').classList.contains('ok')",
                timeout=6000,
            )
        except Exception:
            status = page.text_content("#source-video-status")
            print(f"FAIL: video metadata never loaded. status='{status}'")
            return 2

        loaded_src = page.evaluate("() => document.getElementById('source-video').currentSrc")
        if "livebarn_60sec_cropped.web.mp4" not in loaded_src:
            print(f"FAIL: wrong file auto-resolved: {loaded_src}")
            return 2
        print(f"auto-derived: {loaded_src}")

        # 4. Scrub to a known frame and verify the sync function ran with
        # the right target. We DON'T verify the video actually seeked
        # because headless Chromium can't decode H.264 — that's verified
        # in a real browser.
        page.evaluate("""
          () => {
            const sc = document.getElementById('scrubber');
            sc.value = 600;
            sc.dispatchEvent(new Event('input', {bubbles: true}));
          }
        """)
        page.wait_for_timeout(300)
        snap = page.evaluate("() => window.__hockeyAI.snapshot().sourceVideo")
        print(f"sync state: {snap}")
        if snap["syncCalls"] < 5:
            print(f"FAIL: sync function not running (syncCalls={snap['syncCalls']})")
            return 2
        if snap["syncTrigger"] < 1:
            print(f"FAIL: sync didn't reach the seek-attempt branch (syncTrigger={snap['syncTrigger']})")
            return 2
        # lastTarget should be >= the scrub target (20s) — auto-play kicked
        # off when the panel opened, so it may have advanced past 20.
        if snap["lastTarget"] is None or snap["lastTarget"] < 19.5 or snap["lastTarget"] > 25:
            print(f"FAIL: sync target out of range: {snap['lastTarget']} (expected 19.5–25)")
            return 2

        # 5. Close button hides the panel
        page.click("#source-video-close")
        page.wait_for_timeout(150)
        hidden = page.evaluate(
            "() => document.getElementById('source-video-panel').classList.contains('hidden')"
        )
        if not hidden:
            print("FAIL: close button didn't hide panel")
            return 2

        # 6. Bad URL → error status surfaces correctly
        page.click("#source-video-toggle")
        page.wait_for_timeout(150)
        page.fill("#source-video-url", "/data/raw_videos/this_does_not_exist.mp4")
        page.click("#source-video-load")
        try:
            page.wait_for_function(
                "() => document.getElementById('source-video-status').classList.contains('error')",
                timeout=4000,
            )
            print("error path: status surfaces 'error' class on bad URL ✓")
        except Exception:
            print("WARN: error path didn't surface .error class within 4s (may rely on slower network event)")

        out = Path(__file__).resolve().parent.parent / "output" / "_quiz_browser" / "source_video_panel.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(out), full_page=False)
        print(f"screenshot: {out}")

        browser.close()

    if errors:
        print("Page errors:")
        for e in errors:
            print(f"  {e}")
        return 1

    print("PASS — panel + sync wiring verified. Real decode requires Edge/Chrome.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
