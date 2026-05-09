"""Tiny static HTTP server for the 3D viewer.

The viewer needs to fetch *_positions.json files; browsers block fetch()
on file:// URLs, so we have to serve everything over http. This script
binds 127.0.0.1:8000 to the project root, prints a deep-link URL with
the most recently produced positions JSON, and opens it in the default
browser.
"""

import http.server
import mimetypes
import os
import socketserver
import sys
import webbrowser
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORT = 8000

# Windows registry maps .js → text/plain by default; ES modules will refuse
# to execute under that MIME type ("strict MIME type checking" per HTML
# spec) and the script silently dies with no error event. Force the right
# mappings before SimpleHTTPRequestHandler reads them.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/css", ".css")


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    """Disable caching during dev — we iterate viewer.js often and stale
    cached copies were causing 'no errors but nothing renders' confusion."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def find_latest_positions_json() -> Path | None:
    out = PROJECT_ROOT / "output"
    if not out.is_dir():
        return None
    candidates = sorted(out.glob("*_positions.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> int:
    os.chdir(PROJECT_ROOT)
    handler = NoCacheHandler

    latest = find_latest_positions_json()
    if latest is not None:
        rel = latest.relative_to(PROJECT_ROOT).as_posix()
        url = f"http://localhost:{PORT}/viewer/index.html?data=/{quote(rel)}"
        print(f"Auto-opening latest positions: {rel}")
    else:
        url = f"http://localhost:{PORT}/viewer/index.html"
        print("No *_positions.json found in output/. Open the picker in the page.")

    print(f"Serving {PROJECT_ROOT} on {url}")
    print("Ctrl+C to stop.")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    # ThreadingHTTPServer instead of single-threaded TCPServer — the
    # browser keeps a long-lived connection open via keep-alive, which
    # was blocking new requests (curl / Playwright would see connection
    # refused while the actual user tab held the only thread).
    server_cls = http.server.ThreadingHTTPServer
    server_cls.allow_reuse_address = True
    with server_cls(("127.0.0.1", PORT), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
