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
    """Static handler with two things SimpleHTTPRequestHandler doesn't do:

    1. Disable browser caching of viewer/* — we iterate viewer.js often
       and stale cached copies were causing "no errors but nothing
       renders" confusion.
    2. Serve HTTP Range requests on video files so the <video> element
       in the source-video panel can stream. Without 206 responses,
       Chrome's video pipeline stalls in readyState=0 forever even
       though the server happily returns the full file on a plain GET.
       (Spent 30 mins debugging this — the bug looks like a viewer.js
       issue but is entirely server-side.)
    """

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        """Override to handle HTTP Range. Returns the open file handle
        positioned at the start of the requested byte range, OR None on
        error / 304 / non-range path (caller treats None as 'response
        already sent, don't write body').
        """
        range_header = self.headers.get("Range")
        if not range_header:
            return super().send_head()

        # Resolve the path the same way the base handler does. translate_path
        # strips query strings and resolves the URL to a filesystem path.
        path = self.translate_path(self.path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        try:
            fs = os.fstat(f.fileno())
            file_size = fs.st_size
        except OSError:
            f.close()
            self.send_error(500, "Could not stat file")
            return None

        # Parse "bytes=START-END" (END is optional)
        try:
            units, _, range_spec = range_header.partition("=")
            if units.strip().lower() != "bytes":
                raise ValueError("only 'bytes' range unit supported")
            start_s, _, end_s = range_spec.partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else file_size - 1
            if start < 0 or end >= file_size or start > end:
                raise ValueError("range out of bounds")
        except (ValueError, AttributeError):
            self.send_response(416)  # Range Not Satisfiable
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            f.close()
            return None

        length = end - start + 1
        self.send_response(206)  # Partial Content
        ctype = self.guess_type(path)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()

        # Seek + return; copyfile() in the base handler will stream the rest.
        # But we only want LENGTH bytes — implement our own bounded copy.
        f.seek(start)
        try:
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            f.close()
        # Returning None tells SimpleHTTPRequestHandler we already wrote
        # the body — it won't try to copy more.
        return None


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
