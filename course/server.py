"""Local web server for the course: ``http.server`` plus one tiny JSON endpoint.

GET  /                 -> redirect to the built site
GET  /progress.json    -> the learner's progress (served fresh, never cached)
POST /api/event        -> record a site event (hint revealed, solution forced,
                          mark as read) into progress.json
Everything else is served as static files from the repository root.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from . import progress
from .registry import ROOT

SITE_PATH = "/site/build/index.html"
_lock = threading.Lock()


class CourseHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # keep the terminal quiet
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", SITE_PATH)
            self.end_headers()
            return
        if path == "/progress.json":
            with _lock:
                data = progress.load()
            self._send_json(data)
            return
        super().do_GET()

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/event":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 10_000:
                raise ValueError("event too large")
            event = json.loads(self.rfile.read(length) or b"{}")
            with _lock:
                data = progress.record_site_event(event)
            self._send_json(data)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def serve(port: int = 8000, open_browser: bool = True) -> None:
    handler = partial(CourseHandler, directory=str(ROOT))
    # Bind to localhost only: this server can write progress.json.
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://localhost:{port}{SITE_PATH}"
    print(f"Serving the course at {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
