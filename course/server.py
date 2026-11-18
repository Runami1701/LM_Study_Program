"""Local web server for the course: ``http.server`` plus one tiny JSON endpoint.

GET  /                 -> redirect to the built site
GET  /progress.json    -> the learner's progress (served fresh, never cached)
POST /api/event        -> record a site event (hint revealed, solution forced,
                          mark as read) into progress.json
Everything else is served as static files from the repository root.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from . import progress
from .registry import ROOT, load_lessons

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
        if path == "/api/tutor/status":
            from . import tutor
            self._send_json(tutor.status())
            return
        if path == "/api/tutor/history":
            from urllib.parse import parse_qs, urlparse
            from . import tutor
            lesson = (parse_qs(urlparse(self.path).query).get("lesson") or [""])[0]
            try:
                c = tutor.load_chat(lesson)
                self._send_json({"messages": c["messages"], "checks": c["checks"], "has_summary": bool(c["summary"])})
            except KeyError:
                self._send_json({"error": "unknown lesson"}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/tutor/models":
            from . import tutor
            try:
                self._send_json({"models": tutor.free_openrouter_models()})
            except Exception as exc:  # offline, or the service changed
                self._send_json({"models": [], "error": str(exc)})
            return
        if path == "/progress.json":
            with _lock:
                data = progress.load()
            self._send_json(data)
            return
        super().do_GET()

    def _run_check(self):
        """POST /api/check {"lesson": id}: run `python -m course check <id>` and return its output."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            lesson = str(json.loads(self.rfile.read(length) or b"{}").get("lesson", ""))
        except (ValueError, json.JSONDecodeError):
            lesson = ""
        if lesson not in {l.id for l in load_lessons()}:
            self._send_json({"error": f"unknown lesson {lesson!r}"}, HTTPStatus.BAD_REQUEST)
            return
        env = dict(os.environ, NO_COLOR="1", PYTHONIOENCODING="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "course", "check", lesson],
                cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=900, stdin=subprocess.DEVNULL,
            )
            output, ok = (proc.stdout + proc.stderr).strip(), proc.returncode == 0
        except subprocess.TimeoutExpired:
            output, ok = "The check took longer than 15 minutes and was stopped.", False
        self._send_json({"ok": ok, "output": output})

    def _tutor(self, action: str):
        from . import tutor
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 200_000:
                raise tutor.TutorError("message too long")
            body = json.loads(self.rfile.read(length) or b"{}")
            if action == "config":
                tutor.save_config(str(body.get("provider", "")), str(body.get("model", "")).strip(),
                                  str(body.get("api_key", "")).strip(), str(body.get("base_url", "")).strip())
                self._send_json(tutor.status())
            elif action == "ask":
                self._send_json({"text": tutor.ask(str(body.get("lesson", "")), str(body.get("question", "")))})
            elif action == "clear":
                tutor.clear_chat(str(body.get("lesson", "")))
                self._send_json({"ok": True})
            elif action == "handoff":
                text, path = tutor.handoff()
                try:
                    shown = path.relative_to(ROOT).as_posix()
                except ValueError:
                    shown = str(path)
                self._send_json({"markdown": text, "path": shown})
            elif action == "check":
                self._send_json({"text": tutor.check_think(str(body.get("lesson", "")), int(body.get("index", -1)), str(body.get("answer", "")))})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (tutor.TutorError, KeyError, ValueError, json.JSONDecodeError) as exc:
            msg = exc.args[0] if exc.args else str(exc)
            self._send_json({"error": str(msg)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self):
        route = self.path.split("?", 1)[0]
        if route.startswith("/api/tutor/"):
            self._tutor(route.rsplit("/", 1)[1])
            return
        if route == "/api/check":
            self._run_check()
            return
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
