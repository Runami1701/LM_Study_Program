"""Local web server for the course: ``http.server`` plus one tiny JSON endpoint.

GET  /                 -> redirect to the built site
GET  /progress.json    -> the learner's progress (served fresh, never cached)
POST /api/event        -> record a site event (hint revealed, solution forced,
                          mark as read) into progress.json
Everything else is served as static files from the repository root.

By default the server listens on this computer only (127.0.0.1). With
``--host 0.0.0.0`` other devices on the home network can open it too; then
every request needs the passcode saved in access.txt (see ``access_code``),
because the site can run Python on this machine.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from . import progress
from .registry import ROOT, load_lessons

SITE_PATH = "/site/build/index.html"
ACCESS_FILE = ROOT / "access.txt"
COOKIE = "course_access"
# Never served as files: the tutor's API key and the network passcode.
PRIVATE_FILES = {(ROOT / "tutor.json").resolve(), ACCESS_FILE.resolve()}
LOCAL_NAMES = ("localhost", "127.0.0.1")
_lock = threading.Lock()

LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Course passcode</title>
<style>body{{font-family:system-ui,sans-serif;max-width:22rem;margin:4rem auto;padding:0 1rem}}
input,button{{font-size:1.1rem;padding:.5rem;width:100%;box-sizing:border-box;margin-top:.5rem}}
.err{{color:#b00020}}</style></head><body>
<h1>Course passcode</h1>
<p>Type the passcode shown on the computer running the course (also in its <code>access.txt</code>).</p>
{error}<form method="post" action="/login">
<input name="code" autocomplete="current-password" autofocus required>
<button type="submit">Open the course</button></form></body></html>"""


def access_code() -> str:
    """The network passcode: made once, then read from access.txt."""
    if ACCESS_FILE.exists():
        code = ACCESS_FILE.read_text(encoding="utf-8").strip()
        if code:
            return code
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"          # no 0/o, 1/l/i look-alikes
    code = "-".join("".join(secrets.choice(alphabet) for _ in range(4)) for _ in range(3))
    ACCESS_FILE.write_text(code + "\n", encoding="utf-8")
    try:
        os.chmod(ACCESS_FILE, 0o600)
    except OSError:
        pass
    return code


def _lan_addresses() -> list[str]:
    names = [f"{socket.gethostname()}.local"]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))          # no packet is sent; this picks the LAN interface
            names.append(s.getsockname()[0])
    except OSError:
        pass
    return names


class CourseHandler(SimpleHTTPRequestHandler):
    access: str | None = None    # set when serving on the network; None = this computer only

    def log_message(self, format, *args):  # keep the terminal quiet
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # --- network passcode -------------------------------------------------------------
    def _signed_in(self) -> bool:
        if self.access is None:
            return True
        for part in (self.headers.get("Cookie") or "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == COOKIE and hmac.compare_digest(value.encode(), self.access.encode()):
                return True
        return False

    def _send_login(self, error: str = "", status=HTTPStatus.OK):
        body = LOGIN_PAGE.format(error=f'<p class="err">{error}</p>' if error else "").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _login(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1000:
            self._send_login("Request too large.", HTTPStatus.BAD_REQUEST)
            return
        form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
        code = (form.get("code") or [""])[0].strip().lower()
        if not hmac.compare_digest(code.encode(), (self.access or "").encode()):
            time.sleep(1)                         # slow down guessing
            self._send_login("That passcode is wrong. Try again.", HTTPStatus.FORBIDDEN)
            return
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Set-Cookie", f"{COOKIE}={self.access}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict")
        self.send_header("Location", SITE_PATH)
        self.end_headers()

    def _refuse_signed_out(self, path: str) -> bool:
        """When serving on the network: send signed-out visitors to the passcode page."""
        if self._signed_in():
            return False
        if path.startswith("/api/") or path == "/progress.json":
            self._send_json({"error": "Sign in first: reload the page and type the passcode."}, HTTPStatus.UNAUTHORIZED)
        elif path == "/login":
            self._send_login()
        else:
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/login")
            self.end_headers()
        return True

    def _send_json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if self._refuse_signed_out(path):
            return
        if path == "/login":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", SITE_PATH)
            self.end_headers()
            return
        if os.path.normcase(self.translate_path(path)) in {os.path.normcase(str(p)) for p in PRIVATE_FILES}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if path in ("/", "/index.html"):
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", SITE_PATH)
            self.end_headers()
            return
        if path == "/api/file":
            self._get_file()
            return
        if path == "/api/job":
            from urllib.parse import parse_qs, urlparse
            from . import jobs
            kind = (parse_qs(urlparse(self.path).query).get("kind") or [""])[0]
            if kind not in jobs.JOBS:
                self._send_json({"error": "unknown task"}, HTTPStatus.BAD_REQUEST)
            else:
                self._send_json(jobs.status(kind))
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
        if path == "/api/tutor/ollama":
            from . import tutor
            self._send_json(tutor.ollama_status())
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

    def _body(self, limit: int = 2_000_000) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > limit:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(length) or b"{}")

    @staticmethod
    def _workspace_path(rel: str):
        """Only existing .py files inside workspace/ may be read or written."""
        path = (ROOT / rel).resolve()
        ws = (ROOT / "workspace").resolve()
        if ws not in path.parents or path.suffix != ".py" or not path.is_file():
            raise ValueError(f"not an exercise file: {rel}")
        return path

    def _get_file(self):
        from urllib.parse import parse_qs, urlparse
        rel = (parse_qs(urlparse(self.path).query).get("path") or [""])[0]
        try:
            path = self._workspace_path(rel)
            self._send_json({"path": rel, "text": path.read_text(encoding="utf-8")})
        except (ValueError, OSError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _save_file(self):
        try:
            body = self._body()
            path = self._workspace_path(str(body.get("path", "")))
            text = str(body.get("text", ""))
            tmp = path.with_suffix(".py.tmp")
            tmp.write_text(text, encoding="utf-8", newline="\n")
            os.replace(tmp, path)
            self._send_json({"ok": True})
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _reset(self):
        from .reset import apply_reset
        from .registry import get_lesson
        try:
            lesson = get_lesson(str(self._body().get("lesson", "")))
            apply_reset(lesson)
            self._send_json({"ok": True})
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _job(self, action: str):
        from . import jobs
        try:
            body = self._body()
            if action == "start":
                self._send_json(jobs.start(str(body.get("kind", ""))))
            elif action == "stop":
                self._send_json(jobs.stop())
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _from_course_page(self) -> bool:
        """Accept actions only from the course page itself. Other websites can make a browser
        send requests to localhost; they fail these checks (wrong Host or Origin, or no JSON).
        On the network, the passcode cookie (SameSite=Strict) is checked before this, and the
        Origin must match the address the page was opened at."""
        host = self.headers.get("Host") or ""
        origin = self.headers.get("Origin")
        if self.access is None:
            if host.split(":")[0] not in LOCAL_NAMES:
                return False
            if origin and origin.split("://", 1)[-1].split(":")[0] not in LOCAL_NAMES:
                return False
        elif origin and origin.split("://", 1)[-1].rstrip("/") != host:
            return False
        return (self.headers.get("Content-Type") or "").startswith("application/json")

    def _play(self):
        """Run a Python snippet from the playground on this computer, with a time limit."""
        try:
            code = str(self._body(200_000).get("code", ""))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        play = ROOT / "runs" / "playground.py"
        play.parent.mkdir(parents=True, exist_ok=True)
        play.write_text(code, encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(ROOT))
        try:
            proc = subprocess.run([sys.executable, str(play)], cwd=ROOT, env=env, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=60, stdin=subprocess.DEVNULL)
            out, ok = (proc.stdout + proc.stderr), proc.returncode == 0
            out = out.replace(str(play), "playground")
        except subprocess.TimeoutExpired:
            out, ok = "Stopped: the code ran longer than 60 seconds.", False
        self._send_json({"ok": ok, "output": out[-20000:] or "(no output; use print(...) to show values)"})

    def _run_check(self):
        """POST /api/check {"lesson": id}: run `python -m course check <id>` and return its output."""
        try:
            body = self._body()
            lesson, answer = str(body.get("lesson", "")), str(body.get("answer", ""))
        except (ValueError, json.JSONDecodeError):
            lesson, answer = "", ""
        if lesson not in {l.id for l in load_lessons()}:
            self._send_json({"error": f"unknown lesson {lesson!r}"}, HTTPStatus.BAD_REQUEST)
            return
        env = dict(os.environ, NO_COLOR="1", PYTHONIOENCODING="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "course", "check", lesson],
                cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=900, input=(answer + "\n") if answer else "",
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
            if action == "setup":
                picked = tutor.setup(str(body.get("provider", "")), str(body.get("api_key", "")), str(body.get("model", "")).strip())
                self._send_json({**tutor.status(), "picked": picked["model"]})
            elif action == "config":
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
        if self.access is not None and route == "/login":
            self._login()
            return
        if self._refuse_signed_out(route):
            return
        if not self._from_course_page():
            self._send_json({"error": "refused: this request did not come from the course page"}, HTTPStatus.FORBIDDEN)
            return
        if route == "/api/play":
            self._play()
            return
        if route.startswith("/api/tutor/"):
            self._tutor(route.rsplit("/", 1)[1])
            return
        if route == "/api/check":
            self._run_check()
            return
        if route == "/api/file":
            self._save_file()
            return
        if route == "/api/reset":
            self._reset()
            return
        if route.startswith("/api/job/"):
            self._job(route.rsplit("/", 1)[1])
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


def serve(port: int = 8000, open_browser: bool = True, host: str = "127.0.0.1") -> None:
    from .reset import sync_workspace

    for change in sync_workspace():          # new lessons after an update: add their stubs
        print(f"  {change}")
    on_network = host not in LOCAL_NAMES
    handler_cls = type("Handler", (CourseHandler,), {"access": access_code() if on_network else None})
    handler = partial(handler_cls, directory=str(ROOT))
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://localhost:{port}{SITE_PATH}"
    print(f"Serving the course at {url}")
    if on_network:
        for name in _lan_addresses():
            print(f"  from other devices at home: http://{name}:{port}/")
        print(f"  passcode: {handler_cls.access}   (saved in {ACCESS_FILE.name})")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
