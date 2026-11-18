"""Read and write ``progress.json`` (the learner's progress, gitignored).

Shape::

    {
      "version": 1,
      "machine": {"gflops": 123.4, "measured_at": "..."},
      "lessons": {
        "s1-l01": {
          "status": "passed",          # not_started | in_progress | passed
          "attempts": 3,               # number of `course check` runs
          "passed_at": "2026-...",
          "hints_used": 2,             # highest hint level revealed on the site
          "solution_forced": false,    # "show anyway" was clicked before a pass
          "read": true                 # "mark as read" on the site
        }
      },
      "boss": {"s1-boss": {"passed": true, "value": 2.05}}
    }
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .registry import ROOT

PROGRESS_PATH = ROOT / "progress.json"

STATUSES = ("not_started", "in_progress", "passed")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def empty() -> dict:
    return {"version": 1, "machine": {}, "lessons": {}, "boss": {}}


def load(path: Path = PROGRESS_PATH) -> dict:
    if not path.exists():
        return empty()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupt file should never crash the course; keep a backup and start over.
        backup = path.with_suffix(".json.bak")
        try:
            path.replace(backup)
        except OSError:
            pass
        return empty()
    base = empty()
    base.update(data)
    return base


def save(data: dict, path: Path = PROGRESS_PATH) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def lesson_entry(data: dict, lesson_id: str) -> dict:
    entry = data["lessons"].setdefault(lesson_id, {})
    entry.setdefault("status", "not_started")
    entry.setdefault("attempts", 0)
    entry.setdefault("hints_used", 0)
    entry.setdefault("solution_forced", False)
    entry.setdefault("read", False)
    return entry


def status_of(data: dict, lesson_id: str) -> str:
    return data.get("lessons", {}).get(lesson_id, {}).get("status", "not_started")


def record_check(lesson_id: str, passed: bool) -> dict:
    data = load()
    entry = lesson_entry(data, lesson_id)
    entry["attempts"] += 1
    entry["last_check_at"] = now_iso()
    if passed:
        if entry["status"] != "passed":
            entry["passed_at"] = now_iso()
        entry["status"] = "passed"
    elif entry["status"] == "not_started":
        entry["status"] = "in_progress"
    save(data)
    return data


def record_site_event(event: dict) -> dict:
    """Apply an event posted by the web course (hint revealed, solution forced,
    mark as read, think answered). Only whitelisted fields are accepted."""
    lesson_id = str(event.get("lesson", ""))
    if not lesson_id:
        raise ValueError("event needs a 'lesson'")
    data = load()
    entry = lesson_entry(data, lesson_id)
    if "hints_used" in event:
        entry["hints_used"] = max(entry["hints_used"], int(event["hints_used"]))
    if event.get("solution_forced") and entry["status"] != "passed":
        entry["solution_forced"] = True
    if event.get("read"):
        entry["read"] = True
    touched = any(k in event for k in ("hints_used", "solution_forced", "read", "think"))
    if touched and entry["status"] == "not_started":
        entry["status"] = "in_progress"
    save(data)
    return data


def record_gflops(gflops: float) -> None:
    data = load()
    data["machine"]["gflops"] = round(float(gflops), 2)
    data["machine"]["measured_at"] = now_iso()
    save(data)
