"""Long-running tasks started from the website: downloads, training runs, text samples.

Only the commands listed in JOBS can run, one at a time. Output goes to
runs/jobs/<kind>.log and the website polls it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .registry import ROOT

# kind -> (button label, what it does, python arguments, extra environment)
JOBS: dict[str, tuple[str, str, list[str], dict]] = {
    "data-shakespeare": ("Download Tiny Shakespeare", "About 1 MB, once.", ["-m", "course", "data", "shakespeare"], {}),
    "data-tinystories": ("Download TinyStories", "About 120 MB, once. Needed for Stage 3.", ["-m", "course", "data", "tinystories"], {}),
    "train-s1": ("Train the Stage 1 model", "Uses your code in workspace/stage1. About 1–5 minutes.", ["-m", "workspace.stage1.train_mlp"], {}),
    "sample-s1": ("Write text with the Stage 1 model", "500 characters at temperature 0.8, from runs/s1.", ["-m", "workspace.stage1.sample", "0.8"], {}),
    "train-s2": ("Train the transformer (Stage 2)", "Uses your code in workspace/stage2. One to two hours.", ["-m", "workspace.stage2.train_tf"], {}),
    "resume-s2": ("Continue Stage 2 from its checkpoint", "After a stop or crash: starts from runs/s2/ckpt.npz.", ["-m", "workspace.stage2.train_tf"], {"COURSE_RESUME": "1"}),
    "train-s3": ("Train the BPE model (Stage 3)", "Learns the tokenizer, then trains. Several hours: run it overnight.", ["-m", "workspace.stage3.train_bpe_lm"], {}),
    "resume-s3": ("Continue Stage 3 from its checkpoint", "After a stop or crash: starts from runs/s3/ckpt.npz.", ["-m", "workspace.stage3.train_bpe_lm"], {"COURSE_RESUME": "1"}),
}

LOG_DIR = ROOT / "runs" / "jobs"
_lock = threading.Lock()
_current: dict = {}          # kind, proc, started, log


def describe(kind: str) -> dict:
    label, what, _, _ = JOBS[kind]
    return {"kind": kind, "label": label, "description": what}


def start(kind: str) -> dict:
    if kind not in JOBS:
        raise ValueError(f"unknown task {kind!r}")
    with _lock:
        if _current and _current["proc"].poll() is None:
            raise RuntimeError(f"“{JOBS[_current['kind']][0]}” is still running. Stop it first, or wait until it ends.")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = LOG_DIR / f"{kind}.log"
        env = dict(os.environ, PYTHONUNBUFFERED="1", NO_COLOR="1", PYTHONIOENCODING="utf-8", **JOBS[kind][3])
        f = open(log, "w", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, *JOBS[kind][2]], cwd=ROOT, env=env, stdout=f,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        _current.clear()
        _current.update(kind=kind, proc=proc, started=time.time(), log=log, file=f)
    return status(kind)


def stop() -> dict:
    with _lock:
        if _current and _current["proc"].poll() is None:
            _current["stopped"] = True
            _current["proc"].terminate()
            try:
                _current["proc"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                _current["proc"].kill()
    return status(_current.get("kind", ""))


def _tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 20000))
        text = f.read().decode("utf-8", "replace")
    text = text.replace("\r\n", "\n").split("\n")
    text = [t.split("\r")[-1] for t in text]          # progress lines overwrite themselves with \r
    return "\n".join(text[-lines:]).rstrip()


def status(kind: str) -> dict:
    """State of a task: running / finished (with exit code) / not started; plus its last output."""
    running = bool(_current) and _current["kind"] == kind and _current["proc"].poll() is None
    out = {"kind": kind, "running": running, "busy_with": None, "log": _tail(LOG_DIR / f"{kind}.log") if kind else ""}
    if _current and _current["proc"].poll() is None and _current["kind"] != kind:
        out["busy_with"] = JOBS[_current["kind"]][0]
    if _current and _current["kind"] == kind:
        out["seconds"] = round(time.time() - _current["started"])
        code = _current["proc"].poll()
        if code is not None:
            out["exit_code"] = code
            out["stopped"] = bool(_current.get("stopped"))
            if not _current["file"].closed:
                _current["file"].close()
    return out
