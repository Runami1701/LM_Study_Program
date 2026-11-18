"""Run a lesson's tests and print a friendly, teaching-oriented summary.

pytest runs in-process with its own terminal output captured; we collect each
test's outcome through a small plugin and print one line per learner function.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import pytest

from . import progress
from .registry import ROOT, Lesson, get_lesson, load_lessons, next_lesson


# --------------------------------------------------------------------------- output

def _can_encode(s: str) -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        s.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


OK, BAD, SKIP = ("✓", "✗", "–") if _can_encode("✓✗–") else ("+", "x", "-")
ARROW = "→" if _can_encode("→") else "->"


def _color(code: str, text: str) -> str:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


green = lambda t: _color("32", t)  # noqa: E731
red = lambda t: _color("31", t)  # noqa: E731
dim = lambda t: _color("2", t)  # noqa: E731
bold = lambda t: _color("1", t)  # noqa: E731


# ------------------------------------------------------------------ result collection

@dataclass
class TestResult:
    nodeid: str
    fn: str | None
    outcome: str  # passed | failed | skipped | error
    message: str = ""


class _Collector:
    def __init__(self, lesson: Lesson):
        self.lesson = lesson
        self.results: list[TestResult] = []
        self.collect_errors: list[str] = []

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        rep = outcome.get_result()
        marker = item.get_closest_marker("fn")
        fn = marker.args[0] if marker and marker.args else None
        if rep.when == "call" or (rep.when == "setup" and not rep.passed):
            if rep.passed:
                self.results.append(TestResult(item.nodeid, fn, "passed"))
            elif rep.skipped:
                self.results.append(TestResult(item.nodeid, fn, "skipped"))
            else:
                msg = explain_exception(call.excinfo, fn, self.lesson)
                self.results.append(TestResult(item.nodeid, fn, "failed", msg))

    def pytest_collectreport(self, report):
        if report.failed:
            self.collect_errors.append(str(report.longrepr))


def _learner_frame(excinfo) -> str | None:
    """Return 'workspace/x.py:LINE' for the deepest traceback frame inside the
    learner's code, so errors point at their own line."""
    best = None
    for entry in excinfo.traceback:
        path = Path(str(entry.path))
        try:
            rel = path.resolve().relative_to(ROOT)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] in ("workspace", "solutions"):
            best = f"{rel.as_posix()}:{entry.lineno + 1}"
    return best


def explain_exception(excinfo, fn: str | None, lesson: Lesson) -> str:
    if excinfo is None:
        return "failed (no exception info)"
    etype, value = excinfo.type, excinfo.value

    if issubclass(etype, NotImplementedError):
        name = str(value).strip() or fn or "?"
        name = name.split()[0].strip("`'\"()")
        where = lesson.file_for_function(name) or (lesson.files[0] if lesson.files else "workspace/")
        return f"Not implemented yet: `{name}` in {where}"

    if issubclass(etype, SyntaxError):
        loc = f"{Path(value.filename or '?').name}:{value.lineno}"
        return f"SyntaxError in {loc}: {value.msg}"

    if issubclass(etype, AssertionError):
        text = str(value)
        # pytest appends the rewritten expression after the message; keep only ours.
        for sep in ("\nassert ", "\n+  where", "\nAssertionError"):
            text = text.split(sep)[0]
        return text.strip() or "assertion failed"

    where = _learner_frame(excinfo)
    first = str(value).strip().splitlines()[0] if str(value).strip() else ""
    msg = f"{etype.__name__}: {first}" if first else etype.__name__
    return f"{msg}  (raised at {where})" if where else msg


# ------------------------------------------------------------------------- running

def run_lesson_tests(lesson: Lesson, impl: str = "workspace", verbose: bool = False):
    if lesson.test_path is None or not lesson.test_path.exists():
        raise FileNotFoundError(f"no test file for {lesson.id}: {lesson.tests}")
    collector = _Collector(lesson)
    args = [
        str(lesson.test_path),
        "-q",
        "--no-header",
        "-p", "no:cacheprovider",
        f"--impl={impl}",
        "--rootdir", str(ROOT),
        "-c", str(ROOT / "pyproject.toml"),
    ]
    old_cwd = Path.cwd()
    buf = io.StringIO()
    t0 = time.perf_counter()
    try:
        os.chdir(ROOT)
        if verbose:
            code = pytest.main(args, plugins=[collector])
        else:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                code = pytest.main(args, plugins=[collector])
    finally:
        os.chdir(old_cwd)
    elapsed = time.perf_counter() - t0
    return collector, int(code), elapsed, buf.getvalue()


def print_summary(lesson: Lesson, collector: _Collector, impl: str, elapsed: float) -> bool:
    print(bold(f"{lesson.id} — {lesson.title}") + dim(f"   [{impl}, {elapsed:.1f}s]"))
    if collector.collect_errors:
        print(red(f"  {BAD} the test file could not be loaded:"))
        for err in collector.collect_errors:
            for line in err.strip().splitlines()[-6:]:
                print("     " + line)
        return False
    if not collector.results:
        print(red(f"  {BAD} no tests ran"))
        return False

    order = list(lesson.functions)
    groups: dict[str, list[TestResult]] = {}
    for r in collector.results:
        key = r.fn or "other"
        groups.setdefault(key, []).append(r)
        if key not in order:
            order.append(key)

    width = max(len(k) for k in order if k in groups)
    all_ok = True
    for key in order:
        rs = groups.get(key)
        if not rs:
            continue
        failed = [r for r in rs if r.outcome == "failed"]
        passed = [r for r in rs if r.outcome == "passed"]
        if failed:
            all_ok = False
            first = failed[0].message.splitlines() or [""]
            print(f"  {red(BAD)} {key.ljust(width)}  {first[0]}")
            for extra in first[1:6]:
                print(f"    {' ' * width}  {extra}")
            if len(failed) > 1 and not failed[0].message.startswith("Not implemented yet"):
                print(dim(f"    {' ' * width}  (+{len(failed) - 1} more failing test(s) for {key})"))
        elif passed:
            print(f"  {green(OK)} {key.ljust(width)}  " + dim(f"{len(passed)} test(s) passed"))
        else:
            print(f"  {SKIP} {key.ljust(width)}  " + dim("skipped"))
    return all_ok


# --------------------------------------------------------------- lesson-specific hooks

def _after_pass_s0_l00(impl: str) -> None:
    import numpy as np

    mod = importlib.import_module(f"{impl}.stage0.numpy_basics")
    rng = np.random.default_rng(0)
    try:
        gflops = float(mod.matmul_gflops(1024, 5, rng))
    except Exception as exc:  # pragma: no cover - defensive
        print(red(f"  could not measure GFLOPS: {exc}"))
        return
    print(f"  Your machine: {bold(f'{gflops:.1f} GFLOPS')} (float32 matmul, n = 1024)")
    if impl == "workspace":
        progress.record_gflops(gflops)
        print(dim("  Saved to progress.json; stage 3 uses it to estimate training time."))


AFTER_PASS = {"s0-l00": _after_pass_s0_l00}


def check_boss(lesson: Lesson, impl: str) -> bool:
    """Boss checks read a training run's results.json instead of running tests.

    Front matter:  boss: {results: runs/s1/results.json, metric: val_loss,
                          threshold: 2.1, must_beat: runs/s0/..., ask: "question?"}
    """
    import json

    spec = lesson.boss
    print(bold(f"{lesson.id} — {lesson.title}"))
    path = ROOT / spec["results"]
    if not path.exists():
        print(red(f"  {BAD} {spec['results']} not found — run the training script first:"))
        print(f"      {spec.get('run', '')}")
        return False
    res = json.loads(path.read_text(encoding="utf-8"))
    metric = spec.get("metric", "val_loss")
    value = res.get(metric)
    if value is None:
        print(red(f"  {BAD} {spec['results']} has no '{metric}' entry"))
        return False
    ok = True
    threshold = float(spec["threshold"])
    mark = green(OK) if value <= threshold else red(BAD)
    ok &= value <= threshold
    print(f"  {mark} {metric} = {value:.4f}   (target <= {threshold})")
    if spec.get("must_beat"):
        other = ROOT / spec["must_beat"]
        if other.exists():
            ov = json.loads(other.read_text(encoding="utf-8")).get(spec.get("must_beat_metric", metric))
            if ov is not None:
                beat = value < ov
                ok &= beat
                print(f"  {green(OK) if beat else red(BAD)} beats {spec['must_beat']} ({ov:.4f})")
        else:
            print(dim(f"  ({spec['must_beat']} not found; skipping the comparison)"))
    if "seconds" in res:
        print(dim(f"  training time: {res['seconds'] / 60:.1f} min"))
    if res.get("sample"):
        print(dim("  sample:"))
        for line in str(res["sample"]).splitlines()[:12]:
            print("    " + line)
    if ok and spec.get("ask"):
        answer = input(f"  {spec['ask']} [y/n] ").strip().lower()
        ok &= answer in ("y", "yes")
    if impl == "workspace":
        data = progress.load()
        data["boss"][lesson.id] = {"passed": bool(ok), "value": round(float(value), 4), "at": progress.now_iso()}
        progress.save(data)
        progress.record_check(lesson.id, ok)
    print(green("  Boss defeated!") if ok else dim("  Not yet. Tune the run (see the lesson) and try again."))
    return ok


def check(lesson_id: str, impl: str = "workspace", verbose: bool = False, quiet_next: bool = False) -> bool:
    lesson = get_lesson(lesson_id)
    if lesson.kind == "boss":
        ok = check_boss(lesson, impl)
        nxt = next_lesson(lesson.id)
        if ok and nxt and not quiet_next:
            print(f"  Next: {bold(nxt.id)} — {nxt.title}")
        return ok
    os.environ["COURSE_IMPL"] = impl
    try:
        collector, _code, elapsed, raw = run_lesson_tests(lesson, impl, verbose)
    except FileNotFoundError as exc:
        print(red(str(exc)))
        return False
    except Exception:  # pragma: no cover - should not happen, but never hide it
        traceback.print_exc()
        return False

    ok = print_summary(lesson, collector, impl, elapsed)
    if impl == "workspace":
        progress.record_check(lesson.id, ok)

    if ok:
        print(green(f"  All checks passed for {lesson.id}."))
        hook = AFTER_PASS.get(lesson.id)
        if hook:
            hook(impl)
        nxt = next_lesson(lesson.id)
        if not quiet_next:
            if nxt:
                print(f"  Next: {bold(nxt.id)} — {nxt.title}")
            else:
                print("  That was the last lesson available so far.")
    else:
        print(dim(f"  Fix the first {BAD} above, then run: python -m course check {lesson.id}"))
        if not verbose:
            print(dim("  (add --verbose to see pytest's full output)"))
    return ok


def check_all(impl: str = "workspace", keep_going: bool = False, verbose: bool = False) -> bool:
    all_ok = True
    for lesson in load_lessons():
        if not lesson.tests:  # boss checks need a finished training run; run them by name
            continue
        ok = check(lesson.id, impl=impl, verbose=verbose, quiet_next=True)
        print()
        all_ok &= ok
        if not ok and not keep_going:
            print(f"Stopped at {lesson.id}. Use --keep-going to run every lesson.")
            return False
    return all_ok
