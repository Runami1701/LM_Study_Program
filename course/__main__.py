"""``python -m course <command>`` — the course command line.

Commands: list, check, serve, progress, reset, data, build.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

from . import progress
from .registry import ROOT, STAGE_NAMES, get_lesson, load_lessons


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_course_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _status_icon(status: str, kind: str) -> str:
    from .check import _can_encode

    if not _can_encode("○◐●★"):
        return {"passed": "[x]", "in_progress": "[~]"}.get(status, "[ ]")
    if status == "passed":
        return "★" if kind == "boss" else "●"
    return {"in_progress": "◐"}.get(status, "○")


def cmd_list(args) -> int:
    data = progress.load()
    stage = None
    for lesson in load_lessons():
        if lesson.stage != stage:
            stage = lesson.stage
            print(f"\n{STAGE_NAMES.get(stage, stage)}")
        st = progress.status_of(data, lesson.id)
        mins = f"~{lesson.est_minutes} min" if lesson.est_minutes else ""
        print(f"  {_status_icon(st, lesson.kind)} {lesson.id:<8} {lesson.title:<44} {mins}")
    print()
    return 0


def cmd_check(args) -> int:
    from .check import check, check_all

    if args.all:
        ok = check_all(impl=args.impl, keep_going=args.keep_going, verbose=args.verbose)
    elif args.lesson:
        try:
            ok = check(args.lesson, impl=args.impl, verbose=args.verbose)
        except KeyError as exc:
            print(exc.args[0])
            return 2
    else:
        print("Usage: python -m course check <lesson-id>   (or --all)")
        return 2
    return 0 if ok else 1


def cmd_build(args) -> int:
    build = _load_script("build_site")
    out = build.build()
    print(f"Built {out.relative_to(ROOT).as_posix()}")
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    build = _load_script("build_site")
    if build.is_stale():
        out = build.build()
        print(f"Built {out.relative_to(ROOT).as_posix()}")
    serve(port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_progress(args) -> int:
    data = progress.load()
    lessons = [l for l in load_lessons() if l.kind == "lesson"]
    entries = data.get("lessons", {})
    passed = [l for l in lessons if entries.get(l.id, {}).get("status") == "passed"]
    hints = sum(int(entries.get(l.id, {}).get("hints_used", 0)) for l in lessons)
    forced = [l.id for l in lessons if entries.get(l.id, {}).get("solution_forced")]
    print(f"Lessons passed:   {len(passed)} / {len(lessons)}")
    print(f"Hints revealed:   {hints}")
    print(f"Solutions forced: {len(forced)}" + (f"  ({', '.join(forced)})" if forced else ""))
    gflops = data.get("machine", {}).get("gflops")
    print(f"Machine GFLOPS:   {gflops if gflops is not None else '— (pass s0-l00 to measure)'}")
    boss = data.get("boss", {})
    if boss:
        for bid, res in boss.items():
            print(f"Boss {bid}: {'passed' if res.get('passed') else 'not yet'} ({res.get('value')})")
    nxt = next((l for l in load_lessons() if entries.get(l.id, {}).get("status") != "passed"), None)
    if nxt:
        print(f"Next up:          {nxt.id} — {nxt.title}")
    return 0


def cmd_reset(args) -> int:
    from .reset import apply_reset, plan_reset

    try:
        lesson = get_lesson(args.lesson)
    except KeyError as exc:
        print(exc.args[0])
        return 2
    plans = plan_reset(lesson)
    if not plans:
        print(f"{lesson.id} has no stub functions to reset.")
        return 0
    print(f"This will restore these functions to their original stubs ({lesson.id}):")
    for ws, replaced, appended in plans:
        names = ", ".join(replaced + appended)
        print(f"  {ws.relative_to(ROOT).as_posix()}: {names}")
    print("Your other code in these files is not touched.")
    if not args.yes:
        answer = input("Type 'yes' to continue: ").strip().lower()
        if answer not in ("yes", "y"):
            print("Cancelled.")
            return 1
    apply_reset(lesson)
    print("Done.")
    return 0


def cmd_sync(args) -> int:
    from .reset import sync_workspace

    changes = sync_workspace()
    for c in changes:
        print("  " + c)
    print("Workspace is up to date." if not changes else f"{len(changes)} change(s); your existing code was not touched.")
    return 0


def cmd_tutor(args) -> int:
    from getpass import getpass

    from . import tutor

    st = tutor.status()
    print(f"Tutor now: {st['provider'] or 'not set up'}" + (f", model {st['model']}" if st["provider"] else ""))
    if args.handoff:
        text, path = tutor.handoff()
        print(text)
        print(f"Written to {path}")
        return 0
    if args.test:
        try:
            print(tutor.chat([{"role": "user", "content": "In one sentence: what does np.add.at do?"}], max_tokens=80))
            return 0
        except tutor.TutorError as exc:
            print(f"Tutor test failed: {exc}")
            return 1
    print("Pick one (you only need one):")
    print("  1) OpenRouter: free, online. You need a free key from https://openrouter.ai/keys")
    print("  2) Ollama: on this computer, no key. Install from https://ollama.com/download, then run: ollama pull qwen2.5:3b")
    provider = {"1": "openrouter", "2": "ollama"}.get(input("Type 1 or 2: ").strip())
    if not provider:
        print("Cancelled.")
        return 1
    key = getpass("Paste your OpenRouter key (hidden), then press Enter: ").strip() if provider == "openrouter" else ""
    print("Connecting and testing ...")
    try:
        picked = tutor.setup(provider, key)
    except tutor.TutorError as exc:
        print(f"Not connected: {exc}")
        return 1
    print(f"Connected. The tutor uses {picked['model']}.")
    return 0


def cmd_data(args) -> int:
    dl = _load_script("download_data")
    return dl.main(args.which)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m course", description="NumPy LM course")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="all lessons with status").set_defaults(func=cmd_list)

    p = sub.add_parser("check", help="run a lesson's checks")
    p.add_argument("lesson", nargs="?", help="lesson id, e.g. s1-l01")
    p.add_argument("--all", action="store_true", help="check every lesson in order")
    p.add_argument("--keep-going", action="store_true", help="with --all: don't stop at the first failure")
    p.add_argument("--impl", choices=["workspace", "solutions"], default="workspace")
    p.add_argument("--verbose", "-v", action="store_true", help="show pytest's full output")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("serve", help="build (if needed) and open the web course")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_serve)

    sub.add_parser("build", help="rebuild the web course").set_defaults(func=cmd_build)
    sub.add_parser("progress", help="progress summary").set_defaults(func=cmd_progress)

    p = sub.add_parser("reset", help="restore a lesson's stubs from templates/")
    p.add_argument("lesson")
    p.add_argument("--yes", "-y", action="store_true", help="skip the confirmation")
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("tutor", help="set up the AI tutor (OpenRouter free models or a local Ollama model)")
    p.add_argument("--test", action="store_true", help="send one test question")
    p.add_argument("--handoff", action="store_true", help="write tutor_memory/HANDOFF.md for another tutor or AI")
    p.set_defaults(func=cmd_tutor)

    sub.add_parser("sync", help="add new lesson stubs to workspace/ after pulling an update").set_defaults(func=cmd_sync)

    p = sub.add_parser("data", help="download datasets")
    p.add_argument("which", nargs="?", default="shakespeare", choices=["shakespeare", "tinystories", "all"])
    p.set_defaults(func=cmd_data)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
