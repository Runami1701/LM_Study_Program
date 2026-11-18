"""Start the course with one double-click.

What it does, in order:
1. checks that Python is 3.10 or newer;
2. installs the course and its packages the first time (pip install -e .);
3. downloads Tiny Shakespeare the first time;
4. starts the course website and opens it in your browser.

Windows: double-click start_course.bat.   Linux/macOS: run ./start_course.sh.
You can also run it directly:  python start_course.py
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NEEDED = ["numpy", "pytest", "markdown", "yaml"]


def say(msg: str) -> None:
    print(f"[course] {msg}", flush=True)


def run(args: list[str]) -> int:
    return subprocess.call([sys.executable, *args], cwd=ROOT)


def missing_packages() -> list[str]:
    return [name for name in NEEDED if importlib.util.find_spec(name) is None]


def main() -> int:
    if sys.version_info < (3, 10):
        say(f"This course needs Python 3.10 or newer. You have {sys.version.split()[0]}.")
        say("Install a newer Python from https://www.python.org/downloads/ and try again.")
        return 1

    missing = missing_packages()
    if missing:
        say(f"First start: installing the course and its packages ({', '.join(missing)} missing).")
        say("This takes a minute or two and needs an internet connection.")
        if run(["-m", "pip", "install", "-e", "."]) != 0:
            say("Installing failed. Read the error above, or ask for help with it.")
            return 1

    if not (ROOT / "data" / "tinyshakespeare.txt").exists():
        say("First start: downloading the training text (about 1 MB).")
        if run(["-m", "course", "data"]) != 0:
            say("The download failed. The course still opens; run it again later to retry.")

    say("Starting the course. Your browser opens in a moment.")
    say("Keep this window open while you study. Close it (or press Ctrl+C) to stop.")
    return run(["-m", "course", "serve"])


if __name__ == "__main__":
    try:
        code = main()
    except KeyboardInterrupt:
        code = 0
    if code != 0:
        input("\nPress Enter to close this window.")
    sys.exit(code)
