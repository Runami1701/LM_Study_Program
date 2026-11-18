"""Download the course datasets into data/ (one-time; everything else is offline).

Usage:  python -m course data            (Tiny Shakespeare)
        python scripts/download_data.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

DATASETS = {
    "shakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "path": DATA / "tinyshakespeare.txt",
        "n_chars": 1_115_394,
        "vocab_size": 65,
    },
    # TinyStories is added in Phase 5 (stage 3).
}


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "numpy-lm-course"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def fetch_shakespeare(force: bool = False) -> Path:
    spec = DATASETS["shakespeare"]
    dest = spec["path"]
    DATA.mkdir(exist_ok=True)
    if dest.exists() and not force:
        print(f"Tiny Shakespeare already present: {dest.relative_to(ROOT).as_posix()}")
    else:
        print("Downloading Tiny Shakespeare ...")
        _download(spec["url"], dest)
    text = dest.read_text(encoding="utf-8")
    vocab = len(set(text))
    ok = len(text) == spec["n_chars"] and vocab == spec["vocab_size"]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    print(f"  {len(text):,} characters, vocabulary size {vocab}, sha256 {digest}…")
    if not ok:
        print(f"  WARNING: expected {spec['n_chars']:,} characters and V = {spec['vocab_size']}.")
    return dest


def main(which: str = "shakespeare") -> int:
    try:
        if which in ("shakespeare", "all"):
            fetch_shakespeare()
    except OSError as exc:
        print(f"Download failed: {exc}")
        print("Check your internet connection and try again.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "shakespeare"))
