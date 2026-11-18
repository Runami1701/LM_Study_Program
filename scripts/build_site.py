"""Build the web course: lessons/**/*.md -> site/build/index.html.

The output is one self-contained HTML file: CSS, JS, highlight.js, KaTeX and
all lesson content are inlined. Only KaTeX's fonts are loaded from
site/vendor/katex/fonts/ by relative path, so it works offline.

Usage:  python scripts/build_site.py      (or: python -m course build)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course.content import (  # noqa: E402
    expand_includes,
    extract_signature,
    md_to_html,
    parse_lesson,
    split_think,
)
from course.registry import STAGES, load_lessons  # noqa: E402

SRC = ROOT / "site" / "src"
VENDOR = ROOT / "site" / "vendor"
OUT = ROOT / "site" / "build" / "index.html"

WATCH = [
    ROOT / "lessons",
    ROOT / "site" / "src",
    ROOT / "templates",
    ROOT / "solutions",
    ROOT / "scripts" / "build_site.py",
    ROOT / "course" / "content.py",
    ROOT / "course" / "registry.py",
]


def _mtime(p: Path) -> float:
    if p.is_dir():
        return max((f.stat().st_mtime for f in p.rglob("*") if f.is_file()), default=0.0)
    return p.stat().st_mtime if p.exists() else 0.0


def is_stale() -> bool:
    if not OUT.exists():
        return True
    built = OUT.stat().st_mtime
    return any(_mtime(p) > built for p in WATCH)


class BuildError(Exception):
    pass


def lesson_payload(lesson) -> dict:
    parsed = parse_lesson(lesson.path)
    if parsed.errors:
        raise BuildError(f"{lesson.path.name}: " + "; ".join(parsed.errors))

    def one(name: str) -> str:
        blocks = parsed.of(name)
        return md_to_html(blocks[0].body) if blocks else ""

    thinks = []
    for b in parsed.of("think"):
        q, a = split_think(b.body)
        thinks.append({"q": md_to_html(q), "a": md_to_html(a)})

    hints = sorted(parsed.of("hint"), key=lambda b: int(b.attrs.get("level", 0)))

    sol_blocks = parsed.of("solution")
    solution = ""
    if sol_blocks:
        body, errors = expand_includes(sol_blocks[0].body)
        if errors:
            raise BuildError(f"{lesson.path.name}: " + "; ".join(errors))
        solution = md_to_html(body)

    signatures = []
    for rel in lesson.files:
        tpl = lesson.template_file(rel)
        code = []
        for fn in lesson.functions:
            if lesson.file_for_function(fn) == rel:
                try:
                    code.append(extract_signature(tpl, fn))
                except (KeyError, OSError):
                    pass
        if code:
            signatures.append({"file": rel, "code": "\n\n".join(code)})

    return {
        "id": lesson.id,
        "title": lesson.title,
        "stage": lesson.stage,
        "order": lesson.order,
        "kind": lesson.kind,
        "prereqs": lesson.prereqs,
        "files": lesson.files,
        "functions": lesson.functions,
        "est_minutes": lesson.est_minutes,
        "has_check": bool(lesson.tests),
        "goal": one("goal"),
        "notation": one("notation"),
        "concept": one("concept"),
        "think": thinks,
        "hints": [md_to_html(h.body) for h in hints],
        "attempt": one("attempt"),
        "signatures": signatures,
        "solution": solution,
        "breakit": [md_to_html(b.body) for b in parsed.of("breakit")],
        "recap": one("recap"),
    }


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _script_safe(text: str) -> str:
    """Make text safe to inline inside <script> ... </script>."""
    return text.replace("</script", "<\\/script").replace("<!--", "<\\!--")


def build() -> Path:
    lessons = load_lessons()
    stage_ids = {l.stage for l in lessons}
    data = {
        "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stages": [{"id": sid, "name": name} for sid, name in STAGES if sid in stage_ids],
        "lessons": [lesson_payload(l) for l in lessons],
    }
    katex_css = _read(VENDOR / "katex" / "katex.min.css").replace(
        "url(fonts/", "url(../vendor/katex/fonts/"
    )
    page = _read(SRC / "index.html")
    replacements = {
        "/*@KATEX_CSS@*/": katex_css,
        "/*@STYLE@*/": _read(SRC / "style.css"),
        "/*@KATEX_JS@*/": _script_safe(_read(VENDOR / "katex" / "katex.min.js")),
        "/*@HLJS@*/": _script_safe(_read(VENDOR / "highlight" / "highlight.min.js")),
        # "<" is a valid JSON escape for "<", so no "</script>" can appear.
        "/*@DATA@*/": json.dumps(data, ensure_ascii=False).replace("<", "\\u003c"),
        "/*@APP@*/": _script_safe(_read(SRC / "app.js")),
    }
    for key, value in replacements.items():
        if key not in page:
            raise BuildError(f"site/src/index.html is missing the placeholder {key}")
        page = page.replace(key, value)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    try:
        out = build()
    except BuildError as exc:
        print(f"Build failed: {exc}")
        sys.exit(1)
    print(f"Built {out.relative_to(ROOT).as_posix()}")
