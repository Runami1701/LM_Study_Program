"""Content QA for lessons (PLAN.md §10).

Checks every lessons/**/*.md for:
  - front matter fields; the files, functions and test file it names exist
    (in workspace/, solutions/ and templates/)
  - required sections; exactly 3 hint levels (1, 2, 3); >= 2 think; >= 1 breakit
  - no code fences outside `solution`, except ```pseudo blocks in hint level 3
  - balanced math delimiters and braces; every formula renders in KaTeX
    (via Node if available, otherwise a warning and the balance check only)
  - (warning) symbols in display math that the notation table doesn't mention
  - @include lines in solutions resolve
With --release: workspace/ stubs must be byte-identical to templates/.

Usage:  python scripts/lint_lessons.py [--release] [lesson files...]
Exit code 1 if any error.
"""

from __future__ import annotations

import ast
import filecmp
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course.content import (  # noqa: E402
    expand_includes,
    find_math,
    parse_lesson,
    protect_math,
    split_think,
)

REQUIRED_META = ("id", "title", "stage", "order")
LESSON_META = ("files", "functions", "tests")
SINGLE = ("goal", "notation", "concept", "attempt", "solution", "recap")
GREEK = {
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon",
    "phi", "varphi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi",
    "Pi", "Sigma", "Phi", "Psi", "Omega",
}
_FENCE_OPEN = re.compile(r"^\s*(```|~~~)\s*([\w-]*)", re.M)
_TEXTLIKE = re.compile(r"\\(?:text|mathrm|texttt|operatorname|mathbf|textbf|textit|begin|end)\s*\{[^{}]*\}")


class Report:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, where: str, msg: str):
        self.errors.append(f"{where}: {msg}")

    def warn(self, where: str, msg: str):
        self.warnings.append(f"{where}: {msg}")


def top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def braces_balanced(tex: str) -> bool:
    depth = 0
    i = 0
    while i < len(tex):
        c = tex[i]
        if c == "\\":
            i += 2  # skip escaped char, e.g. \{ or \}
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


def symbols_in(tex: str) -> set[str]:
    """Best-effort: single Latin letters and Greek commands used in a formula."""
    t = _TEXTLIKE.sub(" ", tex)
    greek = {m for m in re.findall(r"\\([A-Za-z]+)", t) if m in GREEK}
    t = re.sub(r"\\[A-Za-z]+", " ", t)  # drop all commands
    letters = set(re.findall(r"[A-Za-z]", t))
    return letters | {"\\" + g for g in greek}


def notation_symbols(body: str) -> set[str]:
    syms: set[str] = set()
    for line in body.splitlines():
        if not line.strip().startswith("|") or set(line.strip()) <= set("|-: "):
            continue
        first = line.strip().strip("|").split("|")[0]
        syms |= {"\\" + g for g in re.findall(r"\\([A-Za-z]+)", first) if g in GREEK}
        syms |= set(re.findall(r"[A-Za-z]", re.sub(r"\\[A-Za-z]+", " ", first)))
    return syms


def lint_lesson(path: Path, rep: Report, formulas: list[dict]) -> None:
    rel = path.relative_to(ROOT).as_posix()
    try:
        lesson = parse_lesson(path)
    except Exception as exc:  # front matter errors etc.
        rep.error(rel, f"cannot parse: {exc}")
        return
    for e in lesson.errors:
        rep.error(rel, e)
    meta = lesson.meta
    kind = meta.get("kind", "lesson")

    # ---- front matter
    for key in REQUIRED_META + (LESSON_META if kind == "lesson" else ()):
        if key not in meta:
            rep.error(rel, f"front matter is missing '{key}'")
    files = list(meta.get("files") or [])
    functions = list(meta.get("functions") or [])
    for f in files:
        if not f.startswith("workspace/"):
            rep.error(rel, f"files entry must start with workspace/: {f}")
            continue
        sub = Path(f).relative_to("workspace")
        for base in ("workspace", "solutions", "templates"):
            p = ROOT / base / sub
            if not p.exists():
                rep.error(rel, f"missing {base}/{sub.as_posix()}")
    if files and functions:
        for base in ("workspace", "solutions", "templates"):
            names: set[str] = set()
            for f in files:
                p = ROOT / base / Path(f).relative_to("workspace")
                if p.exists():
                    try:
                        names |= top_level_names(p)
                    except SyntaxError as exc:
                        rep.error(rel, f"{base}/{Path(f).relative_to('workspace').as_posix()} has a syntax error: {exc}")
            for fn in functions:
                if fn not in names:
                    rep.error(rel, f"function '{fn}' is not defined in {base}/ ({', '.join(files)})")
    tests = meta.get("tests")
    if tests and not (ROOT / tests).exists():
        rep.error(rel, f"test file {tests} does not exist")

    # ---- sections
    counts = {}
    for b in lesson.blocks:
        counts[b.name] = counts.get(b.name, 0) + 1
    if kind == "lesson":
        for name in SINGLE:
            if counts.get(name, 0) != 1:
                rep.error(rel, f"needs exactly one '::: {name}' section (found {counts.get(name, 0)})")
        if counts.get("think", 0) < 2:
            rep.error(rel, f"needs at least 2 '::: think' sections (found {counts.get('think', 0)})")
        if counts.get("breakit", 0) < 1:
            rep.error(rel, "needs at least 1 '::: breakit' section")
        levels = sorted(b.attrs.get("level", "?") for b in lesson.of("hint"))
        if levels != ["1", "2", "3"]:
            rep.error(rel, f"needs exactly 3 hints with level=1, 2, 3 (found levels {levels})")

    for b in lesson.blocks:
        where = f"{rel}:{b.line} ({b.name})"
        # ---- think format
        if b.name == "think":
            try:
                q, a = split_think(b.body)
                if not q or not a:
                    rep.error(where, "empty question or answer")
            except ValueError as exc:
                rep.error(where, str(exc))
        # ---- code fences
        fences = _FENCE_OPEN.findall(b.body)
        opening_langs = [lang for i, (_, lang) in enumerate(fences) if i % 2 == 0]
        if b.name == "solution":
            pass
        elif b.name == "hint" and b.attrs.get("level") == "3":
            bad = [l for l in opening_langs if l != "pseudo"]
            if bad:
                rep.error(where, "code fences in hint 3 must be marked ```pseudo (pseudocode, never real code)")
        elif opening_langs:
            rep.error(where, "no code fences allowed outside the solution (use inline shapes, or ```pseudo in hint 3)")
        if b.name == "solution":
            _, inc_errors = expand_includes(b.body)
            for e in inc_errors:
                rep.error(where, e)

        # ---- math
        protected, store = protect_math(b.body)
        stripped = re.sub(r"(?ms)^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$", "", protected)
        stripped = re.sub(r"`[^`\n]+`", "", stripped)
        stripped = stripped.replace("\\$", "")
        if "$" in stripped:
            line = next((ln for ln in stripped.splitlines() if "$" in ln), "")
            rep.error(where, f"unbalanced or malformed math delimiter near: {line.strip()[:70]!r}")
        for n, (tex, display) in enumerate(find_math(b.body)):
            fid = f"{where} formula {n + 1}"
            if not braces_balanced(tex):
                rep.error(fid, f"unbalanced braces in {tex[:60]!r}")
            formulas.append({"id": fid, "tex": tex, "display": display})

    # ---- notation coverage (best effort)
    notation = lesson.of("notation")
    if notation:
        known = notation_symbols(notation[0].body)
        missing: dict[str, int] = {}
        for b in lesson.blocks:
            for tex, display in find_math(b.body):
                if not display:
                    continue
                for s in symbols_in(tex) - known:
                    missing[s] = missing.get(s, 0) + 1
        if missing:
            listed = ", ".join(sorted(missing))
            rep.warn(rel, f"symbols in display math not in the notation table: {listed}")


def katex_check(formulas: list[dict], rep: Report) -> None:
    node = shutil.which("node")
    if not node:
        rep.warn("katex", "Node.js not found: formulas were only checked for balanced braces, not rendered")
        return
    script = ROOT / "scripts" / "katex_check.js"
    proc = subprocess.run(
        [node, str(script)], input=json.dumps(formulas), capture_output=True, text=True, encoding="utf-8"
    )
    if proc.returncode != 0:
        rep.error("katex", f"katex_check.js failed: {proc.stderr.strip()[:300]}")
        return
    for item in json.loads(proc.stdout or "[]"):
        if "error" in item:
            rep.error(item["id"], f"KaTeX: {item['error']}")
        else:
            rep.warn(item["id"], f"KaTeX: {item['warning']}")


def release_check(rep: Report) -> None:
    ws, tp = ROOT / "workspace", ROOT / "templates"
    for f in sorted(tp.rglob("*.py")):
        other = ws / f.relative_to(tp)
        if not other.exists():
            rep.error("release", f"workspace is missing {other.relative_to(ROOT).as_posix()}")
        elif not filecmp.cmp(f, other, shallow=False):
            rep.error("release", f"{other.relative_to(ROOT).as_posix()} differs from its template")
    for f in sorted(ws.rglob("*.py")):
        if not (tp / f.relative_to(ws)).exists():
            rep.error("release", f"{f.relative_to(ROOT).as_posix()} has no template")


def main(argv: list[str]) -> int:
    release = "--release" in argv
    paths = [Path(a).resolve() for a in argv if not a.startswith("--")]
    if not paths:
        paths = sorted((ROOT / "lessons").rglob("*.md"))
    rep = Report()
    formulas: list[dict] = []
    for p in paths:
        lint_lesson(p, rep, formulas)
    katex_check(formulas, rep)
    if release:
        release_check(rep)

    for w in rep.warnings:
        print(f"warning  {w}")
    for e in rep.errors:
        print(f"ERROR    {e}")
    print(f"\n{len(paths)} lesson(s), {len(formulas)} formula(s): "
          f"{len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
