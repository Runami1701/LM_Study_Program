"""Parse lesson Markdown (front matter + ``::: directive`` blocks) and convert it
to HTML with math protected from the Markdown converter.

Used by scripts/build_site.py and scripts/lint_lessons.py.
"""

from __future__ import annotations

import ast
import html
import re
from dataclasses import dataclass, field
from pathlib import Path

from .registry import ROOT, split_front_matter

SECTION_NAMES = (
    "goal", "notation", "concept", "think", "hint", "attempt",
    "solution", "breakit", "recap",
)

_OPEN = re.compile(r"^:::\s*([a-z]+)\s*(.*?)\s*$")
_CLOSE = re.compile(r"^:::\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_INCLUDE = re.compile(r"^@include\s+(\S+?)#(\w+)\s*$")


@dataclass
class Block:
    name: str
    attrs: dict
    body: str
    line: int  # 1-based line in the .md file where the block opens


@dataclass
class ParsedLesson:
    path: Path
    meta: dict
    blocks: list[Block] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    body_offset: int = 0  # number of lines of front matter

    def of(self, name: str) -> list[Block]:
        return [b for b in self.blocks if b.name == name]


def parse_attrs(text: str) -> dict:
    return {k: v.strip("\"'") for k, v in re.findall(r"(\w+)=(\S+)", text)}


def parse_lesson(path: Path) -> ParsedLesson:
    raw = path.read_text(encoding="utf-8")
    meta, body = split_front_matter(raw)
    offset = raw.lstrip("\ufeff").count("\n") - body.count("\n")
    lesson = ParsedLesson(path=path, meta=meta, body_offset=offset)

    current: Block | None = None
    buf: list[str] = []
    in_fence = False
    for i, line in enumerate(body.splitlines(), start=offset + 1):
        if current is not None and _FENCE.match(line):
            in_fence = not in_fence
        if current is None:
            m = _OPEN.match(line)
            if m:
                current = Block(m.group(1), parse_attrs(m.group(2)), "", i)
                buf = []
                in_fence = False
            elif line.strip():
                lesson.errors.append(f"line {i}: text outside any ::: block: {line.strip()[:60]!r}")
            continue
        if not in_fence and _CLOSE.match(line):
            current.body = "\n".join(buf).strip("\n")
            lesson.blocks.append(current)
            current = None
            continue
        if not in_fence and _OPEN.match(line):
            lesson.errors.append(f"line {i}: '::: {_OPEN.match(line).group(1)}' opened inside "
                                 f"'::: {current.name}' (line {current.line}) — missing ':::'?")
        buf.append(line)
    if current is not None:
        lesson.errors.append(f"line {current.line}: '::: {current.name}' is never closed")
    for b in lesson.blocks:
        if b.name not in SECTION_NAMES:
            lesson.errors.append(f"line {b.line}: unknown section '::: {b.name}'")
    return lesson


# ------------------------------------------------------------------ think blocks

def split_think(body: str) -> tuple[str, str]:
    """'Q: ...\\nA: ...' -> (question, answer)."""
    m = re.match(r"\s*Q:\s*(.*?)^\s*A:\s*(.*)\Z", body, re.S | re.M)
    if not m:
        raise ValueError("think block must contain 'Q:' then a line starting with 'A:'")
    return m.group(1).strip(), m.group(2).strip()


# --------------------------------------------------------------- source extraction

def _node_span(node) -> tuple[int, int]:
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return start, node.end_lineno


def find_def(path: Path, name: str):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return src, node
    return src, None


def extract_source(path: Path, name: str) -> str:
    src, node = find_def(path, name)
    if node is None:
        raise KeyError(f"{name} not found in {path}")
    start, end = _node_span(node)
    return "".join(src.splitlines(keepends=True)[start - 1 : end]).rstrip() + "\n"


def _signature_of(lines: list[str], node) -> list[str]:
    """def-line(s) + docstring of a function node, followed by '...'."""
    start, _ = _node_span(node)
    body0 = node.body[0]
    has_doc = (
        isinstance(body0, ast.Expr)
        and isinstance(getattr(body0, "value", None), ast.Constant)
        and isinstance(body0.value.value, str)
    )
    stop = body0.end_lineno if has_doc else body0.lineno - 1
    out = lines[start - 1 : stop]
    indent = " " * (body0.col_offset)
    out.append(f"{indent}...\n")
    return out


def extract_signature(path: Path, name: str) -> str:
    """Signature + docstring of a stub (function, or class with its methods)."""
    src, node = find_def(path, name)
    if node is None:
        raise KeyError(f"{name} not found in {path}")
    lines = src.splitlines(keepends=True)
    if isinstance(node, ast.ClassDef):
        start, _ = _node_span(node)
        out = lines[start - 1 : node.lineno]
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
            out = lines[start - 1 : body[0].end_lineno]
            body = body[1:]
        for child in body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append("\n")
                out.extend(_signature_of(lines, child))
        return "".join(out).rstrip() + "\n"
    return "".join(_signature_of(lines, node)).rstrip() + "\n"


def expand_includes(body: str) -> tuple[str, list[str]]:
    """Replace '@include path#name' lines by a fenced python block of that source."""
    errors = []
    out = []
    for line in body.splitlines():
        m = _INCLUDE.match(line.strip())
        if not m:
            out.append(line)
            continue
        rel, name = m.groups()
        try:
            code = extract_source(ROOT / rel, name)
            out.append("```python\n" + code + "```")
        except (OSError, KeyError, SyntaxError) as exc:
            errors.append(f"@include {rel}#{name}: {exc}")
            out.append(f"*(missing include: {rel}#{name})*")
    return "\n".join(out), errors


# ------------------------------------------------------------- math protection

# Code first (so a '$' inside code is never math), then $$display$$, then $inline$.
_TOKEN = re.compile(
    r"(?P<fence>^[ \t]*(```|~~~).*?^[ \t]*\2[ \t]*$)"
    r"|(?P<code>`[^`\n]+`)"
    r"|(?P<display>\$\$(?P<dbody>.+?)\$\$)"
    r"|(?P<inline>(?<![\\$])\$(?P<ibody>[^\s$](?:[^$\n]*?[^\s$\\])?)\$(?!\$))",
    re.S | re.M,
)


def find_math(text: str) -> list[tuple[str, bool]]:
    """All math in a Markdown string as (tex, is_display), skipping code."""
    found = []
    for m in _TOKEN.finditer(text):
        if m.group("display"):
            found.append((m.group("dbody").strip(), True))
        elif m.group("inline"):
            found.append((m.group("ibody"), False))
    return found


def protect_math(text: str) -> tuple[str, list[tuple[str, bool]]]:
    store: list[tuple[str, bool]] = []

    def repl(m):
        if m.group("fence") or m.group("code"):
            return m.group(0)
        if m.group("display"):
            store.append((m.group("dbody").strip(), True))
        else:
            store.append((m.group("ibody"), False))
        return f"MATHPH{len(store) - 1:04d}Z"

    return _TOKEN.sub(repl, text), store


def restore_math(html_text: str, store: list[tuple[str, bool]]) -> str:
    def render(idx: int) -> str:
        tex, display = store[idx]
        if display:
            return f'<div class="math math-display">{html.escape(tex)}</div>'
        return f'<span class="math math-inline">{html.escape(tex)}</span>'

    html_text = re.sub(r"<p>\s*MATHPH(\d{4})Z\s*</p>", lambda m: render(int(m.group(1))), html_text)
    return re.sub(r"MATHPH(\d{4})Z", lambda m: render(int(m.group(1))), html_text)


_md = None


def md_to_html(text: str) -> str:
    """Markdown -> HTML with $...$ / $$...$$ protected and restored as math spans."""
    global _md
    import markdown

    if _md is None:
        _md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"])
    _md.reset()
    protected, store = protect_math(text)
    out = _md.convert(protected)
    return restore_math(out, store)
