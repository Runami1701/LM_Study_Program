"""Restore a lesson's stub functions from ``templates/``.

Only the top-level functions/classes listed in the lesson's ``functions`` front
matter are replaced; everything else in the learner's file (other lessons'
functions, their own helpers, imports) is left exactly as it is.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .registry import ROOT, Lesson


def _def_spans(source: str) -> dict[str, tuple[int, int]]:
    """name -> (start_line, end_line), 1-based inclusive, decorators included."""
    tree = ast.parse(source)
    spans = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            spans[node.name] = (start, node.end_lineno)
    return spans


def _extract(source: str, span: tuple[int, int]) -> list[str]:
    lines = source.splitlines(keepends=True)
    return lines[span[0] - 1 : span[1]]


def plan_reset(lesson: Lesson) -> list[tuple[Path, list[str], list[str]]]:
    """Return [(workspace_file, replaced_names, appended_names)] without writing."""
    plans = []
    for rel in lesson.files:
        ws = ROOT / rel
        tpl = lesson.template_file(rel)
        if not tpl.exists():
            continue
        tpl_src = tpl.read_text(encoding="utf-8")
        tpl_spans = _def_spans(tpl_src)
        names = [n for n in lesson.functions if n in tpl_spans]
        if not names:
            continue
        if not ws.exists():
            plans.append((ws, [], names))
            continue
        ws_spans = _def_spans(ws.read_text(encoding="utf-8"))
        replaced = [n for n in names if n in ws_spans]
        appended = [n for n in names if n not in ws_spans]
        plans.append((ws, replaced, appended))
    return plans


def apply_reset(lesson: Lesson) -> list[tuple[Path, list[str], list[str]]]:
    plans = plan_reset(lesson)
    for ws, replaced, appended in plans:
        rel = ws.relative_to(ROOT)
        tpl = lesson.template_file(rel.as_posix())
        tpl_src = tpl.read_text(encoding="utf-8")
        tpl_spans = _def_spans(tpl_src)
        if not ws.exists():
            ws.parent.mkdir(parents=True, exist_ok=True)
            ws.write_text(tpl_src, encoding="utf-8")
            continue
        src = ws.read_text(encoding="utf-8")
        lines = src.splitlines(keepends=True)
        ws_spans = _def_spans(src)
        # Replace from the bottom up so earlier line numbers stay valid.
        for name in sorted(replaced, key=lambda n: ws_spans[n][0], reverse=True):
            start, end = ws_spans[name]
            lines[start - 1 : end] = _extract(tpl_src, tpl_spans[name])
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        for name in appended:
            lines.append("\n\n")
            lines.extend(_extract(tpl_src, tpl_spans[name]))
        ws.write_text("".join(lines), encoding="utf-8")
    return plans


def sync_workspace() -> list[str]:
    """Add stubs that exist in templates/ but not in workspace/ (new files, and
    new functions appended to existing files). Never changes existing code."""
    changes = []
    tp_root, ws_root = ROOT / "templates", ROOT / "workspace"
    for tpl in sorted(tp_root.rglob("*.py")):
        ws = ws_root / tpl.relative_to(tp_root)
        rel = ws.relative_to(ROOT).as_posix()
        if not ws.exists():
            ws.parent.mkdir(parents=True, exist_ok=True)
            ws.write_text(tpl.read_text(encoding="utf-8"), encoding="utf-8")
            changes.append(f"created {rel}")
            continue
        tpl_src = tpl.read_text(encoding="utf-8")
        tpl_spans = _def_spans(tpl_src)
        ws_src = ws.read_text(encoding="utf-8")
        try:
            ws_names = set(_def_spans(ws_src))
        except SyntaxError:
            changes.append(f"skipped {rel} (it has a syntax error; fix it and run sync again)")
            continue
        missing = [n for n in tpl_spans if n not in ws_names]
        if not missing:
            continue
        add = ws_src if ws_src.endswith("\n") else ws_src + "\n"
        for name in missing:
            add += "\n\n" + "".join(_extract(tpl_src, tpl_spans[name]))
        ws.write_text(add, encoding="utf-8")
        changes.append(f"added {', '.join(missing)} to {rel}")
    return changes
