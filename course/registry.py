"""Lesson registry.

The single source of truth for lessons is the YAML front matter at the top of
each ``lessons/**/*.md`` file. This module reads that front matter and exposes
the lessons in course order, plus helpers to map a lesson's functions to the
files that hold them.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LESSONS_DIR = ROOT / "lessons"

# Stage order and display names. A stage with no lesson files yet is simply
# not shown.
STAGES: list[tuple[str, str]] = [
    ("n", "NumPy basics"),
    ("s0", "Stage 0 · Setup"),
    ("s1", "Stage 1 · MLP character model"),
    ("a", "Interlude A · Autograd engine"),
    ("s2", "Stage 2 · Character transformer"),
    ("s3", "Stage 3 · Subword (BPE) transformer"),
]
STAGE_INDEX = {sid: i for i, (sid, _) in enumerate(STAGES)}
STAGE_NAMES = dict(STAGES)


@dataclass
class Lesson:
    id: str
    title: str
    stage: str
    order: int
    path: Path
    prereqs: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    tests: str | None = None
    est_minutes: int | None = None
    kind: str = "lesson"  # "lesson" or "boss"
    boss: dict = field(default_factory=dict)  # boss lessons: results file, metric, threshold

    @property
    def sort_key(self) -> tuple[int, int]:
        return (STAGE_INDEX.get(self.stage, 99), self.order)

    @property
    def test_path(self) -> Path | None:
        return ROOT / self.tests if self.tests else None

    def template_file(self, workspace_rel: str) -> Path:
        """Map ``workspace/x/y.py`` to ``templates/x/y.py``."""
        return ROOT / "templates" / Path(workspace_rel).relative_to("workspace")

    def solution_file(self, workspace_rel: str) -> Path:
        return ROOT / "solutions" / Path(workspace_rel).relative_to("workspace")

    def file_for_function(self, name: str) -> str | None:
        """Return the workspace-relative file (as written in front matter)
        whose template defines ``name``."""
        for rel in self.files:
            tpl = self.template_file(rel)
            if tpl.exists() and name in top_level_names(tpl):
                return rel
        return self.files[0] if self.files else None


def split_front_matter(text: str) -> tuple[dict, str]:
    """Split ``---\\nyaml\\n---\\nbody`` into (meta, body)."""
    text = text.lstrip("﻿")
    if not text.startswith("---"):
        raise ValueError("lesson file must start with YAML front matter ('---')")
    lines = text.splitlines(keepends=True)
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        raise ValueError("unterminated front matter")
    meta = yaml.safe_load("".join(lines[1:end])) or {}
    body = "".join(lines[end + 1 :])
    return meta, body


def top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


@lru_cache(maxsize=1)
def load_lessons() -> tuple[Lesson, ...]:
    lessons = []
    for md in sorted(LESSONS_DIR.rglob("*.md")):
        meta, _ = split_front_matter(md.read_text(encoding="utf-8"))
        lessons.append(
            Lesson(
                id=str(meta["id"]),
                title=str(meta["title"]),
                stage=str(meta["stage"]),
                order=int(meta.get("order", 0)),
                path=md,
                prereqs=list(meta.get("prereqs") or []),
                files=list(meta.get("files") or []),
                functions=list(meta.get("functions") or []),
                tests=meta.get("tests"),
                est_minutes=meta.get("est_minutes"),
                kind=str(meta.get("kind", "lesson")),
                boss=dict(meta.get("boss") or {}),
            )
        )
    lessons.sort(key=lambda l: l.sort_key)
    return tuple(lessons)


def get_lesson(lesson_id: str) -> Lesson:
    for lesson in load_lessons():
        if lesson.id == lesson_id:
            return lesson
    known = ", ".join(l.id for l in load_lessons())
    raise KeyError(f"unknown lesson '{lesson_id}'. Known lessons: {known}")


def next_lesson(lesson_id: str) -> Lesson | None:
    lessons = load_lessons()
    for i, lesson in enumerate(lessons):
        if lesson.id == lesson_id and i + 1 < len(lessons):
            return lessons[i + 1]
    return None
