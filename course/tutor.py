"""The AI tutor: live questions and answer checking through any OpenAI-compatible
chat API: OpenRouter (free models, needs a free API key) or a local model served
by Ollama (no key, nothing leaves your computer).

Settings live in tutor.json at the repository root (gitignored). The website's
tutor panel writes it; `python -m course tutor` does the same in a terminal.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from datetime import datetime, timezone

from . import progress
from .content import parse_lesson, split_think
from .registry import ROOT, get_lesson, load_lessons

CONFIG_PATH = ROOT / "tutor.json"
MEMORY = ROOT / "tutor_memory"          # chats/<lesson>.json and HANDOFF.md (gitignored)
KEEP_RECENT = 8                          # messages sent to the model word for word
COMPRESS_AT = 16                         # longer chats get their oldest part summarized

PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "needs_key": True,
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "needs_key": False,
        "default_model": "qwen2.5:3b",
    },
}

SYSTEM = """You are the tutor of numpy-lm-course: a course in which one learner builds a
language model from scratch in pure NumPy (an MLP, an autograd engine, a transformer,
a BPE tokenizer). The learner knows the math (linear algebra, calculus, backpropagation)
but is new to NumPy and not fluent in coding.

How you answer:
- Short, plain sentences. Name who does what. Define every term the first time.
- Use small concrete examples with real numbers and array shapes, e.g. x (B, k) = (2, 3).
- Write math with $...$ (inline) or $$...$$ (on its own line). Write code in `backticks`.
- Nudge, don't tell. For questions about the learner's exercise, give the next hint only:
  first the idea, then which NumPy feature to use, then pseudocode. Never write the full
  solution code unless the learner says they have already tried and asks for it.
- Stay within this lesson's topic and notation. If you are not sure, say so.
- Answer in the language the learner writes in."""

CHECK = """Compare the learner's answer to a think-question with the model answer.
Reply in this exact form:
VERDICT: one of CORRECT, PARTLY, NOT YET
Then 2 to 4 short sentences: what the learner got right, what is missing or wrong, and
one nudge toward the missing idea. Do not simply copy the model answer. Judge the idea,
not the wording: a correct answer in different words is CORRECT."""


class TutorError(Exception):
    pass


# --------------------------------------------------------------------- settings

def load_config() -> dict:
    cfg = {}
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cfg = {}
    provider = cfg.get("provider") or ("openrouter" if os.environ.get("OPENROUTER_API_KEY") else "")
    if provider in PROVIDERS:
        spec = PROVIDERS[provider]
        cfg.setdefault("base_url", spec["base_url"])
        cfg.setdefault("model", spec["default_model"])
        if provider == "openrouter" and not cfg.get("api_key"):
            cfg["api_key"] = os.environ.get("OPENROUTER_API_KEY", "")
    cfg["provider"] = provider
    return cfg


def save_config(provider: str, model: str = "", api_key: str = "", base_url: str = "") -> dict:
    if provider not in PROVIDERS:
        raise TutorError(f"unknown provider {provider!r}; choose one of {', '.join(PROVIDERS)}")
    old = load_config() if CONFIG_PATH.exists() else {}
    spec = PROVIDERS[provider]
    cfg = {
        "provider": provider,
        "base_url": base_url or spec["base_url"],
        "model": model or spec["default_model"],
        # keep a saved key when the form leaves the key box empty
        "api_key": api_key or (old.get("api_key", "") if old.get("provider") == provider else ""),
    }
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return cfg


def status() -> dict:
    """What the website may show: never the key itself."""
    cfg = load_config()
    p = cfg.get("provider")
    ready = bool(p) and (not PROVIDERS[p]["needs_key"] or bool(cfg.get("api_key")))
    return {"ready": ready, "provider": p or None, "model": cfg.get("model") if p else None,
            "has_key": bool(cfg.get("api_key")), "providers": {k: v["default_model"] for k, v in PROVIDERS.items()}}


# ------------------------------------------------------------------------ calls

def _post(url: str, body: dict, key: str, timeout: int = 120) -> dict:
    headers = {"Content-Type": "application/json", "X-Title": "numpy-lm-course"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        if exc.code == 401:
            raise TutorError("The API key was rejected. Check it in the tutor settings.") from exc
        if exc.code == 429:
            raise TutorError("The free model is busy or its daily limit is reached. Wait a minute, or pick another free model.") from exc
        if exc.code == 404:
            raise TutorError(f"The model was not found. Pick another one in the tutor settings. ({detail})") from exc
        raise TutorError(f"The model service answered with error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        hint = " Is Ollama running? Start it, and pull the model with: ollama pull <model>" if "11434" in url else " Check your internet connection."
        raise TutorError(f"Could not reach {url.split('/v1')[0]}.{hint}") from exc
    except TimeoutError as exc:
        raise TutorError("The model took longer than 2 minutes to answer. Try again or pick a faster model.") from exc


def chat(messages: list[dict], cfg: dict | None = None, max_tokens: int = 700) -> str:
    cfg = cfg or load_config()
    if not cfg.get("provider"):
        raise TutorError("The tutor is not set up yet. Open the tutor settings and choose a model.")
    data = _post(cfg["base_url"].rstrip("/") + "/chat/completions",
                 {"model": cfg["model"], "messages": messages, "temperature": 0.3, "max_tokens": max_tokens},
                 cfg.get("api_key", ""))
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise TutorError(f"Unexpected answer from the model service: {str(data)[:200]}") from exc
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # some local models print their reasoning


def free_openrouter_models() -> list[str]:
    """IDs of the models OpenRouter currently offers for free (no key needed to list)."""
    req = urllib.request.Request("https://openrouter.ai/api/v1/models", headers={"User-Agent": "numpy-lm-course"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        models = json.loads(resp.read().decode("utf-8")).get("data", [])
    free = [m["id"] for m in models
            if str(m.get("pricing", {}).get("prompt")) in ("0", "0.0") and str(m.get("pricing", {}).get("completion")) in ("0", "0.0")]
    return sorted(free)


# --------------------------------------------------------------------- lessons

def lesson_context(lesson_id: str, limit: int = 9000) -> str:
    """The lesson the tutor may see: everything except the solution."""
    lesson = get_lesson(lesson_id)
    parsed = parse_lesson(lesson.path)
    parts = [f"Lesson {lesson.id}: {lesson.title}"]
    if lesson.files:
        parts.append("Exercise file(s): " + ", ".join(lesson.files) + "; functions: " + ", ".join(lesson.functions))
    for b in parsed.blocks:
        if b.name in ("solution",):
            continue
        label = b.name + (f" {b.attrs.get('level')}" if b.name == "hint" else "")
        parts.append(f"[{label}]\n{b.body.strip()}")
    text = "\n\n".join(parts)
    return text[:limit] + ("\n[... lesson text shortened ...]" if len(text) > limit else "")


SUMMARIZE = """You keep the notes of a tutoring conversation, for whoever tutors this learner
next (a person or another AI). Update the notes with the new messages. Write at most 120 words,
in plain sentences, under these labels:
Understood: what the learner has shown they understand.
Confused: misconceptions or questions that kept coming back.
Hints given: which hints or explanations the tutor already gave.
Open: anything left unanswered.
Leave a label out if there is nothing for it."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _chat_path(lesson_id: str) -> Path:
    get_lesson(lesson_id)                     # rejects unknown ids (and any path tricks)
    return MEMORY / "chats" / f"{lesson_id}.json"


def load_chat(lesson_id: str) -> dict:
    path = _chat_path(lesson_id)
    chat_ = {"lesson": lesson_id, "summary": "", "messages": [], "checks": {}}
    if path.exists():
        try:
            chat_.update(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    return chat_


def save_chat(chat_: dict) -> None:
    path = _chat_path(chat_["lesson"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(chat_, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def clear_chat(lesson_id: str) -> None:
    path = _chat_path(lesson_id)
    if path.exists():
        path.unlink()


def _transcript(messages: list) -> str:
    return "\n".join(f"{'Learner' if m['role'] == 'user' else 'Tutor'}: {m['content']}" for m in messages)


def _summarize(old_summary: str, messages: list, checks: dict | None = None) -> str:
    extra = ""
    if checks:
        extra = "\n\nThink-question checks:\n" + "\n".join(
            f"Q{int(i) + 1}: learner wrote {c['answer']!r}; verdict: {c['result'].splitlines()[0] if c['result'] else '?'}"
            for i, c in sorted(checks.items()))
    prompt = (f"Current notes:\n{old_summary or '(none yet)'}\n\nNew messages:\n{_transcript(messages)[:12000]}{extra}")
    return chat([{"role": "system", "content": SUMMARIZE}, {"role": "user", "content": prompt}], max_tokens=300)


def _compress(chat_: dict) -> None:
    """Fold the oldest messages into the running summary, so memory stays small."""
    if len(chat_["messages"]) <= COMPRESS_AT:
        return
    old, keep = chat_["messages"][:-KEEP_RECENT], chat_["messages"][-KEEP_RECENT:]
    try:
        chat_["summary"] = _summarize(chat_["summary"], old)
        chat_["messages"] = keep
        chat_["summarized_at"] = _now()
    except TutorError:
        pass                                   # keep everything; try again next time


def _other_lessons_notes(lesson_id: str, limit: int = 1500) -> str:
    notes = []
    for path in sorted((MEMORY / "chats").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.stem == lesson_id:
            continue
        try:
            summ = json.loads(path.read_text(encoding="utf-8")).get("summary", "")
        except (OSError, json.JSONDecodeError):
            continue
        if summ:
            notes.append(f"[{path.stem}] {summ}")
    text = "\n".join(notes)
    return text[:limit]


def ask(lesson_id: str, question: str) -> str:
    question = (question or "").strip()
    if not question:
        raise TutorError("Type a question first.")
    chat_ = load_chat(lesson_id)
    system = SYSTEM + "\n\nThe lesson the learner is reading:\n\n" + lesson_context(lesson_id)
    if chat_["summary"]:
        system += "\n\nNotes from earlier in this lesson's conversation:\n" + chat_["summary"]
    others = _other_lessons_notes(lesson_id)
    if others:
        system += "\n\nNotes from the learner's other lessons (for context only):\n" + others
    messages = [{"role": "system", "content": system}]
    messages += [{"role": m["role"], "content": m["content"][:3000]} for m in chat_["messages"][-KEEP_RECENT:]]
    messages.append({"role": "user", "content": question[:3000]})
    answer = chat(messages)
    chat_["messages"] += [{"role": "user", "content": question, "at": _now()},
                          {"role": "assistant", "content": answer, "at": _now()}]
    _compress(chat_)
    save_chat(chat_)
    return answer


def check_think(lesson_id: str, index: int, answer: str) -> str:
    answer = (answer or "").strip()
    if not answer:
        raise TutorError("Write your answer in the box first.")
    thinks = parse_lesson(get_lesson(lesson_id).path).of("think")
    if not 0 <= index < len(thinks):
        raise TutorError("Unknown question.")
    q, model_answer = split_think(thinks[index].body)
    messages = [
        {"role": "system", "content": SYSTEM + "\n\n" + CHECK},
        {"role": "user", "content": f"Lesson: {get_lesson(lesson_id).title}\n\nQuestion:\n{q}\n\nModel answer:\n{model_answer}\n\nLearner's answer:\n{answer[:3000]}"},
    ]
    result = chat(messages, max_tokens=350)
    chat_ = load_chat(lesson_id)
    chat_["checks"][str(index)] = {"answer": answer, "result": result, "at": _now()}
    save_chat(chat_)
    return result


# ---------------------------------------------------------------- handoff

def handoff(use_model: bool = True) -> tuple[str, Path]:
    """Write tutor_memory/HANDOFF.md: everything another tutor (human or AI) needs to take over."""
    data = progress.load()
    lessons = load_lessons()
    entries = data.get("lessons", {})
    passed = [l.id for l in lessons if entries.get(l.id, {}).get("status") == "passed"]
    current = next((l for l in lessons if entries.get(l.id, {}).get("status") != "passed"), None)
    lines = [
        "# Tutoring handoff: numpy-lm-course",
        "",
        f"Written {_now()}. Read this to take over tutoring this learner.",
        "",
        "## The learner and the course",
        "",
        "- Builds a language model from scratch in pure NumPy: MLP (stage 1), an autograd engine, a transformer",
        "  (stage 2), a BPE transformer (stage 3). Lessons are in `lessons/`, the learner's code in `workspace/`.",
        "- Knows the math well; new to NumPy and not fluent in coding. Wants plain short sentences, worked numeric",
        "  examples, and hints before answers (never the finished solution unless asked after trying).",
        "",
        "## Progress",
        "",
        f"- Lessons passed: {len(passed)} of {len(lessons)}" + (f" (last: {passed[-1]})" if passed else ""),
        f"- Next lesson: {current.id} · {current.title}" if current else "- All lessons passed.",
        f"- Hints revealed: {sum(int(entries.get(l.id, {}).get('hints_used', 0)) for l in lessons)}; "
        f"solutions opened before passing: {', '.join(l.id for l in lessons if entries.get(l.id, {}).get('solution_forced')) or 'none'}",
        "",
        "## Per lesson",
        "",
    ]
    order = {l.id: i for i, l in enumerate(lessons)}
    paths = sorted((MEMORY / "chats").glob("*.json"), key=lambda p: order.get(p.stem, 999))
    if not paths:
        lines.append("No tutor conversations yet.")
    for path in paths:
        chat_ = load_chat(path.stem)
        title = get_lesson(path.stem).title
        lines += [f"### {path.stem} · {title}", ""]
        summary = chat_["summary"]
        if use_model and (chat_["messages"] or chat_["checks"]):
            try:
                summary = _summarize(summary, chat_["messages"], chat_["checks"])
                chat_["summary"], chat_["messages"] = summary, chat_["messages"][-KEEP_RECENT:]
                chat_["summarized_at"] = _now()
                save_chat(chat_)
            except TutorError as exc:
                lines.append(f"(Could not ask the model for a fresh summary: {exc})")
        if summary:
            lines += [summary.strip(), ""]
        recent = [m["content"] for m in chat_["messages"] if m["role"] == "user"][-5:]
        if recent:
            lines += ["Recent questions:", ""] + [f"- {q.splitlines()[0][:200]}" for q in recent] + [""]
        for i, c in sorted(chat_["checks"].items(), key=lambda kv: int(kv[0])):
            verdict = (c["result"].splitlines() or ["?"])[0]
            lines.append(f"- Think Q{int(i) + 1}: {verdict}")
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    MEMORY.mkdir(exist_ok=True)
    out = MEMORY / "HANDOFF.md"
    out.write_text(text, encoding="utf-8")
    return text, out
