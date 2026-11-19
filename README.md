# numpy-lm-course

Build a language model from scratch in pure NumPy, one small lesson at a time:
an MLP character model, then an autograd engine, a character transformer, and a
subword (BPE) transformer. Every forward pass, backward pass, optimizer and
training loop is yours to write.

The course has two halves:

- **A local web course** (`python -m course serve`): each lesson's concept and
  derivations, think-questions, a 3-level hint ladder, and a solution that
  unlocks when your code passes.
- **A code workspace** (`workspace/`): a stub file per lesson, checked by
  `python -m course check <lesson>`.

Everything runs on the CPU and works offline once the data is downloaded.
Windows, Linux and macOS all work the same way.

## Start the course

Double-click **`start_course.bat`** (Windows) or run **`./start_course.sh`** (Linux/macOS).

The first start installs the course and its packages and downloads the training
text, which needs an internet connection and takes a minute or two. Every start
then opens the course in your browser at http://localhost:8000. Keep the window
open while you study; close it to stop.

You need Python 3.10 or newer (https://www.python.org/downloads/). Start with
lesson **n-l01**.

To do the same by hand: `pip install -e .`, then `python -m course data`, then
`python -m course serve`.

## Run it on a Raspberry Pi (open it from any device at home)

On a Raspberry Pi 4 or 5 with 64-bit Raspberry Pi OS, run as your normal user:

```bash
git clone https://github.com/Runami1701/LM_Study_Program.git && cd LM_Study_Program
./scripts/pi_setup.sh
```

The script installs the course in `.venv`, downloads the training text, and adds a service
(`lm-course`) that starts the course on every boot. At the end it prints the address, such as
`http://raspberrypi.local:8000/`, and a passcode. Open that address on your laptop or phone
and type the passcode once; the browser remembers it.

The passcode exists because the course runs Python on the Pi: without it, anyone on your
Wi-Fi could. It is saved in `access.txt`; delete that file and restart the service
(`sudo systemctl restart lm-course`) to get a new one. After a `git pull`, run
`./scripts/pi_setup.sh` again.

On a Pi, Stage 1 still takes minutes, but Stage 2 and 3 training take several times longer
than on a laptop (use the Continue buttons after a stop). For the tutor, use OpenRouter:
Ollama models run very slowly on a Pi 4.

To do the same on any computer by hand: `python -m course serve --host 0.0.0.0`.

## The AI tutor (optional)

Press the chat button in the top bar to open the tutor column. Ask about the lesson you are
reading, or press **Check with the tutor** under a think-question to have your answer judged
(the verdict appears under that question). The tutor sees the lesson but not its solution, and
gives hints rather than finished code.

**Setup (once).** Open the tutor column. Until the tutor is set up, it shows a setup card.
Pick **one** of the two (you do not need both):

- **OpenRouter** (free, online): open https://openrouter.ai/keys, sign in, press *Create key*,
  copy the key, paste it into the box **"Paste your key here"** in the setup card, press
  **Connect**. The tutor picks a free model and tests it.
- **Ollama** (on your computer, no key, offline): install it from https://ollama.com/download,
  run `ollama pull qwen2.5:3b` once in a terminal (about 2 GB), then press **Connect**.

To change it later, press the gear in the tutor column. In a terminal: `python -m course tutor`.

Chats are saved in `tutor_memory/` on your computer (not in git). Long chats are folded into
a running summary the tutor keeps reading. **Summary for another AI** writes
`tutor_memory/HANDOFF.md`: your progress and, per lesson, what you understood, what confused
you, which hints you got and what is still open, so another AI or a human tutor can take over.
The same file: `python -m course tutor --handoff`.

## How to take a lesson

Everything happens in the course website; you never need the terminal.

1. Read the lesson. Answer each think-question in its box, then compare with the model
   answer (or press **Check with the tutor**).
2. Write your code in the editor under **Your attempt**: it opens the lesson's file from
   `workspace/`. **Save** with the button or Ctrl+S. Use the hints one at a time when stuck.
3. Press **Run check** (or **Save and check**). It tells you which function is wrong and why;
   on a pass the solution unlocks and the next lesson is named.
4. Lessons with long tasks (downloads, training, writing text) have a **Run** section with
   buttons, live output and a Stop button. Stage 2 and 3 can continue from their checkpoint.
5. **Reset my code for this lesson** puts the original stubs back (it asks first).

When you pull a new version of the course, the start window adds new lesson stubs to your
`workspace/` automatically; your code is kept.

The commands below do the same things from a terminal, if you ever want them.

## Commands

| Command | What it does |
| --- | --- |
| `python -m course list` | all lessons with their status |
| `python -m course check <id>` | check one lesson (`--all` for every lesson, `--verbose` for pytest's output) |
| `python -m course serve` | build the site if needed and open it (`--port 8000`) |
| `python -m course progress` | lessons passed, hints used, solutions opened early |
| `python -m course reset <id>` | restore that lesson's stub functions (asks first; other code is kept) |
| `python -m course data` | download the datasets into `data/` |
| `python -m course sync` | add new lesson stubs to `workspace/` after you pull an update (your code is kept) |
| `python -m course build` | rebuild the site |

## Layout

```
lessons/      lesson text (Markdown); the only learner-facing prose
workspace/    YOUR code: stubs to fill in
templates/    pristine copies of the stubs (used by `reset`)
solutions/    reference implementations (peek only after a pass)
tests/        the checks; tests/helpers.py has the friendly asserts
course/       the CLI and tooling (not learner code)
site/         the web course (src/, vendored KaTeX + highlight.js, build/)
scripts/      build_site.py, lint_lessons.py, download_data.py
```

## For course development

```bash
COURSE_IMPL=solutions pytest
```

```bash
python scripts/lint_lessons.py --release
```

The first runs every check against the reference solutions (all must pass);
`pytest --impl solutions` does the same on Windows. The second checks lesson
structure, renders every formula with KaTeX (needs Node.js; otherwise it falls
back to a delimiter check), and confirms `workspace/` matches `templates/`.

Vendored libraries: KaTeX 0.16.11 (MIT) and highlight.js 11.10.0 (BSD-3-Clause);
their licenses are in `site/vendor/`.
