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

## How to take a lesson

1. Read the lesson on the site. Answer each think-question in the box before
   comparing with the model answer.
2. Open the file named under **Your attempt** (in `workspace/`) and implement
   the functions. Use the hints one at a time only when you are stuck.
3. Run the check, e.g.

   ```bash
   python -m course check s1-l01
   ```

   It tells you which function is wrong and why. On a pass, the solution
   unlocks on the site within a few seconds and the next lesson is printed.
4. Try the **Break it** experiments: they show what each piece is protecting
   you from.

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
