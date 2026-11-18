/* NumPy LM course — client app.
 *
 * Progress sources:
 *   - progress.json (via `python -m course serve`): check results, hints, forced solutions.
 *   - localStorage: think answers, theme, and a local copy of hint/solution events
 *     (the only source when the page is opened as a plain file).
 */
(function () {
  "use strict";

  const DATA = JSON.parse(document.getElementById("course-data").textContent);
  const LESSONS = DATA.lessons;
  const BY_ID = Object.fromEntries(LESSONS.map((l) => [l.id, l]));
  const STAGE_NAME = Object.fromEntries(DATA.stages.map((s) => [s.id, s.name]));
  const SERVED = location.protocol === "http:" || location.protocol === "https:";
  const POLL_MS = 3000;

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  // ------------------------------------------------------------------ storage
  const store = {
    get(key, fallback) {
      try {
        const v = localStorage.getItem(key);
        return v === null ? fallback : JSON.parse(v);
      } catch (e) {
        return fallback;
      }
    },
    set(key, value) {
      try {
        localStorage.setItem(key, JSON.stringify(value));
      } catch (e) {
        /* private mode or blocked storage: the page still works */
      }
    },
  };

  let server = null; // parsed progress.json, when served
  let serverText = "";
  const local = store.get("course:local", { lessons: {} });
  if (!local.lessons) local.lessons = {};
  let currentId = null;

  function localEntry(id) {
    if (!local.lessons[id]) local.lessons[id] = { hints: 0, forced: false, read: false, touched: false };
    return local.lessons[id];
  }
  const saveLocal = () => store.set("course:local", local);
  const serverEntry = (id) => (server && server.lessons && server.lessons[id]) || {};

  function status(id) {
    const s = serverEntry(id).status;
    if (s === "passed") return "passed";
    const l = local.lessons[id];
    if (s === "in_progress" || (l && (l.hints > 0 || l.forced || l.read || l.touched))) return "in_progress";
    return "not_started";
  }
  const isPassed = (id) => status(id) === "passed";
  const hintsUsed = (id) => Math.max(localEntry(id).hints || 0, serverEntry(id).hints_used || 0);
  const isForced = (id) => !!(localEntry(id).forced || serverEntry(id).solution_forced);
  const isRead = (id) => !!(localEntry(id).read || serverEntry(id).read);

  function postEvent(ev) {
    if (!SERVED) return;
    fetch("/api/event", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(ev),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d && !d.error) applyServer(d);
      })
      .catch(() => {});
  }

  function poll() {
    fetch("/progress.json", { cache: "no-store" })
      .then((r) => (r.ok ? r.text() : null))
      .then((t) => {
        if (t !== null && t !== serverText) {
          serverText = t;
          try {
            applyServer(JSON.parse(t));
          } catch (e) {
            /* half-written file: try again next tick */
          }
        }
      })
      .catch(() => {});
  }

  function applyServer(d) {
    server = d;
    refreshStatus();
  }

  // -------------------------------------------------------------------- icons
  const ICON_MOON =
    '<svg viewBox="0 0 20 20" width="17" height="17" aria-hidden="true"><path d="M15.5 12.6A6.5 6.5 0 0 1 7.4 4.5a6.5 6.5 0 1 0 8.1 8.1z" fill="currentColor"/></svg>';
  const ICON_SUN =
    '<svg viewBox="0 0 20 20" width="17" height="17" aria-hidden="true"><circle cx="10" cy="10" r="3.6" fill="currentColor"/><g stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M10 1.8v2.1M10 16.1v2.1M1.8 10h2.1M16.1 10h2.1M4.2 4.2l1.5 1.5M14.3 14.3l1.5 1.5M4.2 15.8l1.5-1.5M14.3 5.7l1.5-1.5"/></g></svg>';
  const ICON_LOCK =
    '<svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true"><rect x="3" y="7" width="10" height="7" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>';

  // -------------------------------------------------------------------- theme
  function effectiveTheme() {
    const t = document.documentElement.getAttribute("data-theme");
    if (t) return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function paintThemeButton() {
    const dark = effectiveTheme() === "dark";
    const btn = $("#theme-btn");
    btn.innerHTML = dark ? ICON_SUN : ICON_MOON;
    btn.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
    btn.title = btn.getAttribute("aria-label");
  }
  $("#theme-btn").addEventListener("click", () => {
    const next = effectiveTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    store.set("course:theme", next);
    paintThemeButton();
  });

  // ------------------------------------------------------------------ sidebar
  function prereqsMissing(L) {
    return (L.prereqs || []).filter((p) => BY_ID[p] && !isPassed(p));
  }

  function renderSidebar() {
    let h = "";
    for (const st of DATA.stages) {
      h += `<div class="stage-name">${esc(st.name)}</div>`;
      for (const L of LESSONS.filter((l) => l.stage === st.id)) {
        const s = status(L.id);
        const missing = prereqsMissing(L);
        const cls = ["status", s, L.kind === "boss" ? "boss" : ""].join(" ");
        const label = { passed: "passed", in_progress: "in progress", not_started: "not started" }[s];
        h +=
          `<a class="lesson-link${missing.length ? " prereq-open" : ""}" href="#${L.id}"` +
          (L.id === currentId ? ' aria-current="page"' : "") +
          ` title="${esc(L.title)} (${label})${missing.length ? " — prerequisites not passed yet: " + esc(missing.join(", ")) : ""}">` +
          `<span class="${cls}" aria-hidden="true"></span>` +
          `<span class="lid">${esc(L.id)}</span><span class="ltitle">${esc(L.title)}</span>` +
          (missing.length ? `<span class="lock">${ICON_LOCK}</span>` : "") +
          `</a>`;
      }
    }
    $("#sidebar").innerHTML = h;
  }

  function renderTopbar() {
    const total = LESSONS.length;
    const done = LESSONS.filter((l) => isPassed(l.id)).length;
    $("#progress-fill").style.width = total ? `${(100 * done) / total}%` : "0";
    $("#progress-text").textContent = `${done} / ${total}`;
    const L = BY_ID[currentId];
    $("#current-label").textContent = L ? `${L.id} · ${L.title}` : "";
  }

  // ------------------------------------------------------------------- lesson
  function section(key, title, inner, extraClass = "") {
    if (!inner) return "";
    return `<section class="sec ${extraClass}" id="sec-${key}"><h2>${title}</h2>${inner}</section>`;
  }

  function renderLesson(id) {
    if (editorDirty) editorSave();
    const L = BY_ID[id];
    currentId = id;
    const idx = LESSONS.indexOf(L);
    const prev = LESSONS[idx - 1];
    const next = LESSONS[idx + 1];
    const missing = prereqsMissing(L);

    let h = `<article class="lesson">`;
    h += `<header class="lesson-head">
      <div class="eyebrow">${esc(STAGE_NAME[L.stage] || L.stage)} · ${esc(L.id)}</div>
      <h1>${esc(L.title)}</h1>
      <div class="meta">
        ${L.est_minutes ? `<span>~${L.est_minutes} min</span>` : ""}
        ${L.files.length ? `<span>${L.files.map((f) => `<code>${esc(f)}</code>`).join(" ")}</span>` : ""}
        ${missing.length ? `<span class="warn">Prerequisites not passed yet: ${missing.map(esc).join(", ")}</span>` : ""}
      </div>
    </header>`;

    h += section("goal", "Goal", L.goal ? `<div class="goal-box">${L.goal}</div>` : "");
    h += section("notation", "Notation", L.notation);
    h += section("numpy", "Code you need here", L.numpy ? `<div class="numpy-box">${L.numpy}</div>` : "");
    h += section("concept", "Concept", L.concept);

    if (L.think.length) {
      h += section(
        "think",
        "Think",
        `<p class="muted">Write your answer first, then compare.</p>` +
          L.think
            .map(
              (t, i) => `<div class="think" data-i="${i}">
            <div class="think-q"><span class="qnum">Q${i + 1}</span><div>${t.q}</div></div>
            <textarea class="think-input" data-i="${i}" placeholder="Your answer…" aria-label="Your answer to question ${i + 1}"></textarea>
            <div class="row"><button class="btn small" type="button" data-act="reveal-think" data-i="${i}">Compare with model answer</button>${SERVED ? `<button class="btn small" type="button" data-act="tutor-check" data-i="${i}">Check with the tutor</button>` : ""}</div>
            <div class="think-check" data-i="${i}" hidden></div>
            <div class="think-a" data-i="${i}" hidden><div class="label">Model answer</div>${t.a}</div>
          </div>`
            )
            .join("")
      );
    }

    if (L.hints.length) {
      h += section(
        "hints",
        "Hints",
        `<p class="muted">Stuck? Reveal one hint at a time; each unlocks the next.</p>
         <div class="row hint-buttons">${L.hints
           .map((_, i) => `<button class="btn small" type="button" data-act="hint" data-level="${i + 1}">Hint ${i + 1}</button>`)
           .join("")}</div>` +
          L.hints
            .map((hh, i) => `<div class="hint" data-level="${i + 1}" hidden><div class="label">Hint ${i + 1}</div>${hh}</div>`)
            .join("")
      );
    }

    const sigs = L.signatures
      .map(
        (s) =>
          `<div class="sig"><div class="sig-file">${esc(s.file)}</div><pre><code class="language-python">${esc(
            s.code
          )}</code></pre></div>`
      )
      .join("");
    const editor = L.files.length
      ? `<div class="editor" id="editor">
           <div class="editor-tabs" role="tablist">${L.files.map((f, i) => `<button type="button" role="tab" class="editor-tab" data-act="open-file" data-file="${esc(f)}" aria-selected="${i === 0}">${esc(f.split("/").pop())}</button>`).join("")}<span class="editor-path" id="editor-path"></span></div>
           <textarea id="code" class="code" spellcheck="false" autocapitalize="off" autocomplete="off" wrap="off" aria-label="Your code"></textarea>
           <div class="row editor-bar">
             <button class="btn primary small" type="button" data-act="save">Save</button>
             <button class="btn small" type="button" data-act="save-check">Save and check</button>
             <span class="save-state" id="save-state"></span>
             <span class="grow"></span>
             <button class="btn small" type="button" data-act="reset-ask">Reset my code for this lesson</button>
           </div>
           <div class="confirm" id="reset-confirm" hidden>
             <p>This puts back the original empty functions of this lesson (${L.functions.map((f) => `<code>${esc(f)}</code>`).join(", ")}). Your other code is not touched.</p>
             <div class="row"><button class="btn small danger" type="button" data-act="reset-yes">Yes, reset them</button><button class="btn small" type="button" data-act="reset-no">Keep my code</button></div>
           </div>
         </div>`
      : "";
    h += section("attempt", "Your attempt", L.attempt + (SERVED ? editor : sigs));

    if (SERVED && L.jobs && L.jobs.length) {
      h += section(
        "run",
        "Run",
        `<div class="jobs">${L.jobs
          .map((j) => `<div class="job"><button class="btn small" type="button" data-act="job-start" data-kind="${j.kind}">${esc(j.label)}</button><span class="muted small-note">${esc(j.description)}</span></div>`)
          .join("")}</div>
         <div class="row job-bar" id="job-bar" hidden><span id="job-state" class="small-note"></span><button class="btn small danger" type="button" data-act="job-stop">Stop</button></div>
         <pre id="job-log" class="check-output job-log" hidden></pre>`
      );
    }

    if (L.has_check) {
      const cmd = `python -m course check ${L.id}`;
      h += section(
        "check",
        "Check",
        (SERVED && L.ask ? `<div class="boss-ask"><p><b>${esc(L.ask)}</b></p>
           <div class="row" role="group" aria-label="Your answer"><button class="btn small" type="button" data-act="ask-answer" data-answer="y" aria-pressed="false">Yes</button><button class="btn small" type="button" data-act="ask-answer" data-answer="n" aria-pressed="false">No</button>
           <span class="muted small-note">Read the sample the check shows, then answer and press Run check again.</span></div></div>` : "") +
        (SERVED ? `<div class="row run-row"><button class="btn primary" type="button" data-act="run-check">Run check</button>
           <span class="muted small-note">${L.kind === "boss" ? "Reads your training results." : `Tests your code in <code>${esc(L.files.join(", "))}</code>.`}</span></div>
           <pre id="check-output" class="check-output" hidden></pre>` : `<div class="cmd"><code>${esc(cmd)}</code>
           <button class="btn small" type="button" data-act="copy" data-text="${esc(cmd)}">Copy</button></div>`) +
        `<div id="check-status" class="check-status" aria-live="polite"></div>`
      );
    }

    if (L.solution) {
      h += section(
        "solution",
        "Solution",
        `<div id="sol-locked" class="locked">
           <p>The reference solution unlocks when <code>python -m course check ${esc(L.id)}</code> passes.</p>
           <button class="btn small" type="button" data-act="force-ask">I've tried — show me anyway</button>
           <div id="force-confirm" class="confirm" hidden>
             <p>Opening the solution before your code passes is recorded in your progress. Have you tried all three hints? Wrestling with it a little longer is where most of the learning happens.</p>
             <div class="row">
               <button class="btn small danger" type="button" data-act="force-yes">Yes, show the solution</button>
               <button class="btn small" type="button" data-act="force-no">Keep trying</button>
             </div>
           </div>
         </div>
         <div id="sol-body" hidden><div id="forced-note" class="forced-note" hidden>Opened before passing. Read it, close it, then write your own version from memory.</div>${L.solution}</div>`
      );
    }

    if (L.breakit.length) {
      h += section("breakit", "Break it", L.breakit.map((b) => `<div class="breakit">${b}</div>`).join(""));
    }
    h += section("recap", "Recap", L.recap);

    h += `<footer class="lesson-foot">
      ${prev ? `<a class="navlink" href="#${prev.id}"><small>← Previous</small>${esc(prev.title)}</a>` : "<span></span>"}
      <button class="btn small" type="button" data-act="mark-read" id="mark-read">Mark as read</button>
      ${next ? `<a class="navlink" href="#${next.id}" style="text-align:right"><small>Next →</small>${esc(next.title)}</a>` : "<span></span>"}
    </footer>`;
    h += `</article>`;

    const main = $("#main");
    main.innerHTML = h;

    // Restore think answers.
    $$(".think-input", main).forEach((ta) => {
      const saved = store.get(`think:${id}:${ta.dataset.i}`, "");
      if (saved) ta.value = saved;
    });

    renderMath(main);
    highlight(main);
    refreshStatus();
    bossAnswer = "";
    if (SERVED && L.files.length) { editorDirty = false; editorOpen(L.files[0], true); }
    if (SERVED && L.jobs && L.jobs.length) {
      const known = L.jobs.map((j) => j.kind);
      jobKind = known.includes(jobKind) ? jobKind : null;
      if (!jobKind) {
        Promise.all(known.map((k) => fetch(`/api/job?kind=${k}`).then((r) => r.json()).catch(() => ({}))))
          .then((all) => { const run = all.find((x) => x.running) || all.find((x) => x.log); if (run && run.kind) { jobKind = run.kind; $("#job-bar").hidden = false; $("#job-log").hidden = false; jobPoll(); } });
      } else { $("#job-bar").hidden = false; $("#job-log").hidden = false; jobPoll(); }
    }
    window.scrollTo(0, 0);
    document.title = `${L.id} · ${L.title} — NumPy LM Course`;
    tutorShowLesson();
  }

  function renderMath(root) {
    if (!window.katex) return;
    $$(".math", root).forEach((el) => {
      const tex = el.textContent;
      try {
        window.katex.render(tex, el, {
          displayMode: el.classList.contains("math-display"),
          throwOnError: false,
          strict: "ignore",
        });
      } catch (e) {
        el.textContent = tex; // never lose the source
      }
    });
  }

  function highlight(root) {
    if (!window.hljs) return;
    window.hljs.configure({ ignoreUnescapedHTML: true });
    $$("pre code.language-python", root).forEach((el) => window.hljs.highlightElement(el));
  }

  // Updates the parts of the page that depend on progress, without re-rendering
  // the lesson (so typing in a textarea is never interrupted by a poll).
  function refreshStatus() {
    renderSidebar();
    renderTopbar();
    const L = BY_ID[currentId];
    if (!L) return;
    const id = L.id;

    // hints
    const used = hintsUsed(id);
    $$(".hint", $("#main")).forEach((el) => {
      el.hidden = Number(el.dataset.level) > used;
    });
    $$('[data-act="hint"]', $("#main")).forEach((btn) => {
      const lvl = Number(btn.dataset.level);
      btn.disabled = lvl > used + 1;
      btn.classList.toggle("done", lvl <= used);
      btn.textContent = lvl <= used ? `Hint ${lvl} ✓` : `Hint ${lvl}`;
    });

    // check status
    const box = $("#check-status");
    if (box) {
      const e = serverEntry(id);
      box.className = "check-status";
      if (isPassed(id)) {
        box.classList.add("passed");
        const when = e.passed_at ? new Date(e.passed_at).toLocaleString() : "";
        box.textContent = `✓ Passed${when ? " on " + when : ""}${e.attempts ? ` · ${e.attempts} check run(s)` : ""}`;
      } else if (!SERVED) {
        box.textContent =
          "Check results appear here when the course is opened with “python -m course serve”.";
      } else if (e.attempts) {
        box.textContent = `Not passing yet · ${e.attempts} check run(s). Fix the first ✗ in the terminal output and run it again.`;
      } else {
        box.textContent = "Not checked yet. Write your code, then run the command above.";
      }
    }

    // solution gate
    const locked = $("#sol-locked");
    const body = $("#sol-body");
    if (locked && body) {
      const open = isPassed(id) || isForced(id);
      locked.hidden = open;
      body.hidden = !open;
      $("#forced-note").hidden = !(isForced(id) && !isPassed(id));
    }

    const mr = $("#mark-read");
    if (mr) {
      const read = isRead(id);
      mr.textContent = read ? "Read ✓" : "Mark as read";
      mr.classList.toggle("done", read);
    }
  }

  // ------------------------------------------------------------------ actions
  $("#main").addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-act]");
    if (!btn) return;
    const id = currentId;
    const act = btn.dataset.act;

    if (act === "reveal-think") {
      const box = $(`.think-a[data-i="${btn.dataset.i}"]`);
      box.hidden = !box.hidden;
      btn.textContent = box.hidden ? "Compare with model answer" : "Hide model answer";
    } else if (act === "hint") {
      const lvl = Number(btn.dataset.level);
      if (lvl > hintsUsed(id) + 1) return;
      const e = localEntry(id);
      e.hints = Math.max(e.hints || 0, lvl);
      e.touched = true;
      saveLocal();
      postEvent({ lesson: id, hints_used: e.hints });
      refreshStatus();
    } else if (act === "copy") {
      const text = btn.dataset.text;
      const done = () => {
        btn.textContent = "Copied";
        setTimeout(() => (btn.textContent = "Copy"), 1400);
      };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(done, () => fallbackCopy(text, done));
      } else {
        fallbackCopy(text, done);
      }
    } else if (act === "run-check") {
      const out = $("#check-output");
      btn.disabled = true;
      btn.textContent = "Checking…";
      out.hidden = false;
      out.textContent = "Running the checks for " + id + " …";
      fetch("/api/check", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ lesson: id, answer: bossAnswer }) })
        .then((r) => r.json())
        .then((d) => {
          out.textContent = d.output || d.error || "(no output)";
          out.classList.toggle("passed", !!d.ok);
        })
        .catch(() => (out.textContent = "Could not reach the course server. Is the start window still open?"))
        .finally(() => {
          btn.disabled = false;
          btn.textContent = "Run check";
          poll();
        });
    } else if (act === "ask-answer") {
      bossAnswer = btn.dataset.answer;
      $$('[data-act="ask-answer"]').forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    } else if (act === "open-file") {
      editorOpen(btn.dataset.file);
    } else if (act === "save") {
      editorSave();
    } else if (act === "save-check") {
      editorSave().then((ok) => ok && $('[data-act="run-check"]') && $('[data-act="run-check"]').click());
    } else if (act === "reset-ask") {
      $("#reset-confirm").hidden = false;
    } else if (act === "reset-no") {
      $("#reset-confirm").hidden = true;
    } else if (act === "reset-yes") {
      fetch("/api/reset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ lesson: id }) })
        .then((r) => r.json())
        .then((d) => {
          $("#reset-confirm").hidden = true;
          if (d.error) { $("#save-state").textContent = d.error; return; }
          editorDirty = false;
          editorOpen(editorFile, true);
          $("#save-state").textContent = "Reset to the original stubs.";
        });
    } else if (act === "job-start") {
      jobStart(btn.dataset.kind);
    } else if (act === "job-stop") {
      fetch("/api/job/stop", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }).then(() => jobPoll());
    } else if (act === "tutor-check") {
      const ta = $(`.think-input[data-i="${btn.dataset.i}"]`);
      tutorCheck(Number(btn.dataset.i), ta ? ta.value : "");
    } else if (act === "force-ask") {
      $("#force-confirm").hidden = false;
      btn.hidden = true;
    } else if (act === "force-no") {
      $("#force-confirm").hidden = true;
      $('[data-act="force-ask"]').hidden = false;
    } else if (act === "force-yes") {
      const e = localEntry(id);
      e.forced = true;
      e.touched = true;
      saveLocal();
      postEvent({ lesson: id, solution_forced: true });
      refreshStatus();
    } else if (act === "mark-read") {
      const e = localEntry(id);
      e.read = true;
      e.touched = true;
      saveLocal();
      postEvent({ lesson: id, read: true });
      refreshStatus();
    }
  });

  function fallbackCopy(text, done) {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand("copy");
      done();
    } catch (e) {
      /* ignore */
    }
    ta.remove();
  }

  let thinkTimer = null;
  $("#main").addEventListener("input", (ev) => {
    const ta = ev.target.closest(".think-input");
    if (!ta) return;
    const id = currentId;
    clearTimeout(thinkTimer);
    thinkTimer = setTimeout(() => {
      store.set(`think:${id}:${ta.dataset.i}`, ta.value);
      const e = localEntry(id);
      if (!e.touched) {
        e.touched = true;
        saveLocal();
        postEvent({ lesson: id, think: true });
        refreshStatus();
      }
    }, 400);
  });

  // ---------------------------------------------------------- mobile sidebar
  function setMenu(open) {
    $("#sidebar").classList.toggle("open", open);
    $("#scrim").hidden = !open;
    $("#menu-btn").setAttribute("aria-expanded", String(open));
  }
  $("#menu-btn").addEventListener("click", () => setMenu(!$("#sidebar").classList.contains("open")));
  $("#scrim").addEventListener("click", () => setMenu(false));
  $("#sidebar").addEventListener("click", (ev) => {
    if (ev.target.closest(".lesson-link")) setMenu(false);
  });

  // ------------------------------------------------------------------ routing
  function defaultLesson() {
    const firstOpen = LESSONS.find((l) => !isPassed(l.id));
    return (firstOpen || LESSONS[0]).id;
  }
  function route() {
    const id = decodeURIComponent(location.hash.replace(/^#/, ""));
    if (BY_ID[id]) {
      if (id !== currentId) renderLesson(id);
    } else {
      renderLesson(store.get("course:last", null) && BY_ID[store.get("course:last", null)] ? store.get("course:last") : defaultLesson());
    }
    store.set("course:last", currentId);
  }
  window.addEventListener("hashchange", route);

  // ------------------------------------------------------------------- editor
  // The lesson's exercise file, edited in place. Only files in workspace/ can be saved.
  let editorFile = null, editorDirty = false, bossAnswer = "";

  async function editorOpen(file, force) {
    if (!$("#code")) return;
    if (editorDirty && !force) await editorSave();
    editorFile = file;
    $$(".editor-tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.file === file)));
    $("#editor-path").textContent = file;
    $("#code").value = "Loading…";
    try {
      const d = await (await fetch(`/api/file?path=${encodeURIComponent(file)}`, { cache: "no-store" })).json();
      $("#code").value = d.error ? `# ${d.error}` : d.text;
      $("#save-state").textContent = "";
    } catch (e) {
      $("#code").value = "# Could not load the file. Is the course window still open?";
    }
    editorDirty = false;
  }

  async function editorSave() {
    if (!editorFile || !$("#code")) return false;
    $("#save-state").textContent = "Saving…";
    try {
      const r = await fetch("/api/file", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: editorFile, text: $("#code").value }) });
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      editorDirty = false;
      $("#save-state").textContent = "Saved.";
      return true;
    } catch (e) {
      $("#save-state").textContent = `Not saved: ${e.message || e}`;
      return false;
    }
  }

  $("#main").addEventListener("keydown", (ev) => {
    if (ev.target.id !== "code") return;
    if (ev.key === "Tab" && !ev.shiftKey) {
      ev.preventDefault();
      const t = ev.target, a = t.selectionStart, b = t.selectionEnd;
      t.setRangeText("    ", a, b, "end");
      editorDirty = true;
    } else if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "s") {
      ev.preventDefault();
      editorSave();
    }
  });
  $("#main").addEventListener("input", (ev) => {
    if (ev.target.id === "code" && !editorDirty) { editorDirty = true; $("#save-state").textContent = "Unsaved changes"; }
  });
  window.addEventListener("beforeunload", (ev) => { if (editorDirty) { ev.preventDefault(); ev.returnValue = ""; } });

  // --------------------------------------------------------------------- jobs
  let jobKind = null, jobTimer = null;

  async function jobStart(kind) {
    if (editorDirty) await editorSave();
    const r = await fetch("/api/job/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind }) });
    const d = await r.json();
    jobKind = kind;
    $("#job-bar").hidden = false;
    $("#job-log").hidden = false;
    if (d.error) { $("#job-state").textContent = d.error; return; }
    jobPoll();
  }

  async function jobPoll() {
    clearTimeout(jobTimer);
    if (!jobKind || !$("#job-log")) return;
    let d;
    try { d = await (await fetch(`/api/job?kind=${encodeURIComponent(jobKind)}`, { cache: "no-store" })).json(); } catch (e) { return; }
    const log = $("#job-log");
    if (!log) return;
    log.textContent = d.log || "(starting…)";
    log.scrollTop = log.scrollHeight;
    const mins = d.seconds ? ` · ${Math.floor(d.seconds / 60)} min ${d.seconds % 60} s` : "";
    $("#job-state").textContent = d.running ? `Running${mins}. You can keep reading; this page updates every 2 seconds.`
      : d.stopped ? `Stopped${mins}.` : d.exit_code === 0 ? `Finished${mins}.` : d.exit_code != null ? `Stopped with an error (code ${d.exit_code}). Read the last lines above.` : "";
    log.classList.toggle("passed", d.exit_code === 0);
    $('[data-act="job-stop"]').hidden = !d.running;
    $$('[data-act="job-start"]').forEach((b) => (b.disabled = d.running));
    if (d.running) jobTimer = setTimeout(jobPoll, 2000);
    else poll();
  }

  // -------------------------------------------------------------------- tutor
  // A chat column on the right. One conversation per lesson, kept in this browser.
  let tutorState = { ready: false, provider: null, model: null, providers: {} };

  function verdictHtml(text) {
    const m = /^\s*VERDICT:\s*(CORRECT|PARTLY|NOT YET)\s*\n?/i.exec(text);
    if (!m) return renderRich(text);
    const v = m[1].toUpperCase();
    return `<span class="verdict ${v === "CORRECT" ? "correct" : v === "PARTLY" ? "partly" : "notyet"}">${v}</span>` + renderRich(text.slice(m[0].length));
  }

  function showCheck(i, html, isNote) {
    const box = $(`.think-check[data-i="${i}"]`);
    if (!box) return;
    box.hidden = false;
    box.innerHTML = `<div class="label">Tutor's check</div>` + (isNote ? `<p class="muted">${html}</p>` : html);
    renderMath(box);
  }

  function renderRich(text) {
    const parts = [];
    const keep = (html) => { parts.push(html); return `\u0000${parts.length - 1}\u0000`; };
    let t = String(text);
    t = t.replace(/```[\w-]*\n?([\s\S]*?)```/g, (_, c) => keep(`<pre><code>${esc(c.replace(/\s+$/, ""))}</code></pre>`));
    t = t.replace(/\$\$([\s\S]+?)\$\$/g, (_, m) => keep(`<div class="math math-display">${esc(m.trim())}</div>`));
    t = t.replace(/\\\[([\s\S]+?)\\\]/g, (_, m) => keep(`<div class="math math-display">${esc(m.trim())}</div>`));
    t = t.replace(/\\\((.+?)\\\)/g, (_, m) => keep(`<span class="math math-inline">${esc(m)}</span>`));
    t = t.replace(/\$([^$\n]+?)\$/g, (_, m) => keep(`<span class="math math-inline">${esc(m)}</span>`));
    t = t.replace(/`([^`\n]+)`/g, (_, c) => keep(`<code>${esc(c)}</code>`));
    t = esc(t).replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    const html = t.split(/\n{2,}/).map((para) => `<p>${para.replace(/\n/g, "<br>")}</p>`).join("");
    return html.replace(/\u0000(\d+)\u0000/g, (_, i) => parts[+i]);
  }

  function tutorMessage(role, text, extra = "") {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    if (role === "tutor") {
      div.innerHTML = renderRich(text);
    } else if (role === "user") {
      div.innerHTML = extra + renderRich(text);
    } else {
      div.textContent = text;
    }
    $("#tutor-log").appendChild(div);
    renderMath(div);
    $("#tutor-log").scrollTop = $("#tutor-log").scrollHeight;
    return div;
  }

  async function tutorShowLesson() {
    if (!SERVED || !currentId) return;
    const id = currentId;
    const L = BY_ID[id];
    let hist = { messages: [], checks: {}, has_summary: false };
    try { hist = await (await fetch(`/api/tutor/history?lesson=${encodeURIComponent(id)}`, { cache: "no-store" })).json(); } catch (e) { /* offline */ }
    if (id !== currentId) return;
    const log = $("#tutor-log");
    log.innerHTML = "";
    tutorMessage("note", `Lesson ${L.id} · ${L.title}. The tutor sees this lesson (not its solution) and gives hints rather than finished code. Chats are saved on this computer.`);
    if (hist.has_summary) tutorMessage("note", "Earlier messages of this chat were folded into a summary; the tutor still remembers them.");
    if (!tutorState.ready) tutorMessage("note", "The tutor is not set up yet: follow the steps above.");
    if (!tutorState.ready && $("#tutor-setup").hidden) tutorSettings(true);
    for (const turn of hist.messages || []) tutorMessage(turn.role === "assistant" ? "tutor" : "user", turn.content);
    Object.entries(hist.checks || {}).forEach(([i, c]) => {
      const ta = $(`.think-input[data-i="${i}"]`);
      if (ta && !ta.value) ta.value = c.answer;
      showCheck(i, verdictHtml(c.result));
    });
    $("#tutor-sub").textContent = tutorState.ready ? `${tutorState.provider} · ${tutorState.model}` : "not set up";
  }

  function tutorOpen(open) {
    $("#tutor").hidden = !open;
    $("#tutor-btn").setAttribute("aria-expanded", String(open));
    $("#tutor-btn").setAttribute("aria-label", open ? "Hide the tutor chat" : "Show the tutor chat");
    store.set("course:tutor-open", open);
    if (open) setTimeout(() => $("#tutor-input").focus(), 0);
  }

  async function tutorPost(path, body) {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const d = await r.json().catch(() => ({ error: "The course server sent an unreadable answer." }));
    if (!r.ok || d.error) throw new Error(d.error || `error ${r.status}`);
    return d;
  }

  let tutorBusy = false;
  async function tutorRun(userText, request, shown) {
    if (tutorBusy) return;
    if (!tutorState.ready) { tutorOpen(true); tutorSettings(true); tutorMessage("note", "Set up the tutor first (the steps above)."); return; }
    tutorBusy = true;
    $("#tutor-send").disabled = true;
    const id = currentId;
    tutorMessage("user", shown || userText);
    const waiting = tutorMessage("note", "The tutor is thinking…");
    waiting.classList.add("typing");
    try {
      const d = await request();
      waiting.remove();
      if (id === currentId) tutorMessage("tutor", d.text);
    } catch (e) {
      waiting.remove();
      tutorMessage("error", String(e.message || e));
    } finally {
      tutorBusy = false;
      $("#tutor-send").disabled = false;
    }
  }

  function tutorAsk(question) {
    question = question.trim();
    if (!question) return;
    tutorRun(question, () => tutorPost("/api/tutor/ask", { lesson: currentId, question }));
  }

  async function tutorCheck(i, answer) {
    if (!tutorState.ready) { tutorOpen(true); tutorSettings(true); showCheck(i, "Set up the tutor first (the form in the tutor column).", true); return; }
    if (!answer.trim()) { showCheck(i, "Write your answer in the box first, then press “Check with the tutor”.", true); return; }
    const btn = $(`[data-act="tutor-check"][data-i="${i}"]`);
    btn.disabled = true;
    showCheck(i, "The tutor is reading your answer…", true);
    try {
      const d = await tutorPost("/api/tutor/check", { lesson: currentId, index: i, answer });
      showCheck(i, verdictHtml(d.text));
    } catch (e) {
      showCheck(i, esc(String(e.message || e)), true);
    } finally {
      btn.disabled = false;
    }
  }

  let setupProvider = null;

  function tutorSettings(open) {
    $("#tutor-setup").hidden = !open;
    $("#tutor-gear").setAttribute("aria-expanded", String(open));
    if (!open) return;
    $("#setup-msg").textContent = "";
    $("#setup-msg").className = "setup-msg";
    $("#setup-cancel").hidden = !tutorState.ready;
    pickProvider(tutorState.provider || null);
  }

  async function pickProvider(prov) {
    setupProvider = prov;
    $$(".choice").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.prov === prov)));
    $("#setup-openrouter").hidden = prov !== "openrouter";
    $("#setup-ollama").hidden = prov !== "ollama";
    $("#setup-actions").hidden = !prov;
    $("#setup-more").hidden = !prov;
    $("#t-model").value = "";
    $("#t-models").innerHTML = "";
    if (prov === "openrouter") {
      $("#t-key").value = "";
      $("#t-key-saved").textContent = tutorState.has_key && tutorState.provider === "openrouter" ? "A key is already saved; leave the box empty to keep it." : "";
      setTimeout(() => $("#t-key").focus(), 0);
      try {
        const d = await (await fetch("/api/tutor/models")).json();
        (d.models || []).forEach((m) => $("#t-models").insertAdjacentHTML("beforeend", `<option value="${esc(m)}">`));
      } catch (e) { /* the list is optional */ }
    } else if (prov === "ollama") {
      $("#ollama-state").textContent = "Looking for Ollama…";
      try {
        const st = await (await fetch("/api/tutor/ollama", { cache: "no-store" })).json();
        (st.models || []).forEach((m) => $("#t-models").insertAdjacentHTML("beforeend", `<option value="${esc(m)}">`));
        $("#ollama-state").textContent = !st.running
          ? "Ollama is not running yet."
          : st.models.length ? `Ollama is running with ${st.models.length} model(s): ${st.models.slice(0, 3).join(", ")}.` : "Ollama is running, but no model is downloaded yet (step 2).";
      } catch (e) {
        $("#ollama-state").textContent = "";
      }
    }
  }

  async function tutorConnect() {
    if (!setupProvider) return;
    const btn = $("#setup-connect");
    const msg = $("#setup-msg");
    btn.disabled = true;
    msg.className = "setup-msg";
    msg.textContent = setupProvider === "openrouter" ? "Connecting to OpenRouter and testing a free model… (up to a minute)" : "Connecting to Ollama and testing the model…";
    try {
      const d = await tutorPost("/api/tutor/setup", { provider: setupProvider, api_key: $("#t-key").value, model: $("#t-model").value.trim() });
      tutorState = d;
      msg.className = "setup-msg ok";
      msg.textContent = `Connected. The tutor uses ${d.picked}. Ask your first question below.`;
      $("#t-key").value = "";
      tutorShowLesson();
      setTimeout(() => tutorSettings(false), 1600);
    } catch (e) {
      msg.className = "setup-msg bad";
      msg.textContent = String(e.message || e);
      if (setupProvider === "ollama") pickProvider("ollama");
    } finally {
      btn.disabled = false;
    }
  }

  async function tutorRefresh() {
    try {
      tutorState = await (await fetch("/api/tutor/status", { cache: "no-store" })).json();
    } catch (e) { /* not served */ }
    tutorShowLesson();
  }

  if (SERVED) {
    $("#tutor-btn").hidden = false;
    $("#tutor-btn").addEventListener("click", () => tutorOpen($("#tutor").hidden));
    $("#tutor-close").addEventListener("click", () => tutorOpen(false));
    $("#tutor-gear").addEventListener("click", () => tutorSettings($("#tutor-setup").hidden));
    $("#setup-cancel").addEventListener("click", () => tutorSettings(false));
    $$(".choice").forEach((b) => b.addEventListener("click", () => pickProvider(b.dataset.prov)));
    $("#setup-connect").addEventListener("click", tutorConnect);
    $("#t-key").addEventListener("keydown", (ev) => { if (ev.key === "Enter") { ev.preventDefault(); tutorConnect(); } });
    $("#copy-pull").addEventListener("click", (ev) => {
      const b = ev.currentTarget, text = "ollama pull qwen2.5:3b";
      const done = () => { b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1400); };
      if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(done, () => fallbackCopy(text, done));
      else fallbackCopy(text, done);
    });
    $("#tutor-form").addEventListener("submit", (ev) => {
      ev.preventDefault();
      const q = $("#tutor-input").value;
      $("#tutor-input").value = "";
      tutorAsk(q);
    });
    $("#tutor-input").addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); $("#tutor-form").requestSubmit(); }
    });
    $("#tutor-clear").addEventListener("click", async () => {
      try { await tutorPost("/api/tutor/clear", { lesson: currentId }); } catch (e) { /* ignore */ }
      $$(".think-check").forEach((b) => (b.hidden = true));
      tutorShowLesson();
    });
    $("#tutor-handoff").addEventListener("click", async (ev) => {
      const btn = ev.currentTarget;
      if (!tutorState.ready) { tutorSettings(true); return; }
      btn.disabled = true;
      const waiting = tutorMessage("note", "Writing a summary of all your tutor chats for another AI…");
      try {
        const d = await tutorPost("/api/tutor/handoff", {});
        waiting.remove();
        const div = tutorMessage("note", "");
        div.className = "msg tutor handoff";
        div.innerHTML = `<p><strong>Saved to <code>${esc(d.path)}</code>.</strong> Give that file to another AI (or paste it) and it can take over.</p>
          <div class="row"><button class="btn small" type="button">Copy the summary</button></div><pre>${esc(d.markdown)}</pre>`;
        $("button", div).addEventListener("click", (e) => {
          const b = e.currentTarget;
          const done = () => { b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy the summary"), 1400); };
          if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(d.markdown).then(done, () => fallbackCopy(d.markdown, done));
          else fallbackCopy(d.markdown, done);
        });
      } catch (e) {
        waiting.remove();
        tutorMessage("error", String(e.message || e));
      } finally {
        btn.disabled = false;
      }
    });
    tutorRefresh();
    if (store.get("course:tutor-open", window.innerWidth > 1250)) tutorOpen(true);
  }

  // -------------------------------------------------------------------- start
  function showNotice(html) {
    const n = $("#notice");
    n.innerHTML = `<span>${html}</span><button class="btn small" type="button">Dismiss</button>`;
    n.hidden = false;
    $("button", n).addEventListener("click", () => (n.hidden = true));
  }

  paintThemeButton();
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", paintThemeButton);
  }
  if (!LESSONS.length) {
    $("#main").innerHTML = "<p>No lessons found. Run <code>python -m course build</code>.</p>";
    return;
  }
  if (SERVED) {
    poll();
    setInterval(poll, POLL_MS);
  } else {
    showNotice(
      "Opened as a file: check results from the terminal won't appear here. Run <code>python -m course serve</code> for live progress."
    );
  }
  route();
})();
