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
            <div class="row"><button class="btn small" type="button" data-act="reveal-think" data-i="${i}">Compare with model answer</button></div>
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
    h += section("attempt", "Your attempt", L.attempt + sigs);

    if (L.has_check) {
      const cmd = `python -m course check ${L.id}`;
      h += section(
        "check",
        "Check",
        `<div class="cmd"><code>${esc(cmd)}</code>
           <button class="btn small" type="button" data-act="copy" data-text="${esc(cmd)}">Copy</button></div>
         <div id="check-status" class="check-status" aria-live="polite"></div>`
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
    window.scrollTo(0, 0);
    document.title = `${L.id} · ${L.title} — NumPy LM Course`;
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
