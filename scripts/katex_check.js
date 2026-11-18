// Render every formula with KaTeX and report errors (used by lint_lessons.py).
// stdin:  JSON [{ "id": "...", "tex": "...", "display": true }, ...]
// stdout: JSON [{ "id": "...", "error": "..." } | { "id": "...", "warning": "..." }, ...]
"use strict";
const path = require("path");
const katex = require(path.join(__dirname, "..", "site", "vendor", "katex", "katex.min.js"));

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => (input += chunk));
process.stdin.on("end", () => {
  const items = JSON.parse(input || "[]");
  const out = [];
  for (const item of items) {
    const warnings = [];
    try {
      katex.renderToString(item.tex, {
        displayMode: !!item.display,
        throwOnError: true,
        strict: (code, msg) => {
          warnings.push(`${code}: ${msg}`);
          return "ignore";
        },
      });
    } catch (e) {
      out.push({ id: item.id, error: String(e.message || e) });
      continue;
    }
    for (const w of warnings) out.push({ id: item.id, warning: w });
  }
  process.stdout.write(JSON.stringify(out));
});
