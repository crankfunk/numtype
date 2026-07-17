// Item 11 / S2 — Post-Emit-Rewrite (docs/item-11-api-paket-spec.md D-S2.2).
//
// Runs after `tsc -p tsconfig.build.json` over ./dist/**. Rewrites relative
// `.ts` import extensions to `.js` in the exact syntactic positions that
// TypeScript 7.0.2's `rewriteRelativeImportExtensions` does NOT cover:
//   Blocker 1 — `.d.ts` declaration files keep `from "./x.ts"` verbatim.
//   Blocker 2 — `new URL("./x.ts", import.meta.url)` string literals (the
//               threaded worker URL) stay `.ts` in the emitted `.js`.
//   plus dynamic `import("./x.ts")` (the lazy threads load in ndarray.ts).
//
// Precision is binding: ONLY relative paths (`./` or `../`) ending in `.ts`,
// and ONLY inside these three module-position forms. Prose/doc-comment
// mentions of `.ts`, and non-relative or non-`.ts` URLs (e.g. the `.wasm`
// artifact URLs in loader.ts/threaded.ts/backend-api.ts), are left untouched.
// Zero runtime deps (node:fs, node:path, node:url only).

import { readdirSync, readFileSync, writeFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const DIST = join(fileURLToPath(new URL(".", import.meta.url)), "..", "dist");

// Three syntax-scoped forms. The capture keeps the opening token + the
// relative path body; only the trailing `.ts"` (or `.ts'`) becomes `.js"`.
// `\.\.?\/` requires the path to START relative (`./` or `../`) — a bare
// `"x.ts"` or a bare-specifier package import is never touched.
const PATTERNS = [
  // import ... from "./x.ts"  /  export ... from "../x.ts"
  /(\bfrom\s*["'])(\.\.?\/[^"']*)\.ts(["'])/g,
  // import("./x.ts")  (dynamic)
  /(\bimport\(\s*["'])(\.\.?\/[^"']*)\.ts(["'])/g,
  // new URL("./x.ts", ...)
  /(\bnew\s+URL\(\s*["'])(\.\.?\/[^"']*)\.ts(["'])/g,
];

// Comment-aware, line-based (zero-dep). The three PATTERNS are applied ONLY to
// genuine code lines — never to a line inside a block comment (`/* … */`, incl.
// JSDoc `/** … */`) or a whole-line `//` comment. This closes the S2-verify
// finding (Baustein B) that a JSDoc example such as
// `* import { Dim } from "./dim.ts"` — prose that happens to match a PATTERN —
// would otherwise be falsely rewritten to `.js`, silently misdocumenting the
// SOURCE file name. tsc emits import/export/`new URL(...)` statements as their
// own code lines and JSDoc as `*`-prefixed block lines, so line granularity is
// safe for the emitted corpus (a rewrite target never shares a line with the
// prose that must be preserved).
export function rewrite(src) {
  const lines = src.split("\n");
  let inBlock = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (inBlock) {
      if (line.includes("*/")) inBlock = false; // closing line: still skip it
      continue;
    }
    const trimmed = line.trimStart();
    if (trimmed.startsWith("//") || trimmed.startsWith("*") || trimmed.startsWith("/*")) {
      if (trimmed.startsWith("/*") && !line.includes("*/")) inBlock = true; // multi-line block opens
      continue; // comment line — never rewrite
    }
    let out = line;
    for (const re of PATTERNS) out = out.replace(re, "$1$2.js$3");
    lines[i] = out;
  }
  return lines.join("\n");
}

function walk(dir) {
  const files = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) files.push(...walk(full));
    else if (entry.endsWith(".js") || entry.endsWith(".d.ts")) files.push(full);
  }
  return files;
}

function main() {
  let changed = 0;
  let scanned = 0;
  for (const file of walk(DIST)) {
    scanned++;
    const src = readFileSync(file, "utf8");
    const out = rewrite(src);
    if (out !== src) {
      writeFileSync(file, out);
      changed++;
    }
  }
  console.log(`postbuild-dist: rewrote .ts→.js module refs in ${changed}/${scanned} emitted files`);
}

// Run the emit walk only when invoked directly (`node scripts/postbuild-dist.mjs`);
// stays importable for testing `rewrite` in isolation (S2 prosa-immunity check).
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main();
