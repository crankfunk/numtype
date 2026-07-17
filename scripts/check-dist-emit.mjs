// Item 11 / S3 — dist emit-precision gate (D-S2.2 + S2-Verify A-1 coverage note).
//
// Independent check (NOT reusing postbuild-dist.mjs's rewrite logic, so a bug
// in THAT logic can't hide here): after `build:dist`, scan the emitted
// `dist/**/*.{js,d.ts}` for any REMAINING relative `.ts` module reference in
// import/export/`import()`/`new URL` position. If one survives, the post-emit
// rewrite missed it (a Blocker-1/2 regression) — exit 1 with the location.
//
// This is the ONLY gate that catches a Blocker-1 rewrite regression: the S3
// consumer type-smoke stays green even with `.ts` extensions in the `.d.ts`
// under bundler/nodenext (S2-Verify A-1), so it CANNOT stand in for this check.
//
// Comment-aware (S3-Verify F5): `.ts` mentions inside comments (e.g.
// `// ported from "../x.ts"`) must NOT fail the build — this uses the same
// block-comment tracking the rewriter does, kept as an INDEPENDENT copy so the
// two never share MATCH logic (the independence that lets this catch a rewriter
// bug). Line-scoped (S3-Verify F3): matches per physical line; tsc emits
// import/export/`import()`/`new URL(...)` single-line only, so a multi-line
// offender is structurally impossible in the emitted corpus — a documented
// latent limit, not an active gap. Zero runtime deps.

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const DIST = join(fileURLToPath(new URL(".", import.meta.url)), "..", "dist");

// Same three syntax-scoped forms as the rewrite, matching a SURVIVING `.ts`.
const OFFENDERS = [
  /\bfrom\s*["']\.\.?\/[^"']*\.ts["']/,
  /\bimport\(\s*["']\.\.?\/[^"']*\.ts["']/,
  /\bnew\s+URL\(\s*["']\.\.?\/[^"']*\.ts["']/,
];

function walk(dir) {
  const files = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) files.push(...walk(full));
    else if (entry.endsWith(".js") || entry.endsWith(".d.ts")) files.push(full);
  }
  return files;
}

const hits = [];
for (const file of walk(DIST)) {
  const lines = readFileSync(file, "utf8").split("\n");
  let inBlock = false;
  lines.forEach((line, i) => {
    if (inBlock) {
      if (line.includes("*/")) inBlock = false;
      return; // inside (or closing) a block comment — never a real offender
    }
    const trimmed = line.trimStart();
    if (trimmed.startsWith("//") || trimmed.startsWith("*") || trimmed.startsWith("/*")) {
      if (trimmed.startsWith("/*") && !line.includes("*/")) inBlock = true;
      return; // comment line — a `.ts` mention here is prose, not a module ref
    }
    if (OFFENDERS.some((re) => re.test(line))) hits.push(`${file}:${i + 1}: ${line.trim()}`);
  });
}

if (hits.length > 0) {
  console.error(
    `check-dist-emit: FAIL — ${hits.length} relative .ts module reference(s) survived the post-emit rewrite ` +
      `(a consumer would get "Cannot find module" under strict resolvers):\n  ${hits.join("\n  ")}`,
  );
  process.exit(1);
}
console.log("check-dist-emit: OK — no relative .ts module references remain in dist/");
