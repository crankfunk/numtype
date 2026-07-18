// Item 12 (CI) — COVENANT S1 import guard (zero-dep, CI-runnable).
//
// Mirrors the canonical rule in `graph-a-lama.rules.json` (`covenant-s1`):
//   forbidden import  from: ^spike/src/   to: ^spike/(tests|bench|demo)
// i.e. runtime source under spike/src/ must never import from test/bench/demo.
// The graph-a-lama `lint` gate stays the canonical LOCAL check (its binary is
// not CI-installable); this is its CI-runnable, zero-dep twin. `rules.json`
// remains the single source of truth — if the rule changes there, change here.
//
// CRITICAL (Item-12 Baustein-0 finding D-1): every import specifier under
// spike/src/** is RELATIVE (`./` / `../`) — a raw specifier NEVER begins with
// the literal "spike/", so matching the pattern against the raw specifier string
// would be structurally VACUOUS (green forever). So each specifier is RESOLVED
// against its importing file's directory to a repo-relative path BEFORE the test.
//
// ROBUSTNESS (Item-12 Verify-B findings, closed in-slice): the scan is TEXT-WIDE
// over a comment-stripped view of each source, NOT line-by-line — so it catches
// import forms split across physical lines: multi-line dynamic `import(\n "x" \n)`
// AND multi-line static `import … from\n "x"` (both valid TS; the repo has no
// formatter forcing single-line, so a naive line-scan is a real bypass). Comments
// are removed by a small STRING-AWARE state machine (not a line regex): a `//` or
// `/*` inside a string is not treated as a comment, and an import-looking string
// inside a `/* … */` (on one line or many) or `// …` comment is neither a false
// positive nor a bypass. Not a full TS parser, but robust for the import surface
// that graph-a-lama's AST lint covers. The self-test below pins these forms.
//
// Uses ONLY the project's hand-written ambient node shim (spike/src/ambient.d.ts;
// `@types/node` is deliberately not a dependency): node:assert `ok`/`strictEqual`,
// node:fs `readdirSync({recursive})`/`readFileSync`, node:path `join`/`dirname`,
// node:url `fileURLToPath`. Repo-relative paths via prefix-stripping (macOS/Linux,
// separator "/"), avoiding node:path.relative/.sep.

import { test } from "node:test";
import assert from "node:assert";
import { readdirSync, readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url)); // <repo>/spike/tests-runtime
const REPO_ROOT = join(HERE, "..", ".."); // <repo>
const SRC_DIR = join(HERE, "..", "src"); // <repo>/spike/src
const FORBIDDEN = /^spike\/(tests|bench|demo)/;

// Repo-relative POSIX path of an absolute path under REPO_ROOT.
function toRepoRel(absPath: string): string {
  return absPath.startsWith(REPO_ROOT + "/") ? absPath.slice(REPO_ROOT.length + 1) : absPath;
}

// Specifier forms: static `import … from "x"` / `import type … from` / re-export
// `export … from` (all end in `from "x"`), dynamic `import("x")`, side-effect
// `import "x"`. `\s` matches newlines, so over a text-wide scan these span lines.
const SPECIFIER_RES: RegExp[] = [
  /\bfrom\s*["']([^"']+)["']/g,
  /\bimport\s*\(\s*["']([^"']+)["']/g,
  /\bimport\s+["']([^"']+)["']/g,
];

// Replace comments with layout-preserving blanks (spaces + kept newlines) while
// PRESERVING string literals (an import specifier IS a string). String-aware, so
// `//`/`/*` inside a string is not a comment, and an import-looking string inside
// a comment is blanked. Character positions are preserved in→out, so line numbers
// survive.
function stripComments(src: string): string {
  let out = "";
  const n = src.length;
  let state: "code" | "line" | "block" | "sq" | "dq" | "tpl" = "code";
  let i = 0;
  while (i < n) {
    const c = src[i]!;
    const c2 = i + 1 < n ? src[i + 1]! : "";
    if (state === "code") {
      if (c === "/" && c2 === "/") { out += "  "; state = "line"; i += 2; continue; }
      if (c === "/" && c2 === "*") { out += "  "; state = "block"; i += 2; continue; }
      if (c === "'") { out += c; state = "sq"; i++; continue; }
      if (c === '"') { out += c; state = "dq"; i++; continue; }
      if (c === "`") { out += c; state = "tpl"; i++; continue; }
      out += c; i++; continue;
    }
    if (state === "line") {
      if (c === "\n") { out += "\n"; state = "code"; i++; continue; }
      out += " "; i++; continue;
    }
    if (state === "block") {
      if (c === "*" && c2 === "/") { out += "  "; state = "code"; i += 2; continue; }
      out += c === "\n" ? "\n" : " "; i++; continue;
    }
    // string states (sq/dq/tpl): preserve verbatim, honor backslash escapes
    const close = state === "sq" ? "'" : state === "dq" ? '"' : "`";
    if (c === "\\") { out += c + c2; i += 2; continue; }
    out += c;
    if (c === close) state = "code";
    i++;
  }
  return out;
}

interface SpecRef {
  specifier: string;
  line: number;
}

function importSpecifiers(source: string): SpecRef[] {
  const code = stripComments(source);
  const results: SpecRef[] = [];
  for (const re of SPECIFIER_RES) {
    re.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(code)) !== null) {
      const spec = m[1];
      if (spec) results.push({ specifier: spec, line: code.slice(0, m.index).split("\n").length });
    }
  }
  return results;
}

// Non-vacuity self-test: the scanner catches the multi-line + comment forms a
// naive line-scan misses (Verify-B findings), without polluting spike/src.
test("S1 guard scanner: multi-line + comment forms (non-vacuity)", () => {
  const dyn = importSpecifiers('const m = await import(\n  "../tests-runtime/x"\n);');
  assert.ok(dyn.some((s) => s.specifier === "../tests-runtime/x"), "multi-line dynamic import missed");
  const stat = importSpecifiers('import { y } from\n  "../tests-runtime/x";');
  assert.ok(stat.some((s) => s.specifier === "../tests-runtime/x"), "multi-line static import missed");
  const sameLineBlock = importSpecifiers('/* import "../tests-runtime/x" here */\nconst z = 1;');
  assert.strictEqual(sameLineBlock.length, 0, "same-line block comment falsely matched");
  const strWithSlashes = importSpecifiers('const u = "http://x/y";\nconst a = 1;');
  assert.strictEqual(strWithSlashes.length, 0, "string literal with // falsely matched or dropped");
});

// Every .ts file under spike/src, as absolute paths (recursive listing returns
// paths relative to SRC_DIR).
const srcFiles = readdirSync(SRC_DIR, { recursive: true })
  .filter((p) => p.endsWith(".ts"))
  .map((p) => join(SRC_DIR, p));

const violations: string[] = [];
for (const file of srcFiles) {
  const rel = toRepoRel(file);
  const src = readFileSync(file, "utf8");
  for (const { specifier, line } of importSpecifiers(src)) {
    if (!specifier.startsWith(".")) continue; // bare / node: builtins are not project paths
    const resolvedRel = toRepoRel(join(dirname(file), specifier)); // join() normalizes the `..`
    if (FORBIDDEN.test(resolvedRel)) {
      violations.push(`${rel}:${line} imports "${specifier}" -> ${resolvedRel}`);
    }
  }
}

test("COVENANT S1: spike/src never imports from spike/(tests|bench|demo)", () => {
  assert.ok(
    violations.length === 0,
    `COVENANT S1 violation(s) (mirrors graph-a-lama.rules.json 'covenant-s1'):\n  ${violations.join("\n  ")}`,
  );
});
