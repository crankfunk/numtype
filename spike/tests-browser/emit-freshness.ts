/**
 * F1 (verify-round finding, docs/phase-d-vorarbeiten-v3-ergebnisse.md
 * "In-Slice-Schließungen nach der Verify-Runde"): a stale-`.emit`-freshness
 * guard. `pnpm test:browser` (the wrapper) always emits fresh
 * (`rm -rf .emit && tsc -p tsconfig.emit.json`), but a DIRECT
 * `pnpm exec playwright test spike/tests-browser/smoke.test.ts` happily
 * serves whatever `.emit/` tree happens to be sitting on disk — including
 * one emitted from a stale `spike/src` state, silently. That is a proven
 * false-pass vector (a mutated `spike/src/wasm/resident.ts` compiled into an
 * OLD `.emit/` still reports green).
 *
 * Design: no separate "stamp" file is written. `tsc -p tsconfig.emit.json`
 * always writes EVERY output file fresh (no incremental/composite build is
 * configured, and `test:browser` always `rm -rf`s `.emit/` first) — so every
 * file under `.emit/**\/*.js` already carries an implicit "when was this
 * emitted" stamp in its own mtime, and every one of those mtimes lands
 * within the same short `tsc` invocation. Comparing the max mtime across
 * `spike/src/**\/*.ts` (every file that can affect the emitted output,
 * including `ambient.d.ts` — `*.ts` matches it too, it is a legitimate
 * compilation input) against the mtime of one representative emitted file
 * (the exact module the test loads, `.emit/src/ndarray.js`) is sufficient:
 * if `.emit/` was produced strictly after the newest source edit, freshness
 * holds; if any source file was touched afterward (a stale `.emit/` left
 * over from a previous wrapper run, or a hand-rolled `tsc` invocation that
 * bypassed the wrapper entirely), the source's mtime will be strictly newer
 * and this throws.
 *
 * Uses the same "local cast over the ambient `node:fs/promises` import"
 * technique as `spike/tests-runtime/test-scripts-guard.test.ts` (which
 * needs `readdir`/`stat`, not just the ambiently-declared `readFile`) —
 * scoped to this file only, `spike/src/ambient.d.ts` stays untouched
 * (D-V3.5: this fix is test-only).
 */

interface FsPromisesExtra {
  readdir(url: URL, options: { recursive: true }): Promise<string[]>;
  stat(url: URL): Promise<{ readonly mtimeMs: number }>;
}

async function fsp(): Promise<FsPromisesExtra> {
  return (await import("node:fs/promises")) as unknown as FsPromisesExtra;
}

/** Max mtime (ms since epoch) across every `*.ts` file under `srcDirUrl`
 * (recursive) — `srcDirUrl` must be a directory URL (trailing slash). */
async function computeMaxSrcMtimeMs(srcDirUrl: URL): Promise<number> {
  const fs = await fsp();
  const entries = await fs.readdir(srcDirUrl, { recursive: true });
  const tsFiles = entries.filter((e) => e.endsWith(".ts"));
  let max = 0;
  for (const rel of tsFiles) {
    const { mtimeMs } = await fs.stat(new URL(rel, srcDirUrl));
    if (mtimeMs > max) max = mtimeMs;
  }
  return max;
}

/**
 * Throws with a clear, actionable message if `emitDirUrl` (the tsc-emitted
 * product tree the browser test serves) is missing or older than the
 * newest file under `srcDirUrl`. `referenceRelPath` is the one emitted file
 * used as the "when was this emitted" witness — any file works (they all
 * land within the same `tsc` invocation), so this uses the exact module the
 * test suite loads (`src/ndarray.js`) rather than introducing a second,
 * unrelated file into the freshness contract.
 */
export async function assertEmitFresh(emitDirUrl: URL, srcDirUrl: URL, referenceRelPath: string): Promise<void> {
  const fs = await fsp();
  const maxSrcMtime = await computeMaxSrcMtimeMs(srcDirUrl);

  let emitMtime: number;
  try {
    const stat = await fs.stat(new URL(referenceRelPath, emitDirUrl));
    emitMtime = stat.mtimeMs;
  } catch {
    throw new Error(
      `spike/tests-browser/.emit is missing (expected to find ${referenceRelPath}) — ` +
        "run `pnpm test:browser`, do not invoke playwright directly against an absent emit.",
    );
  }

  if (maxSrcMtime > emitMtime) {
    throw new Error(
      "spike/tests-browser/.emit is stale: a file under spike/src/**/*.ts was modified after the last " +
        "emission — run `pnpm test:browser` (it always re-emits fresh), do not invoke playwright directly " +
        "against a stale .emit.",
    );
  }
}
