/**
 * Guard against the "explicit test-file-list" footgun: `package.json`'s
 * `test:core`/`test:resident`/`test:resident:gc`/`test:threaded` scripts
 * each run `node --test <explicit file list>` rather than a glob, so a new
 * test file dropped into `spike/tests-runtime/` that nobody adds to one of
 * those lists simply never runs — silently, with no error anywhere. This
 * has nearly happened more than once. This test parses `package.json` and
 * checks three invariants:
 *
 *   (a) every `spike/tests-runtime/*.test.ts` file is listed in EXACTLY ONE
 *       of `test:core` / `test:resident` / `test:threaded` (Kern 06: widened
 *       from two lists to three — so every file runs somewhere, exactly
 *       once at that granularity — `test:resident:gc` is a deliberate
 *       re-run variant, see (c));
 *   (b) every file referenced by `test:core`, `test:resident`,
 *       `test:resident:gc`, or `test:threaded` actually exists on disk
 *       (catches stale/renamed entries left behind);
 *   (c) `test:resident:gc`'s file list is a subset of `test:resident`'s —
 *       `:gc` intentionally re-runs a subset of resident tests under
 *       `--expose-gc`, so overlap there is allowed, not a violation of (a).
 *
 * No product dependencies, no WASM import — this test only reads
 * `package.json` and the `spike/tests-runtime/` directory listing, so it's
 * a fast, standalone sanity check (see the negative-demonstration procedure
 * this hardening's report describes: a temporary unregistered `.test.ts`
 * file must make this test fail, naming the file).
 *
 * Only `node:fs/promises`' `readFile` is declared in `spike/src/
 * ambient.d.ts` (no `@types/node` dependency, per project constraint); this
 * file needs `readdir`/`stat` too, which the real Node module has but the
 * ambient declaration doesn't expose. Rather than widen the shared ambient
 * shim for one test file, the extra methods are accessed through a local
 * cast on the dynamically-imported module — scoped to this file only. Repo
 * paths are resolved from `import.meta.url` (never CWD) using `URL`
 * relative resolution, exactly like `spike/src/wasm/loader.ts`'s
 * `WASM_URL` — no `node:path`/`node:url` needed.
 */
import { test } from "node:test";
import assert from "node:assert";

// This file lives at <repo-root>/spike/tests-runtime/test-scripts-guard.test.ts.
// `new URL(".", import.meta.url)` = this file's directory;
// `new URL("../../", import.meta.url)` = two directories up = repo root.
const testsRuntimeDirUrl = new URL(".", import.meta.url);
const repoRootUrl = new URL("../../", import.meta.url);

interface FsPromisesExtra {
  readFile(url: URL, encoding: string): Promise<string>;
  readdir(url: URL): Promise<string[]>;
  stat(url: URL): Promise<unknown>;
}

const fsp = (await import("node:fs/promises")) as unknown as FsPromisesExtra;

async function fileExists(url: URL): Promise<boolean> {
  try {
    await fsp.stat(url);
    return true;
  } catch {
    return false;
  }
}

interface PackageJson {
  readonly scripts?: Record<string, string>;
}

async function readPackageJson(): Promise<PackageJson> {
  const raw = await fsp.readFile(new URL("package.json", repoRootUrl), "utf8");
  return JSON.parse(raw) as PackageJson;
}

/** Extract every `*.test.ts` token from a script string, in order. Scripts
 * are `node --test <files...>` (possibly preceded by `pnpm build:wasm &&`
 * etc.) — splitting on whitespace and keeping tokens that end in
 * `.test.ts` is robust to whatever comes before the file list. */
function extractTestFileTokens(script: string): string[] {
  return script.split(/\s+/).filter((tok) => tok.endsWith(".test.ts"));
}

/** A listed token is a repo-relative path like
 * `spike/tests-runtime/add.test.ts`; reduce to just the basename for
 * comparison against the directory listing. */
function basename(token: string): string {
  return token.split("/").pop() ?? token;
}

const pkg = await readPackageJson();
const scripts = pkg.scripts ?? {};

const testCoreScript = scripts["test:core"];
const testResidentScript = scripts["test:resident"];
const testResidentGcScript = scripts["test:resident:gc"];
const testThreadedScript = scripts["test:threaded"];

assert.ok(testCoreScript, `package.json is missing a "test:core" script`);
assert.ok(testResidentScript, `package.json is missing a "test:resident" script`);
assert.ok(testResidentGcScript, `package.json is missing a "test:resident:gc" script`);
assert.ok(testThreadedScript, `package.json is missing a "test:threaded" script`);

const testCoreFiles = extractTestFileTokens(testCoreScript);
const testResidentFiles = extractTestFileTokens(testResidentScript);
const testResidentGcFiles = extractTestFileTokens(testResidentGcScript);
const testThreadedFiles = extractTestFileTokens(testThreadedScript);

test("every spike/tests-runtime/*.test.ts file is listed in exactly one of test:core / test:resident / test:threaded", async () => {
  const entries = await fsp.readdir(testsRuntimeDirUrl);
  const onDisk = entries.filter((f: string) => f.endsWith(".test.ts"));
  assert.ok(onDisk.length > 0, `no *.test.ts files found under ${testsRuntimeDirUrl.href} — readdir likely misconfigured`);

  const coreBasenames = new Set(testCoreFiles.map(basename));
  const residentBasenames = new Set(testResidentFiles.map(basename));
  const threadedBasenames = new Set(testThreadedFiles.map(basename));

  const unregistered: string[] = [];
  const inMultiple: string[] = [];
  for (const file of onDisk) {
    const memberships = [coreBasenames.has(file), residentBasenames.has(file), threadedBasenames.has(file)];
    const count = memberships.filter(Boolean).length;
    if (count === 0) {
      unregistered.push(file);
    } else if (count > 1) {
      inMultiple.push(file);
    }
  }

  assert.strictEqual(
    unregistered.length,
    0,
    `these test files exist in spike/tests-runtime/ but are not listed in test:core, test:resident, or ` +
      `test:threaded (they silently never run): ${unregistered.join(", ")}`,
  );
  assert.strictEqual(
    inMultiple.length,
    0,
    `these test files are listed in MORE THAN ONE of test:core / test:resident / test:threaded (should be exactly one): ${inMultiple.join(", ")}`,
  );
});

test("every file referenced by test:core / test:resident / test:resident:gc / test:threaded exists on disk", async () => {
  const allReferenced = [
    ...testCoreFiles.map((f) => ({ list: "test:core", file: f })),
    ...testResidentFiles.map((f) => ({ list: "test:resident", file: f })),
    ...testResidentGcFiles.map((f) => ({ list: "test:resident:gc", file: f })),
    ...testThreadedFiles.map((f) => ({ list: "test:threaded", file: f })),
  ];

  const missing: { list: string; file: string }[] = [];
  for (const ref of allReferenced) {
    const exists = await fileExists(new URL(ref.file, repoRootUrl));
    if (!exists) missing.push(ref);
  }

  assert.strictEqual(
    missing.length,
    0,
    `these test files are referenced in package.json but do not exist on disk: ` +
      missing.map(({ list, file }) => `${file} (in ${list})`).join(", "),
  );
});

test("test:resident:gc's file list is a subset of test:resident's", () => {
  const residentBasenames = new Set(testResidentFiles.map(basename));
  const extra = testResidentGcFiles.filter((f) => !residentBasenames.has(basename(f)));

  assert.strictEqual(
    extra.length,
    0,
    `test:resident:gc references files not present in test:resident (gc is meant to be a ` +
      `re-run subset): ${extra.join(", ")}`,
  );
});
