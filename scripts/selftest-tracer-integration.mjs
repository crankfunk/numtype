// Integration self-tests 12 and 13 of docs/view-test-klassifikation-spec.md (D4).
//
// These two cannot be checked against a synthetic ledger: they are claims about
// the TRACER's behaviour under a real `node --test` run, so they need one.
// Both were added because the adversarial spec verifier showed that the first
// ten self-tests would NOT have caught the corresponding failure modes.
//
//   12  `.shape` alone must NOT register as readback. An earlier draft used
//       `.shape` as the provenance signal; it is a plain data property that the
//       library itself reads ~4x per toArray(), so the signal would have been
//       pure noise. This test pins the narrowing — as a DISCRIMINATING PAIR:
//       one case reads only `.shape` (expect 0), its twin calls `toArray()`
//       (expect > 0). A single negative case would pass even if readback were
//       broken and always 0.
//   13  A test case that spawns a nested Node child process must not corrupt the
//       ledger. NODE_OPTIONS is inherited, so the child loads this preload too —
//       the real corpus does exactly this (the real-tsc diagnostic-pin tests
//       shell out to `tsc`).
//
//   node scripts/selftest-tracer-integration.mjs

import { mkdtempSync, mkdirSync, writeFileSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { spawnSync } from "node:child_process";

const REPO = new URL("..", import.meta.url).pathname.replace(/\/$/, "");
const PRELOAD = `${REPO}/scripts/trace-view-tests.mjs`;
const RESIDENT = `${REPO}/spike/src/wasm/resident.ts`;
const LOADER = `${REPO}/spike/src/wasm/loader.ts`;

const work = mkdtempSync(`${tmpdir()}/nt-tracer-selftest-`);
const outDir = `${work}/ledger`;
mkdirSync(outDir, { recursive: true });

// The fixture lives OUTSIDE spike/tests-runtime on purpose: that directory is
// guarded by test-scripts-guard.test.ts, which fails on unregistered test files.
const fixture = `${work}/fixture.test.ts`;
writeFileSync(
  fixture,
  `import { test } from "node:test";
import assert from "node:assert";
import { spawnSync } from "node:child_process";
import { initCore } from ${JSON.stringify(LOADER)};
import { WNDArray } from ${JSON.stringify(RESIDENT)};

const core = await initCore();

// --- self-test 12, negative half: build a genuinely non-contiguous view and
// read ONLY .shape. Must yield readback === 0.
test("SELFTEST12-shape-only", () => {
  const w = WNDArray.fromArray(core, [3, 4], [0,1,2,3,4,5,6,7,8,9,10,11]);
  const v = w.transpose();
  assert.strictEqual(v.shape.length, 2);
  const s = v.shape;
  assert.ok(s[0] === 4);
  v.dispose();
  w.dispose();
});

// --- self-test 12, positive half: same view, but read it back via toArray().
// Must yield readback > 0. Without this half, a permanently-zero readback would
// pass the negative half and look like a working signal.
test("SELFTEST12-toarray", () => {
  const w = WNDArray.fromArray(core, [3, 4], [0,1,2,3,4,5,6,7,8,9,10,11]);
  const v = w.transpose();
  const data = v.toArray();
  assert.strictEqual(data.length, 12);
  v.dispose();
  w.dispose();
});

// --- self-test 13: spawn a nested Node child process from inside a test case.
test("SELFTEST13-nested-subprocess", () => {
  const r = spawnSync(process.execPath, ["-e", "process.exit(0)"], { encoding: "utf8" });
  assert.strictEqual(r.status, 0);
});
`,
);

const env = {
  ...process.env,
  NT_TRACE_OUT_DIR: outDir,
  NT_TRACE_MODE: "trace",
  NODE_OPTIONS: `--import file://${PRELOAD}`,
};

const run = spawnSync(process.execPath, ["--test", fixture], { cwd: REPO, env, encoding: "utf8" });
console.log(`fixture run exit=${run.status}`);
if (run.status !== 0) {
  console.error(run.stdout);
  console.error(run.stderr);
  console.error("FAIL: fixture run did not pass — cannot evaluate self-tests 12/13");
  process.exit(1);
}

const cases = [];
for (const f of readdirSync(outDir)) {
  if (!f.startsWith("ledger-") || !f.endsWith(".jsonl")) continue;
  for (const line of readFileSync(`${outDir}/${f}`, "utf8").split("\n")) {
    if (!line.trim()) continue;
    const rec = JSON.parse(line);
    if (rec.ev === "case") cases.push(rec);
  }
}

const failures = [];
const check = (id, what, got, want) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) failures.push(`${id}: ${what} -> got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
  console.log(`${ok ? "ok  " : "FAIL"} ${id}  ${what}`);
};

const byName = (n) => cases.filter((c) => c.name === n);

// 12 — the discriminating pair
{
  const neg = byName("SELFTEST12-shape-only");
  const pos = byName("SELFTEST12-toarray");
  check(12, "shape-only case appears exactly once", neg.length, 1);
  check(12, "toArray case appears exactly once", pos.length, 1);
  check(12, "shape-only: view WAS recorded as non-trivial", neg[0]?.nontrivial, true);
  check(12, "shape-only: readback === 0", neg[0]?.readback, 0);
  check(12, "toArray: readback > 0", (pos[0]?.readback ?? 0) > 0, true);
}

// 13 — nested subprocess must not corrupt the ledger
{
  const nested = byName("SELFTEST13-nested-subprocess");
  check(13, "nested-subprocess case appears exactly once", nested.length, 1);
  check(13, "total case count is exactly 3", cases.length, 3);
  const keys = new Set(cases.map((c) => `${c.file} ${c.name}`));
  check(13, "no duplicate case identities", keys.size, cases.length);
}

rmSync(work, { recursive: true, force: true });

if (failures.length > 0) {
  console.error(`\n${failures.length} INTEGRATION SELF-TEST FAILURE(S):`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("\nintegration self-tests 12/13 green");
