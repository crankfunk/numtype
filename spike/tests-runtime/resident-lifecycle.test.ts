/**
 * Kern 02 lifecycle tests: the hard part of residency per the spec.
 * Covers everything the differential suite (`resident.test.ts`) doesn't:
 * dispose semantics, use-after-dispose, leak-free error paths, and the
 * deterministic leak-plateau bound. The GC-backstop test lives separately
 * in `resident-gc.test.ts` (needs `--expose-gc`, see that file's header).
 *
 * Each test below uses its own freshly-initialized `core` (a fresh
 * `WebAssembly.Instance`, fresh linear memory) so that one test's
 * allocation history never taints another's `byteLength` observations.
 */
import assert from "node:assert";
import { test } from "node:test";
import { initCore } from "../src/wasm/loader.ts";
import { WNDArray } from "../src/wasm/resident.ts";

// --- use-after-dispose: throws, names the op, never touches WASM memory ---
test("use-after-dispose: toArray throws naming the op", async () => {
  const core = await initCore();
  const a = WNDArray.fromArray(core, [2, 3], [1, 2, 3, 4, 5, 6]);
  a.dispose();
  assert.throws(() => a.toArray(), /WNDArray\.toArray:.*disposed/);
});

test("use-after-dispose: toNestedArray throws naming the op", async () => {
  const core = await initCore();
  const a = WNDArray.fromArray(core, [2, 3], [1, 2, 3, 4, 5, 6]);
  a.dispose();
  assert.throws(() => a.toNestedArray(), /WNDArray\.toNestedArray:.*disposed/);
});

test("use-after-dispose: add throws naming the op when `this` is disposed", async () => {
  const core = await initCore();
  const a = WNDArray.fromArray(core, [2, 3], [1, 2, 3, 4, 5, 6]);
  const b = WNDArray.fromArray(core, [2, 3], [1, 1, 1, 1, 1, 1]);
  a.dispose();
  assert.throws(() => a.add(b), /WNDArray\.add:.*disposed/);
  b.dispose();
});

test("use-after-dispose: add throws naming the op when `other` is disposed", async () => {
  const core = await initCore();
  const a = WNDArray.fromArray(core, [2, 3], [1, 2, 3, 4, 5, 6]);
  const b = WNDArray.fromArray(core, [2, 3], [1, 1, 1, 1, 1, 1]);
  b.dispose();
  assert.throws(() => a.add(b), /WNDArray\.add:.*disposed/);
  a.dispose();
});

test("use-after-dispose: matmul/sum/transpose all throw naming their op", async () => {
  const core = await initCore();
  const a = WNDArray.fromArray(core, [2, 2], [1, 2, 3, 4]);
  const b = WNDArray.fromArray(core, [2, 2], [1, 0, 0, 1]);
  a.dispose();
  assert.throws(() => a.matmul(b), /WNDArray\.matmul:.*disposed/);
  assert.throws(() => a.sum(), /WNDArray\.sum:.*disposed/);
  assert.throws(() => a.sum(0), /WNDArray\.sum:.*disposed/);
  assert.throws(() => a.transpose(), /WNDArray\.transpose:.*disposed/);
  b.dispose();
});

// --- double-dispose is a safe no-op -----------------------------------------
test("double-dispose is a safe no-op (does not double-free)", async () => {
  const core = await initCore();
  const a = WNDArray.fromArray(core, [4, 4], new Array(16).fill(1));
  assert.strictEqual(a.disposed, false);
  a.dispose();
  assert.strictEqual(a.disposed, true);
  // A second (and third) dispose must not throw and must not attempt to
  // free the same allocation again (verified indirectly: if it did
  // double-free, the WASM allocator would be corrupted and a subsequent
  // fresh allocation would very likely crash or produce garbage — the
  // `zeros` call below exercises exactly that).
  a.dispose();
  a.dispose();
  assert.strictEqual(a.disposed, true);

  const check = WNDArray.zeros(core, [4, 4]);
  try {
    assert.deepStrictEqual(Array.from(check.toArray()), new Array(16).fill(0));
  } finally {
    check.dispose();
  }
});

// --- an empty (zero-element) array is NOT "already disposed" ---------------
// `ptr === 0` is also nt_alloc's legitimate "zero-byte allocation" sentinel
// (spec: a zero-length shape/data argument is valid, never a failure) — the
// `disposed` flag, not `ptr`, must be the sole source of truth.
test("a legitimately empty array (shape with a 0 dim) is not disposed at construction", async () => {
  const core = await initCore();
  const empty = WNDArray.fromArray(core, [0, 3], []);
  assert.strictEqual(empty.disposed, false);
  assert.deepStrictEqual(Array.from(empty.toArray()), []);
  empty.dispose();
  assert.strictEqual(empty.disposed, true);
});

// --- failing op: leak-free, inputs stay valid -------------------------------
// Shapes typed as plain `number[]` (not literal tuples) so `S`/`B` infer as
// dynamic rank and `Guard<...>` degrades to the plain `WNDArray<B>` type —
// deliberately bypassing the compile-time shape guard so the mismatch is
// caught at RUNTIME instead (exactly what this test wants to exercise; the
// compile-time path is already what `assert.throws` would otherwise never
// even reach, since `tsc` would reject the call before it runs — same
// technique `negative-paths.test.ts` uses for the v1 backend).
test("a failing op (shape mismatch) leaves both inputs usable afterward", async () => {
  const core = await initCore();
  const aShape: number[] = [2, 3];
  const bShape: number[] = [2, 4];
  const a = WNDArray.fromArray(core, aShape, [1, 2, 3, 4, 5, 6]);
  const b = WNDArray.fromArray(core, bShape, new Array(8).fill(0));

  assert.throws(() => a.add(b), (err: unknown) => {
    if (!(err instanceof Error)) return false;
    assert.ok(err.message.includes("2,3"), `should name shape [2,3]: ${err.message}`);
    assert.ok(err.message.includes("2,4"), `should name shape [2,4]: ${err.message}`);
    return true;
  });

  // inputs remain fully valid and usable — not disposed, not corrupted
  assert.strictEqual(a.disposed, false);
  assert.strictEqual(b.disposed, false);
  assert.deepStrictEqual(Array.from(a.toArray()), [1, 2, 3, 4, 5, 6]);
  assert.deepStrictEqual(Array.from(b.toArray()), new Array(8).fill(0));

  a.dispose();
  b.dispose();
});

test("matmul inner-dim mismatch leaves both inputs usable afterward", async () => {
  const core = await initCore();
  const aShape: number[] = [2, 3];
  const bShape: number[] = [4, 2];
  const a = WNDArray.fromArray(core, aShape, [1, 2, 3, 4, 5, 6]);
  const b = WNDArray.fromArray(core, bShape, new Array(8).fill(1));

  assert.throws(() => a.matmul(b), /inner dimensions 3 and 4/);

  assert.strictEqual(a.disposed, false);
  assert.strictEqual(b.disposed, false);
  assert.deepStrictEqual(Array.from(a.toArray()), [1, 2, 3, 4, 5, 6]);

  a.dispose();
  b.dispose();
});

test("failing ops leak nothing: repeated failing add calls do not grow WASM memory", async () => {
  const core = await initCore();
  const aShape: number[] = [2, 3];
  const bShape: number[] = [2, 4];
  const a = WNDArray.fromArray(core, aShape, [1, 2, 3, 4, 5, 6]);
  const b = WNDArray.fromArray(core, bShape, new Array(8).fill(0));

  // warmup: let the allocator reach its steady-state block layout for this
  // (shape-scratch + would-be-output) allocation pattern first.
  for (let i = 0; i < 20; i++) {
    assert.throws(() => a.add(b));
  }
  const byteLengthAfterWarmup = core.memory.buffer.byteLength;

  for (let i = 0; i < 500; i++) {
    assert.throws(() => a.add(b));
  }
  const byteLengthAfter500More = core.memory.buffer.byteLength;

  assert.strictEqual(
    byteLengthAfter500More,
    byteLengthAfterWarmup,
    `expected no growth from 500 more failing calls: ${byteLengthAfterWarmup} -> ${byteLengthAfter500More}`,
  );

  a.dispose();
  b.dispose();
});

// --- kernel-status failure path (status != 0 AFTER allocation) --------------
// The shape-mismatch tests above are all caught by the TS-side
// pre-validation BEFORE anything is allocated — they never reach the kernel.
// This test exercises the deeper spec path "kernel returns non-zero status
// -> the already-allocated output buffer is freed before throwing": a
// rank-33 shape passes every TS-side check (broadcasting handles any rank)
// but exceeds the kernel's MAX_RANK = 32, so nt_add itself rejects it with
// status 2 — after the output buffer was allocated. The would-be output is
// deliberately large (128KB) so that a missing free would move the
// page-granular byteLength within 500 calls (a leaked 8-byte output never
// would). Coverage gap found in independent verification of Kern 02.
test("kernel-status failure (rank > 32) frees the output buffer and leaves inputs usable", async () => {
  const core = await initCore();
  const shape33: number[] = [...new Array<number>(31).fill(1), 128, 128];
  const bShape: number[] = [1];
  const a = WNDArray.fromArray(core, shape33, new Array<number>(128 * 128).fill(7));
  const b = WNDArray.fromArray(core, bShape, [35]);

  for (let i = 0; i < 20; i++) {
    assert.throws(() => a.add(b), /status 2/);
  }
  const byteLengthAfterWarmup = core.memory.buffer.byteLength;

  for (let i = 0; i < 500; i++) {
    assert.throws(() => a.add(b), /status 2/);
  }
  assert.strictEqual(
    core.memory.buffer.byteLength,
    byteLengthAfterWarmup,
    "expected no growth across 500 kernel-rejected calls — output buffer must be freed on the status!=0 path",
  );

  // inputs untouched and usable
  assert.strictEqual(a.disposed, false);
  assert.strictEqual(b.disposed, false);
  const aBack = a.toArray();
  assert.strictEqual(aBack.length, 128 * 128);
  assert.strictEqual(aBack[0], 7);
  assert.deepStrictEqual(Array.from(b.toArray()), [35]);

  a.dispose();
  b.dispose();
});

// --- leak plateau (deterministic) ------------------------------------------
// >=1000 op+dispose cycles (a mixed add/matmul/transpose/sum chain, varying
// operand sizes to stress the allocator's free-list reuse across
// heterogeneous block sizes) must reach a `core.memory.buffer.byteLength`
// plateau: identical at cycle 100 (already past the initial warmup) and at
// cycle 1000. Empirically verified before writing this assertion (see
// docs/kern-02-ergebnisse.md): with full dispose discipline, WASM's default
// allocator (dlmalloc, per crates/core/src/abi.rs's doc comment) reuses
// freed blocks perfectly for this workload — byteLength does not grow AT
// ALL past its very first plateau point, not just "eventually stabilizes".
test("leak plateau: >=1000 op+dispose cycles reach an exact byteLength plateau", async () => {
  const core = await initCore();
  const sizes = [4, 8, 16, 32, 64, 8, 4, 128, 16, 32, 2, 20, 6];

  function runCycle(i: number): void {
    const n = sizes[i % sizes.length] ?? 8;
    const shape = [n, n];
    const aVals = new Array(n * n).fill(1);
    const bVals = new Array(n * n).fill(2);

    const a = WNDArray.fromArray(core, shape, aVals);
    const b = WNDArray.fromArray(core, shape, bVals);
    const c = a.add(b);
    const e = WNDArray.fromArray(core, shape, aVals);
    const d = c.matmul(e);
    const f = d.transpose();
    const g = f.sum(0);

    a.dispose();
    b.dispose();
    c.dispose();
    e.dispose();
    d.dispose();
    f.dispose();
    g.dispose();
  }

  for (let i = 0; i < 100; i++) runCycle(i);
  const byteLengthAt100 = core.memory.buffer.byteLength;

  for (let i = 100; i < 1000; i++) runCycle(i);
  const byteLengthAt1000 = core.memory.buffer.byteLength;

  assert.strictEqual(
    byteLengthAt1000,
    byteLengthAt100,
    `expected byteLength to plateau: cycle 100 = ${byteLengthAt100}, cycle 1000 = ${byteLengthAt1000}`,
  );
});
