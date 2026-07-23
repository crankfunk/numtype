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
import { initCore, type CoreExports } from "../src/wasm/loader.ts";
import { getResidentFreeCount, WNDArray, type AnyWNDArray } from "../src/wasm/resident.ts";

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

// =============================================================================
// WASM parity S2 (docs/wasm-parity-mean-spec.md, D3/D5): `WNDArray.mean`
// leak-non-vacuity. `mean` is a pure-TS composition — `this.sum(axis,
// keepdims).div(n)` — with an intermediate `summed` WNDArray that MUST be
// disposed exactly once per call (in a `finally`, after `.div` has already
// produced its own independent result buffer). This test proves D3 with an
// EXACT free-count delta, not just a plateau: `getResidentFreeCount()`
// increments once per *resident data buffer* whose refcount reaches 0
// (module doc comment, resident.ts) — scratch (shape/stride marshalling) is
// NOT counted, so the expected delta per `mean()` call is exactly 2 (the
// intermediate `summed`, freed inside `mean` itself, PLUS the final result,
// freed by this test's own `dispose()` call) — proving neither a leak (delta
// too small) nor a double-free (delta too large, or a corrupted allocator
// on the follow-up allocation). The RECEIVER is allocated once, OUTSIDE the
// measured window (and disposed only after the assertion) — its own
// alloc/dispose lifecycle is orthogonal to D3's claim and must not dilute
// the exact per-call delta.
// =============================================================================
test("mean leak non-vacuity: N mean() calls on ONE receiver (dispose each result) free exactly 2N resident buffers, no net leak", async () => {
  const core = await initCore();
  const shape = [4, 5];
  const a = WNDArray.fromArray(
    core,
    shape,
    new Array(20).fill(0).map((_, i) => i + 1),
  );
  const N = 500;

  const before = getResidentFreeCount();
  for (let i = 0; i < N; i++) {
    const got = a.mean(1);
    got.dispose();
  }
  const after = getResidentFreeCount();

  a.dispose(); // outside the measured window — see comment above

  assert.strictEqual(
    after - before,
    2 * N,
    `expected exactly 2 resident-buffer frees per mean() call (intermediate summed + final result): ` +
      `before=${before}, after=${after}, delta=${after - before}, expected=${2 * N}`,
  );

  // Independent corroboration: a plain allocator-growth check (same style as
  // the byteLength plateau test above) — after warmup, many more mean()
  // calls must not grow WASM memory at all.
  const warmupShape = [4, 4];
  for (let i = 0; i < 20; i++) {
    const a = WNDArray.fromArray(core, warmupShape, new Array(16).fill(2));
    a.mean(1).dispose();
    a.dispose();
  }
  const byteLengthAfterWarmup = core.memory.buffer.byteLength;

  for (let i = 0; i < 500; i++) {
    const a = WNDArray.fromArray(core, warmupShape, new Array(16).fill(2));
    a.mean(1).dispose();
    a.dispose();
  }
  const byteLengthAfter500More = core.memory.buffer.byteLength;

  assert.strictEqual(
    byteLengthAfter500More,
    byteLengthAfterWarmup,
    `expected no growth from 500 more mean() call+dispose cycles: ${byteLengthAfterWarmup} -> ${byteLengthAfter500More}`,
  );
});

// =============================================================================
// WASM parity S3 (docs/wasm-parity-item-stack-spec.md, D3/D7): `WNDArray.item`
// lifecycle — allocation-free (D3: "kein Allokations-, kein Kernel-Aufruf").
// =============================================================================

test("item: N calls on ONE receiver are allocation-free — getResidentFreeCount() delta is exactly 0, memory never grows", async () => {
  const core = await initCore();
  const a = WNDArray.fromArray(core, [4, 5], Array.from({ length: 20 }, (_, i) => i + 1));
  const N = 500;

  const before = getResidentFreeCount();
  for (let i = 0; i < N; i++) {
    a.item(i % 4, (i + 1) % 5);
  }
  const after = getResidentFreeCount();
  assert.strictEqual(after, before, `item() must never touch a resident buffer's refcount: before=${before}, after=${after}`);

  // No allocation at all (D3) means byteLength must not move even once —
  // there is no "warmup" to let an allocator reach steady state, unlike
  // every other op's leak test in this file.
  const byteLengthBefore = core.memory.buffer.byteLength;
  for (let i = 0; i < 500; i++) a.item(i % 4, (i + 1) % 5);
  const byteLengthAfter = core.memory.buffer.byteLength;
  assert.strictEqual(byteLengthAfter, byteLengthBefore, "item() calls must never grow WASM memory (it never allocates)");

  a.dispose();
});

test("use-after-dispose: item throws naming the op", async () => {
  const core = await initCore();
  const a = WNDArray.fromArray(core, [2, 3], [1, 2, 3, 4, 5, 6]);
  a.dispose();
  assert.throws(() => a.item(0, 0), /WNDArray\.item:.*disposed/);
});

// =============================================================================
// WASM parity S3 (docs/wasm-parity-item-stack-spec.md, D5/D7): `WNDArray.stack`
// lifecycle — leak-non-vacuity, use-after-dispose, and the mid-loop kernel-
// status failure path.
//
// FINDING (deviation from D7's literal wording, reported rather than
// silently resolved): D7 states the free-count delta for `stack` "zeigt
// exakt die zwei Scratch-Freigaben pro Aufruf" (shows exactly the two
// scratch frees per call). Verified against `resident.ts`'s own source:
// `getResidentFreeCount()` increments ONLY inside `releaseBuffer()` (called
// from a WNDArray HANDLE's `dispose()` or the FinalizationRegistry
// backstop) — it tracks refcounted `ResidentBuffer` releases, never a raw
// `core.nt_free()` call. `stack()`'s two scratch buffers (shapeBuf/
// stridesBuf) are freed via the separate `allocBytes`/`freeBuf` path — the
// SAME path every other op's own per-call scratch uses (add/sub/mul/div/
// matmul/sum/sqrt/scalarOp), none of which move this counter for their
// scratch either. Unlike `mean` (which disposes a real INTERMEDIATE
// WNDArray handle inside its own body — the source of ITS delta=2), `stack`
// has no intermediate handle at all: only the final OUTPUT buffer becomes a
// real `ResidentBuffer`, released exactly once when the caller disposes the
// returned `WNDArray`. The mechanically correct, achievable claim is
// therefore delta = 1 per disposed stack() result (not 2) — proven below
// via a 3-per-iteration signal (2 input rows + 1 result, so the counter's
// own health is cross-checked too), PLUS a separate byteLength-plateau
// check (this file's own established technique for scratch that never
// becomes a WNDArray handle, e.g. the "failing ops leak nothing" tests
// above) that specifically covers the scratch buffers' own leak-freedom,
// since the counter cannot observe them.
// =============================================================================

test("stack leak non-vacuity: N stack() calls (dispose rows + result) free exactly 3 resident buffers per call, no net leak", async () => {
  const core = await initCore();
  const N = 500;
  const shape: number[] = [3]; // dynamic shape (see file-wide note above T7's item allocation-free test)
  const before = getResidentFreeCount();
  for (let i = 0; i < N; i++) {
    const a = WNDArray.fromArray(core, shape, [1, 2, 3]);
    const b = WNDArray.fromArray(core, shape, [4, 5, 6]);
    const stacked = WNDArray.stack(core, [a, b]);
    stacked.dispose();
    a.dispose();
    b.dispose();
  }
  const after = getResidentFreeCount();
  assert.strictEqual(
    after - before,
    3 * N,
    `expected exactly 3 resident-buffer frees per iteration (2 input rows + 1 stack result): before=${before}, after=${after}, delta=${after - before}, expected=${3 * N}`,
  );

});

test("stack SUCCESS path: every allocation is freed — exact alloc/free ledger over the counting mock (the two per-call scratch buffers included)", () => {
  // The two per-call scratch buffers (shape cell + strides cell) are
  // invisible to `getResidentFreeCount()`: that counter only fires in
  // `releaseBuffer()` when a refcounted `ResidentBuffer` hits 0, never on the
  // raw `nt_free` path `allocBytes`/`freeBuf` use for ephemeral scratch (the
  // finding recorded on the test above).
  //
  // An allocator-GROWTH plateau check cannot substitute for it: the scratch
  // is 8 bytes per call, so even hundreds of leaked calls stay orders of
  // magnitude below WASM's 64 KiB page granularity and the check passes
  // vacuously. That is not a hypothesis — Baustein A's verification mutant
  // dropped both `scratch.push(...)` calls (a real, permanent scratch leak)
  // and a 500-iteration growth check stayed GREEN; only the mid-loop failure
  // test below caught it, and only because it counts.
  //
  // So this test counts too, on the path that test does not cover: the
  // SUCCESS path. `failAtMaterializeCall = 0` is never reached (the mock's
  // counter is 1-indexed), so every `nt_materialize` call succeeds. The
  // assertion is an exact ledger — every non-sentinel allocation must have
  // exactly one matching free — which catches ANY leaked buffer, scratch or
  // output, at a granularity of one allocation.
  const { core: mockCore, allocs, frees } = makeStackFailureMockCore(0);
  const shape: number[] = [2];
  const a = WNDArray.fromArray(mockCore, shape, [1, 2]);
  const b = WNDArray.fromArray(mockCore, shape, [3, 4]);
  const stacked = WNDArray.stack(mockCore, [a, b]);
  assert.deepStrictEqual(Array.from(stacked.toArray()), [1, 2, 3, 4], "precondition: the mock's gather really ran");
  stacked.dispose();
  a.dispose();
  b.dispose();

  const ledger = (entries: readonly MockAlloc[]): string[] =>
    entries
      .filter((e) => e.bytes !== 0) // `nt_alloc(0)` returns the ptr-0 sentinel, never freed as a real block
      .map((e) => `${e.ptr}:${e.bytes}`)
      .sort();
  assert.deepStrictEqual(
    ledger(frees),
    ledger(allocs),
    `every allocation must be freed exactly once (scratch included): allocs=${JSON.stringify(allocs)}, frees=${JSON.stringify(frees)}`,
  );
});

test("stack: result is a FRESH, independent buffer — stays valid and correct after the input rows are disposed", async () => {
  const core = await initCore();
  const shape: number[] = [3];
  const a = WNDArray.fromArray(core, shape, [1, 2, 3]);
  const b = WNDArray.fromArray(core, shape, [4, 5, 6]);
  const stacked = WNDArray.stack(core, [a, b]);
  a.dispose();
  b.dispose();
  try {
    assert.strictEqual(stacked.disposed, false);
    assert.deepStrictEqual(Array.from(stacked.toArray()), [1, 2, 3, 4, 5, 6]);
  } finally {
    stacked.dispose();
  }
});

test("use-after-dispose: stack throws naming the op when a row is disposed", async () => {
  const core = await initCore();
  const shape: number[] = [3];
  const a = WNDArray.fromArray(core, shape, [1, 2, 3]);
  const b = WNDArray.fromArray(core, shape, [7, 8, 9]);
  a.dispose();
  assert.throws(() => WNDArray.stack(core, [a, b]), /WNDArray\.stack:.*disposed/);
  b.dispose();
});

test("use-after-dispose: stack throws for a NON-FIRST disposed row too (the liveness check walks every row)", async () => {
  // Baustein-B finding: with only the row-index-0 case above, a mutant that
  // checks liveness for `rows[0]` alone instead of looping over every row
  // passed all 1280 tests. That mutant is not benign — a disposed row at
  // index 1 then reads freed (possibly already reused) WASM memory and
  // `stack` returns silently wrong data with no diagnostic at all, the exact
  // failure class this slice treats as its most dangerous. The real
  // implementation loops correctly; this test is what keeps it that way.
  const core = await initCore();
  const shape: number[] = [3];
  const a = WNDArray.fromArray(core, shape, [1, 2, 3]);
  const b = WNDArray.fromArray(core, shape, [7, 8, 9]);
  const c = WNDArray.fromArray(core, shape, [4, 5, 6]);
  b.dispose(); // the MIDDLE row, not the first
  assert.throws(() => WNDArray.stack(core, [a, b, c]), /WNDArray\.stack:.*disposed/);
  // and the last row, so no single-position shortcut can satisfy this file
  const d = WNDArray.fromArray(core, shape, [1, 1, 1]);
  d.dispose();
  assert.throws(() => WNDArray.stack(core, [a, c, d]), /WNDArray\.stack:.*disposed/);
  a.dispose();
  c.dispose();
});

// -----------------------------------------------------------------------
// D7: "eine mitten im Loop scheiternde Zeile gibt Ausgabepuffer UND Scratch
// frei" — deterministically forceable, per backend-oom.test.ts's own
// precedent: a real WASM kernel-status failure inside `stack()`'s per-row
// loop cannot be forced against the real artifact (every row constructed
// through the public API is a valid in-bounds rank-1 view), so this mocks
// `CoreExports` directly — same bump-allocator-over-a-real-ArrayBuffer
// technique backend-oom.test.ts uses for the v1 backend's OOM paths, here
// injecting an `nt_materialize` FAILURE (not an `nt_alloc` failure) at a
// specific call index instead. `WNDArray.stack` takes `core` as an explicit
// parameter (D2), so the mock plugs in directly — no monkey-patching.
// -----------------------------------------------------------------------

interface MockAlloc {
  readonly ptr: number;
  readonly bytes: number;
}

/** A minimal, functional `CoreExports` mock: `nt_alloc`/`nt_free` are a real
 * bump allocator over a real backing `ArrayBuffer` (so `WNDArray.fromArray`
 * and `stack`'s own scratch/output allocations work genuinely, not just
 * type-check); `nt_materialize` performs the real rank-1 gather EXCEPT on
 * its `failAtCall`-th invocation (1-indexed across the whole test), where it
 * returns a non-zero status without touching `outDataPtr` — reproducing the
 * exact "kernel rejects an already-allocated output buffer" path `stack`'s
 * own `status !== 0` branch exists for. Every other `CoreExports` member is
 * a `notImplemented` stub (throws if ever called) — `stack()` never reaches
 * them, so a call would indicate a real bug, not a mock gap. */
function makeStackFailureMockCore(failAtMaterializeCall: number): {
  readonly core: CoreExports;
  readonly allocs: MockAlloc[];
  readonly frees: MockAlloc[];
} {
  const buffer = new ArrayBuffer(4 << 20); // 4 MiB — ample for these small test shapes
  let bump = 8; // keep ptr 0 reserved (matches allocBytes' "ptr === 0 => OOM" convention)
  let materializeCallCount = 0;
  const allocs: MockAlloc[] = [];
  const frees: MockAlloc[] = [];

  function nt_alloc(bytes: number): number {
    if (bytes === 0) {
      allocs.push({ ptr: 0, bytes: 0 });
      return 0;
    }
    const ptr = bump;
    bump += bytes;
    if (bump > buffer.byteLength) throw new Error(`mock backing buffer too small (need ${bump} bytes)`);
    allocs.push({ ptr, bytes });
    return ptr;
  }
  function nt_free(ptr: number, bytes: number): void {
    frees.push({ ptr, bytes });
  }
  function nt_materialize(
    shapePtr: number,
    rank: number,
    stridesPtr: number,
    offset: number,
    dataPtr: number,
    dataLen: number,
    outDataPtr: number,
    outLen: number,
  ): number {
    materializeCallCount++;
    if (materializeCallCount === failAtMaterializeCall) {
      return 7; // arbitrary non-zero status
    }
    const shapeView = new Uint32Array(buffer, shapePtr, rank);
    const stridesView = new Uint32Array(buffer, stridesPtr, rank);
    const dataView = new Float64Array(buffer, dataPtr, dataLen);
    const outView = new Float64Array(buffer, outDataPtr, outLen);
    const d = shapeView[0] ?? 0;
    const stride = stridesView[0] ?? 0;
    for (let k = 0; k < d; k++) outView[k] = dataView[offset + k * stride] ?? 0;
    return 0;
  }
  function notImplemented(name: string): (...args: number[]) => number {
    return (): number => {
      throw new Error(`mock: ${name} unexpectedly called`);
    };
  }

  const core: CoreExports = {
    memory: { buffer } as WebAssembly.Memory,
    nt_alloc,
    nt_free,
    nt_materialize,
    nt_add: notImplemented("nt_add"),
    nt_matmul: notImplemented("nt_matmul"),
    nt_sum_all: notImplemented("nt_sum_all"),
    nt_sum_axis: notImplemented("nt_sum_axis"),
    nt_transpose: notImplemented("nt_transpose"),
    nt_fill: notImplemented("nt_fill"),
    nt_add_strided: notImplemented("nt_add_strided"),
    nt_matmul_strided: notImplemented("nt_matmul_strided"),
    nt_sum_all_strided: notImplemented("nt_sum_all_strided"),
    nt_sum_axis_strided: notImplemented("nt_sum_axis_strided"),
    nt_matmul_blocked: notImplemented("nt_matmul_blocked"),
    nt_sub_strided: notImplemented("nt_sub_strided"),
    nt_mul_strided: notImplemented("nt_mul_strided"),
    nt_div_strided: notImplemented("nt_div_strided"),
    nt_dot_strided: notImplemented("nt_dot_strided"),
    nt_norm_sq_strided: notImplemented("nt_norm_sq_strided"),
    nt_sqrt_strided: notImplemented("nt_sqrt_strided"),
    nt_scalar_add_strided: notImplemented("nt_scalar_add_strided"),
    nt_scalar_sub_strided: notImplemented("nt_scalar_sub_strided"),
    nt_scalar_mul_strided: notImplemented("nt_scalar_mul_strided"),
    nt_scalar_div_strided: notImplemented("nt_scalar_div_strided"),
  };

  return { core, allocs, frees };
}

test("stack: a row that fails mid-loop (kernel status != 0) frees BOTH the output buffer and the two scratch buffers, leaves every input row untouched", () => {
  const d = 4;
  const { core: mockCore, allocs, frees } = makeStackFailureMockCore(2); // fail on the 2nd nt_materialize call (row index 1)

  const rows: AnyWNDArray[] = [];
  for (let i = 0; i < 3; i++) {
    rows.push(WNDArray.fromArray(mockCore, [d], new Array(d).fill(i + 1)));
  }

  const allocsBefore = allocs.length;
  const freesBefore = frees.length;

  assert.throws(() => WNDArray.stack(mockCore, rows), /wasm resident nt_materialize \(stack\): status 7 for row 1/);

  const stackAllocs = allocs.slice(allocsBefore);
  const stackFrees = frees.slice(freesBefore);

  // stack() itself makes exactly 3 allocations: outDataBuf, shapeBuf, stridesBuf.
  assert.strictEqual(stackAllocs.length, 3, `expected exactly 3 allocations from stack() itself, got ${stackAllocs.length}`);
  // ALL three must be freed on the failure path: the output buffer via the
  // explicit `status !== 0` branch, shapeBuf/stridesBuf via `finally`.
  assert.strictEqual(stackFrees.length, 3, `expected exactly 3 frees on the failure path (output + 2 scratch), got ${stackFrees.length}`);
  const remaining = [...stackFrees];
  for (const a of stackAllocs) {
    const idx = remaining.findIndex((f) => f.ptr === a.ptr && f.bytes === a.bytes);
    assert.ok(idx !== -1, `allocation {ptr:${a.ptr}, bytes:${a.bytes}} was never freed on the failure path`);
    remaining.splice(idx, 1);
  }
  assert.strictEqual(remaining.length, 0, `unexpected/extra frees on the failure path: ${JSON.stringify(remaining)}`);

  // A failing op never touches ANY input row (same contract as add/sub/mul/
  // div/matmul above) — row 0 (already gathered before the failure) and the
  // never-reached rows 1/2 are all still fully live and usable.
  for (const r of rows) {
    assert.strictEqual(r.disposed, false);
  }
  assert.deepStrictEqual(Array.from(rows[0]!.toArray()), [1, 1, 1, 1]);
  for (const r of rows) r.dispose();
});
