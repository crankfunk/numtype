/**
 * Kern 06 differential tests: `threadedMatmul` must stay bit-identical to
 * BOTH the naive `runtime.ts` reference AND the single-threaded
 * `nt_matmul_blocked` (same artifact family, via the stable core), across
 * worker counts {1,2,3,4} and every input the parallel bit-identity law
 * covers — contiguous, transposed views, batch, broadcast batch, k=0,
 * size-0, odd/prime dims. Plus lifecycle (dispose → workers exited,
 * free-counter plateau, second pool works), sequential reuse, and the
 * crash-detection deadline path. See docs/kern-06-threads-spec.md §5.
 *
 * Four persistent pools (one per worker count) are created ONCE at module
 * level and reused across every case in the differential grid — this is
 * itself part of what "sequential reuse (many matmuls on one pool)" means
 * for the {1,2,3,4} pools; a SEPARATE dedicated test below additionally
 * exercises many matmuls on one freshly-created pool in isolation. All
 * pools are disposed in `after()` so the test process exits by itself (a
 * live `Worker` keeps Node's event loop alive) — verified by actually
 * running `pnpm test:threaded` and observing the process exit (see the
 * results doc).
 *
 * Wired into `pnpm test:threaded` (own explicit file list — the guard's
 * rule (a) now covers exactly one of test:core / test:resident /
 * test:threaded).
 */
import assert from "node:assert";
import { after, test } from "node:test";
import { itemRuntime, keepDimsShape, matmulRuntime, meanRuntime, scalarElementwiseRuntime, sqrtRuntime, stackRuntime, transposeRuntime } from "../src/runtime.ts";
import { initCore, type CoreExports } from "../src/wasm/loader.ts";
import { WNDArray, type AnyWNDArray } from "../src/wasm/resident.ts";
import {
  alignDown16,
  getPoisonCleanupFreeCount,
  getThreadedPoolFreeCount,
  initThreadedCore,
  THREADED_MATMUL_MIN_POOL_WORK,
  threadedMatmul,
  ThreadedPool,
} from "../src/wasm/threaded.ts";
import { assertDataBitIdentical, assertShapeEqual, wideSpecs } from "./assert-helpers.ts";
import { genBroadcastShapes, genData, genDataSpecial, makeRng, type Rng } from "./prng.ts";

const stableCore: CoreExports = await initCore();

/** Since the size-based auto-routing follow-up, `threadedMatmul` routes
 * calls below `THREADED_MATMUL_MIN_POOL_WORK` (batch·m·k·n) to the
 * single-threaded main-thread kernel. Every test in this file whose POINT
 * is the worker-dispatch path (the differential grid's deliberately small
 * shapes, lifecycle, crash/poison scenarios) pins that route explicitly —
 * otherwise they would silently all run on main and prove nothing about
 * the pool. The auto-routing tests at the bottom of this file are the ones
 * that exercise the router itself. */
const FORCE_POOL = { minPoolWork: 0 } as const;

const WORKER_COUNTS = [1, 2, 3, 4] as const;
const pools = new Map<number, ThreadedPool>();
for (const wc of WORKER_COUNTS) {
  pools.set(wc, await initThreadedCore(wc));
}

after(async () => {
  for (const pool of pools.values()) await pool.dispose();
});

interface Operand {
  readonly arr: AnyWNDArray;
  readonly owners: readonly AnyWNDArray[];
}

function makeContiguous(core: CoreExports, shape: readonly number[], refData: Float64Array): Operand {
  const arr = WNDArray.fromArray(core, shape, refData);
  return { arr, owners: [arr] };
}

/** A transpose VIEW whose logical shape/content is `shape`/`refData` — same
 * involution trick as blocked.test.ts/strided.test.ts. */
function makeView(core: CoreExports, shape: readonly number[], refData: Float64Array): Operand {
  const baseShape = [...shape].reverse();
  const baseData = transposeRuntime(shape, refData).data;
  const base = WNDArray.fromArray(core, baseShape, baseData);
  const view = base.transpose();
  return { arr: view, owners: [base, view] };
}

function makeOperand(core: CoreExports, asView: boolean, shape: readonly number[], refData: Float64Array): Operand {
  return asView ? makeView(core, shape, refData) : makeContiguous(core, shape, refData);
}

function disposeAll(...operands: Operand[]): void {
  for (const op of operands) for (const h of op.owners) h.dispose();
}

/** Full differential case: naive `runtime.ts` reference, single-threaded
 * `nt_matmul_blocked` reference (stable core), then `threadedMatmul` on
 * every pool in `WORKER_COUNTS` — every pair (worker counts, and each vs.
 * both references) must be bit-identical. */
function runCase(name: string, aShape: number[], bShape: number[], aView: boolean, bView: boolean, rng: Rng): void {
  test(name, () => {
    const aData = genData(rng, aShape);
    const bData = genData(rng, bShape);
    const ref = matmulRuntime(aShape, aData, bShape, bData);

    const aStable = makeOperand(stableCore, aView, aShape, aData);
    const bStable = makeOperand(stableCore, bView, bShape, bData);
    let refBlockedShape: readonly number[];
    let refBlockedData: Float64Array;
    try {
      const gotStable = aStable.arr.matmul(bStable.arr);
      try {
        refBlockedShape = gotStable.shape;
        refBlockedData = gotStable.toArray();
      } finally {
        gotStable.dispose();
      }
    } finally {
      disposeAll(aStable, bStable);
    }
    assertShapeEqual(ref.shape, refBlockedShape, `${name}: runtime.ts vs nt_matmul_blocked shape`);
    assertDataBitIdentical(ref.data, refBlockedData, `${name}: runtime.ts vs nt_matmul_blocked`);

    const perWorkerCount: Float64Array[] = [];
    for (const wc of WORKER_COUNTS) {
      const pool = pools.get(wc)!;
      const a = makeOperand(pool.core, aView, aShape, aData);
      const b = makeOperand(pool.core, bView, bShape, bData);
      try {
        const got = threadedMatmul(pool, a.arr, b.arr, FORCE_POOL);
        try {
          assertShapeEqual(ref.shape, got.shape, `${name} workers=${wc}`);
          const gotData = got.toArray();
          assertDataBitIdentical(ref.data, gotData, `${name} workers=${wc} vs runtime.ts`);
          assertDataBitIdentical(refBlockedData, gotData, `${name} workers=${wc} vs nt_matmul_blocked`);
          perWorkerCount.push(gotData);
        } finally {
          got.dispose();
        }
      } finally {
        disposeAll(a, b);
      }
    }
    for (let i = 1; i < perWorkerCount.length; i++) {
      assertDataBitIdentical(
        perWorkerCount[0]!,
        perWorkerCount[i]!,
        `${name}: workers=${WORKER_COUNTS[0]} vs workers=${WORKER_COUNTS[i]}`,
      );
    }
  });
}

// MC tile size from crates/core/src/kernels/matmul_blocked.rs — dims below
// are chosen relative to it (and to typical worker-pool row splits) so
// cases actually straddle both the kernel's own tile boundaries AND the
// row-range split boundaries `computeRowRanges` produces.
const MC = 32;

// --- contiguous + transposed-view random cases, straddling MC boundaries --
{
  const rng = makeRng(0x5448524444524e44n); // "THRD" + "RND" packed, arbitrary distinct seed
  const CASE_COUNT = 24;
  for (let c = 0; c < CASE_COUNT; c++) {
    const m = rng.nextInt(1, 3 * MC);
    const k = rng.nextInt(1, 2 * MC);
    const n = rng.nextInt(1, 3 * MC);
    let aView = rng.nextBool();
    let bView = rng.nextBool();
    if (!aView && !bView) {
      if (rng.nextBool()) aView = true;
      else bView = true;
    }
    runCase(
      `threaded matmul random case ${c}: [${m},${k}]${aView ? "ᵛ" : ""} @ [${k},${n}]${bView ? "ᵛ" : ""}`,
      [m, k],
      [k, n],
      aView,
      bView,
      rng,
    );
  }
}

// --- explicit row-split-boundary cases: m at/around MC boundaries, since
// that's what computeRowRanges actually splits on (m directly determines
// the flat row space for a non-batch matmul). ---
{
  const rng = makeRng(0x5448524444424e4442n); // "THRD" + "BNDB" packed, arbitrary distinct seed
  const mBoundary = [1, MC - 1, MC, MC + 1, 2 * MC - 1, 2 * MC, 2 * MC + 1, 3 * MC + 5, 4 * MC + 1];
  for (let i = 0; i < mBoundary.length; i++) {
    const m = mBoundary[i]!;
    const k = 40 + (i % 5) * 7;
    const n = 30 + (i % 4) * 11;
    const aView = i % 2 === 0;
    const bView = i % 3 === 0;
    runCase(`threaded matmul row-boundary case ${i}: [${m},${k}]${aView ? "ᵛ" : ""} @ [${k},${n}]${bView ? "ᵛ" : ""}`, [m, k], [k, n], aView, bView, rng);
  }
}

// --- batch and broadcast-batch: totalRows = batchSize * m, so the split
// must correctly straddle BATCH boundaries too (a single partial call
// spanning parts of two batches — proven at the kernel level already, this
// re-proves it end-to-end through the real worker dispatch path). ---
{
  const rng = makeRng(0x5448524444424154n); // "THRD" + "BAT" packed, arbitrary distinct seed
  const CASE_COUNT = 10;
  for (let c = 0; c < CASE_COUNT; c++) {
    const m = rng.nextInt(MC - 1, 2 * MC + 3);
    const k = rng.nextInt(10, 60);
    const n = rng.nextInt(10, 60);
    const { aShape: batchA, bShape: batchB } = genBroadcastShapes(rng, 2);
    const aShape = [...batchA, m, k];
    const bShape = [...batchB, k, n];
    let aView = rng.nextBool();
    let bView = rng.nextBool();
    if (!aView && !bView) {
      if (rng.nextBool()) aView = true;
      else bView = true;
    }
    runCase(
      `threaded matmul batch case ${c}: a=[${aShape.join(",")}]${aView ? "ᵛ" : ""} b=[${bShape.join(",")}]${bView ? "ᵛ" : ""}`,
      aShape,
      bShape,
      aView,
      bView,
      rng,
    );
  }
}

// --- k=0 / size-0 shapes: every worker's partial call must zero-fill its
// rows and short-circuit correctly (m=0/n=0 skip dispatch on main entirely
// per threadedMatmul's own short-circuit; k=0 dispatches but every output
// must be exactly 0.0, not merely close). ---
{
  const rng = makeRng(0x54485244445a4552n); // "THRD" + "ZER" packed, arbitrary distinct seed
  const zeroCases: Array<{ aShape: number[]; bShape: number[]; label: string }> = [
    { aShape: [0, 5], bShape: [5, 4], label: "m=0" },
    { aShape: [70, 0], bShape: [0, 66], label: "k=0" },
    { aShape: [65, 70], bShape: [70, 0], label: "n=0" },
    { aShape: [3, 0, 5], bShape: [3, 5, 4], label: "batch k=0" },
  ];
  for (const { aShape, bShape, label } of zeroCases) {
    test(`threaded matmul size-zero: ${label} a=[${aShape.join(",")}] b=[${bShape.join(",")}]`, () => {
      const aData = genData(rng, aShape);
      const bData = genData(rng, bShape);
      const ref = matmulRuntime(aShape, aData, bShape, bData);
      for (const wc of WORKER_COUNTS) {
        const pool = pools.get(wc)!;
        const a = WNDArray.fromArray(pool.core, aShape, aData);
        const b = WNDArray.fromArray(pool.core, bShape, bData);
        try {
          const got = threadedMatmul(pool, a, b, FORCE_POOL);
          try {
            assertShapeEqual(ref.shape, got.shape, `${label} workers=${wc}`);
            const gotData = got.toArray();
            assertDataBitIdentical(ref.data, gotData, `${label} workers=${wc}`);
            if (label === "k=0" || label === "batch k=0") {
              assert.ok(gotData.every((v) => v === 0), `${label}: every output must be exactly 0.0, workers=${wc}`);
            }
          } finally {
            got.dispose();
          }

          // DEFAULT opts too (verify finding: FORCE_POOL alone left this
          // untested): every size-zero shape has work volume 0, so the
          // router sends these to the MAIN thread — that route must be
          // bit-identical as well, and must dispatch no worker.
          const seqBefore = pool.workers.map((pw) => pw.postedSeq);
          const gotAuto = threadedMatmul(pool, a, b);
          try {
            assertShapeEqual(ref.shape, gotAuto.shape, `${label} workers=${wc} (auto-routed)`);
            const gotAutoData = gotAuto.toArray();
            assertDataBitIdentical(ref.data, gotAutoData, `${label} workers=${wc} (auto-routed)`);
            if (label === "k=0" || label === "batch k=0") {
              assert.ok(gotAutoData.every((v) => v === 0), `${label} (auto-routed): every output must be exactly 0.0, workers=${wc}`);
            }
          } finally {
            gotAuto.dispose();
          }
          assert.deepStrictEqual(
            pool.workers.map((pw) => pw.postedSeq),
            seqBefore,
            `${label} workers=${wc}: a size-zero call (volume 0) must not dispatch any worker on the default route`,
          );
        } finally {
          a.dispose();
          b.dispose();
        }
      }
    });
  }
}

// --- odd / prime dims: no accidental alignment with MC=32 or worker count
// anywhere (rows, batch, or split boundaries). ---
{
  const rng = makeRng(0x5448524444444f4444n); // "THRD" + "ODD" packed, arbitrary distinct seed
  const primeCases: Array<[number, number, number]> = [
    [97, 53, 71],
    [131, 17, 89],
    [3, 251, 3],
    [1009, 3, 5],
  ];
  for (let i = 0; i < primeCases.length; i++) {
    const [m, k, n] = primeCases[i]!;
    const aView = i % 2 === 0;
    const bView = i % 2 === 1;
    runCase(`threaded matmul prime-dims case ${i}: [${m},${k}]${aView ? "ᵛ" : ""} @ [${k},${n}]${bView ? "ᵛ" : ""}`, [m, k], [k, n], aView, bView, rng);
  }
}

// --- pool lifecycle: dispose -> workers exited, free-counter plateau,
// second pool still works. Uses its OWN freshly-created pool (not the
// module-level ones, which stay alive until `after()`). ---
test("pool lifecycle: dispose frees workers (free-counter increments), plateaus on a second dispose, and a fresh pool afterward still works", async () => {
  const before = getThreadedPoolFreeCount();
  const p = await initThreadedCore(2);

  const a = WNDArray.fromArray(p.core, [2, 2] as const, [1, 2, 3, 4]);
  const b = WNDArray.fromArray(p.core, [2, 2] as const, [1, 0, 0, 1]); // identity
  const c = threadedMatmul(p, a, b, FORCE_POOL);
  assertDataBitIdentical(new Float64Array([1, 2, 3, 4]), c.toArray(), "sanity matmul before dispose");
  a.dispose();
  b.dispose();
  c.dispose();

  await p.dispose();
  const afterFirst = getThreadedPoolFreeCount();
  assert.strictEqual(afterFirst, before + p.workerCount, "dispose() must free exactly workerCount (stack, ctrl) pairs");
  for (const pw of p.workers) assert.strictEqual(pw.alive, false, "every worker must have exited after dispose()");

  await p.dispose(); // second call: must be a no-op (plateau), not a double-free
  const afterSecond = getThreadedPoolFreeCount();
  assert.strictEqual(afterSecond, afterFirst, "a second dispose() must not change the free counter (plateau, no double-free)");

  const p2 = await initThreadedCore(2);
  const a2 = WNDArray.fromArray(p2.core, [2, 2] as const, [5, 6, 7, 8]);
  const b2 = WNDArray.fromArray(p2.core, [2, 2] as const, [1, 0, 0, 1]);
  const c2 = threadedMatmul(p2, a2, b2, FORCE_POOL);
  assertDataBitIdentical(new Float64Array([5, 6, 7, 8]), c2.toArray(), "a fresh pool created after a disposed one still works");
  a2.dispose();
  b2.dispose();
  c2.dispose();
  await p2.dispose();
});

// --- sequential reuse: many matmuls on one persistent (freshly created,
// isolated) pool. ---
test("sequential reuse: many matmuls on one persistent pool", async () => {
  const p = await initThreadedCore(3);
  try {
    const rng = makeRng(0x5345514552455553n); // "SEQ" + "REUS" packed, arbitrary distinct seed
    for (let i = 0; i < 20; i++) {
      const m = rng.nextInt(1, 90);
      const k = rng.nextInt(1, 90);
      const n = rng.nextInt(1, 90);
      const aData = genData(rng, [m, k]);
      const bData = genData(rng, [k, n]);
      const ref = matmulRuntime([m, k], aData, [k, n], bData);
      const a = WNDArray.fromArray(p.core, [m, k], aData);
      const b = WNDArray.fromArray(p.core, [k, n], bData);
      try {
        const got = threadedMatmul(p, a, b, FORCE_POOL);
        try {
          assertShapeEqual(ref.shape, got.shape, `sequential reuse case ${i}`);
          assertDataBitIdentical(ref.data, got.toArray(), `sequential reuse case ${i}`);
        } finally {
          got.dispose();
        }
      } finally {
        a.dispose();
        b.dispose();
      }
    }
  } finally {
    await p.dispose();
  }
});

// --- crash detection: a worker dying mid-job surfaces as a thrown Error
// within the configured deadline, never a hang. See threaded.ts's module
// doc "Crash detection" note for why this specifically exercises the
// DEADLINE path (not the cheap pre-dispatch alive check): `terminate()` is
// fired WITHOUT awaiting it, then `threadedMatmul` is called in the very
// same tick, so the worker is dispatched a job while genuinely dying —
// deterministic because `alive` has not yet flipped to `false` (verified:
// the underlying event only fires once control returns to the event loop,
// confirmed empirically while designing this protocol). ---
test("worker death mid-job surfaces as a thrown Error within the deadline, not a hang", async () => {
  const shortTimeoutMs = 1500;
  const p = await initThreadedCore(2, shortTimeoutMs);
  try {
    const rng = makeRng(0x4352415348n); // "CRASH"
    const m = 300,
      k = 150,
      n = 150; // large enough that termination should race ahead of completion
    const aData = genData(rng, [m, k]);
    const bData = genData(rng, [k, n]);
    const a = WNDArray.fromArray(p.core, [m, k], aData);
    const b = WNDArray.fromArray(p.core, [k, n], bData);

    assert.strictEqual(p.workers[1]!.alive, true, "precondition: worker 1 must look alive before terminate()");
    const terminatePromise = p.workers[1]!.worker.terminate();

    const t0 = Date.now();
    assert.throws(() => threadedMatmul(p, a, b, FORCE_POOL), /did not complete within|not alive/, "threadedMatmul must throw, not hang, when a worker dies mid-job");
    const elapsed = Date.now() - t0;
    assert.ok(elapsed < shortTimeoutMs + 5000, `crash detection took ${elapsed}ms, expected roughly the ${shortTimeoutMs}ms deadline`);

    await terminatePromise;
    a.dispose();
    b.dispose();
  } finally {
    await p.dispose();
  }
});

// --- pre-dispatch liveness check: a worker known dead from a PRIOR call
// (event already processed, `alive === false`) is refused immediately,
// without waiting for the deadline at all. ---
test("a worker already known dead (from a prior call) is refused immediately, not after the deadline", async () => {
  const p = await initThreadedCore(2, 5000);
  try {
    await p.workers[0]!.worker.terminate();
    await new Promise((resolve) => setTimeout(resolve, 100)); // let the 'exit' event actually flush
    assert.strictEqual(p.workers[0]!.alive, false, "precondition: worker 0 must be known-dead after the yield");

    const a = WNDArray.fromArray(p.core, [4, 3] as const, genData(makeRng(1n), [4, 3]));
    const b = WNDArray.fromArray(p.core, [3, 2] as const, genData(makeRng(2n), [3, 2]));
    const t0 = Date.now();
    assert.throws(() => threadedMatmul(p, a, b, FORCE_POOL), /not alive/);
    const elapsed = Date.now() - t0;
    assert.ok(elapsed < 500, `pre-dispatch liveness check took ${elapsed}ms, expected near-instant (not the 5000ms deadline)`);
    a.dispose();
    b.dispose();
  } finally {
    await p.dispose();
  }
});

// --- post-verify fix: deferred buffer freeing on a pool-compromising error
// (use-after-free regression test). A fresh-context verify pass found that
// the ORIGINAL code freed the per-call output/scratch buffers immediately
// upon throwing, even when an EARLIER-dispatched worker (lower index, so
// already posted a job in the dispatch loop's iteration order) could still
// be actively writing into those exact buffers on its own OS thread. This
// test reproduces that scenario deterministically:
//   - worker 1 is killed and the kill is allowed to SETTLE (yielded, like
//     the "already known dead" test above) BEFORE calling threadedMatmul,
//     so `alive` correctly reads `false` at dispatch time — no timing race
//     for the kill itself.
//   - worker 0 (index 0, checked/dispatched FIRST in dispatchAndRun's
//     iteration order) is given a LARGE job (4000x500x500) so it is
//     genuinely still computing on its own OS thread at the moment the
//     dispatch loop reaches worker 1 and throws — deterministic because a
//     job this size takes tens to hundreds of milliseconds to actually
//     compute, vastly longer than the microseconds between "worker 0's job
//     is posted" and "worker 1's dead-check throws" in the SAME synchronous
//     call.
// Asserts: the throw is immediate; `getPoisonCleanupFreeCount()` has NOT
// advanced right after the throw (buffers demonstrably NOT freed while
// worker 0 might still be writing); a second call fails fast (poisoned);
// after `dispose()` (which awaits the SAME cleanup `poison()` started) the
// counter HAS advanced to the expected plateau. No sleeps used to
// synchronize the race itself — only the (already-required) settle-wait
// for the worker-1 kill, exactly like the existing pre-dispatch-dead test.
test("worker death mid-dispatch (a still-computing EARLIER worker) defers buffer freeing until cleanup completes — no use-after-free", async () => {
  const p = await initThreadedCore(2, 5000);
  try {
    await p.workers[1]!.worker.terminate();
    await new Promise((resolve) => setTimeout(resolve, 100));
    assert.strictEqual(p.workers[1]!.alive, false, "precondition: worker 1 must be known-dead before dispatch");
    assert.strictEqual(p.workers[0]!.alive, true, "precondition: worker 0 must still be alive (it will get dispatched a real job)");

    const rng = makeRng(0x444546455245n); // "DEFERE" (arbitrary)
    const m = 4000;
    const k = 500;
    const n = 500;
    const aData = genData(rng, [m, k]);
    const bData = genData(rng, [k, n]);
    const a = WNDArray.fromArray(p.core, [m, k], aData);
    const b = WNDArray.fromArray(p.core, [k, n], bData);

    const beforeThrow = getPoisonCleanupFreeCount();
    assert.throws(() => threadedMatmul(p, a, b, FORCE_POOL), /not alive/, "must throw synchronously (pool poisoned), not hang");
    const rightAfterThrow = getPoisonCleanupFreeCount();
    assert.strictEqual(
      rightAfterThrow,
      beforeThrow,
      "buffers must NOT be freed synchronously at throw time — worker 0 (dispatched a large job) may still be writing them",
    );
    assert.strictEqual(p.isPoisoned, true, "pool must be marked poisoned immediately");

    // A second call on the poisoned pool must fail fast, not attempt
    // another dispatch (and must not touch the free counter either — it
    // never gets far enough to allocate anything).
    const t0 = Date.now();
    assert.throws(() => threadedMatmul(p, a, b), /poisoned/, "a poisoned pool must refuse further calls");
    assert.ok(Date.now() - t0 < 200, "poisoned-pool refusal must be near-instant");
    assert.strictEqual(getPoisonCleanupFreeCount(), beforeThrow, "the refused second call must not free (or double-free) anything");

    // Await the SAME cleanup dispatchAndRun's poison() call started
    // (dispose() chains onto pool.poisonCleanup) — only now must the
    // deferred buffers actually be freed.
    await p.dispose();
    const afterCleanup = getPoisonCleanupFreeCount();
    assert.ok(
      afterCleanup > beforeThrow,
      `poison cleanup must have freed the in-flight buffers once dispose() completed (before=${beforeThrow}, after=${afterCleanup})`,
    );
    // Exactly 5 buffers were in flight for this call: aShape/aStrides/
    // bShape/bStrides scratch + the output buffer.
    assert.strictEqual(afterCleanup - beforeThrow, 5, `expected exactly 5 buffers freed by the deferred cleanup, got ${afterCleanup - beforeThrow}`);
    assert.strictEqual(p.workers[0]!.alive, false, "worker 0 must have been terminated as part of the poison cleanup");

    a.dispose();
    b.dispose();
  } finally {
    await p.dispose(); // no-op if already disposed above
  }
});

// --- post-verify fix: 16-byte stack alignment -------------------------------
test("alignDown16 rounds down to a multiple of 16, never past the lower bound", () => {
  assert.strictEqual(alignDown16(0), 0);
  assert.strictEqual(alignDown16(16), 16);
  assert.strictEqual(alignDown16(17), 16);
  assert.strictEqual(alignDown16(31), 16);
  assert.strictEqual(alignDown16(32), 32);
  assert.strictEqual(alignDown16(1000), 992); // 1000 = 62*16 + 8
  assert.strictEqual(alignDown16(1048576), 1048576); // 1 MiB is itself a multiple of 16
  for (let x = 1048576; x < 1048576 + 64; x++) {
    const aligned = alignDown16(x);
    assert.strictEqual(aligned % 16, 0, `alignDown16(${x}) = ${aligned} is not a multiple of 16`);
    assert.ok(aligned <= x && aligned > x - 16, `alignDown16(${x}) = ${aligned} is not the nearest lower multiple of 16`);
  }
});

test("every spawned worker's stackTop is actually 16-byte aligned end-to-end", async () => {
  // nt_alloc only guarantees 8-byte alignment (crates/core/src/abi.rs's own
  // documented contract) — allocate an extra 8-byte buffer first so the
  // NEXT allocation (the worker stack) has a real chance of landing on a
  // non-16-aligned address if alignDown16 were not applied, rather than
  // relying on the allocator's own incidental behavior to prove the point.
  const p = await initThreadedCore(3);
  try {
    const stray = p.core.nt_alloc(8);
    assert.notStrictEqual(stray, 0);
    for (const pw of p.workers) {
      const stackPtr = pw.stackPtr;
      // Recompute the SAME way initThreadedCore does, to check the actual
      // value handed to the worker (stackPtr itself is only 8-aligned; the
      // aligned TOP is what matters and what this test pins).
      const rawTop = stackPtr + 1024 * 1024;
      const alignedTop = alignDown16(rawTop);
      assert.strictEqual(alignedTop % 16, 0, `worker stackTop ${alignedTop} (raw ${rawTop}, stackPtr ${stackPtr}) is not 16-byte aligned`);
    }
    p.core.nt_free(stray, 8);
  } finally {
    await p.dispose();
  }
});

// --- post-re-verify: dedicated regression test for the WAIT-TIMEOUT poison
// branch (the earlier deferred-free test only covered the DISPATCH-TIME
// dead-worker branch, where the still-alive worker is structurally the
// last index in a 2-worker pool — this proves the OTHER throw site inside
// dispatchAndRun, where the still-computing worker has a HIGHER index than
// the one whose timeout triggers the throw, the only way it CAN happen
// given the sequential index-order wait loop). Deterministic via a robust
// wall-clock margin, not sleep-racing: BOTH workers get an equally large,
// genuinely slow job (calibrated empirically: a healthy 2-worker run of
// this exact size took ~2.2s on the reference machine), `matmulTimeoutMs`
// is set far below that (150ms — a ~15x margin on the reference machine).
// A slower machine only makes the real job take LONGER relative to the
// fixed absolute 150ms deadline, so the margin can only grow, never shrink
// or flip the scenario negative — the test cannot become flaky by running
// on a slower host, only by running on a host so fast that a
// 12000x1000 @ 1000x1000 matmul completes in under ~150ms total (not
// plausible for a scalar/SIMD128 WASM kernel on any current hardware).
test("wait-loop timeout (a later-index worker still genuinely computing) defers buffer freeing until cleanup completes", async () => {
  const shortTimeoutMs = 150;
  const p = await initThreadedCore(2, shortTimeoutMs);
  try {
    const rng = makeRng(0x54494d454f5554n); // "TIMEOUT"
    const m = 12000;
    const k = 1000;
    const n = 1000;
    const aData = genData(rng, [m, k]);
    const bData = genData(rng, [k, n]);
    const a = WNDArray.fromArray(p.core, [m, k], aData);
    const b = WNDArray.fromArray(p.core, [k, n], bData);

    const beforeThrow = getPoisonCleanupFreeCount();
    const t0 = Date.now();
    assert.throws(() => threadedMatmul(p, a, b, FORCE_POOL), /did not complete within/, "must throw due to the wait-loop timeout, not the pre-dispatch alive check");
    const elapsed = Date.now() - t0;
    // Sanity: the throw must actually be GATED by the deadline (roughly
    // shortTimeoutMs), not instant — proves this exercised the timeout
    // branch, not some other early-exit path.
    assert.ok(elapsed >= shortTimeoutMs - 50, `threw too early (${elapsed}ms) — expected to be gated by the ${shortTimeoutMs}ms deadline`);
    assert.ok(elapsed < shortTimeoutMs + 10_000, `threw too late (${elapsed}ms)`);

    const rightAfterThrow = getPoisonCleanupFreeCount();
    assert.strictEqual(
      rightAfterThrow,
      beforeThrow,
      "buffers must NOT be freed synchronously at throw time — the later-index worker (still computing its large share) may still be writing them",
    );
    assert.strictEqual(p.isPoisoned, true, "pool must be marked poisoned immediately");

    const t1 = Date.now();
    assert.throws(() => threadedMatmul(p, a, b), /poisoned/, "a poisoned pool must refuse further calls");
    assert.ok(Date.now() - t1 < 200, "poisoned-pool refusal must be near-instant, not another deadline wait");
    assert.strictEqual(getPoisonCleanupFreeCount(), beforeThrow, "the refused second call must not free (or double-free) anything");

    await p.dispose();
    const afterCleanup = getPoisonCleanupFreeCount();
    assert.strictEqual(
      afterCleanup - beforeThrow,
      5,
      `expected exactly 5 buffers freed by the deferred cleanup (4 scratch + 1 output), got ${afterCleanup - beforeThrow}`,
    );

    a.dispose();
    b.dispose();
  } finally {
    await p.dispose(); // no-op if already disposed above
  }
});

// --- post-re-verify: disposed-pool guard in threadedMatmul ------------------
test("threadedMatmul on an already-disposed pool throws immediately, not undefined behavior", async () => {
  const p = await initThreadedCore(2);
  const a = WNDArray.fromArray(p.core, [2, 2] as const, [1, 0, 0, 1]);
  const b = WNDArray.fromArray(p.core, [2, 2] as const, [5, 6, 7, 8]);

  await p.dispose();
  assert.strictEqual(p.isDisposed, true, "precondition: pool must be disposed");

  const t0 = Date.now();
  // Deliberately DEFAULT opts: [2,2]@[2,2] is far below the auto-routing
  // threshold, so this also pins that the disposed check runs BEFORE the
  // route decision — a disposed pool refuses even calls the router would
  // have run entirely on the main thread.
  assert.throws(() => threadedMatmul(p, a, b), /disposed/, "threadedMatmul on a disposed pool must throw, naming the lifecycle state");
  assert.ok(Date.now() - t0 < 200, "disposed-pool refusal must be near-instant, not a hang or undefined behavior");

  a.dispose();
  b.dispose();
});

// --- Kern 06 follow-up: size-based auto-routing ------------------------------
// `threadedMatmul` routes calls with work volume (batch·m·k·n) below
// `THREADED_MATMUL_MIN_POOL_WORK` to the single-threaded main-thread kernel
// (a.matmul(b) over the pool's core) instead of the worker pool. The route
// actually taken is observed via the workers' `postedSeq` counters — they
// increment ONLY when dispatchAndRun posts jobs into the pool, so an
// unchanged counter proves no worker was involved.

function postedSeqs(p: ThreadedPool): number[] {
  return p.workers.map((pw) => pw.postedSeq);
}

test("auto-routing: a small call (default threshold) runs on the main thread — no worker dispatch, bit-identical result", async () => {
  const p = await initThreadedCore(2);
  try {
    const rng = makeRng(0x524f555445534d4cn); // "ROUTESML" packed, arbitrary distinct seed
    const aData = genData(rng, [8, 8]);
    const bData = genData(rng, [8, 8]);
    const ref = matmulRuntime([8, 8], aData, [8, 8], bData);
    const a = WNDArray.fromArray(p.core, [8, 8], aData);
    const b = WNDArray.fromArray(p.core, [8, 8], bData);
    try {
      assert.ok(8 * 8 * 8 < THREADED_MATMUL_MIN_POOL_WORK, "precondition: this case must sit below the default threshold");
      const before = postedSeqs(p);

      const got = threadedMatmul(p, a, b); // default opts -> router decides -> main
      try {
        assertShapeEqual(ref.shape, got.shape, "auto-routed small call shape");
        assertDataBitIdentical(ref.data, got.toArray(), "auto-routed small call vs runtime.ts");
      } finally {
        got.dispose();
      }
      assert.deepStrictEqual(postedSeqs(p), before, "no worker may have been dispatched for a below-threshold call");

      // Explicit Infinity pins the same route regardless of the constant.
      const got2 = threadedMatmul(p, a, b, { minPoolWork: Infinity });
      got2.dispose();
      assert.deepStrictEqual(postedSeqs(p), before, "minPoolWork: Infinity must never dispatch workers");
    } finally {
      a.dispose();
      b.dispose();
    }
  } finally {
    await p.dispose();
  }
});

test("auto-routing: the threshold is inclusive (>= dispatches through the pool), exact boundary", async () => {
  const p = await initThreadedCore(2);
  try {
    const rng = makeRng(0x524f555445424e44n); // "ROUTEBND" packed, arbitrary distinct seed
    const aData = genData(rng, [16, 16]);
    const bData = genData(rng, [16, 16]);
    const ref = matmulRuntime([16, 16], aData, [16, 16], bData);
    const a = WNDArray.fromArray(p.core, [16, 16], aData);
    const b = WNDArray.fromArray(p.core, [16, 16], bData);
    const volume = 16 * 16 * 16; // 4096
    try {
      const before = postedSeqs(p);

      // work == minPoolWork -> POOL (inclusive lower bound).
      const viaPool = threadedMatmul(p, a, b, { minPoolWork: volume });
      try {
        assertDataBitIdentical(ref.data, viaPool.toArray(), "boundary case via pool vs runtime.ts");
      } finally {
        viaPool.dispose();
      }
      assert.deepStrictEqual(
        postedSeqs(p),
        before.map((s) => s + 1),
        "work volume == minPoolWork must dispatch through the pool (every worker gets a job posted, even an empty range)",
      );

      // work one below minPoolWork -> MAIN.
      const afterPool = postedSeqs(p);
      const viaMain = threadedMatmul(p, a, b, { minPoolWork: volume + 1 });
      try {
        assertDataBitIdentical(ref.data, viaMain.toArray(), "boundary case via main vs runtime.ts");
      } finally {
        viaMain.dispose();
      }
      assert.deepStrictEqual(postedSeqs(p), afterPool, "work volume < minPoolWork must not dispatch any worker");
    } finally {
      a.dispose();
      b.dispose();
    }
  } finally {
    await p.dispose();
  }
});

test("auto-routing criterion is work volume (batch·m·k·n), not row count", async () => {
  const p = await initThreadedCore(2);
  try {
    const rng = makeRng(0x524f555445564f4cn); // "ROUTEVOL" packed, arbitrary distinct seed

    // ONE output row, but k·n = 2048·2048 -> volume 4.19M, far above the
    // default threshold: must go through the pool despite rows=1 (row count
    // alone would call this "too small to parallelize").
    {
      const aData = genData(rng, [1, 2048]);
      const bData = genData(rng, [2048, 2048]);
      const ref = matmulRuntime([1, 2048], aData, [2048, 2048], bData);
      const a = WNDArray.fromArray(p.core, [1, 2048], aData);
      const b = WNDArray.fromArray(p.core, [2048, 2048], bData);
      try {
        assert.ok(1 * 2048 * 2048 >= THREADED_MATMUL_MIN_POOL_WORK, "precondition: single-row case must sit at/above the default threshold");
        const before = postedSeqs(p);
        const got = threadedMatmul(p, a, b);
        try {
          assertDataBitIdentical(ref.data, got.toArray(), "single-row above-threshold call vs runtime.ts");
        } finally {
          got.dispose();
        }
        assert.deepStrictEqual(
          postedSeqs(p),
          before.map((s) => s + 1),
          "a single-row call above the volume threshold must still dispatch through the pool",
        );
      } finally {
        a.dispose();
        b.dispose();
      }
    }

    // MANY rows (2048), but k·n = 8·8 -> volume 0.13M < threshold: must run
    // on main despite the large row count (rows alone would misroute this
    // into the pool, where dispatch overhead loses).
    {
      const aData = genData(rng, [2048, 8]);
      const bData = genData(rng, [8, 8]);
      const ref = matmulRuntime([2048, 8], aData, [8, 8], bData);
      const a = WNDArray.fromArray(p.core, [2048, 8], aData);
      const b = WNDArray.fromArray(p.core, [8, 8], bData);
      try {
        assert.ok(2048 * 8 * 8 < THREADED_MATMUL_MIN_POOL_WORK, "precondition: many-rows case must sit below the default threshold");
        const before = postedSeqs(p);
        const got = threadedMatmul(p, a, b);
        try {
          assertDataBitIdentical(ref.data, got.toArray(), "many-rows below-threshold call vs runtime.ts");
        } finally {
          got.dispose();
        }
        assert.deepStrictEqual(postedSeqs(p), before, "a many-rows call below the volume threshold must not dispatch any worker");
      } finally {
        a.dispose();
        b.dispose();
      }
    }
  } finally {
    await p.dispose();
  }
});

test("auto-routing: the batch product counts toward the work volume (exact boundary through a batched call)", async () => {
  const p = await initThreadedCore(2);
  try {
    const rng = makeRng(0x524f555445424154n); // "ROUTEBAT" packed, arbitrary distinct seed
    const aShape = [8, 64, 64];
    const bShape = [64, 64];
    const volume = 8 * 64 * 64 * 64; // 2_097_152 — batch·m·k·n, NOT m·k·n
    const aData = genData(rng, aShape);
    const bData = genData(rng, bShape);
    const ref = matmulRuntime(aShape, aData, bShape, bData);
    const a = WNDArray.fromArray(p.core, aShape, aData);
    const b = WNDArray.fromArray(p.core, bShape, bData);
    try {
      const before = postedSeqs(p);

      // minPoolWork exactly at batch·m·k·n -> pool. If the router forgot the
      // batch product (m·k·n = 262144 only), this would route to main and
      // the postedSeq assertion below would fail.
      const viaPool = threadedMatmul(p, a, b, { minPoolWork: volume });
      try {
        assertShapeEqual(ref.shape, viaPool.shape, "batched boundary call shape");
        assertDataBitIdentical(ref.data, viaPool.toArray(), "batched boundary call vs runtime.ts");
      } finally {
        viaPool.dispose();
      }
      assert.deepStrictEqual(
        postedSeqs(p),
        before.map((s) => s + 1),
        "batch·m·k·n == minPoolWork must dispatch through the pool (batch product must count)",
      );

      // One above -> main.
      const afterPool = postedSeqs(p);
      const viaMain = threadedMatmul(p, a, b, { minPoolWork: volume + 1 });
      viaMain.dispose();
      assert.deepStrictEqual(postedSeqs(p), afterPool, "batch·m·k·n < minPoolWork must not dispatch any worker");
    } finally {
      a.dispose();
      b.dispose();
    }
  } finally {
    await p.dispose();
  }
});

test("auto-routing: a poisoned pool refuses even a small call the router would have run on main — lifecycle is size-independent", async () => {
  const p = await initThreadedCore(2, 5000);
  try {
    // Poison the pool: kill worker 1, let the exit event settle, then any
    // forced-pool dispatch discovers the dead worker and poisons (same
    // recipe as the deferred-free test above, just with a tiny job — the
    // poisoning itself is not the subject here).
    await p.workers[1]!.worker.terminate();
    await new Promise((resolve) => setTimeout(resolve, 100));
    const a = WNDArray.fromArray(p.core, [2, 2] as const, [1, 2, 3, 4]);
    const b = WNDArray.fromArray(p.core, [2, 2] as const, [1, 0, 0, 1]);
    assert.throws(() => threadedMatmul(p, a, b, FORCE_POOL), /not alive/, "setup: forced-pool dispatch must discover the dead worker and poison");
    assert.strictEqual(p.isPoisoned, true, "setup: pool must now be poisoned");

    // The actual subject: DEFAULT opts on a far-below-threshold call. The
    // router WOULD run this on the main thread (which still works fine on a
    // poisoned pool's core) — but the size-independent lifecycle contract
    // says threadedMatmul on a poisoned pool throws, period.
    assert.throws(() => threadedMatmul(p, a, b), /poisoned/, "a poisoned pool must refuse even main-routable calls — no size-dependent contract");

    a.dispose();
    b.dispose();
  } finally {
    await p.dispose();
  }
});

// ---------------------------------------------------------------------------
// WASM parity S0 (docs/wasm-parity-sqrt-spec.md, D6): `WNDArray.sqrt()`
// threaded-vs-stable parity. `sqrt` is NOT dispatched through the worker
// pool (the pool only ever routes `threadedMatmul` — see the spec's
// Nicht-Ziele); it runs directly on the resident core, on BOTH the stable
// and threads artifacts, since they're the same crate. `pool.core` is a
// real `CoreExports` for the threads-compiled artifact (see `dispatchAndRun`
// in threaded.ts, and this file's own `makeOperand(pool.core, ...)` calls
// above for matmul reference-building) — `WNDArray.sqrt()` runs on it
// exactly like every other non-matmul resident op already does. Extends
// the file's established threaded-vs-stable differential to `sqrt`,
// reusing the persistent `pools`/`stableCore` already set up above.
// ---------------------------------------------------------------------------

function runSqrtCase(name: string, shape: number[], asView: boolean, rng: Rng, special = false): void {
  test(name, () => {
    const data = special ? genDataSpecial(rng, shape) : genData(rng, shape);
    const ref = sqrtRuntime(data);

    const stableOperand = makeOperand(stableCore, asView, shape, data);
    let stableData: Float64Array;
    try {
      const got = stableOperand.arr.sqrt();
      try {
        // D-V2.3 fallout (see elementwise.test.ts's identical comment):
        // `got: AnyWNDArray = WNDArray<any>`, so `.shape` is `Readonly<any>`
        // — doesn't structurally collapse to `any`, so no longer matches
        // `readonly number[]` (TS2740). Cast only; runtime value unaffected.
        assertShapeEqual(shape, got.shape as readonly number[], `${name}: stable shape`);
        stableData = got.toArray();
      } finally {
        got.dispose();
      }
    } finally {
      disposeAll(stableOperand);
    }
    assertDataBitIdentical(ref, stableData, `${name}: runtime.ts vs stable resident`);

    for (const wc of WORKER_COUNTS) {
      const pool = pools.get(wc)!;
      const operand = makeOperand(pool.core, asView, shape, data);
      try {
        const got = operand.arr.sqrt();
        try {
          assertShapeEqual(shape, got.shape as readonly number[], `${name} workers=${wc}`);
          const gotData = got.toArray();
          assertDataBitIdentical(ref, gotData, `${name} workers=${wc} vs runtime.ts`);
          assertDataBitIdentical(stableData, gotData, `${name} workers=${wc} vs stable resident`);
        } finally {
          got.dispose();
        }
      } finally {
        disposeAll(operand);
      }
    }
  });
}

{
  const rng = makeRng(0x5351525f5448524en); // "SQR_THRN"-ish
  runSqrtCase("sqrt threaded parity: contiguous [2,3,4]", [2, 3, 4], false, rng);
  // Pflicht view case (spec D6): a transposed view on every pool AND stable.
  runSqrtCase("sqrt threaded parity: transposed view [4,3,2]", [4, 3, 2], true, rng);
  runSqrtCase("sqrt threaded parity: rank-0 scalar", [], false, rng);
  runSqrtCase("sqrt threaded parity: size-0 dim [0,5]", [0, 5], false, rng);
  // C-2 (Baustein-C-Befund der Verify-Runde): prove the "bit-identical incl.
  // IEEE special values" M1 claim ON the threads artifact directly, not only by
  // the same-crate argument — NaN / +-0 / +-Inf / subnormals through the
  // threaded resident core, contiguous AND on a transposed view.
  runSqrtCase("sqrt threaded parity: special values, contiguous [2,3,4]", [2, 3, 4], false, rng, true);
  runSqrtCase("sqrt threaded parity: special values, transposed view [4,3,2]", [4, 3, 2], true, rng, true);
}

// ---------------------------------------------------------------------------
// WASM parity S1 (docs/wasm-parity-scalar-spec.md, D6): `WNDArray.{add,sub,
// mul,div}(s)` threaded-vs-stable parity. Same non-pool-routed reasoning as
// `sqrt` above (spec's Nicht-Ziele: the pool only ever routes
// `threadedMatmul`) — the scalar ops run directly on the resident core, on
// BOTH the stable and threads artifacts, since they're the same crate.
// Extends the file's established threaded-vs-stable differential to the
// four scalar ops, reusing the persistent `pools`/`stableCore` set up above.
// ---------------------------------------------------------------------------

type ScalarOp = "add" | "sub" | "mul" | "div";
const SCALAR_OPS: readonly ScalarOp[] = ["add", "sub", "mul", "div"];

function callResidentScalar(op: ScalarOp, w: AnyWNDArray, s: number): AnyWNDArray {
  if (op === "add") return w.add(s);
  if (op === "sub") return w.sub(s);
  if (op === "mul") return w.mul(s);
  return w.div(s);
}

function runScalarCase(name: string, op: ScalarOp, shape: number[], asView: boolean, rng: Rng, s: number, special = false): void {
  test(name, () => {
    const data = special ? genDataSpecial(rng, shape) : genData(rng, shape);
    const ref = scalarElementwiseRuntime(op, data, s);

    const stableOperand = makeOperand(stableCore, asView, shape, data);
    let stableData: Float64Array;
    try {
      const got = callResidentScalar(op, stableOperand.arr, s);
      try {
        assertShapeEqual(shape, got.shape as readonly number[], `${name}: stable shape`);
        stableData = got.toArray();
      } finally {
        got.dispose();
      }
    } finally {
      disposeAll(stableOperand);
    }
    assertDataBitIdentical(ref, stableData, `${name}: runtime.ts vs stable resident`);

    for (const wc of WORKER_COUNTS) {
      const pool = pools.get(wc)!;
      const operand = makeOperand(pool.core, asView, shape, data);
      try {
        const got = callResidentScalar(op, operand.arr, s);
        try {
          assertShapeEqual(shape, got.shape as readonly number[], `${name} workers=${wc}`);
          const gotData = got.toArray();
          assertDataBitIdentical(ref, gotData, `${name} workers=${wc} vs runtime.ts`);
          assertDataBitIdentical(stableData, gotData, `${name} workers=${wc} vs stable resident`);
        } finally {
          got.dispose();
        }
      } finally {
        disposeAll(operand);
      }
    }
  });
}

{
  const rng = makeRng(0x5343414c5f5448524en); // "SCAL_THRN"-ish
  for (const op of SCALAR_OPS) {
    runScalarCase(`${op}(s) threaded parity: contiguous [2,3,4]`, op, [2, 3, 4], false, rng, 2.5);
    // Pflicht view case (spec D6): a transposed view on every pool AND stable.
    runScalarCase(`${op}(s) threaded parity: transposed view [4,3,2]`, op, [4, 3, 2], true, rng, 2.5);
    // C-2 lesson (from the S0/sqrt verify round): at least one genDataSpecial
    // case per op, contiguous AND view, directly on the threads artifact.
    runScalarCase(`${op}(s) threaded parity: special values, contiguous [2,3,4]`, op, [2, 3, 4], false, rng, 2.5, true);
    runScalarCase(`${op}(s) threaded parity: special values, transposed view [4,3,2]`, op, [4, 3, 2], true, rng, 2.5, true);
  }
}

// ---------------------------------------------------------------------------
// WASM parity S2 (docs/wasm-parity-mean-spec.md, D6): `WNDArray.mean(axis?,
// keepdims?)` threaded-vs-stable parity. Same non-pool-routed reasoning as
// `sqrt`/the scalar ops above (spec's Nicht-Ziele: the pool only ever routes
// `threadedMatmul`; `mean` is a pure-TS composition of `sum` + `div`, no new
// kernel at all) — it runs directly on the resident core, on BOTH the stable
// and threads artifacts, since they're the same crate. Extends the file's
// established threaded-vs-stable differential to `mean`, reusing the
// persistent `pools`/`stableCore` set up above.
//
// F1 methodology (spec addendum, CRITICAL — same as resident.test.ts's/
// special-values.test.ts's own mean blocks): `meanRuntime` takes no
// `keepdims` parameter and always returns the REDUCED shape — shape is
// compared against `keepdims ? keepDimsShape(shape, axis) : ref.shape`,
// never directly against `meanRuntime(...).shape` for keepdims=true.
// ---------------------------------------------------------------------------

function runMeanCase(name: string, shape: number[], axis: number | undefined, keepdims: boolean, asView: boolean, rng: Rng, special = false): void {
  test(name, () => {
    const data = special ? genDataSpecial(rng, shape) : genData(rng, shape);
    const ref = meanRuntime(shape, data, axis);
    const expectedShape = keepdims ? keepDimsShape(shape, axis) : ref.shape;

    const stableOperand = makeOperand(stableCore, asView, shape, data);
    let stableData: Float64Array;
    try {
      const got = stableOperand.arr.mean(axis, keepdims);
      try {
        assertShapeEqual(expectedShape, got.shape as readonly number[], `${name}: stable shape`);
        stableData = got.toArray();
      } finally {
        got.dispose();
      }
    } finally {
      disposeAll(stableOperand);
    }
    assertDataBitIdentical(ref.data, stableData, `${name}: runtime.ts vs stable resident`);

    for (const wc of WORKER_COUNTS) {
      const pool = pools.get(wc)!;
      const operand = makeOperand(pool.core, asView, shape, data);
      try {
        const got = operand.arr.mean(axis, keepdims);
        try {
          assertShapeEqual(expectedShape, got.shape as readonly number[], `${name} workers=${wc}`);
          const gotData = got.toArray();
          assertDataBitIdentical(ref.data, gotData, `${name} workers=${wc} vs runtime.ts`);
          assertDataBitIdentical(stableData, gotData, `${name} workers=${wc} vs stable resident`);
        } finally {
          got.dispose();
        }
      } finally {
        disposeAll(operand);
      }
    }
  });
}

{
  const rng = makeRng(0x4d45414e5f5448524en); // "MEAN_THRN"-ish
  runMeanCase("mean() threaded parity: contiguous [2,3,4]", [2, 3, 4], undefined, false, false, rng);
  // Pflicht view case (spec D6): a transposed view on every pool AND stable.
  runMeanCase("mean() threaded parity: transposed view [4,3,2]", [4, 3, 2], undefined, false, true, rng);
  runMeanCase("mean() threaded parity: rank-0 scalar", [], undefined, false, false, rng);
  runMeanCase("mean() threaded parity: size-0 dim [0,5]", [0, 5], undefined, false, false, rng);
  // Mandatory axis case (spec D5/task): both a plain axis form and a
  // keepdims form, contiguous AND view.
  runMeanCase("mean(axis) threaded parity: contiguous [2,3,4]", [2, 3, 4], 1, false, false, rng);
  runMeanCase("mean(axis) threaded parity: transposed view [4,3,2]", [4, 3, 2], -1, false, true, rng);
  runMeanCase("mean(axis, keepdims=true) threaded parity: contiguous [2,3,4]", [2, 3, 4], 0, true, false, rng);
  // C-2 lesson (from the S0/sqrt verify round): at least one genDataSpecial
  // case, contiguous AND view, directly on the threads artifact — niladic
  // and axis form.
  runMeanCase("mean() threaded parity: special values, contiguous [2,3,4]", [2, 3, 4], undefined, false, false, rng, true);
  runMeanCase("mean() threaded parity: special values, transposed view [4,3,2]", [4, 3, 2], undefined, false, true, rng, true);
  runMeanCase("mean(axis) threaded parity: special values, contiguous [2,3,4]", [2, 3, 4], 1, false, false, rng, true);
}

// ---------------------------------------------------------------------------
// WASM parity S3 (docs/wasm-parity-item-stack-spec.md, D6/T8): `WNDArray.item`
// threaded-vs-stable parity. `item` runs NO WASM code at all (D3: a plain
// strided TS read over `core.memory.buffer` — the fourth M1 case this
// campaign has hit) — it works identically on BOTH the stable and threads
// artifacts, since it never touches a kernel export. Extends the file's
// established threaded-vs-stable differential to `item`, reusing the
// persistent `pools`/`stableCore` set up above.
// ---------------------------------------------------------------------------

function runItemCase(name: string, shape: number[], asView: boolean, rng: Rng, special = false): void {
  test(name, () => {
    const data = special ? genDataSpecial(rng, shape) : genData(rng, shape);
    const indices = shape.map((d) => rng.nextInt(0, Math.max(d - 1, 0)));
    const ref = itemRuntime(shape, data, indices);

    const stableOperand = makeOperand(stableCore, asView, shape, data);
    let stableResult: number;
    try {
      stableResult = stableOperand.arr.item(...indices);
    } finally {
      disposeAll(stableOperand);
    }
    assert.ok(Object.is(ref, stableResult), `${name}: stable expected ${ref}, got ${stableResult}`);

    for (const wc of WORKER_COUNTS) {
      const pool = pools.get(wc)!;
      const operand = makeOperand(pool.core, asView, shape, data);
      try {
        const got = operand.arr.item(...indices);
        assert.ok(Object.is(ref, got), `${name} workers=${wc}: expected ${ref} (runtime.ts), got ${got}`);
        assert.ok(Object.is(stableResult, got), `${name} workers=${wc}: expected ${stableResult} (stable), got ${got}`);
      } finally {
        disposeAll(operand);
      }
    }
  });
}

{
  const rng = makeRng(0x4954454d5f5448524en); // "ITEM_THRN"-ish
  runItemCase("item threaded parity: contiguous [2,3,4]", [2, 3, 4], false, rng);
  // Pflicht view case (spec D6/T8): a transposed view on every pool AND stable.
  runItemCase("item threaded parity: transposed view [4,3,2]", [4, 3, 2], true, rng);
  runItemCase("item threaded parity: rank-0 scalar", [], false, rng);
  // C-2 lesson (from the S0/sqrt verify round): at least one genDataSpecial
  // case, contiguous AND view, directly on the threads artifact.
  runItemCase("item threaded parity: special values, contiguous [2,3,4]", [2, 3, 4], false, rng, true);
  runItemCase("item threaded parity: special values, transposed view [4,3,2]", [4, 3, 2], true, rng, true);
}

// ---------------------------------------------------------------------------
// WASM parity S3 (docs/wasm-parity-item-stack-spec.md, D6/T8): `WNDArray.stack`
// threaded-vs-stable parity. `stack` is NOT dispatched through the worker
// pool (spec's Nicht-Ziele: the pool only ever routes `threadedMatmul`) —
// each row's `nt_materialize` call runs directly against the resident core,
// on BOTH the stable and threads artifacts, since they're the same crate.
// Extends the file's established threaded-vs-stable differential to `stack`,
// reusing the persistent `pools`/`stableCore` set up above.
// ---------------------------------------------------------------------------

/** Builds `rowsData.length` `WNDArray` rows on `core`: contiguous where
 * `asView[i]` is false, else a `[d,2]` base's column-0 integer-index view
 * (non-natural stride, the "geschnittene Zeile" D7 requires) — same
 * technique `resident.test.ts`'s own randomized stack grid uses. */
function buildStackRows(core: CoreExports, rowsData: readonly Float64Array[], asView: readonly boolean[]): { rows: AnyWNDArray[]; owners: AnyWNDArray[] } {
  const rows: AnyWNDArray[] = [];
  const owners: AnyWNDArray[] = [];
  const d = rowsData[0]?.length ?? 0;
  const rowShape: number[] = [d]; // dynamic: a plain runtime test, no need to pay the literal-tuple type cost
  const baseShape: number[] = [d, 2];
  for (let i = 0; i < rowsData.length; i++) {
    const data = rowsData[i]!;
    if (asView[i]) {
      const baseData: number[] = [];
      for (let k = 0; k < d; k++) baseData.push(data[k]!, 0);
      const base = WNDArray.fromArray(core, baseShape, baseData);
      owners.push(base);
      rows.push(base.slice(...wideSpecs(null, 0)));
    } else {
      rows.push(WNDArray.fromArray(core, rowShape, data));
    }
  }
  return { rows, owners };
}

function runStackCase(name: string, n: number, d: number, asView: boolean[], rng: Rng, special = false): void {
  test(name, () => {
    const rowsData: Float64Array[] = [];
    for (let i = 0; i < n; i++) rowsData.push(special ? genDataSpecial(rng, [d]) : genData(rng, [d]));
    const ref = stackRuntime(rowsData.map((data) => ({ shape: [d], data })));

    const { rows: stableRows, owners: stableOwners } = buildStackRows(stableCore, rowsData, asView);
    let stableData: Float64Array;
    try {
      const stacked = WNDArray.stack(stableCore, stableRows);
      try {
        stableData = stacked.toArray();
      } finally {
        stacked.dispose();
      }
    } finally {
      for (const r of stableRows) r.dispose();
      for (const o of stableOwners) o.dispose();
    }
    assertDataBitIdentical(ref.data, stableData, `${name}: runtime.ts vs stable resident`);

    for (const wc of WORKER_COUNTS) {
      const pool = pools.get(wc)!;
      const { rows, owners } = buildStackRows(pool.core, rowsData, asView);
      try {
        const stacked = WNDArray.stack(pool.core, rows);
        try {
          const gotData = stacked.toArray();
          assertDataBitIdentical(ref.data, gotData, `${name} workers=${wc} vs runtime.ts`);
          assertDataBitIdentical(stableData, gotData, `${name} workers=${wc} vs stable resident`);
        } finally {
          stacked.dispose();
        }
      } finally {
        for (const r of rows) r.dispose();
        for (const o of owners) o.dispose();
      }
    }
  });
}

{
  const rng = makeRng(0x5354434b5f5448524en); // "STCK_THRN"-ish
  runStackCase("stack threaded parity: contiguous rows, n=4 d=5", 4, 5, [false, false, false, false], rng);
  // Pflicht view case (spec D6/T8): at least one view row on every pool AND stable.
  runStackCase("stack threaded parity: mixed contiguous+view rows, n=4 d=5", 4, 5, [false, true, false, true], rng);
  runStackCase("stack threaded parity: n=1", 1, 5, [false], rng);
  runStackCase("stack threaded parity: d=0", 3, 0, [false, false, false], rng);
  // C-2 lesson (from the S0/sqrt verify round): at least one genDataSpecial
  // case, contiguous AND view, directly on the threads artifact.
  runStackCase("stack threaded parity: special values, contiguous rows, n=4 d=5", 4, 5, [false, false, false, false], rng, true);
  runStackCase("stack threaded parity: special values, mixed contiguous+view rows, n=4 d=5", 4, 5, [false, true, false, true], rng, true);
}


// ---------------------------------------------------------------------------
// WASM parity S4 (docs/wasm-parity-argmax-spec.md, D1/D7): `WNDArray.argmax`
// threaded-vs-stable parity. `argmax` adds two REAL kernels
// (`nt_argmax_{all,axis}_strided`), and the threads artifact is built from the
// same crate — so this block proves the two artifacts agree bit-for-bit on the
// new exports, exactly as the sqrt/scalar blocks above do for theirs. Extends
// the file's established threaded-vs-stable differential, reusing the
// persistent `pools`/`stableCore` set up above.
//
// Oracle methodology (spec D1, same as resident.test.ts's own argmax block):
// DATA against `argmaxRuntime`; the keepdims SHAPE via structural invariants,
// never `keepDimsShape` (which is what the method under test itself calls).
// ---------------------------------------------------------------------------

import { argmaxRuntime } from "../src/runtime.ts";

/** The message of a call that must throw (local to this S4 block). */
function throwMsg(fn: () => unknown): string {
  try {
    fn();
  } catch (e) {
    return e instanceof Error ? e.message : String(e);
  }
  throw new Error("expected fn() to throw, but it did not");
}

/** keepdims SHAPE via structural invariants only — deliberately NOT
 * `keepDimsShape` (circular for argmax; spec D1(2)). */
function assertArgmaxKeepdimsShape(
  inputShape: readonly number[],
  axis: number | undefined,
  keptShape: readonly number[],
  reducedShape: readonly number[],
  ctx: string,
): void {
  const prod = (s: readonly number[]): number => s.reduce((acc, d) => acc * d, 1);
  assert.strictEqual(keptShape.length, inputShape.length, `${ctx}: keepdims must preserve rank`);
  assert.strictEqual(prod(keptShape), prod(reducedShape), `${ctx}: keepdims must not change the element count`);
  if (axis === undefined) {
    assert.ok(keptShape.every((d) => d === 1), `${ctx}: full reduction with keepdims must be all-ones, got [${keptShape.join(",")}]`);
    return;
  }
  const normAxis = axis < 0 ? inputShape.length + axis : axis;
  assert.strictEqual(keptShape[normAxis], 1, `${ctx}: the reduced axis must be size 1 under keepdims`);
  assert.deepStrictEqual(
    [...keptShape.slice(0, normAxis), ...keptShape.slice(normAxis + 1)],
    [...reducedShape],
    `${ctx}: keepdims shape minus the reduced axis must equal the non-keepdims shape`,
  );
}

function runArgmaxCase(name: string, shape: number[], axis: number | undefined, keepdims: boolean, asView: boolean, rng: Rng, special = false): void {
  test(name, () => {
    const data = special ? genDataSpecial(rng, shape) : genData(rng, shape);
    const ref = argmaxRuntime(shape, data, axis);

    const assertShape = (got: readonly number[], ctx: string): void => {
      if (keepdims) assertArgmaxKeepdimsShape(shape, axis, got, ref.shape, ctx);
      else assertShapeEqual(ref.shape, got, ctx);
    };

    const stableOperand = makeOperand(stableCore, asView, shape, data);
    let stableData: Float64Array;
    let stableNiladic: number;
    try {
      stableNiladic = stableOperand.arr.argmax();
      const got = stableOperand.arr.argmax(axis, keepdims);
      try {
        assertShape(got.shape as readonly number[], `${name}: stable shape`);
        stableData = got.toArray();
      } finally {
        got.dispose();
      }
    } finally {
      disposeAll(stableOperand);
    }
    assertDataBitIdentical(ref.data, stableData, `${name}: runtime.ts vs stable resident`);
    assert.strictEqual(stableNiladic, argmaxRuntime(shape, data, undefined).data[0], `${name}: stable niladic vs runtime.ts`);

    for (const wc of WORKER_COUNTS) {
      const pool = pools.get(wc)!;
      const operand = makeOperand(pool.core, asView, shape, data);
      try {
        assert.strictEqual(operand.arr.argmax(), stableNiladic, `${name} workers=${wc}: niladic argmax() vs stable`);
        const got = operand.arr.argmax(axis, keepdims);
        try {
          assertShape(got.shape as readonly number[], `${name} workers=${wc}`);
          const gotData = got.toArray();
          assertDataBitIdentical(ref.data, gotData, `${name} workers=${wc} vs runtime.ts`);
          assertDataBitIdentical(stableData, gotData, `${name} workers=${wc} vs stable resident`);
        } finally {
          got.dispose();
        }
      } finally {
        disposeAll(operand);
      }
    }
  });
}

{
  const rng = makeRng(0x41524758_5f544852n); // "ARGX_THR"
  runArgmaxCase("argmax() threaded parity: contiguous [2,3,4]", [2, 3, 4], undefined, false, false, rng);
  // Pflicht view case (spec D7): a transposed view on every pool AND stable —
  // the case where the LOGICAL index provably differs from the memory offset.
  runArgmaxCase("argmax() threaded parity: transposed view [4,3,2]", [4, 3, 2], undefined, false, true, rng);
  runArgmaxCase("argmax() threaded parity: rank-0 scalar", [], undefined, false, false, rng);
  // Axis form: plain, negative, keepdims — contiguous AND view.
  runArgmaxCase("argmax(axis) threaded parity: contiguous [2,3,4]", [2, 3, 4], 1, false, false, rng);
  runArgmaxCase("argmax(axis) threaded parity: transposed view [4,3,2]", [4, 3, 2], -1, false, true, rng);
  runArgmaxCase("argmax(axis, keepdims=true) threaded parity: contiguous [2,3,4]", [2, 3, 4], 0, true, false, rng);
  runArgmaxCase("argmax(axis, keepdims=true) threaded parity: transposed view [4,3,2]", [4, 3, 2], 2, true, true, rng);
  runArgmaxCase("argmax(undefined, keepdims=true) threaded parity: contiguous [2,3,4]", [2, 3, 4], undefined, true, false, rng);
  // C-2 lesson (from the S0/sqrt verify round): at least one genDataSpecial
  // case, contiguous AND view, directly on the threads artifact — niladic and
  // axis form. For argmax the special values are not decoration: the total
  // order is DEFINED on them (NaN maximal, ties by first index).
  runArgmaxCase("argmax() threaded parity: special values, contiguous [2,3,4]", [2, 3, 4], undefined, false, false, rng, true);
  runArgmaxCase("argmax() threaded parity: special values, transposed view [4,3,2]", [4, 3, 2], undefined, false, true, rng, true);
  runArgmaxCase("argmax(axis) threaded parity: special values, contiguous [2,3,4]", [2, 3, 4], 1, false, false, rng, true);
  runArgmaxCase("argmax(axis) threaded parity: special values, transposed view [4,3,2]", [4, 3, 2], -2, true, true, rng, true);
}

test("argmax threaded parity: the empty-reduction throws carry the SAME stem on every pool as on the stable core", async () => {
  const shape: number[] = [0, 3];
  const stableEmpty = WNDArray.fromArray(stableCore, shape, []);
  let stableMsg: string;
  try {
    stableMsg = throwMsg(() => stableEmpty.argmax());
    assert.strictEqual(stableMsg, "argmax: attempt to get argmax of an empty array", "sanity: exact expected wording");
  } finally {
    stableEmpty.dispose();
  }
  for (const wc of WORKER_COUNTS) {
    const pool = pools.get(wc)!;
    const e = WNDArray.fromArray(pool.core, shape, []);
    try {
      assert.strictEqual(throwMsg(() => e.argmax()), stableMsg, `workers=${wc}: niladic empty stem`);
      assert.strictEqual(throwMsg(() => e.argmax(0)), stableMsg, `workers=${wc}: zero-length-axis stem`);
    } finally {
      e.dispose();
    }
  }
});

// =============================================================================
// WASM parity S5 (docs/wasm-parity-topk-spec.md, D1/D7): `WNDArray.topk`
// threaded-vs-stable parity — the last op of the S0-S5 campaign. `topk` adds
// ONE real kernel (`nt_topk_strided`), and the threads artifact is built from
// the same crate, so parity is expected by construction; this block PROVES it
// rather than asserting it, on every pool in `WORKER_COUNTS`.
//
// Two things make this block stricter than the argmax one above it:
//  - `topk` returns real DATA VALUES, so the comparison is over raw 64-bit
//    patterns (`assertDataBitIdentical`'s `Object.is` cannot tell two NaN
//    payloads apart);
//  - the receiver is exercised as a genuinely STRIDED rank-1 view, which
//    `makeView`'s transpose-involution cannot produce at rank 1 (transposing
//    a vector is a no-op).
// =============================================================================

import { topkRuntime } from "../src/runtime.ts";

/** Raw 64-bit read from a `Float64Array`'s own buffer — payload-preserving
 * (never the `new Float64Array([x])` round trip). */
function topkBitsAt(a: Float64Array, i: number): bigint {
  return new DataView(a.buffer, a.byteOffset, a.byteLength).getBigUint64(i * 8, true);
}

function assertTopkBitsIdentical(expected: Float64Array, actual: Float64Array, ctx: string): void {
  assert.strictEqual(actual.length, expected.length, `${ctx}: length`);
  for (let i = 0; i < expected.length; i++) {
    assert.strictEqual(topkBitsAt(actual, i), topkBitsAt(expected, i), `${ctx}: bits differ at ${i}: reference=${expected[i]} wasm=${actual[i]}`);
  }
}

/** A rank-1 operand: either contiguous, or a genuine stride-3 view over a
 * padded buffer whose LOGICAL content is exactly `refData`. */
function makeTopkOperand(core: CoreExports, asView: boolean, refData: Float64Array): Operand {
  if (!asView) {
    const arr = WNDArray.fromArray(core, [refData.length], refData);
    return { arr, owners: [arr] };
  }
  const padded = new Float64Array(refData.length * 3);
  for (let i = 0; i < padded.length; i++) padded[i] = -12345.5;
  for (let i = 0; i < refData.length; i++) padded[i * 3] = refData[i] ?? 0;
  const base = WNDArray.fromArray(core, [refData.length * 3], padded);
  const view = base.slice(...wideSpecs({ step: 3 })) as AnyWNDArray;
  return { arr: view, owners: [base, view] };
}

function runTopkCase(name: string, n: number, k: number, asView: boolean, rng: Rng, special = false): void {
  test(name, () => {
    const data = special ? genDataSpecial(rng, [n]) : genData(rng, [n]);
    const ref = topkRuntime([n], data, k);

    const stableOperand = makeTopkOperand(stableCore, asView, data);
    let stableValues: Float64Array;
    let stableIndices: Float64Array;
    try {
      const got = stableOperand.arr.topk(k as never);
      try {
        assertShapeEqual([k], got.values.shape as readonly number[], `${name}: stable values shape`);
        assertShapeEqual([k], got.indices.shape as readonly number[], `${name}: stable indices shape`);
        stableValues = got.values.toArray();
        stableIndices = got.indices.toArray();
      } finally {
        got.values.dispose();
        got.indices.dispose();
      }
    } finally {
      disposeAll(stableOperand);
    }
    assertTopkBitsIdentical(ref.values, stableValues, `${name}: runtime.ts vs stable resident (values)`);
    assertTopkBitsIdentical(ref.indices, stableIndices, `${name}: runtime.ts vs stable resident (indices)`);

    for (const wc of WORKER_COUNTS) {
      const pool = pools.get(wc)!;
      const operand = makeTopkOperand(pool.core, asView, data);
      try {
        const got = operand.arr.topk(k as never);
        try {
          assertShapeEqual([k], got.values.shape as readonly number[], `${name} workers=${wc}: values shape`);
          const gotValues = got.values.toArray();
          const gotIndices = got.indices.toArray();
          assertTopkBitsIdentical(ref.values, gotValues, `${name} workers=${wc} values vs runtime.ts`);
          assertTopkBitsIdentical(ref.indices, gotIndices, `${name} workers=${wc} indices vs runtime.ts`);
          assertTopkBitsIdentical(stableValues, gotValues, `${name} workers=${wc} values vs stable resident`);
          assertTopkBitsIdentical(stableIndices, gotIndices, `${name} workers=${wc} indices vs stable resident`);
        } finally {
          got.values.dispose();
          got.indices.dispose();
        }
      } finally {
        disposeAll(operand);
      }
    }
  });
}

{
  const rng = makeRng(0x544f504b_5f544852n); // "TOPK_THR"
  runTopkCase("topk threaded parity: contiguous n=12 k=4", 12, 4, false, rng);
  // Pflicht view case (spec D7): a genuinely strided rank-1 receiver on every
  // pool AND stable — where the LOGICAL index provably differs from the
  // memory offset, and a confusion would corrupt the VALUES too.
  runTopkCase("topk threaded parity: strided view n=12 k=4", 12, 4, true, rng);
  runTopkCase("topk threaded parity: k = 0 (empty result)", 8, 0, false, rng);
  runTopkCase("topk threaded parity: k = 1", 8, 1, false, rng);
  runTopkCase("topk threaded parity: k = n (whole vector, sorted)", 8, 8, false, rng);
  runTopkCase("topk threaded parity: k = n on a strided view", 8, 8, true, rng);
  // C-2 lesson (from the S0/sqrt verify round): at least one genDataSpecial
  // case, contiguous AND view, directly on the threads artifact. For `topk`
  // the special values are not decoration — the total order is DEFINED on
  // them (NaN first, ties by ascending index, +0/-0 equal).
  runTopkCase("topk threaded parity: special values, contiguous n=10 k=5", 10, 5, false, rng, true);
  runTopkCase("topk threaded parity: special values, strided view n=10 k=5", 10, 5, true, rng, true);
  runTopkCase("topk threaded parity: special values, k = n", 9, 9, false, rng, true);
}

test("topk threaded parity: a CONSTRUCTED tie raster agrees on every pool (randomized data cannot reach these branches)", () => {
  const NAN = Number.NaN;
  const CASES: readonly { data: number[]; k: number }[] = [
    { data: [5, 5, 5, 5, 5], k: 3 },
    { data: [NAN, NAN, NAN, NAN], k: 2 },
    { data: [1, NAN, 2, NAN, 3, NAN], k: 2 },
    { data: [-0, 0, -0, 0, -1], k: 2 },
    { data: [3, NAN, 3, -0, 0, NAN, 3], k: 5 },
  ];
  for (const c of CASES) {
    const ref = topkRuntime([c.data.length], Float64Array.from(c.data), c.k);
    for (const wc of WORKER_COUNTS) {
      const pool = pools.get(wc)!;
      const a = WNDArray.fromArray(pool.core, [c.data.length], c.data);
      try {
        const got = a.topk(c.k as never);
        try {
          assertTopkBitsIdentical(ref.values, got.values.toArray(), `tie raster [${c.data.join(",")}] k=${c.k} workers=${wc}: values`);
          assertTopkBitsIdentical(ref.indices, got.indices.toArray(), `tie raster [${c.data.join(",")}] k=${c.k} workers=${wc}: indices`);
        } finally {
          got.values.dispose();
          got.indices.dispose();
        }
      } finally {
        a.dispose();
      }
    }
  }
});

test("topk threaded parity: a NON-CANONICAL NaN payload survives byte-identically on every pool", () => {
  const data = new Float64Array(4);
  data[0] = 1;
  data[2] = Number.POSITIVE_INFINITY;
  data[3] = -7;
  new DataView(data.buffer).setBigUint64(1 * 8, 0x7ff8_0000_cafe_baben, true);
  const ref = topkRuntime([4], data, 4);
  for (const wc of WORKER_COUNTS) {
    const pool = pools.get(wc)!;
    const a = WNDArray.fromArray(pool.core, [4], data);
    try {
      const got = a.topk(4 as never);
      try {
        const values = got.values.toArray();
        assert.strictEqual(topkBitsAt(values, 0), 0x7ff8_0000_cafe_baben, `workers=${wc}: the exact NaN payload must survive the threads artifact too`);
        assertTopkBitsIdentical(ref.values, values, `workers=${wc}: NaN-payload values vs runtime.ts`);
        assertTopkBitsIdentical(ref.indices, got.indices.toArray(), `workers=${wc}: NaN-payload indices vs runtime.ts`);
      } finally {
        got.values.dispose();
        got.indices.dispose();
      }
    } finally {
      a.dispose();
    }
  }
});

test("topk threaded parity: the three prevalidated throws carry the SAME stems on every pool as on the stable core", () => {
  const shape2d: number[] = [2, 3];
  const shape1d: number[] = [3];
  const stableM = WNDArray.fromArray(stableCore, shape2d, [1, 2, 3, 4, 5, 6]);
  const stableV = WNDArray.fromArray(stableCore, shape1d, [1, 2, 3]);
  let rankMsg: string;
  let invalidMsg: string;
  let boundsMsg: string;
  try {
    rankMsg = throwMsg(() => stableM.topk(2 as never));
    invalidMsg = throwMsg(() => stableV.topk(-1 as never));
    boundsMsg = throwMsg(() => stableV.topk(4 as never));
    assert.strictEqual(rankMsg, "topk: expected a 1-D vector (got shape [2,3])", "sanity: exact expected wording");
    assert.strictEqual(invalidMsg, "topk: k must be a non-negative integer (got -1)", "sanity: exact expected wording");
    assert.strictEqual(boundsMsg, "topk: k=4 exceeds the vector length 3", "sanity: exact expected wording");
  } finally {
    stableM.dispose();
    stableV.dispose();
  }
  for (const wc of WORKER_COUNTS) {
    const pool = pools.get(wc)!;
    const m = WNDArray.fromArray(pool.core, shape2d, [1, 2, 3, 4, 5, 6]);
    const v = WNDArray.fromArray(pool.core, shape1d, [1, 2, 3]);
    try {
      assert.strictEqual(throwMsg(() => m.topk(2 as never)), rankMsg, `workers=${wc}: rank stem`);
      assert.strictEqual(throwMsg(() => v.topk(-1 as never)), invalidMsg, `workers=${wc}: invalid-k stem`);
      assert.strictEqual(throwMsg(() => v.topk(4 as never)), boundsMsg, `workers=${wc}: bounds stem`);
    } finally {
      m.dispose();
      v.dispose();
    }
  }
});
