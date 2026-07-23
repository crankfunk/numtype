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
import { matmulRuntime, scalarElementwiseRuntime, sqrtRuntime, transposeRuntime } from "../src/runtime.ts";
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
import { assertDataBitIdentical, assertShapeEqual } from "./assert-helpers.ts";
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

