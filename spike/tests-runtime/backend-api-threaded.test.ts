/**
 * Item 10 — Backend-Wahl-API (docs/item-10-backend-api-spec.md): runtime
 * differential tests for the threaded half of the facade
 * (`NDArray.backend("threaded")` / `ThreadedBackend`). Own file, wired into
 * `pnpm test:threaded` (builds `build:wasm:threads`) — Verify-Blocker 3:
 * `test:resident` does NOT build the threads artifact, so this facade's
 * differential coverage cannot live alongside `backend-api.test.ts`.
 *
 * Covers the testplan's two threaded-backend bullets:
 *  - `ThreadedBackend.matmul` bit-identical to `WNDArray.matmul` (single-
 *    threaded, same pool core) AND the naive `runtime.ts` reference, both
 *    over and under the size-based auto-routing threshold (`opts.minPoolWork`
 *    `0`/`Infinity`, exactly like `threaded.test.ts` pins the underlying
 *    `threadedMatmul` routes).
 *  - Lifecycle: `backend.dispose()` frees the pool (Kern-06 lifecycle
 *    contract unchanged — disposed/poisoned semantics); cross-pool operands
 *    throw (the pre-existing `threadedMatmul` pool-identity check).
 */
import assert from "node:assert";
import { test } from "node:test";
import { NDArray } from "../src/ndarray.ts";
import { matmulRuntime } from "../src/runtime.ts";
import { getThreadedPoolFreeCount } from "../src/wasm/threaded.ts";
import { assertDataBitIdentical, assertShapeEqual } from "./assert-helpers.ts";

// --- ThreadedBackend.matmul: bit-identical, both auto-routing sides --------

test("ThreadedBackend.matmul: bit-identical to WNDArray.matmul and the naive runtime.ts reference, forced through the POOL (minPoolWork: 0)", async () => {
  const aShape = [4, 3];
  const bShape = [3, 5];
  const aData = new Float64Array(Array.from({ length: 12 }, (_, i) => i - 5.5));
  const bData = new Float64Array(Array.from({ length: 15 }, (_, i) => 0.5 * i - 3));
  const ref = matmulRuntime(aShape, aData, bShape, bData);

  const backend = await NDArray.backend("threaded", { workers: 2, minPoolWork: 0 });
  const a = backend.fromArray(aShape, aData);
  const b = backend.fromArray(bShape, bData);
  try {
    const gotPool = backend.matmul(a, b); // backend default minPoolWork: 0 -> forced pool
    try {
      const gotMain = a.matmul(b); // same pool's core, single-threaded main-thread kernel
      try {
        assertShapeEqual(ref.shape, gotPool.shape, "ThreadedBackend.matmul (pool) vs runtime.ts shape");
        assertDataBitIdentical(ref.data, gotPool.toArray(), "ThreadedBackend.matmul (pool) vs runtime.ts");
        assertShapeEqual(gotMain.shape, gotPool.shape, "ThreadedBackend.matmul (pool) vs WNDArray.matmul (main) shape");
        assertDataBitIdentical(gotMain.toArray(), gotPool.toArray(), "ThreadedBackend.matmul (pool) vs WNDArray.matmul (main)");
      } finally {
        gotMain.dispose();
      }
    } finally {
      gotPool.dispose();
    }
  } finally {
    a.dispose();
    b.dispose();
    await backend.dispose();
  }
});

test("ThreadedBackend.matmul: bit-identical to WNDArray.matmul and the naive runtime.ts reference, forced through MAIN (minPoolWork: Infinity)", async () => {
  const aShape = [4, 3];
  const bShape = [3, 5];
  const aData = new Float64Array(Array.from({ length: 12 }, (_, i) => i - 5.5));
  const bData = new Float64Array(Array.from({ length: 15 }, (_, i) => 0.5 * i - 3));
  const ref = matmulRuntime(aShape, aData, bShape, bData);

  const backend = await NDArray.backend("threaded", { workers: 2 });
  const a = backend.fromArray(aShape, aData);
  const b = backend.fromArray(bShape, bData);
  try {
    const gotMainRouted = backend.matmul(a, b, { minPoolWork: Infinity }); // per-call override -> forced main
    try {
      const gotMain = a.matmul(b);
      try {
        assertShapeEqual(ref.shape, gotMainRouted.shape, "ThreadedBackend.matmul (main-routed) vs runtime.ts shape");
        assertDataBitIdentical(ref.data, gotMainRouted.toArray(), "ThreadedBackend.matmul (main-routed) vs runtime.ts");
        assertDataBitIdentical(gotMain.toArray(), gotMainRouted.toArray(), "ThreadedBackend.matmul (main-routed) vs WNDArray.matmul (main)");
      } finally {
        gotMain.dispose();
      }
    } finally {
      gotMainRouted.dispose();
    }
  } finally {
    a.dispose();
    b.dispose();
    await backend.dispose();
  }
});

// --- Lifecycle ---------------------------------------------------------------

test("ThreadedBackend.dispose(): frees the pool (Kern-06 lifecycle unchanged); disposed backend rejects further use", async () => {
  // getThreadedPoolFreeCount() is the deterministic (stack, ctrl)-pair
  // free-count plateau (mirrors resident.ts's getResidentFreeCount(), same
  // rationale as backend-api.test.ts's WasmBackend.dispose() test) — it only
  // moves inside ThreadedPool.dispose()'s own AWAITED worker-teardown loop,
  // never merely from setting `isDisposed`. Measured before creating the
  // backend/pool so the delta below is attributable to exactly this backend's
  // dispose() call, not any other pool created elsewhere in this file.
  const before = getThreadedPoolFreeCount();
  const backend = await NDArray.backend("threaded", { workers: 1 });
  const a = backend.fromArray([2, 2], [1, 2, 3, 4]);
  const b = backend.fromArray([2, 2], [1, 0, 0, 1]);

  assert.strictEqual(backend.pool.isDisposed, false);
  await backend.dispose();
  assert.strictEqual(backend.pool.isDisposed, true);
  // Non-vacuity: `isDisposed` flips synchronously (before ThreadedPool's own
  // first `await`), so a check on `isDisposed` alone cannot prove the async
  // worker-teardown actually ran to completion. The free-count plateau only
  // moves once `ThreadedPool.dispose()`'s awaited loop has actually freed
  // each worker's (stack, ctrl) pair — this backend's pool has `workers: 1`,
  // so `await backend.dispose()` must move the plateau by exactly 1.
  assert.strictEqual(
    getThreadedPoolFreeCount(),
    before + 1,
    "backend.dispose() must free exactly 1 (stack, ctrl) pair (workers: 1) — proves the awaited pool.dispose() actually ran, not just isDisposed being set",
  );

  // A second dispose() is a safe no-op (delegates to ThreadedPool.dispose()'s
  // own idempotent contract) and must not move the free-count plateau again.
  await backend.dispose();
  assert.strictEqual(getThreadedPoolFreeCount(), before + 1, "a second dispose() must not change the free counter (plateau, no double-free)");

  assert.throws(() => backend.matmul(a, b), /ThreadedBackend\.matmul: backend has been disposed/);
  assert.throws(() => backend.fromArray([2], [1, 2]), /ThreadedBackend\.fromArray: backend has been disposed/);
  assert.throws(() => backend.zeros([2]), /ThreadedBackend\.zeros: backend has been disposed/);
  assert.throws(() => backend.ones([2]), /ThreadedBackend\.ones: backend has been disposed/);

  a.dispose();
  b.dispose();
});

test("Cross-Pool-Guard: matmul mixing WNDArrays from two separate ThreadedBackend instances (separate pools) throws", async () => {
  const backendA = await NDArray.backend("threaded", { workers: 1 });
  const backendB = await NDArray.backend("threaded", { workers: 1 });
  const a = backendA.fromArray([2, 2], [1, 2, 3, 4]);
  const b = backendB.fromArray([2, 2], [1, 0, 0, 1]);
  try {
    assert.throws(
      () => backendA.matmul(a, b, { minPoolWork: 0 }),
      /threadedMatmul: second operand is not bound to this pool's threaded core/,
    );
  } finally {
    a.dispose();
    b.dispose();
    await backendA.dispose();
    await backendB.dispose();
  }
});
