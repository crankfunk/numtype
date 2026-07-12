/**
 * Item 10 — Backend-Wahl-API (docs/item-10-backend-api-spec.md): runtime
 * differential tests for the WASM half of the facade (`NDArray.backend
 * ("wasm")` / `WasmBackend`). Wired into `pnpm test:resident` (builds
 * `build:wasm` only — the `"threaded"` env-detection negative case below
 * needs no threads artifact, per the spec's testplan; the actual threaded
 * facade's own differential tests live in `backend-api-threaded.test.ts`,
 * `pnpm test:threaded`).
 *
 * Covers exactly the testplan's five WASM-backend bullets:
 *  - Fassaden-Äquivalenz: `backend("wasm").fromArray/zeros/ones` bit-identical
 *    to direct `WNDArray.fromArray(core, ...)` AND the naive `NDArray`
 *    reference.
 *  - Lifecycle: `backend.dispose()` blocks further creation through that
 *    backend; the arrays it already created keep their own independent
 *    lifecycle (D1: the facade never hides WASM memory management) — proven
 *    via the existing, real `getResidentFreeCount()` plateau signal
 *    (non-vacuous, same methodology as `resident-lifecycle.test.ts`).
 *  - env-Detektion (negativ): `backend("threaded")` throws the pinned
 *    message stem when Node/the threads artifact is missing.
 *  - Interop: `NDArray <-> WNDArray` round trip (`toArray`/`.data`/
 *    `fromArray`) bit-identical, including a materialized transpose/slice.
 *  - Cross-Backend-Guard: two separate `WasmBackend` instances (separate
 *    cores) — an op mixing operands from both throws via the pre-existing
 *    `assertSameCore` guard.
 */
import assert from "node:assert";
import { test } from "node:test";
import { NDArray } from "../src/ndarray.ts";
import { checkThreadedEnv, WasmBackend } from "../src/wasm/backend-api.ts";
import { initCore } from "../src/wasm/loader.ts";
import { getResidentFreeCount, WNDArray } from "../src/wasm/resident.ts";
import { assertDataBitIdentical, assertShapeEqual } from "./assert-helpers.ts";
import { genData, makeRng } from "./prng.ts";

// --- Fassaden-Äquivalenz: fromArray/zeros/ones bit-identical ----------------

test("WasmBackend.fromArray: bit-identical to direct WNDArray.fromArray(core, ...) and the naive NDArray reference", async () => {
  const rng = makeRng(0x4241434b454e445fn); // "BACKEND_"
  const shape = [2, 3, 4];
  const data = genData(rng, shape);

  const backend = await NDArray.backend("wasm");
  const viaBackend = backend.fromArray(shape, data);
  const viaDirect = WNDArray.fromArray(await initCore(), shape, data);
  const naive = NDArray.fromArray(shape, data);
  try {
    assertShapeEqual([...shape], viaBackend.shape, "WasmBackend.fromArray shape");
    assertShapeEqual(viaDirect.shape, viaBackend.shape, "WasmBackend.fromArray vs direct shape");
    assertDataBitIdentical(viaDirect.toArray(), viaBackend.toArray(), "WasmBackend.fromArray vs direct WNDArray.fromArray");
    assertDataBitIdentical(naive.data, viaBackend.toArray(), "WasmBackend.fromArray vs naive NDArray");
  } finally {
    viaBackend.dispose();
    viaDirect.dispose();
  }
});

test("WasmBackend.zeros: bit-identical to direct WNDArray.zeros(core, ...) and the naive NDArray reference", async () => {
  const shape = [2, 3, 4];
  const backend = await NDArray.backend("wasm");
  const viaBackend = backend.zeros(shape);
  const viaDirect = WNDArray.zeros(await initCore(), shape);
  const naive = NDArray.zeros(shape);
  try {
    assertShapeEqual(viaDirect.shape, viaBackend.shape, "WasmBackend.zeros vs direct shape");
    assertDataBitIdentical(viaDirect.toArray(), viaBackend.toArray(), "WasmBackend.zeros vs direct WNDArray.zeros");
    assertDataBitIdentical(naive.data, viaBackend.toArray(), "WasmBackend.zeros vs naive NDArray");
  } finally {
    viaBackend.dispose();
    viaDirect.dispose();
  }
});

test("WasmBackend.ones: bit-identical to direct WNDArray.ones(core, ...) and the naive NDArray reference", async () => {
  const shape = [3, 5];
  const backend = await NDArray.backend("wasm");
  const viaBackend = backend.ones(shape);
  const viaDirect = WNDArray.ones(await initCore(), shape);
  const naive = NDArray.ones(shape);
  try {
    assertShapeEqual(viaDirect.shape, viaBackend.shape, "WasmBackend.ones vs direct shape");
    assertDataBitIdentical(viaDirect.toArray(), viaBackend.toArray(), "WasmBackend.ones vs direct WNDArray.ones");
    assertDataBitIdentical(naive.data, viaBackend.toArray(), "WasmBackend.ones vs naive NDArray");
  } finally {
    viaBackend.dispose();
    viaDirect.dispose();
  }
});

// --- Lifecycle ---------------------------------------------------------------

test("WasmBackend.dispose(): blocks further creation through this backend; arrays already created keep their own independent lifecycle", async () => {
  const backend = await NDArray.backend("wasm");
  const before = getResidentFreeCount();

  const a = backend.fromArray([2, 2], [1, 2, 3, 4]);
  const b = backend.zeros([2]);
  const c = backend.ones([3]);

  backend.dispose();

  // Disposed backend: every creation method throws, naming itself.
  assert.throws(() => backend.fromArray([2], [1, 2]), /WasmBackend\.fromArray: backend has been disposed/);
  assert.throws(() => backend.zeros([2]), /WasmBackend\.zeros: backend has been disposed/);
  assert.throws(() => backend.ones([2]), /WasmBackend\.ones: backend has been disposed/);

  // A second dispose() is a safe no-op (mirrors WNDArray's own contract).
  backend.dispose();

  // Arrays already created through the (now-disposed) backend are still
  // fully live and independently usable — the facade never hides WASM
  // memory management (D1). Disposing them is what actually moves the
  // resident-free-count plateau: a real, non-vacuous signal.
  assert.strictEqual(a.disposed, false);
  assert.deepStrictEqual(Array.from(a.toArray()), [1, 2, 3, 4]);
  a.dispose();
  b.dispose();
  c.dispose();
  assert.strictEqual(getResidentFreeCount(), before + 3, "disposing the three backend-created arrays must move the free-count plateau by exactly 3");
});

// --- env-Detektion (negativ) --------------------------------------------------

// MUST run before any other test in this file that calls
// `NDArray.backend("threaded")` (including the very next test below): Node's
// dynamic `import()` caches the module, and `process.moduleLoadList` only
// ever grows — once `./wasm/threaded.ts` (and therefore `node:worker_threads`)
// has been imported once in this process, a later call can never observe it
// "not yet loaded" again. Placed first so this test's own before/after
// comparison is the first thing in the file to exercise
// `NDArray.backend("threaded")` at all, and therefore the only one able to
// catch an import-before-env-check regression (verified empirically: with
// this test placed AFTER the "pinned message stem" test below — which also
// calls `NDArray.backend("threaded")` with `process` blanked — a real
// import-before-check mutant went UNDETECTED, because that earlier call had
// already loaded `node:worker_threads` and the plateau no longer moved).
test('D2 ordering: backend("threaded") runs the env check BEFORE the dynamic threaded.ts import (threaded.ts / node:worker_threads must not load when the env check fails)', async () => {
  const original = globalThis.process;
  // process.moduleLoadList: internal-but-long-stable Node API listing loaded
  // builtins; read via the retained `original` reference so it stays reachable
  // while `globalThis.process` is blanked to force the "not Node" env-check
  // branch (same blanking the message-stem test below uses).
  const wtCount = () =>
    ((original as unknown as { moduleLoadList: readonly string[] }).moduleLoadList)
      .filter((m) => /worker_threads/.test(m)).length;
  const before = wtCount();
  try {
    (globalThis as unknown as { process: unknown }).process = undefined;
    await NDArray.backend("threaded").catch(() => {}); // must throw (env fail) WITHOUT importing threaded.ts
  } finally {
    (globalThis as unknown as { process: unknown }).process = original;
  }
  assert.strictEqual(
    wtCount(),
    before,
    "env check must run before the dynamic threaded.ts import: threaded.ts (node:worker_threads) loaded despite the env check failing",
  );
});

test('NDArray.backend("threaded"): throws the pinned message stem when not running under Node', async () => {
  const original = globalThis.process;
  try {
    // `typeof process === "undefined"` is the documented (D2) Node feature
    // detection this whole codebase already uses (loader.ts's `instantiate`);
    // blanking the global reproduces the "not Node" branch deterministically,
    // without depending on whether a threads artifact happens to be present
    // on this machine.
    (globalThis as unknown as { process: unknown }).process = undefined;
    await assert.rejects(
      () => NDArray.backend("threaded"),
      (err: unknown) =>
        err instanceof Error &&
        err.message === 'NDArray.backend("threaded"): threaded backend requires Node with the threads artifact (not running under Node)',
    );
  } finally {
    (globalThis as unknown as { process: unknown }).process = original;
  }
});

test('checkThreadedEnv: returns "threads artifact not found" for an unreachable artifact URL', async () => {
  const reason = await checkThreadedEnv(new URL("./does-not-exist.wasm", import.meta.url));
  assert.strictEqual(reason, "threads artifact not found");
});

test("checkThreadedEnv: returns null when Node is running and the artifact URL is reachable", async () => {
  // Points at this very test file (definitely reachable) — proves the
  // "reachable -> null" branch independently of whether the real threads
  // artifact has been built on this machine (test:resident does not build it).
  const reason = await checkThreadedEnv(new URL("./backend-api.test.ts", import.meta.url));
  assert.strictEqual(reason, null);
});

// --- Interop: NDArray <-> WNDArray round trip --------------------------------

test("Interop: NDArray -> WasmBackend.fromArray(shape, nd.data) -> WNDArray, bit-identical", async () => {
  const nd = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
  const backend = await NDArray.backend("wasm");
  const wnd = backend.fromArray(nd.shape, nd.data);
  try {
    assertShapeEqual([...nd.shape], wnd.shape, "interop NDArray->WNDArray shape");
    assertDataBitIdentical(nd.data, wnd.toArray(), "interop NDArray->WNDArray data");
  } finally {
    wnd.dispose();
  }
});

test("Interop: WNDArray.toArray() -> NDArray.fromArray(shape, ...), bit-identical, including a materialized transpose", async () => {
  const backend = await NDArray.backend("wasm");
  const wnd = backend.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
  const wndT = wnd.transpose(); // O(1) view — not yet materialized
  try {
    const nd = NDArray.fromArray(wndT.shape, wndT.toArray()); // toArray() materializes the view's logical row-major order
    assertShapeEqual(wndT.shape, [...nd.shape], "interop WNDArray(transpose)->NDArray shape");
    assertDataBitIdentical(wndT.toArray(), nd.data, "interop WNDArray(transpose)->NDArray data");

    // NDArray itself is always materialized/contiguous (D4) — round-tripping
    // its own transpose (a fresh COPY, per NDArray's own contract) back
    // through the WASM backend must stay bit-identical too.
    const ndT = nd.transpose();
    const wndFromNdT = backend.fromArray(ndT.shape, ndT.data);
    try {
      assertShapeEqual([...ndT.shape], wndFromNdT.shape, "interop NDArray(transpose)->WNDArray shape");
      assertDataBitIdentical(ndT.data, wndFromNdT.toArray(), "interop NDArray(transpose)->WNDArray data");
    } finally {
      wndFromNdT.dispose();
    }
  } finally {
    wndT.dispose();
    wnd.dispose();
  }
});

test("Interop: WNDArray.slice() (view) -> toArray() -> NDArray.fromArray(shape, ...), bit-identical", async () => {
  const backend = await NDArray.backend("wasm");
  const wnd = backend.fromArray([4, 4], Array.from({ length: 16 }, (_, i) => i));
  const wndSlice = wnd.slice({ start: 1, stop: 3 }, { start: 1, stop: 3 }); // O(1) view — not yet materialized
  try {
    const nd = NDArray.fromArray(wndSlice.shape, wndSlice.toArray());
    assertShapeEqual(wndSlice.shape, [...nd.shape], "interop WNDArray(slice)->NDArray shape");
    assertDataBitIdentical(wndSlice.toArray(), nd.data, "interop WNDArray(slice)->NDArray data");
  } finally {
    wndSlice.dispose();
    wnd.dispose();
  }
});

// --- Cross-Backend-Guard -------------------------------------------------------

test("Cross-Backend-Guard: an op mixing WNDArrays from two separate WasmBackend instances (separate cores) throws via assertSameCore", async () => {
  const backendA = await NDArray.backend("wasm");
  const backendB = await NDArray.backend("wasm");
  const a = backendA.fromArray([2, 2], [1, 2, 3, 4]);
  const b = backendB.fromArray([2, 2], [1, 1, 1, 1]);
  try {
    assert.throws(() => a.add(b), /WNDArray\.add: operands belong to different WASM core instances/);
  } finally {
    a.dispose();
    b.dispose();
  }
});
