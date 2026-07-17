/**
 * D-S3.2 (1) — Runtime package smoke. Imports the BUILT package
 * (`dist/index.js` — NOT `spike/src`) in a fresh Node process and exercises
 * basic ops, proving the emitted + rewritten package is consumable
 * end-to-end: the `.wasm` loads over the package-relative loader path, the
 * barrel resolves, and the S1 sum-overload structure carries through into the
 * build. Also asserts the `backend("threaded")`-without-artifact rejection
 * (D-S2.4 / Option 2: threads JS ships, the threads `.wasm` does not).
 *
 * Runs ONLY under `pnpm test:package` (which runs `build:dist` first) — never
 * in the plain `node --test` corpora, where `dist/` is not built. The
 * test-scripts-guard enforces that separation (invariant (e)).
 */
import { test } from "node:test";
import assert from "node:assert";

// This file: <root>/spike/tests-package/package-smoke.test.ts → two up = repo root
// (same depth as spike/tests-runtime/, cf. test-scripts-guard's `../../`).
const distIndexUrl = new URL("../../dist/index.js", import.meta.url);

interface NDArrayLike {
  readonly shape: readonly number[];
  sum(): { toNestedArray(): unknown };
  sum(axis: number): { toNestedArray(): unknown };
  sum(axis: number, keepdims: boolean): { toNestedArray(): unknown };
  add(other: NDArrayLike): { toNestedArray(): unknown };
}
interface WNDArrayLike {
  sum(): { toNestedArray(): unknown };
  add(other: WNDArrayLike): { toNestedArray(): unknown };
}
interface WasmBackendLike {
  fromArray(shape: readonly number[], data: readonly number[]): WNDArrayLike;
}
interface PackageExports {
  NDArray: {
    fromArray(shape: readonly number[], data: readonly number[]): NDArrayLike;
    backend(kind: "wasm"): Promise<WasmBackendLike>;
    backend(kind: "threaded"): Promise<unknown>;
  };
}

async function loadPackage(): Promise<PackageExports> {
  return (await import(distIndexUrl.href)) as unknown as PackageExports;
}

test("built package (dist/index.js) is consumable end-to-end", async () => {
  const { NDArray } = await loadPackage();
  const a = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
  assert.deepStrictEqual([...a.shape], [2, 3], "shape");
  assert.strictEqual(a.sum().toNestedArray(), 21, "sum() full reduction");
  assert.deepStrictEqual(a.sum(0).toNestedArray(), [5, 7, 9], "sum(0)");
  assert.deepStrictEqual(a.sum(1, true).toNestedArray(), [[6], [15]], "sum(1, keepdims)");
  const b = NDArray.fromArray([2, 3], [6, 5, 4, 3, 2, 1]);
  assert.deepStrictEqual(
    a.add(b).toNestedArray(),
    [[7, 7, 7], [7, 7, 7]],
    "add (broadcasting elementwise)",
  );
});

test("backend('wasm') loads the BUNDLED .wasm over the package-relative loader path and computes on it", async () => {
  // S3-Verify F1: the plain JS `NDArray` path (the test above) computes in
  // pure JS and NEVER touches the .wasm — only `backend("wasm")` calls
  // `initCore()`, which reads dist/wasm/numtype_core.wasm via the
  // package-relative loader URL. THIS is the test that actually proves the
  // bundled .wasm loads over the package-relative path (delete it → this test
  // throws, where the JS-path tests stay green).
  const { NDArray } = await loadPackage();
  const backend = await NDArray.backend("wasm");
  const w = backend.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
  assert.strictEqual(w.sum().toNestedArray(), 21, "WASM-backed sum() over the bundled .wasm");
  const wb = backend.fromArray([2, 3], [6, 5, 4, 3, 2, 1]);
  assert.deepStrictEqual(w.add(wb).toNestedArray(), [[7, 7, 7], [7, 7, 7]], "WASM-backed add");
});

test('backend("threaded") without the threads artifact rejects with the pinned message (Option 2)', async () => {
  const { NDArray } = await loadPackage();
  await assert.rejects(
    () => NDArray.backend("threaded"),
    /threaded backend requires Node with the threads artifact/,
    "backend('threaded') must reject (threads .wasm is deliberately NOT bundled in v0)",
  );
});
