/**
 * Item 10 — Backend-Wahl-API (docs/item-10-backend-api-spec.md): type-level
 * tests for `NDArray.backend(kind, opts?)` and the two facade classes it
 * returns. Never executed (only type-checked, like every other
 * `*.test-d.ts` file) — `await NDArray.backend(...)` at module top level is
 * fine here for exactly that reason.
 */
import type { Shape } from "../src/dim.ts";
import { NDArray, type NDArrayView } from "../src/ndarray.ts";
import type { WNDArray } from "../src/wasm/resident.ts";
import type { ThreadedBackend } from "../src/wasm/threaded.ts";
import type { Equal, Expect } from "./test-utils.ts";

// --- D1: overload resolution — each `kind` returns its own precise type ---

const wasmBackend = await NDArray.backend("wasm");
const threadedBackend = await NDArray.backend("threaded");
type T1 = Expect<Equal<typeof threadedBackend, ThreadedBackend>>;

// --- D1: WasmBackend facade — fromArray/zeros/ones correctly typed, and
// return the REAL WNDArray class (not a wrapper) ---------------------------

// D-V2.3 (docs/phase-d-vorarbeiten-spec.md): `.shape` is now `Readonly<S>` —
// pins re-expressed intent-preservingly as `readonly [...]`.
const fa = wasmBackend.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
type T2 = Expect<Equal<(typeof fa)["shape"], readonly [2, 3]>>;

const z = wasmBackend.zeros([2, 4]);
type T3 = Expect<Equal<(typeof z)["shape"], readonly [2, 4]>>;

const o = wasmBackend.ones([3]);
type T4 = Expect<Equal<(typeof o)["shape"], readonly [3]>>;

declare const wndCheck: WNDArray<[2, 3]>;
const wndAssignable: typeof wndCheck = fa; // proves `fa` really is a `WNDArray<[2, 3]>`
void wndAssignable;

// --- D2: ThreadedBackend.matmul carries the SAME Guard<MatMul<S,B>>/OkShape
// as WNDArray.matmul — error at the offending argument, same as the instance
// method (testplan: "trägt denselben Guard<MatMul<S,B>>/OkShape") ----------

declare const ta: WNDArray<[2, 3]>;
declare const tb: WNDArray<[3, 4]>;
const matmulled = threadedBackend.matmul(ta, tb);
type T5 = Expect<Equal<(typeof matmulled)["shape"], readonly [2, 4]>>;

declare const badB: WNDArray<[5, 4]>;
// @ts-expect-error - inner dims 3 vs 5 mismatch: error must land on `badB`, exactly like WNDArray.matmul's own guard
threadedBackend.matmul(ta, badB);

// --- D-V2.2 (docs/phase-d-vorarbeiten-spec.md): a ThreadedBackend-produced
// WNDArray also satisfies NDArrayView<S> — same conformance as the WasmBackend
// handle `fa` above, proven separately since ThreadedBackend has its own
// fromArray/zeros/ones signatures (threaded.ts), not just a re-export -------

const threadedArr = threadedBackend.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
const threadedAsView: NDArrayView<[2, 3]> = threadedArr;
const threadedAsWidenedView: NDArrayView<Shape> = threadedArr;
void threadedAsView;
void threadedAsWidenedView;
