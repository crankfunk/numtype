/**
 * Kern 03 lifecycle tests: refcounted buffer sharing between a base handle
 * and its transpose views. Complements `resident-lifecycle.test.ts` (whose
 * per-handle dispose/use-after-dispose contract is unchanged and stays
 * covered there) with the NEW sharing semantics: the WASM allocation is
 * freed exactly when the LAST handle onto it is released, in any order.
 *
 * Observability: `getResidentFreeCount()` counts actual buffer frees
 * (refcount reaching 0), NOT dispose calls — see its doc comment. All
 * handles created here are explicitly disposed (dispose unregisters from
 * the FinalizationRegistry), so no spontaneous GC finalizer can move the
 * counter mid-file and the deltas below are deterministic.
 */
import assert from "node:assert";
import { test } from "node:test";
import { initCore } from "../src/wasm/loader.ts";
import { getResidentFreeCount, WNDArray } from "../src/wasm/resident.ts";

const core = await initCore();

test("refcount: disposing the base keeps a live view fully usable; free happens on the LAST release", () => {
  const base = WNDArray.fromArray(core, [2, 3], [1, 2, 3, 4, 5, 6]);
  const view = base.transpose();
  const before = getResidentFreeCount();

  base.dispose();
  assert.strictEqual(base.disposed, true);
  assert.strictEqual(view.disposed, false);
  assert.strictEqual(getResidentFreeCount(), before, "buffer must NOT be freed while a view lives");

  // the view still reads and computes correctly after the base is gone
  assert.deepStrictEqual(Array.from(view.toArray()), [1, 4, 2, 5, 3, 6]);
  const s = view.sum();
  assert.deepStrictEqual(Array.from(s.toArray()), [21]);
  s.dispose(); // frees the OP's own output buffer — counter moves by 1 here

  const beforeLast = getResidentFreeCount();
  view.dispose();
  assert.strictEqual(getResidentFreeCount(), beforeLast + 1, "last release frees the shared buffer exactly once");
});

test("refcount: disposing the view first keeps the base usable (symmetric order)", () => {
  const base = WNDArray.fromArray(core, [2, 2], [1, 2, 3, 4]);
  const view = base.transpose();
  const before = getResidentFreeCount();

  view.dispose();
  assert.strictEqual(view.disposed, true);
  assert.strictEqual(base.disposed, false);
  assert.strictEqual(getResidentFreeCount(), before);
  assert.deepStrictEqual(Array.from(base.toArray()), [1, 2, 3, 4]);

  base.dispose();
  assert.strictEqual(getResidentFreeCount(), before + 1);
});

test("refcount: a view-of-view chain (three handles) frees only after all three release", () => {
  const base = WNDArray.fromArray(core, [2, 3], [1, 2, 3, 4, 5, 6]);
  const v1 = base.transpose();
  const v2 = v1.transpose(); // logically == base, same buffer
  const before = getResidentFreeCount();

  base.dispose();
  v1.dispose();
  assert.strictEqual(getResidentFreeCount(), before, "two of three released — buffer still live");
  assert.deepStrictEqual(Array.from(v2.toArray()), [1, 2, 3, 4, 5, 6]);

  v2.dispose();
  assert.strictEqual(getResidentFreeCount(), before + 1);
});

test("refcount: double-dispose of a view releases its reference only once", () => {
  const base = WNDArray.fromArray(core, [2, 2], [1, 1, 1, 1]);
  const view = base.transpose();
  const before = getResidentFreeCount();

  view.dispose();
  view.dispose(); // no-op: must NOT decrement again (that would free under the live base)
  assert.strictEqual(getResidentFreeCount(), before);
  assert.deepStrictEqual(Array.from(base.toArray()), [1, 1, 1, 1], "base unaffected by double-disposed view");

  base.dispose();
  assert.strictEqual(getResidentFreeCount(), before + 1);
});

test("use-after-dispose: a disposed view throws per-handle; ops/views on a disposed handle throw", () => {
  const base = WNDArray.fromArray(core, [2, 2], [1, 2, 3, 4]);
  const view = base.transpose();
  view.dispose();
  assert.throws(() => view.toArray(), /WNDArray\.toArray:.*disposed/);
  assert.throws(() => view.transpose(), /WNDArray\.transpose:.*disposed/);
  assert.throws(() => view.contiguous(), /WNDArray\.contiguous:.*disposed/);
  // creating a view of a disposed handle is also use-after-dispose
  base.dispose();
  assert.throws(() => base.transpose(), /WNDArray\.transpose:.*disposed/);
});

test("contiguous() is independently owned: survives disposal of base and view, then frees separately", () => {
  const base = WNDArray.fromArray(core, [2, 3], [1, 2, 3, 4, 5, 6]);
  const view = base.transpose();
  const copy = view.contiguous();
  const before = getResidentFreeCount();

  base.dispose();
  view.dispose();
  assert.strictEqual(getResidentFreeCount(), before + 1, "base+view shared ONE buffer; copy is separate");
  assert.deepStrictEqual(Array.from(copy.toArray()), [1, 4, 2, 5, 3, 6]);

  copy.dispose();
  assert.strictEqual(getResidentFreeCount(), before + 2);
});

test("a failing op on a view leaves base and view fully usable", () => {
  // dynamic shapes so the compile-time guard degrades and the mismatch is
  // caught at runtime (same technique as resident-lifecycle.test.ts)
  const aShape: number[] = [3, 2];
  const bShape: number[] = [2, 4];
  const base = WNDArray.fromArray(core, aShape, [1, 2, 3, 4, 5, 6]);
  const view = base.transpose(); // shape [2,3]
  const b = WNDArray.fromArray(core, bShape, new Array(8).fill(0));

  assert.throws(() => view.add(b), /2,3.*2,4|broadcast/s);

  assert.strictEqual(base.disposed, false);
  assert.strictEqual(view.disposed, false);
  assert.deepStrictEqual(Array.from(view.toArray()), [1, 3, 5, 2, 4, 6]);

  b.dispose();
  view.dispose();
  base.dispose();
});

test("empty-array sharing: base+view of a size-0 buffer still free exactly once", () => {
  const base = WNDArray.fromArray(core, [0, 3] as number[], []);
  const view = base.transpose();
  const before = getResidentFreeCount();
  base.dispose();
  assert.strictEqual(getResidentFreeCount(), before);
  assert.deepStrictEqual(Array.from(view.toArray()), []);
  view.dispose();
  assert.strictEqual(getResidentFreeCount(), before + 1);
});

test("op outputs never alias operands: result of an op on a view has its own buffer", () => {
  const base = WNDArray.fromArray(core, [2, 2], [1, 2, 3, 4]);
  const view = base.transpose();
  const zeros = WNDArray.zeros(core, [2, 2]);
  const out = view.add(zeros); // out holds the view's logical content [1,3,2,4]
  zeros.dispose();
  const before = getResidentFreeCount();
  // Disposing base+view must not disturb the op result.
  base.dispose();
  view.dispose();
  assert.strictEqual(getResidentFreeCount(), before + 1);
  assert.deepStrictEqual(Array.from(out.toArray()), [1, 3, 2, 4]);
  out.dispose();
});
