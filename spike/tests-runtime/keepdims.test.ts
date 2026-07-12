/**
 * Differential test: `sum(axis, keepdims=true)` on both public surfaces
 * (`NDArray` = naive TS reference, `WNDArray` = resident WASM), full
 * (`axis === undefined`) and per-axis (incl. negative).
 *
 * keepdims is pure SHAPE metadata — the summed data is byte-identical to the
 * non-keepdims result (an axis of length 1 changes neither the row-major order
 * nor `product`). So the checks are, deliberately, NON-CIRCULAR: the expected
 * shape is derived from the trusted non-keepdims `sumRuntime` result plus
 * structural invariants (rank preserved, reduced axis == 1, removing it
 * recovers the non-keep shape, product unchanged), NOT from `keepDimsShape`
 * (the code under test). The data must equal the non-keep reference on BOTH
 * surfaces — which also proves the two surfaces agree with each other.
 */
import assert from "node:assert";
import { test } from "node:test";
import { NDArray } from "../src/ndarray.ts";
import { product, sumRuntime } from "../src/runtime.ts";
import { initCore } from "../src/wasm/loader.ts";
import { WNDArray } from "../src/wasm/resident.ts";
import { assertDataBitIdentical, assertShapeEqual } from "./assert-helpers.ts";
import { genData, makeRng, type Rng } from "./prng.ts";

const core = await initCore();
const CASE_COUNT = 150;

function genShape(rng: Rng, minRank: number, maxRank: number): number[] {
  const rank = rng.nextInt(minRank, maxRank);
  return Array.from({ length: rank }, () => rng.nextInt(1, 8));
}

/** Assert a keepdims shape against the non-keepdims reference, WITHOUT
 * re-deriving the keepdims rule (non-circular). */
function assertKeepShape(
  keepShape: readonly number[],
  inputShape: readonly number[],
  nonKeepShape: readonly number[],
  axis: number | undefined,
  ctx: string,
): void {
  // Rank is always preserved (this is what keepdims IS).
  assert.strictEqual(keepShape.length, inputShape.length, `${ctx}: keepdims must preserve rank`);
  // product invariant: element count never changes.
  assert.strictEqual(product(keepShape), product(nonKeepShape), `${ctx}: product must be unchanged`);
  if (axis === undefined) {
    assert.ok(
      keepShape.every((d) => d === 1),
      `${ctx}: full-reduction keepdims must be all-ones, got [${keepShape.join(",")}]`,
    );
    return;
  }
  const norm = axis < 0 ? inputShape.length + axis : axis;
  assert.strictEqual(keepShape[norm], 1, `${ctx}: reduced axis ${norm} must be size 1`);
  // Removing the (size-1) reduced axis must recover exactly the non-keep shape.
  const removed = [...keepShape.slice(0, norm), ...keepShape.slice(norm + 1)];
  assertShapeEqual(nonKeepShape, removed, `${ctx}: keepShape minus reduced axis == non-keep shape`);
}

// --- full reduction (axis === undefined), rank 0..4 ------------------------
{
  const rng = makeRng(0x4b445f414c4c5f5fn); // "KD_ALL__"
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);

    test(`keepdims sum_all case ${c}: shape=[${shape.join(",")}]`, () => {
      const ref = sumRuntime(shape, data, undefined); // trusted non-keep data
      const ctx = `keepdims sum_all case ${c} shape=[${shape.join(",")}]`;

      const nd = NDArray.fromArray(shape, data).sum(undefined, true);
      assertKeepShape(nd.shape, shape, ref.shape, undefined, `${ctx} [NDArray]`);
      assertDataBitIdentical(ref.data, nd.data, `${ctx} [NDArray]`);

      const w = WNDArray.fromArray(core, shape, Array.from(data));
      try {
        const wKeep = w.sum(undefined, true);
        try {
          assertKeepShape(wKeep.shape, shape, ref.shape, undefined, `${ctx} [WNDArray]`);
          assertDataBitIdentical(ref.data, wKeep.toArray(), `${ctx} [WNDArray]`);
        } finally {
          wKeep.dispose();
        }
      } finally {
        w.dispose();
      }
    });
  }
}

// --- per-axis (rank 1..4), positive and negative axes ----------------------
{
  const rng = makeRng(0x4b445f4158495f5fn); // "KD_AXI__"
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 1, 4);
    const data = genData(rng, shape);
    const rank = shape.length;
    const positiveAxis = rng.nextInt(0, rank - 1);
    const axis = rng.nextBool() ? positiveAxis - rank : positiveAxis;

    test(`keepdims sum_axis case ${c}: shape=[${shape.join(",")}] axis=${axis}`, () => {
      const ref = sumRuntime(shape, data, axis); // trusted non-keep data
      const ctx = `keepdims sum_axis case ${c} shape=[${shape.join(",")}] axis=${axis}`;

      const nd = NDArray.fromArray(shape, data).sum(axis, true);
      assertKeepShape(nd.shape, shape, ref.shape, axis, `${ctx} [NDArray]`);
      assertDataBitIdentical(ref.data, nd.data, `${ctx} [NDArray]`);

      const w = WNDArray.fromArray(core, shape, Array.from(data));
      try {
        const wKeep = w.sum(axis, true);
        try {
          assertKeepShape(wKeep.shape, shape, ref.shape, axis, `${ctx} [WNDArray]`);
          assertDataBitIdentical(ref.data, wKeep.toArray(), `${ctx} [WNDArray]`);
        } finally {
          wKeep.dispose();
        }
      } finally {
        w.dispose();
      }
    });
  }
}

// --- explicit keepdims=false == default (axis removed) on both surfaces -----
{
  const rng = makeRng(0x4b445f46414c5345n); // "KD_FALSE"
  for (let c = 0; c < 60; c++) {
    const shape = genShape(rng, 1, 4);
    const data = genData(rng, shape);
    const rank = shape.length;
    const axis = rng.nextInt(0, rank - 1);

    test(`keepdims=false parity case ${c}: shape=[${shape.join(",")}] axis=${axis}`, () => {
      const ref = sumRuntime(shape, data, axis);
      const ctx = `keepdims=false parity case ${c} shape=[${shape.join(",")}] axis=${axis}`;
      const nd = NDArray.fromArray(shape, data).sum(axis, false);
      assertShapeEqual(ref.shape, nd.shape, `${ctx} [NDArray shape]`);
      assertDataBitIdentical(ref.data, nd.data, `${ctx} [NDArray data]`);

      const w = WNDArray.fromArray(core, shape, Array.from(data));
      try {
        const wSum = w.sum(axis, false);
        try {
          assertShapeEqual(ref.shape, wSum.shape, `${ctx} [WNDArray shape]`);
          assertDataBitIdentical(ref.data, wSum.toArray(), `${ctx} [WNDArray data]`);
        } finally {
          wSum.dispose();
        }
      } finally {
        w.dispose();
      }
    });
  }
}

// --- keepdims on VIEWS (F5 addendum, owner-mandated 2026-07-12): keepdims is
// computed from `this.shape` — for a view that is the VIEW's logical shape,
// never the underlying buffer's. Exercised on the three view kinds WNDArray
// has: transposed (reversed strides), offset-shifted slice (offset != 0), and
// their composition. Reference stays non-circular: the SAME chain on the
// naive NDArray, summed WITHOUT keepdims (the long-established path), plus
// the structural invariants of assertKeepShape. -----------------------------
{
  // transpose view: [3,4] -> T [4,3]
  for (const axis of [0, 1, undefined] as const) {
    test(`keepdims on transpose view: [3,4]^T axis=${axis}`, () => {
      const base = NDArray.fromArray([3, 4], Array.from({ length: 12 }, (_, i) => i + 1));
      const ref = base.transpose().sum(axis); // trusted non-keep reference chain
      const ctx = `keepdims transpose view axis=${axis}`;

      const w = WNDArray.fromArray(core, [3, 4], Array.from(base.data));
      try {
        const view = w.transpose(); // O(1) view, reversed strides
        try {
          const got = view.sum(axis, true);
          try {
            assertKeepShape(got.shape, view.shape, ref.shape, axis, ctx);
            assertDataBitIdentical(ref.data, got.toArray(), ctx);
          } finally {
            got.dispose();
          }
        } finally {
          view.dispose();
        }
      } finally {
        w.dispose();
      }
    });
  }

  // offset-shifted slice view: [3,4] -> rows 1.. -> [2,4] with offset 4 != 0.
  // The axis===undefined case additionally runs nt_sum_all_strided on an
  // offset view — the branch where keepDimsShape meets a nonzero offset.
  for (const axis of [0, -1, undefined] as const) {
    test(`keepdims on offset slice view: [3,4] rows 1.. axis=${axis}`, () => {
      const base = NDArray.fromArray([3, 4], Array.from({ length: 12 }, (_, i) => i * 2 + 1));
      const ref = base.slice({ start: 1 }, null).sum(axis); // trusted non-keep reference chain
      const ctx = `keepdims offset slice view axis=${axis}`;

      const w = WNDArray.fromArray(core, [3, 4], Array.from(base.data));
      try {
        const view = w.slice({ start: 1 }, null); // O(1) view, offset 4
        try {
          const got = view.sum(axis, true);
          try {
            assertKeepShape(got.shape, view.shape, ref.shape, axis, ctx);
            assertDataBitIdentical(ref.data, got.toArray(), ctx);
          } finally {
            got.dispose();
          }
        } finally {
          view.dispose();
        }
      } finally {
        w.dispose();
      }
    });
  }

  // composed view: [2,3,4] -> transpose [4,3,2] -> slice rows 1.. of axis 1
  // -> [4,2,2] with non-natural strides AND nonzero offset.
  test("keepdims on composed transpose+slice view: [2,3,4]^T sliced, axis=1", () => {
    const base = NDArray.fromArray([2, 3, 4], Array.from({ length: 24 }, (_, i) => i - 7));
    const ref = base.transpose().slice(null, { start: 1 }, null).sum(1); // trusted non-keep reference chain
    const ctx = "keepdims composed view axis=1";

    const w = WNDArray.fromArray(core, [2, 3, 4], Array.from(base.data));
    try {
      const t = w.transpose();
      try {
        const view = t.slice(null, { start: 1 }, null);
        try {
          const got = view.sum(1, true);
          try {
            assertKeepShape(got.shape, view.shape, ref.shape, 1, ctx);
            assertDataBitIdentical(ref.data, got.toArray(), ctx);
          } finally {
            got.dispose();
          }
        } finally {
          view.dispose();
        }
      } finally {
        t.dispose();
      }
    } finally {
      w.dispose();
    }
  });
}
