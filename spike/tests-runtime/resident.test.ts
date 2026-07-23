/**
 * Kern 02 differential tests: the resident path (`WNDArray.fromArray` ->
 * op(s) -> `toArray`) must be bit-identical to the naive TS reference for
 * every op (add/matmul/sum/transpose), exactly like the v1 differential
 * suite (`add.test.ts` etc.) — but here the point isn't re-verifying kernel
 * correctness (already proven bit-identical in Kern 01), it's verifying the
 * NEW plumbing: ptr lifecycle, per-call shape marshalling, and view
 * discipline introduced by residency. Same seeded-PRNG methodology as the
 * v1 suite (reused verbatim from `prng.ts`/`assert-helpers.ts`).
 *
 * Also covers "chained residency": multi-op chains that stay resident
 * throughout (intermediate `WNDArray`s disposed along the way), compared
 * bit-identically against the same chain run on the naive reference.
 *
 * Deliberately NOT wired into `pnpm test:core`'s glob (see package.json's
 * `test:resident` script and docs/kern-02-ergebnisse.md for why) — the
 * acceptance criterion is v1's `pnpm test:core` staying at exactly 791/791,
 * so v2's own tests get their own script instead of inflating that count.
 */
import assert from "node:assert";
import { test } from "node:test";
import { elementwiseBinary, keepDimsShape, matmulRuntime, meanRuntime, sumRuntime, transposeRuntime } from "../src/runtime.ts";
import { initCore } from "../src/wasm/loader.ts";
import { WNDArray, type AnyWNDArray } from "../src/wasm/resident.ts";
import { assertDataBitIdentical, assertShapeEqual } from "./assert-helpers.ts";
import { genBroadcastShapes, genData, makeRng, type Rng } from "./prng.ts";

const core = await initCore();
const CASE_COUNT = 120;

function genShape(rng: Rng, minRank: number, maxRank: number): number[] {
  const rank = rng.nextInt(minRank, maxRank);
  return Array.from({ length: rank }, () => rng.nextInt(1, 8));
}

function genSmallBatch(rng: Rng): number[] {
  const rank = rng.nextInt(0, 2);
  return Array.from({ length: rank }, () => rng.nextInt(1, 8));
}

function genMatmulShapes(rng: Rng): { aShape: number[]; bShape: number[] } {
  const k = rng.nextInt(1, 8);
  const m = rng.nextInt(1, 8);
  const n = rng.nextInt(1, 8);
  const aIs1D = rng.nextInt(0, 4) === 0;
  const bIs1D = rng.nextInt(0, 4) === 0;
  if (aIs1D && bIs1D) return { aShape: [k], bShape: [k] };
  if (aIs1D) {
    const batchB = genSmallBatch(rng);
    return { aShape: [k], bShape: [...batchB, k, n] };
  }
  if (bIs1D) {
    const batchA = genSmallBatch(rng);
    return { aShape: [...batchA, m, k], bShape: [k] };
  }
  const { aShape: batchA, bShape: batchB } = genBroadcastShapes(rng, 2);
  return { aShape: [...batchA, m, k], bShape: [...batchB, k, n] };
}

// --- add: resident vs naive, bit-identical ---------------------------------
{
  const rng = makeRng(0x5245535f4144445fn); // "RES_ADD_"
  for (let c = 0; c < CASE_COUNT; c++) {
    const { aShape, bShape } = genBroadcastShapes(rng);
    const aData = genData(rng, aShape);
    const bData = genData(rng, bShape);

    test(`resident add case ${c}: a=[${aShape.join(",")}] b=[${bShape.join(",")}]`, () => {
      const ref = elementwiseBinary(aShape, aData, bShape, bData, (x, y) => x + y);
      const a = WNDArray.fromArray(core, aShape, Array.from(aData));
      const b = WNDArray.fromArray(core, bShape, Array.from(bData));
      try {
        const got = a.add(b);
        try {
          const ctx = `resident add case ${c} a=[${aShape.join(",")}] b=[${bShape.join(",")}]`;
          assertShapeEqual(ref.shape, got.shape, ctx);
          assertDataBitIdentical(ref.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        a.dispose();
        b.dispose();
      }
    });
  }
}

// --- matmul: resident vs naive, bit-identical ------------------------------
{
  const rng = makeRng(0x5245535f4d4d554cn); // "RES_MMUL"
  for (let c = 0; c < CASE_COUNT; c++) {
    const { aShape, bShape } = genMatmulShapes(rng);
    const aData = genData(rng, aShape);
    const bData = genData(rng, bShape);

    test(`resident matmul case ${c}: a=[${aShape.join(",")}] b=[${bShape.join(",")}]`, () => {
      const ref = matmulRuntime(aShape, aData, bShape, bData);
      const a = WNDArray.fromArray(core, aShape, Array.from(aData));
      const b = WNDArray.fromArray(core, bShape, Array.from(bData));
      try {
        const got = a.matmul(b);
        try {
          const ctx = `resident matmul case ${c} a=[${aShape.join(",")}] b=[${bShape.join(",")}]`;
          assertShapeEqual(ref.shape, got.shape, ctx);
          assertDataBitIdentical(ref.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        a.dispose();
        b.dispose();
      }
    });
  }
}

// --- sum: resident vs naive, bit-identical (sum_all + sum_axis) -----------
{
  const rng = makeRng(0x5245535f53554d5fn); // "RES_SUM_"
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);

    test(`resident sum_all case ${c}: shape=[${shape.join(",")}]`, () => {
      const ref = sumRuntime(shape, data, undefined);
      const a = WNDArray.fromArray(core, shape, Array.from(data));
      try {
        const got = a.sum();
        try {
          const ctx = `resident sum_all case ${c} shape=[${shape.join(",")}]`;
          assertShapeEqual(ref.shape, got.shape, ctx);
          assertDataBitIdentical(ref.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        a.dispose();
      }
    });
  }
}
{
  const rng = makeRng(0x5245535f41585f5fn); // "RES_AX__"
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 1, 4);
    const data = genData(rng, shape);
    const rank = shape.length;
    const positiveAxis = rng.nextInt(0, rank - 1);
    const axis = rng.nextBool() ? positiveAxis - rank : positiveAxis;

    test(`resident sum_axis case ${c}: shape=[${shape.join(",")}] axis=${axis}`, () => {
      const ref = sumRuntime(shape, data, axis);
      const a = WNDArray.fromArray(core, shape, Array.from(data));
      try {
        const got = a.sum(axis);
        try {
          const ctx = `resident sum_axis case ${c} shape=[${shape.join(",")}] axis=${axis}`;
          assertShapeEqual(ref.shape, got.shape, ctx);
          assertDataBitIdentical(ref.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        a.dispose();
      }
    });
  }
}

// --- transpose: resident vs naive, bit-identical ---------------------------
{
  const rng = makeRng(0x5245535f5452414en); // "RES_TRAN"-ish
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);

    test(`resident transpose case ${c}: shape=[${shape.join(",")}]`, () => {
      const ref = transposeRuntime(shape, data);
      const a = WNDArray.fromArray(core, shape, Array.from(data));
      try {
        const got = a.transpose();
        try {
          const ctx = `resident transpose case ${c} shape=[${shape.join(",")}]`;
          assertShapeEqual(ref.shape, got.shape, ctx);
          assertDataBitIdentical(ref.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        a.dispose();
      }
    });
  }
}

// --- zeros/ones: light sanity check (not a runtime.ts-backed op, but
// exercises the `nt_fill`-based construction path with no reference
// counterpart other than a plain JS literal). -------------------------------
test("resident zeros/ones: plumbing sanity", () => {
  const z = WNDArray.zeros(core, [3, 4]);
  try {
    assertDataBitIdentical(new Float64Array(12).fill(0), z.toArray(), "zeros [3,4]");
  } finally {
    z.dispose();
  }
  const o = WNDArray.ones(core, [3, 4]);
  try {
    assertDataBitIdentical(new Float64Array(12).fill(1), o.toArray(), "ones [3,4]");
  } finally {
    o.dispose();
  }
});

// --- chained residency: add -> matmul -> transpose -> sum, staying
// resident throughout; intermediates disposed along the way; final result
// compared bit-identically against the same chain run on the naive
// reference. Square shapes throughout so every step is well-defined
// (elementwise add needs equal shapes here to keep the chain's matmul/
// transpose/sum steps unambiguous; broadcasting is already covered by the
// dedicated `add` differential cases above). ------------------------------
{
  const rng = makeRng(0x5245535f4348414en); // "RES_CHAN"-ish
  const CHAIN_CASE_COUNT = 60;
  for (let c = 0; c < CHAIN_CASE_COUNT; c++) {
    const n = rng.nextInt(1, 6);
    const shape = [n, n];
    const aData = genData(rng, shape);
    const bData = genData(rng, shape);
    const eData = genData(rng, shape);

    test(`resident chain case ${c}: add->matmul->transpose->sum(0), n=${n}`, () => {
      // naive reference chain
      const refC = elementwiseBinary(shape, aData, shape, bData, (x, y) => x + y);
      const refD = matmulRuntime(refC.shape, refC.data, shape, eData);
      const refF = transposeRuntime(refD.shape, refD.data);
      const refG = sumRuntime(refF.shape, refF.data, 0);

      // resident chain, disposing every intermediate as we go
      const a = WNDArray.fromArray(core, shape, Array.from(aData));
      const b = WNDArray.fromArray(core, shape, Array.from(bData));
      const rC = a.add(b);
      a.dispose();
      b.dispose();

      const e = WNDArray.fromArray(core, shape, Array.from(eData));
      const rD = rC.matmul(e);
      rC.dispose();
      e.dispose();

      const rF = rD.transpose();
      rD.dispose();

      const rG = rF.sum(0);
      rF.dispose();

      try {
        const ctx = `resident chain case ${c} n=${n}`;
        assertShapeEqual(refG.shape, rG.shape, ctx);
        assertDataBitIdentical(refG.data, rG.toArray(), ctx);
      } finally {
        rG.dispose();
      }

      // every intermediate handle should now report disposed
      assert.strictEqual(a.disposed, true);
      assert.strictEqual(b.disposed, true);
      assert.strictEqual(rC.disposed, true);
      assert.strictEqual(e.disposed, true);
      assert.strictEqual(rD.disposed, true);
      assert.strictEqual(rF.disposed, true);
      assert.strictEqual(rG.disposed, true);
    });
  }
}

// =============================================================================
// WASM parity S2 (docs/wasm-parity-mean-spec.md, D5): `WNDArray.mean` — the
// M1 differential. `mean` is a pure-TS composition (`this.sum(axis,
// keepdims).div(n)`, no new kernel), but bit-identity to `meanRuntime` is
// still proven directly here, not merely argued.
//
// F1 methodology (Baustein-0 finding, spec addendum — CRITICAL): `meanRuntime`
// takes NO `keepdims` parameter and always returns the REDUCED shape, so a
// direct `assertShapeEqual` against `meanRuntime(...).shape` fails every
// keepdims=true case. Data is compared against `meanRuntime(...).data`
// (keepdims-invariant: a size-1 axis never changes the element count); shape
// is compared against `keepdims ? keepDimsShape(shape, axis) : ref.shape` —
// exactly the methodology `scalar-mean.test.ts:285-292`/`:461-466` already
// establishes on the NDArray side.
// =============================================================================

// --- mean(): full reduction, resident vs naive, bit-identical -------------
{
  const rng = makeRng(0x5245535f4d45414en); // "RES_MEAN"
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);
    const keepdims = rng.nextBool();

    test(`resident mean_all case ${c}: shape=[${shape.join(",")}] keepdims=${keepdims}`, () => {
      const ref = meanRuntime(shape, data, undefined);
      const a = WNDArray.fromArray(core, shape, Array.from(data));
      try {
        const got = a.mean(undefined, keepdims);
        try {
          const ctx = `resident mean_all case ${c} shape=[${shape.join(",")}] keepdims=${keepdims}`;
          const expectedShape = keepdims ? keepDimsShape(shape, undefined) : ref.shape;
          assertShapeEqual(expectedShape, got.shape, ctx);
          assertDataBitIdentical(ref.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        a.dispose();
      }
    });
  }
}

// --- mean(axis[, keepdims]): resident vs naive, bit-identical -------------
{
  const rng = makeRng(0x5245535f4d5f4158n); // "RES_M_AX"
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 1, 4);
    const data = genData(rng, shape);
    const rank = shape.length;
    const positiveAxis = rng.nextInt(0, rank - 1);
    const axis = rng.nextBool() ? positiveAxis - rank : positiveAxis;
    const keepdims = rng.nextBool();

    test(`resident mean_axis case ${c}: shape=[${shape.join(",")}] axis=${axis} keepdims=${keepdims}`, () => {
      const ref = meanRuntime(shape, data, axis);
      const a = WNDArray.fromArray(core, shape, Array.from(data));
      try {
        const got = a.mean(axis, keepdims);
        try {
          const ctx = `resident mean_axis case ${c} shape=[${shape.join(",")}] axis=${axis} keepdims=${keepdims}`;
          const expectedShape = keepdims ? keepDimsShape(shape, axis) : ref.shape;
          assertShapeEqual(expectedShape, got.shape, ctx);
          assertDataBitIdentical(ref.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        a.dispose();
      }
    });
  }
}

// --- mean: non-vacuous sum/n vs sum*(1/n) determinism pin (D5), full form --
// Proves the composition's `summed.div(n)` produces `sum/n`, NOT
// `sum*(1/n)` — same discriminator scalar-mean.test.ts's own NDArray-side
// pin uses (n=49, sum=5: `5/49` and `5*(1/49)` diverge in f64). Precondition
// asserted first so the pin is non-vacuous (proves the two formulas actually
// differ before pinning which one `WNDArray.mean` uses).
test("resident mean: sum/n vs sum*(1/n) discriminator, full reduction (n=49, sum=5)", () => {
  const n = 49;
  const viaDiv = 5 / n;
  const viaMul = 5 * (1 / n);
  assert.notStrictEqual(viaDiv, viaMul, "precondition: the two formulas must actually diverge in f64");

  const data = new Float64Array(n);
  data[0] = 5;
  const a = WNDArray.fromArray(core, [n], Array.from(data));
  try {
    const got = a.mean();
    try {
      const gotVal = got.toArray()[0];
      assert.ok(Object.is(gotVal, viaDiv), `mean() must equal sum/n = ${viaDiv}, got ${gotVal}`);
      assert.notStrictEqual(gotVal, viaMul, "mean() must NOT equal the rejected sum*(1/n) formula");
    } finally {
      got.dispose();
    }
  } finally {
    a.dispose();
  }
});

// --- mean: the SAME discriminator at axis granularity (D5) -----------------
test("resident mean: sum/n vs sum*(1/n) discriminator, axis form (n=49, sum=5)", () => {
  const n = 49;
  const viaDiv = 5 / n;
  const viaMul = 5 * (1 / n);
  assert.notStrictEqual(viaDiv, viaMul, "precondition: the two formulas must actually diverge in f64");

  const data = new Float64Array(n); // single row [1, n]: sum=5, rest 0
  data[0] = 5;
  const a = WNDArray.fromArray(core, [1, n], Array.from(data));
  try {
    const got = a.mean(1);
    try {
      const gotVal = got.toArray()[0];
      assert.ok(Object.is(gotVal, viaDiv), `mean(1) must equal sum/n = ${viaDiv}, got ${gotVal}`);
      assert.notStrictEqual(gotVal, viaMul, "mean(1) must NOT equal the rejected sum*(1/n) formula");
    } finally {
      got.dispose();
    }
  } finally {
    a.dispose();
  }
});

// --- mean: size-0 -> NaN, never a throw, both reduction paths (D5) --------
test("resident mean: full reduction of an empty (size-0) receiver is NaN (0/0), never throws", () => {
  const a = WNDArray.fromArray(core, [0], []);
  try {
    const got = a.mean();
    try {
      const v = got.toArray()[0];
      assert.ok(Number.isNaN(v), `expected NaN, got ${v}`);
    } finally {
      got.dispose();
    }
  } finally {
    a.dispose();
  }
});

test("resident mean: a size-0 axis is NaN for every output element (0/0), never throws", () => {
  const a = WNDArray.fromArray(core, [2, 0, 3], []);
  try {
    const got = a.mean(1);
    try {
      assertShapeEqual([2, 3], got.shape, "resident mean(1) over a size-0 axis stays well-defined (shape)");
      assert.ok(
        Array.from(got.toArray()).every((v) => Number.isNaN(v)),
        "resident mean over a size-0 axis must be all-NaN",
      );
    } finally {
      got.dispose();
    }
  } finally {
    a.dispose();
  }
});

// --- mean on VIEWS (Verify-B coverage gap, WASM parity S2): the mean_all/
// mean_axis differential blocks above only exercise contiguous receivers via
// `WNDArray.fromArray` — this closes that gap by running `mean` through the
// same non-contiguous view kinds `keepdims.test.ts` already established for
// `sum` (transposed, sliced, offset-shifted, and their composition), each
// across niladic/positive-axis/negative-axis and keepdims true/false.
//
// F1 methodology (unchanged from mean_all/mean_axis above): `meanRuntime`
// has no keepdims parameter, so DATA is compared against
// `meanRuntime(<view's logical shape>, <view's logical data>, axis).data`,
// and SHAPE against `keepdims ? keepDimsShape(viewShape, axis) : ref.shape`
// — never directly against `meanRuntime(...).shape`. The view's logical
// data is obtained via `view.toArray()` taken BEFORE the `mean` call (the
// same technique the S1 scalar-op view tests use in elementwise.test.ts) —
// an independent read of the view's own already-proven logical content, not
// a second pass through the op under test.

/** Runs `view.mean(axis, keepdims)` and asserts it against `meanRuntime`
 * over the view's own logical shape/data (`view.toArray()`, read BEFORE the
 * mean call). Disposes `got`; the caller owns `view`/any base handles. */
function assertMeanViewMatches(view: AnyWNDArray, axis: number | undefined, keepdims: boolean, ctx: string): void {
  const viewShape = view.shape as readonly number[];
  const viewData = view.toArray(); // logical content of the view, BEFORE the mean call
  const ref = meanRuntime(viewShape, viewData, axis);
  const got = view.mean(axis, keepdims);
  try {
    const expectedShape = keepdims ? keepDimsShape(viewShape, axis) : ref.shape;
    assertShapeEqual(expectedShape, got.shape as readonly number[], ctx);
    assertDataBitIdentical(ref.data, got.toArray(), ctx);
  } finally {
    got.dispose();
  }
}

// transpose view: [3,4] -> T [4,3]
for (const axis of [0, 1, undefined] as const) {
  for (const keepdims of [true, false] as const) {
    test(`resident mean on transpose view: [3,4]^T axis=${axis} keepdims=${keepdims}`, () => {
      const w = WNDArray.fromArray(core, [3, 4], Array.from({ length: 12 }, (_, i) => i + 1));
      try {
        const view = w.transpose(); // O(1) view, reversed strides
        try {
          assertMeanViewMatches(view, axis, keepdims, `resident mean transpose view axis=${axis} keepdims=${keepdims}`);
        } finally {
          view.dispose();
        }
      } finally {
        w.dispose();
      }
    });
  }
}

// sliced view (step slice, non-natural strides, offset 0): [4,3] -> rows
// {0,2} via step 2 -> [2,3].
for (const axis of [0, 1, undefined] as const) {
  for (const keepdims of [true, false] as const) {
    test(`resident mean on sliced view (step): [4,3] step 2 axis=${axis} keepdims=${keepdims}`, () => {
      const baseData = Array.from({ length: 12 }, (_, i) => (i + 1) * (i + 1));
      const w = WNDArray.fromArray(core, [4, 3], baseData);
      try {
        const view = w.slice({ step: 2 }, null); // O(1) view, non-natural strides
        try {
          assertMeanViewMatches(view, axis, keepdims, `resident mean sliced view axis=${axis} keepdims=${keepdims}`);
        } finally {
          view.dispose();
        }
      } finally {
        w.dispose();
      }
    });
  }
}

// offset window (nonzero offset, natural strides): [5,3] -> rows 2.. -> [3,3], offset 6.
for (const axis of [0, -1, undefined] as const) {
  for (const keepdims of [true, false] as const) {
    test(`resident mean on offset window: [5,3] rows 2.. axis=${axis} keepdims=${keepdims}`, () => {
      const baseData = Array.from({ length: 15 }, (_, i) => i - 7);
      const w = WNDArray.fromArray(core, [5, 3], baseData);
      try {
        const view = w.slice({ start: 2 }); // O(1) view, offset 6, natural strides
        try {
          assertMeanViewMatches(view, axis, keepdims, `resident mean offset window axis=${axis} keepdims=${keepdims}`);
        } finally {
          view.dispose();
        }
      } finally {
        w.dispose();
      }
    });
  }
}

// composed view: [2,3,4] -> transpose [4,3,2] -> slice rows 1.. of axis 1 ->
// [4,2,2], non-natural strides AND nonzero offset.
for (const axis of [0, 1, 2, undefined] as const) {
  for (const keepdims of [true, false] as const) {
    test(`resident mean on composed transpose+slice view: [2,3,4]^T sliced axis=${axis} keepdims=${keepdims}`, () => {
      const baseData = Array.from({ length: 24 }, (_, i) => i - 11);
      const w = WNDArray.fromArray(core, [2, 3, 4], baseData);
      try {
        const t = w.transpose();
        try {
          const view = t.slice(null, { start: 1 }, null);
          try {
            assertMeanViewMatches(view, axis, keepdims, `resident mean composed view axis=${axis} keepdims=${keepdims}`);
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
}
