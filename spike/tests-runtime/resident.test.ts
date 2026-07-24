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
import { NDArray } from "../src/ndarray.ts";
import { elementwiseBinary, itemRuntime, keepDimsShape, matmulRuntime, meanRuntime, stackRuntime, sumRuntime, transposeRuntime } from "../src/runtime.ts";
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

// =============================================================================
// WASM parity S3 (docs/wasm-parity-item-stack-spec.md, D3): `WNDArray.item`
// — the M1 differential. `item` never calls into WASM at all (the fourth M1
// case this campaign has hit, spec's own M1-Einordnung section): the offset
// arithmetic runs in plain TS (`itemOffsetStrided`, runtime.ts) and the read
// is a single `Float64Array` index — but bit-identity to `itemRuntime` is
// still proven directly here, not merely argued.
//
// Reference methodology (D7/T3): for a VIEW, the reference is
// `itemRuntime(view.shape, view.toArray(), indices)` — `toArray()` returns
// the view's own logical row-major copy, which is exactly the contiguous
// layout `itemRuntime`'s own argument assumes. Never `itemRuntime` against
// the BASE array's raw (pre-transform) data, which would silently be wrong
// for anything but the identity view.
// =============================================================================

/** Generates a random, valid index tuple for `shape` — every axis mixes
 * positive and (NumPy-normalized) negative forms (spec's "negative Indizes
 * (jede Achse), gemischt positiv/negativ" requirement); every generated
 * index is guaranteed IN BOUNDS after normalization, so this always
 * exercises the success path (the dedicated d===0 test below covers the
 * bounds-throw path on its own). */
function genValidItemIndices(rng: Rng, shape: readonly number[]): number[] {
  return shape.map((d) => {
    const positive = rng.nextInt(0, Math.max(d - 1, 0));
    return d > 0 && rng.nextBool() ? positive - d : positive;
  });
}

/** Runs `view.item(...indices)` and asserts it against `itemRuntime` over
 * the view's own logical shape/data (`view.toArray()`, read BEFORE the
 * `item()` call itself — same "read logical content first, independently"
 * technique `assertMeanViewMatches` above already uses). */
function assertItemMatches(view: AnyWNDArray, indices: readonly number[], ctx: string): void {
  const shape = view.shape as readonly number[];
  const data = view.toArray();
  const ref = itemRuntime(shape, data, indices);
  const got = view.item(...(indices as number[]));
  assert.ok(Object.is(ref, got), `${ctx}: expected ${ref} (itemRuntime reference), got ${got}`);
}

// --- item: contiguous, randomized grid, ranks 0..4, mixed +/- indices ------
{
  const rng = makeRng(0x5245535f4954454dn); // "RES_ITEM"
  const CASE_COUNT = 120;
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);
    const indices = genValidItemIndices(rng, shape);

    test(`item case ${c}: shape=[${shape.join(",")}] indices=[${indices.join(",")}]`, () => {
      const a = WNDArray.fromArray(core, shape, Array.from(data));
      try {
        assertItemMatches(a, indices, `item case ${c} shape=[${shape.join(",")}] indices=[${indices.join(",")}]`);
      } finally {
        a.dispose();
      }
    });
  }
}

// --- item on VIEWS (Arbeitsregel 12, D3/D7 explicit requirement): transpose,
// step-sliced, offset window, and their composition — the same four view
// kinds `assertMeanViewMatches` above already establishes for `mean`. Pflicht-
// Mutant M-a (`this.strides`/`this.offset` -> `computeStrides(this.shape)`/`0`)
// must be caught by these cases, not by the contiguous grid above. ----------

// transpose view: [3,4] -> T [4,3]
{
  const rng = makeRng(0x4954454d5f545241n); // "ITEM_TRA"
  const CASE_COUNT = 30;
  for (let c = 0; c < CASE_COUNT; c++) {
    test(`item on transpose view case ${c}: [3,4]^T`, () => {
      const w = WNDArray.fromArray(core, [3, 4], Array.from({ length: 12 }, (_, i) => i + 1));
      try {
        const view = w.transpose();
        try {
          const indices = genValidItemIndices(rng, view.shape as readonly number[]);
          assertItemMatches(view, indices, `item transpose view case ${c} indices=[${indices.join(",")}]`);
        } finally {
          view.dispose();
        }
      } finally {
        w.dispose();
      }
    });
  }
}

// step-sliced view (non-natural strides, offset 0): [4,3] step 2 -> [2,3]
{
  const rng = makeRng(0x4954454d5f53544en); // "ITEM_STN"-ish
  const CASE_COUNT = 30;
  for (let c = 0; c < CASE_COUNT; c++) {
    test(`item on sliced (step) view case ${c}: [4,3] step 2`, () => {
      const baseData = Array.from({ length: 12 }, (_, i) => (i + 1) * (i + 1));
      const w = WNDArray.fromArray(core, [4, 3], baseData);
      try {
        const view = w.slice({ step: 2 }, null);
        try {
          const indices = genValidItemIndices(rng, view.shape as readonly number[]);
          assertItemMatches(view, indices, `item sliced view case ${c} indices=[${indices.join(",")}]`);
        } finally {
          view.dispose();
        }
      } finally {
        w.dispose();
      }
    });
  }
}

// offset window (nonzero offset, natural strides): [5,3] rows 2.. -> [3,3], offset 6
{
  const rng = makeRng(0x4954454d5f4f4646n); // "ITEM_OFF"
  const CASE_COUNT = 30;
  for (let c = 0; c < CASE_COUNT; c++) {
    test(`item on offset window case ${c}: [5,3] rows 2..`, () => {
      const baseData = Array.from({ length: 15 }, (_, i) => i - 7);
      const w = WNDArray.fromArray(core, [5, 3], baseData);
      try {
        const view = w.slice({ start: 2 });
        try {
          const indices = genValidItemIndices(rng, view.shape as readonly number[]);
          assertItemMatches(view, indices, `item offset window case ${c} indices=[${indices.join(",")}]`);
        } finally {
          view.dispose();
        }
      } finally {
        w.dispose();
      }
    });
  }
}

// composed view: [2,3,4] -> transpose [4,3,2] -> slice rows 1.. of axis 1 -> [4,2,2]
{
  const rng = makeRng(0x4954454d5f434f4dn); // "ITEM_COM"
  const CASE_COUNT = 30;
  for (let c = 0; c < CASE_COUNT; c++) {
    test(`item on composed transpose+slice view case ${c}: [2,3,4]^T sliced`, () => {
      const baseData = Array.from({ length: 24 }, (_, i) => i - 11);
      const w = WNDArray.fromArray(core, [2, 3, 4], baseData);
      try {
        const t = w.transpose();
        try {
          const view = t.slice(null, { start: 1 }, null);
          try {
            const indices = genValidItemIndices(rng, view.shape as readonly number[]);
            assertItemMatches(view, indices, `item composed view case ${c} indices=[${indices.join(",")}]`);
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

// --- item: rank 0 (item(), zero arguments) ----------------------------------
test("item: rank 0 (item(), zero arguments) reads the sole element", () => {
  const a = WNDArray.fromArray(core, [], [42.5]);
  try {
    assertItemMatches(a, [], "item rank 0");
  } finally {
    a.dispose();
  }
});

/** Runs `fn()` and returns the thrown `Error`'s message — fails the test if
 * `fn()` does NOT throw. Declared once, up front (hoisted `function`, used
 * throughout this file's item/stack message-parity tests below) so a
 * non-throwing call surfaces as a clear, dedicated failure instead of being
 * silently swallowed by a surrounding try/catch that expected a throw. */
function throwMessage(fn: () => unknown): string {
  try {
    fn();
  } catch (e) {
    return e instanceof Error ? e.message : String(e);
  }
  throw new Error("expected fn() to throw, but it did not");
}

// --- item: a d===0 axis rejects EVERY index (bounds) — same message across
// the NDArray and WNDArray surfaces (D7's explicit bullet). ------------------
test("item: a d===0 axis rejects every index (bounds), message word-for-word identical across surfaces", () => {
  const shape = [2, 0, 3];
  const nd = NDArray.fromArray(shape, []);
  const wnd = WNDArray.fromArray(core, shape, []);
  try {
    for (const idx of [0, -1, 5, -5]) {
      const ndMsg = throwMessage(() => nd.item(0, idx, 0));
      assert.throws(() => wnd.item(0, idx, 0), (err: unknown) => {
        assert.ok(err instanceof Error);
        assert.strictEqual(err.message, ndMsg, `resident item message must match the naive surface for index ${idx}`);
        return true;
      });
    }
  } finally {
    wnd.dispose();
  }
});

// =============================================================================
// WASM parity S3 (docs/wasm-parity-item-stack-spec.md, D4/T4, Covenant M3):
// cross-surface message parity for `item`/`stack`. D4 deliberately
// DUPLICATES `itemRuntime`/`stackRuntime`'s validation logic into two new
// `runtime.ts` helpers rather than refactoring the oracle functions — the
// price is drift risk, mechanically paid off here: all SIX stems (item:
// arity/integer/bounds; stack: empty/rank/length-mismatch) must be
// WORD-FOR-WORD identical between the NDArray surface (backed by
// `itemRuntime`/`stackRuntime`) and the WNDArray surface (backed by the new
// helpers), checked via STRING EQUALITY — not a prefix/regex match (the
// spec's own explicit requirement). Non-vacuity: Pflicht-Mutant M-c (a
// stem's wording tweaked in the new helper) must turn every one of these
// six tests red.
//
// Every receiver below is built from a plain, non-`const` shape variable
// (`number[]`, not a literal tuple) — the same "widen past the guard"
// technique `resident-lifecycle.test.ts`'s own failing-op tests and
// `scalar-mean.test.ts`'s W4 stem tests already use — so `ItemGuard`/
// `StackCheck`'s compile-time machinery degrades to no-claim and the call
// actually REACHES the runtime throw under test, instead of being rejected
// by `tsc` before this file even compiles. `throwMessage` (above, defined
// once for the whole file) is the shared helper.
// =============================================================================

test("cross-surface message parity: item arity stem, word-for-word (T4, M3)", () => {
  const shape: number[] = [2, 3]; // dynamic rank: bypasses TS2554, reaches the runtime arity check
  const nd = NDArray.fromArray(shape, [1, 2, 3, 4, 5, 6]);
  const wnd = WNDArray.fromArray(core, shape, [1, 2, 3, 4, 5, 6]);
  try {
    const ndMsg = throwMessage(() => nd.item(0));
    const wndMsg = throwMessage(() => wnd.item(0));
    assert.strictEqual(ndMsg, "item: expected 2 indices (got 1)", "sanity: exact expected wording");
    assert.strictEqual(wndMsg, ndMsg, "item arity stem must be word-for-word identical across surfaces");
  } finally {
    wnd.dispose();
  }
});

test("cross-surface message parity: item integer (dot-form) stem, word-for-word (T4, M3)", () => {
  const shape: number[] = [2, 3];
  const nd = NDArray.fromArray(shape, [1, 2, 3, 4, 5, 6]);
  const wnd = WNDArray.fromArray(core, shape, [1, 2, 3, 4, 5, 6]);
  try {
    const ndMsg = throwMessage(() => nd.item(0.5, 0));
    const wndMsg = throwMessage(() => wnd.item(0.5, 0));
    assert.strictEqual(ndMsg, "item: index 0.5 for axis 0 is not an integer", "sanity: exact expected wording");
    assert.strictEqual(wndMsg, ndMsg, "item integer stem must be word-for-word identical across surfaces");
  } finally {
    wnd.dispose();
  }
});

test("cross-surface message parity: item bounds stem, word-for-word (T4, M3)", () => {
  const shape: number[] = [2, 3];
  const nd = NDArray.fromArray(shape, [1, 2, 3, 4, 5, 6]);
  const wnd = WNDArray.fromArray(core, shape, [1, 2, 3, 4, 5, 6]);
  try {
    const ndMsg = throwMessage(() => nd.item(5, 0));
    const wndMsg = throwMessage(() => wnd.item(5, 0));
    assert.strictEqual(ndMsg, "item: index 5 is out of bounds for axis 0 with dim 2", "sanity: exact expected wording");
    assert.strictEqual(wndMsg, ndMsg, "item bounds stem must be word-for-word identical across surfaces");
  } finally {
    wnd.dispose();
  }
});

test("cross-surface message parity: stack empty-rows stem, word-for-word (T4, M3)", () => {
  // A genuinely dynamic-length runtime ARRAY, never a `[]` TUPLE LITERAL
  // (which is a compile-time rejection, F3 — proven separately by a
  // dedicated expect-error pin in ndarray.test-d.ts) — same technique
  // scalar-mean.test.ts's own W4 empty-rows test uses.
  const ndEmpty: NDArray<number[]>[] = [];
  const wndEmpty: AnyWNDArray[] = [];
  const ndMsg = throwMessage(() => NDArray.stack(ndEmpty));
  const wndMsg = throwMessage(() => WNDArray.stack(core, wndEmpty));
  assert.strictEqual(ndMsg, "stack: expected at least one row", "sanity: exact expected wording");
  assert.strictEqual(wndMsg, ndMsg, "stack empty-rows stem must be word-for-word identical across surfaces");
});

test("cross-surface message parity: stack rank stem, word-for-word (T4, M3)", () => {
  const rank2Shape = [2, 3];
  const rank1Shape = [3];
  const ndBad = NDArray.fromArray(rank2Shape, [1, 2, 3, 4, 5, 6]);
  const ndGood = NDArray.fromArray(rank1Shape, [1, 2, 3]);
  const wndBad = WNDArray.fromArray(core, rank2Shape, [1, 2, 3, 4, 5, 6]);
  const wndGood = WNDArray.fromArray(core, rank1Shape, [1, 2, 3]);
  try {
    const ndMsg = throwMessage(() => NDArray.stack([ndBad, ndGood]));
    const wndMsg = throwMessage(() => WNDArray.stack(core, [wndBad, wndGood]));
    assert.strictEqual(ndMsg, "stack: expected 1-D rows (got shape [2,3] at index 0)", "sanity: exact expected wording");
    assert.strictEqual(wndMsg, ndMsg, "stack rank stem must be word-for-word identical across surfaces");
  } finally {
    wndBad.dispose();
    wndGood.dispose();
  }
});

test("cross-surface message parity: stack length-mismatch stem, word-for-word (T4, M3)", () => {
  const lenAShape = [3];
  const lenBShape = [4];
  const ndA = NDArray.fromArray(lenAShape, [1, 2, 3]);
  const ndB = NDArray.fromArray(lenBShape, [1, 2, 3, 4]);
  const wndA = WNDArray.fromArray(core, lenAShape, [1, 2, 3]);
  const wndB = WNDArray.fromArray(core, lenBShape, [1, 2, 3, 4]);
  try {
    const ndMsg = throwMessage(() => NDArray.stack([ndA, ndB]));
    const wndMsg = throwMessage(() => WNDArray.stack(core, [wndA, wndB]));
    assert.strictEqual(ndMsg, "stack: row length mismatch (expected 3, got 4 at index 1)", "sanity: exact expected wording");
    assert.strictEqual(wndMsg, ndMsg, "stack length-mismatch stem must be word-for-word identical across surfaces");
  } finally {
    wndA.dispose();
    wndB.dispose();
  }
});

// =============================================================================
// WASM parity S3 (docs/wasm-parity-item-stack-spec.md, D5): `WNDArray.stack`
// — the M1 differential (the SAME "composed op over an already-proven
// kernel" case S2/`mean` instantiates: `nt_materialize`, in continuous use
// since Kern 03). Randomized N/D plus the explicit edge cases D7 requires
// (N=1, D=0, D=1, large N, aliasing, cross-core, and the memory.grow case
// Pflicht-Mutant M-d needs).
// =============================================================================

// --- randomized differential grid: mixed contiguous/view rows --------------
{
  const rng = makeRng(0x5245535f5354434bn); // "RES_STCK"
  const CASE_COUNT = 60;
  for (let c = 0; c < CASE_COUNT; c++) {
    const n = rng.nextInt(1, 8);
    const d = rng.nextInt(0, 8);
    const rowsData: number[][] = [];
    const asView: boolean[] = [];
    for (let i = 0; i < n; i++) {
      rowsData.push(Array.from(genData(rng, [d])));
      asView.push(d > 0 && rng.nextBool());
    }

    test(`stack case ${c}: n=${n} d=${d}`, () => {
      const rows: AnyWNDArray[] = [];
      const owners: AnyWNDArray[] = [];
      // Dynamic (`number[]`) shape variables, not inline array literals —
      // keeps every `fromArray`/`slice` call on the cheap wide-shape path
      // instead of paying per-case literal-tuple type machinery for a
      // RUNTIME test that never needed the type layer engaged at all.
      const rowShape: number[] = [d];
      const baseShape: number[] = [d, 2];
      try {
        for (let i = 0; i < n; i++) {
          const data = rowsData[i]!;
          if (asView[i]) {
            // View row: a [d,2] base's column 0, via an integer index on
            // axis 1 -> shape [d], stride 2 (non-natural) — the "geschnittene
            // Zeile" D7 requires, distinct from a plain contiguous row.
            const baseData: number[] = [];
            for (let k = 0; k < d; k++) baseData.push(data[k]!, 0);
            const base = WNDArray.fromArray(core, baseShape, baseData);
            owners.push(base);
            rows.push(base.slice(null, 0));
          } else {
            rows.push(WNDArray.fromArray(core, rowShape, data));
          }
        }
        const ref = stackRuntime(rowsData.map((data) => ({ shape: [d], data: new Float64Array(data) })));
        const stacked = WNDArray.stack(core, rows);
        try {
          assertShapeEqual(ref.shape, stacked.shape as readonly number[], `stack case ${c} n=${n} d=${d}`);
          assertDataBitIdentical(ref.data, stacked.toArray(), `stack case ${c} n=${n} d=${d}`);
        } finally {
          stacked.dispose();
        }
      } finally {
        for (const r of rows) r.dispose();
        for (const o of owners) o.dispose();
      }
    });
  }
}

// --- explicit edge cases (D7: "inkl. N=1, D=0, D=1, große N") --------------
//
// Every receiver below uses a DYNAMIC (`number[]`, non-`const`) shape
// variable, not a literal tuple — same "widen past the guard" technique T4
// above uses. These are RUNTIME differential tests (never `.test-d.ts`
// type-level tests), so the STATIC type visible to `tsc` is irrelevant to
// what they prove; keeping receivers dynamically-shaped avoids paying the
// full `StackFold`/`WRowShapesOf` compile-time machinery per literal-shaped
// call site (measured: this is check:diag's dominant cost driver among the
// new tests) for zero loss of runtime coverage.

test("stack: N=1 (single row)", () => {
  const shape: number[] = [4];
  const a = WNDArray.fromArray(core, shape, [1, 2, 3, 4]);
  try {
    const stacked = WNDArray.stack(core, [a]);
    try {
      assertShapeEqual([1, 4], stacked.shape as readonly number[], "stack N=1 shape");
      assertDataBitIdentical(new Float64Array([1, 2, 3, 4]), stacked.toArray(), "stack N=1 data");
    } finally {
      stacked.dispose();
    }
  } finally {
    a.dispose();
  }
});

test("stack: D=0 rows are valid — [[],[]] stacks to shape [2,0]", () => {
  const shape: number[] = [0];
  const a = WNDArray.fromArray(core, shape, []);
  const b = WNDArray.fromArray(core, shape, []);
  try {
    const stacked = WNDArray.stack(core, [a, b]);
    try {
      assertShapeEqual([2, 0], stacked.shape as readonly number[], "stack D=0 shape");
      assert.strictEqual(stacked.toArray().length, 0);
    } finally {
      stacked.dispose();
    }
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("stack: D=1 rows", () => {
  const shape: number[] = [1];
  const a = WNDArray.fromArray(core, shape, [7]);
  const b = WNDArray.fromArray(core, shape, [8]);
  try {
    const stacked = WNDArray.stack(core, [a, b]);
    try {
      assertShapeEqual([2, 1], stacked.shape as readonly number[], "stack D=1 shape");
      assertDataBitIdentical(new Float64Array([7, 8]), stacked.toArray(), "stack D=1 data");
    } finally {
      stacked.dispose();
    }
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("stack: large N (1000 rows), shape + bit-identical data vs stackRuntime", () => {
  const rowCount = 1000;
  const dims = 6;
  const rng = makeRng(0x5354434b5f4c4152n); // "STCK_LAR"-ish
  const rows: AnyWNDArray[] = [];
  const rowsData: Float64Array[] = [];
  for (let i = 0; i < rowCount; i++) {
    const d = genData(rng, [dims]);
    rowsData.push(d);
    rows.push(WNDArray.fromArray(core, [dims], Array.from(d)));
  }
  try {
    const stacked = WNDArray.stack(core, rows);
    try {
      const ref = stackRuntime(rowsData.map((data) => ({ shape: [dims], data })));
      assertShapeEqual([rowCount, dims], stacked.shape as readonly number[], "stack large-N shape");
      assertDataBitIdentical(ref.data, stacked.toArray(), "stack large-N data");
    } finally {
      stacked.dispose();
    }
  } finally {
    for (const r of rows) r.dispose();
  }
});

// --- aliasing: the SAME row twice ------------------------------------------

test("stack: the SAME row twice (aliasing) is valid — reads it independently for each output slot", () => {
  const shape: number[] = [3];
  const a = WNDArray.fromArray(core, shape, [7, 8, 9]);
  try {
    const stacked = WNDArray.stack(core, [a, a]);
    try {
      assertShapeEqual([2, 3], stacked.shape as readonly number[], "stack aliasing shape");
      assertDataBitIdentical(new Float64Array([7, 8, 9, 7, 8, 9]), stacked.toArray(), "stack aliasing data");
    } finally {
      stacked.dispose();
    }
  } finally {
    a.dispose();
  }
});

// --- rows from different buffers (not just aliasing) -----------------------

test("stack: rows from independently allocated buffers", () => {
  const shape: number[] = [2];
  const a = WNDArray.fromArray(core, shape, [1, 2]);
  const b = WNDArray.zeros(core, shape);
  const c = WNDArray.ones(core, shape);
  try {
    const stacked = WNDArray.stack(core, [a, b, c]);
    try {
      assertShapeEqual([3, 2], stacked.shape as readonly number[], "stack independent-buffers shape");
      assertDataBitIdentical(new Float64Array([1, 2, 0, 0, 1, 1]), stacked.toArray(), "stack independent-buffers data");
    } finally {
      stacked.dispose();
    }
  } finally {
    a.dispose();
    b.dispose();
    c.dispose();
  }
});

// --- cross-core error path ---------------------------------------------------

test("stack: a row from a foreign core throws, naming both operand classes", async () => {
  const otherCore = await initCore();
  const shape: number[] = [3];
  const a = WNDArray.fromArray(core, shape, [1, 2, 3]);
  const b = WNDArray.fromArray(otherCore, shape, [4, 5, 6]);
  try {
    assert.throws(() => WNDArray.stack(core, [a, b]), /WNDArray\.stack: operands belong to different WASM core instances/);
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("stack: EVERY row is checked against the `core` PARAMETER, not against row 0's own core", async () => {
  // Baustein-B finding: the test above cannot tell the two designs apart,
  // because its row 0 always shares the `core` parameter's core — a mutant
  // comparing each row against `rows[0].core` instead of the parameter
  // passed all 1261 tests. D2 chose the parameter deliberately (it is what
  // every other WNDArray static takes, and it keeps the empty-rows case
  // well-defined), so the choice needs a test that can observe it: here ALL
  // rows agree with each other and disagree with the parameter, which is
  // exactly the case the `rows[0]`-based variant would wave through.
  const otherCore = await initCore();
  const shape: number[] = [3];
  const a = WNDArray.fromArray(otherCore, shape, [1, 2, 3]);
  const b = WNDArray.fromArray(otherCore, shape, [4, 5, 6]);
  try {
    assert.throws(() => WNDArray.stack(core, [a, b]), /WNDArray\.stack: operands belong to different WASM core instances/);
  } finally {
    a.dispose();
    b.dispose();
  }
});

// =============================================================================
// D8/M-d (bindend, spec addendum): a `stack` call large enough to actually
// trigger `memory.grow` MID-LOOP. This is the case that makes Pflicht-Mutant
// M-d (the per-iteration strides view hoisted out of the loop) fangbar —
// without a case that provably grows memory DURING the per-row loop, a
// hoisted/stale view is never actually exercised (memory that never grows
// never detaches anything). Verified against the real artifact before
// committing this test (Baustein 0 / this scheibe's own probe): N=4,
// D=50000, alternating contiguous (stride 1) and step-2-sliced (stride 2)
// rows RELIABLY grows `core.memory.buffer` during the call (reproduced
// deterministically across repeated runs) — the alternating strides matter
// as much as the growth itself: a mutant that gets "stuck" on a STALE but
// numerically IDENTICAL stride (e.g. all-contiguous rows) would silently
// pass despite being broken.
// =============================================================================

test("stack: a large call that actually triggers memory.grow mid-loop stays byte-identical to stackRuntime (D8/M-d non-vacuity)", () => {
  const n = 4;
  const d = 50000;
  const rowShape: number[] = [d]; // dynamic: avoids paying the literal-tuple type cost for a plain runtime test
  const baseShape: number[] = [d * 2];
  const rowsData: Float64Array[] = [];
  const rows: AnyWNDArray[] = [];
  const owners: AnyWNDArray[] = [];
  for (let i = 0; i < n; i++) {
    if (i % 2 === 0) {
      const data = new Float64Array(d);
      for (let k = 0; k < d; k++) data[k] = i * 1000 + k;
      rowsData.push(data);
      rows.push(WNDArray.fromArray(core, rowShape, data));
    } else {
      const base: number[] = new Array(d * 2);
      for (let k = 0; k < d * 2; k++) base[k] = i * 1000 + k;
      const baseArr = WNDArray.fromArray(core, baseShape, base);
      owners.push(baseArr);
      const view = baseArr.slice({ step: 2 }); // stride 2, differs from the contiguous rows' stride 1
      rows.push(view);
      const exp = new Float64Array(d);
      for (let k = 0; k < d; k++) exp[k] = base[k * 2]!;
      rowsData.push(exp);
    }
  }
  try {
    const byteLengthBefore = core.memory.buffer.byteLength;
    const stacked = WNDArray.stack(core, rows);
    try {
      const byteLengthAfter = core.memory.buffer.byteLength;
      assert.notStrictEqual(
        byteLengthAfter,
        byteLengthBefore,
        `precondition: this case must actually trigger memory.grow during the stack() call (before=${byteLengthBefore}, after=${byteLengthAfter}) — otherwise M-d would not be exercised`,
      );
      const ref = stackRuntime(rowsData.map((data) => ({ shape: [d], data })));
      assertShapeEqual(ref.shape, stacked.shape as readonly number[], "stack memory.grow case shape");
      assertDataBitIdentical(
        ref.data,
        stacked.toArray(),
        "stack memory.grow case data — every row's OWN stride must survive the mid-loop growth, not a stale earlier row's",
      );
    } finally {
      stacked.dispose();
    }
  } finally {
    for (const r of rows) r.dispose();
    for (const o of owners) o.dispose();
  }
});

// =============================================================================
// WASM parity S4 (docs/wasm-parity-argmax-spec.md, D7): `WNDArray.argmax` —
// the M1 differential against `argmaxRuntime`, the pinned correctness oracle
// (spike/src/runtime.ts, deliberately UNCHANGED by this slice: a parity proof
// that rebuilds its own oracle in the same commit proves less).
//
// Oracle methodology (spec D1, BINDING — the Baustein-0 finding that this repo
// carried TWO mutually incompatible precedents for exactly this question):
//  1. DATA is always compared against `argmaxRuntime` — never circular. For a
//     VIEW receiver the reference is built from `view.toArray()`, the view's
//     own logical row-major content, read BEFORE the method under test runs,
//     together with `view.shape`. Never re-derived from the base buffer (the
//     S2/F1 lesson).
//  2. The keepdims SHAPE is asserted through STRUCTURAL INVARIANTS, never
//     against `keepDimsShape` — that helper is exactly what `WNDArray.argmax`
//     itself calls, so using it as the oracle here WOULD be circular (the
//     objection `argmax-topk.test.ts:216-240` already raises on the NDArray
//     side). The NON-keepdims shape, by contrast, comes out of `argmaxRuntime`
//     itself and is a legitimate oracle, so it is compared directly.
//  3. Additionally, an oracle-FREE cross-surface shape pin (further below):
//     `WNDArray.argmax(axis, kd).shape` deep-equals `NDArray.argmax(axis,
//     kd).shape` on the materialized equivalent.
// =============================================================================

import { argmaxRuntime } from "../src/runtime.ts";
import { bitsOf } from "./assert-helpers.ts";
import { genDataSpecial } from "./prng.ts";

/** Raw 64-bit comparison (the spec's explicit "compare the result BITS, not
 * just `===`"). Strictly stronger than `assertDataBitIdentical`'s `Object.is`
 * for NaN payloads and identical everywhere else; argmax results are integral
 * indices, so any bit difference at all is a real divergence. */
function assertIndexBitsIdentical(expected: Float64Array, actual: Float64Array, ctx: string): void {
  assert.strictEqual(actual.length, expected.length, `${ctx}: result length mismatch, expected ${expected.length} got ${actual.length}`);
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i] ?? 0;
    const a = actual[i] ?? 0;
    assert.strictEqual(
      bitsOf(a),
      bitsOf(e),
      `${ctx}: index bits differ at flat ${i}: reference=${e} (0x${bitsOf(e).toString(16)}) wasm=${a} (0x${bitsOf(a).toString(16)})`,
    );
  }
}

/** D1(2): keepdims SHAPE via structural invariants only — deliberately NOT
 * `keepDimsShape` (circular). `reducedShape` is `argmaxRuntime`'s own output
 * shape and IS a legitimate oracle. Mirrors the invariant set
 * `argmax-topk.test.ts` established for the NDArray surface. */
function assertKeepdimsShapeInvariants(
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
    // Every axis was reduced -> every kept axis must be size 1.
    assert.ok(
      keptShape.every((d) => d === 1),
      `${ctx}: full reduction with keepdims must be all-ones, got [${keptShape.join(",")}]`,
    );
    return;
  }
  const normAxis = axis < 0 ? inputShape.length + axis : axis;
  assert.strictEqual(keptShape[normAxis], 1, `${ctx}: the reduced axis ${normAxis} must be size 1 under keepdims`);
  const withoutAxis = [...keptShape.slice(0, normAxis), ...keptShape.slice(normAxis + 1)];
  assert.deepStrictEqual([...withoutAxis], [...reducedShape], `${ctx}: keepdims shape minus the reduced axis must equal the non-keepdims shape`);
  assert.deepStrictEqual(
    [...withoutAxis],
    [...inputShape.slice(0, normAxis), ...inputShape.slice(normAxis + 1)],
    `${ctx}: every non-reduced axis must be unchanged from the input shape`,
  );
}

/** Runs `recv.argmax(axis, keepdims)` and asserts it against `argmaxRuntime`
 * over the receiver's OWN logical shape/data (`toArray()`, read BEFORE the
 * argmax call — D1(1)). Works uniformly for contiguous handles and views:
 * a contiguous handle is just the identity view. Disposes `got`; the caller
 * owns `recv` and any base handles. */
function assertArgmaxMatches(recv: AnyWNDArray, axis: number | undefined, keepdims: boolean, ctx: string): void {
  const shape = recv.shape as readonly number[];
  const data = recv.toArray(); // logical content, BEFORE the argmax call
  const ref = argmaxRuntime(shape, data, axis);
  const got = recv.argmax(axis, keepdims);
  try {
    if (keepdims) assertKeepdimsShapeInvariants(shape, axis, got.shape as readonly number[], ref.shape, ctx);
    else assertShapeEqual(ref.shape, got.shape as readonly number[], ctx);
    assertIndexBitsIdentical(ref.data, got.toArray(), ctx);
  } finally {
    got.dispose();
  }
}

/** The TRULY niladic form (`argmax()`, zero arguments -> plain `number`,
 * D2) — a different overload from `argmax(undefined)`, so it needs its own
 * assertion path. */
function assertArgmaxNiladicMatches(recv: AnyWNDArray, ctx: string): void {
  const shape = recv.shape as readonly number[];
  const data = recv.toArray();
  const ref = argmaxRuntime(shape, data, undefined).data[0] ?? 0;
  const got = recv.argmax();
  assert.strictEqual(typeof got, "number", `${ctx}: niladic argmax() must return a plain number`);
  assert.strictEqual(bitsOf(got), bitsOf(ref), `${ctx}: niladic argmax() must equal argmaxRuntime's flat index ${ref}, got ${got}`);
}

// --- argmax(): full reduction, resident vs naive, bit-identical -----------
// Covers BOTH full-reduction forms per case: the niladic `number` form and
// the 1-/2-arg `argmax(undefined[, keepdims])` WNDArray form.
{
  const rng = makeRng(0x5245535f414d4158n); // "RES_AMAX"
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);
    const keepdims = rng.nextBool();

    test(`resident argmax_all case ${c}: shape=[${shape.join(",")}] keepdims=${keepdims}`, () => {
      const a = WNDArray.fromArray(core, shape, Array.from(data));
      try {
        const ctx = `resident argmax_all case ${c} shape=[${shape.join(",")}] keepdims=${keepdims}`;
        assertArgmaxNiladicMatches(a, `${ctx} (niladic)`);
        assertArgmaxMatches(a, undefined, keepdims, ctx);
      } finally {
        a.dispose();
      }
    });
  }
}

// --- argmax(axis[, keepdims]): resident vs naive, bit-identical -----------
{
  const rng = makeRng(0x5245535f414d4158n + 1n); // "RES_AMAX"+1
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 1, 4);
    const data = genData(rng, shape);
    const rank = shape.length;
    const positiveAxis = rng.nextInt(0, rank - 1);
    const axis = rng.nextBool() ? positiveAxis - rank : positiveAxis;
    const keepdims = rng.nextBool();

    test(`resident argmax_axis case ${c}: shape=[${shape.join(",")}] axis=${axis} keepdims=${keepdims}`, () => {
      const a = WNDArray.fromArray(core, shape, Array.from(data));
      try {
        assertArgmaxMatches(a, axis, keepdims, `resident argmax_axis case ${c} shape=[${shape.join(",")}] axis=${axis} keepdims=${keepdims}`);
      } finally {
        a.dispose();
      }
    });
  }
}

// --- argmax on VIEWS (Arbeitsregel 12 / spec D7: the FOUR view classes) ---
// The interesting receiver for a resident op is the non-contiguous one, and
// for `argmax` specifically it is load-bearing beyond coverage: the returned
// index must be the index into the VIEW's logical row-major flattening, which
// on a transposed/sliced receiver genuinely differs from the memory offset
// (kernel doc, crates/core/src/kernels/argmax.rs). A memory-order bug would
// pass every contiguous case above and fail here.

// transpose view: [3,4] -> T [4,3]
for (const axis of [0, 1, undefined] as const) {
  for (const keepdims of [true, false] as const) {
    test(`resident argmax on transpose view: [3,4]^T axis=${axis} keepdims=${keepdims}`, () => {
      const w = WNDArray.fromArray(core, [3, 4], [5, 2, 9, 1, 7, 3, 0, 8, 4, 6, 11, 10]);
      try {
        const view = w.transpose(); // O(1) view, reversed strides
        try {
          const ctx = `resident argmax transpose view axis=${axis} keepdims=${keepdims}`;
          assertArgmaxMatches(view, axis, keepdims, ctx);
          assertArgmaxNiladicMatches(view, `${ctx} (niladic)`);
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
    test(`resident argmax on sliced view (step): [4,3] step 2 axis=${axis} keepdims=${keepdims}`, () => {
      const baseData = [3, 14, 1, 5, 9, 2, 6, 53, 5, 8, 97, 9];
      const w = WNDArray.fromArray(core, [4, 3], baseData);
      try {
        const view = w.slice({ step: 2 }, null); // O(1) view, non-natural strides
        try {
          const ctx = `resident argmax sliced view axis=${axis} keepdims=${keepdims}`;
          assertArgmaxMatches(view, axis, keepdims, ctx);
          assertArgmaxNiladicMatches(view, `${ctx} (niladic)`);
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
    test(`resident argmax on offset window: [5,3] rows 2.. axis=${axis} keepdims=${keepdims}`, () => {
      const baseData = Array.from({ length: 15 }, (_, i) => ((i * 7) % 13) - 6);
      const w = WNDArray.fromArray(core, [5, 3], baseData);
      try {
        const view = w.slice({ start: 2 }); // O(1) view, offset 6, natural strides
        try {
          const ctx = `resident argmax offset window axis=${axis} keepdims=${keepdims}`;
          assertArgmaxMatches(view, axis, keepdims, ctx);
          assertArgmaxNiladicMatches(view, `${ctx} (niladic)`);
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
    test(`resident argmax on composed transpose+slice view: [2,3,4]^T sliced axis=${axis} keepdims=${keepdims}`, () => {
      const baseData = Array.from({ length: 24 }, (_, i) => ((i * 11) % 23) - 11);
      const w = WNDArray.fromArray(core, [2, 3, 4], baseData);
      try {
        const t = w.transpose();
        try {
          const view = t.slice(null, { start: 1 }, null);
          try {
            const ctx = `resident argmax composed view axis=${axis} keepdims=${keepdims}`;
            assertArgmaxMatches(view, axis, keepdims, ctx);
            assertArgmaxNiladicMatches(view, `${ctx} (niladic)`);
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

// --- special-value raster (spec D7): NaN / +-0 / +-Inf / subnormals through
// the same receiver classes. `argmax`'s total order is defined ON these
// values (NaN maximal, ties by first index), so this block is load-bearing
// for M1, not decoration.
{
  const rng = makeRng(0x414d41585f535056n); // "AMAX_SPV"
  const SHAPES: readonly (readonly number[])[] = [[], [1], [7], [2, 3], [3, 4], [2, 3, 4], [2, 2, 2, 2]];
  for (let c = 0; c < 60; c++) {
    const shape = [...(SHAPES[c % SHAPES.length] ?? [3])];
    const data = genDataSpecial(rng, shape);
    const asView = rng.nextBool() && shape.length >= 2;
    const rank = shape.length;
    const useAxis = rank > 0 && rng.nextBool();
    const positiveAxis = rank > 0 ? rng.nextInt(0, rank - 1) : 0;
    const axis = useAxis ? (rng.nextBool() ? positiveAxis - rank : positiveAxis) : undefined;
    const keepdims = rng.nextBool();

    test(`resident argmax special values case ${c}: shape=[${shape.join(",")}] view=${asView} axis=${axis} keepdims=${keepdims}`, () => {
      const ctx = `resident argmax special case ${c} shape=[${shape.join(",")}] view=${asView} axis=${axis} keepdims=${keepdims}`;
      if (!asView) {
        const a = WNDArray.fromArray(core, shape, data);
        try {
          assertArgmaxMatches(a, axis, keepdims, ctx);
          assertArgmaxNiladicMatches(a, `${ctx} (niladic)`);
        } finally {
          a.dispose();
        }
        return;
      }
      // Transposed VIEW whose logical shape/content is exactly shape/data
      // (the involution trick threaded.test.ts/blocked.test.ts use).
      const baseShape = [...shape].reverse();
      const baseData = transposeRuntime(shape, data).data;
      const base = WNDArray.fromArray(core, baseShape, baseData);
      try {
        const view = base.transpose();
        try {
          assertArgmaxMatches(view, axis, keepdims, ctx);
          assertArgmaxNiladicMatches(view, `${ctx} (niladic)`);
        } finally {
          view.dispose();
        }
      } finally {
        base.dispose();
      }
    });
  }
}

/** Reads a raw 64-bit element straight out of a `Float64Array`'s own backing
 * buffer with no intermediate array-literal construction — the
 * payload-preserving read `argmax-topk.test.ts` had to introduce after
 * bisecting a V8 JIT canonicalization quirk in `bitsOf`'s
 * `new Float64Array([x])` round trip. */
function bitsAtBuf(buffer: ArrayBufferLike, byteOffset: number, i: number): bigint {
  return new DataView(buffer, byteOffset + i * 8, 8).getBigUint64(0, true);
}

test("resident argmax: a NON-CANONICAL NaN payload still counts as maximal (M1 special-value edge)", () => {
  // Build the NaN DIRECTLY in the backing buffer, never via an array literal.
  const data = new Float64Array(5);
  data[0] = 1;
  data[2] = 1e308;
  data[3] = Number.POSITIVE_INFINITY;
  data[4] = -7;
  new DataView(data.buffer).setBigUint64(1 * 8, 0x7ff8_0000_cafe_baben, true);
  assert.ok(Number.isNaN(data[1] ?? 0), "precondition: the constructed element must actually be NaN");
  assert.notStrictEqual(bitsAtBuf(data.buffer, 0, 1), 0x7ff8_0000_0000_0000n, "precondition: payload must be non-canonical for this test to mean anything");

  const shape: number[] = [5];
  const w = WNDArray.fromArray(core, shape, data); // Float64Array source -> memcpy, payload preserved
  try {
    // The payload really made it into WASM memory unchanged (not merely into
    // a JS copy) — read straight from the core's linear memory.
    const d = w.describe();
    assert.strictEqual(bitsAtBuf(core.memory.buffer, d.ptr, 1), 0x7ff8_0000_cafe_baben, "the non-canonical payload must survive the copy INTO wasm memory");

    assert.strictEqual(w.argmax(), 1, "a non-canonical NaN must still beat +Infinity (NaN is maximal)");
    assert.strictEqual(w.argmax(), argmaxRuntime(shape, data, undefined).data[0], "cross-surface: same answer as argmaxRuntime");
  } finally {
    w.dispose();
  }
});

// --- rank 0, size-0 output, and the two empty-reduction throws ------------

test("resident argmax: rank 0 (a lone element) answers 0 in every form", () => {
  const shape: number[] = [];
  const w = WNDArray.fromArray(core, shape, [42]);
  try {
    assert.strictEqual(w.argmax(), 0, "niladic");
    const full = w.argmax(undefined);
    try {
      assertShapeEqual([], full.shape as readonly number[], "rank-0 full reduction shape");
      assert.deepStrictEqual(Array.from(full.toArray()), [0], "rank-0 full reduction data");
    } finally {
      full.dispose();
    }
    const kept = w.argmax(undefined, true);
    try {
      assertShapeEqual([], kept.shape as readonly number[], "rank-0 keepdims shape (rank 0 has no axes to keep)");
    } finally {
      kept.dispose();
    }
  } finally {
    w.dispose();
  }
});

test("resident argmax: a size-0 OUTPUT is valid, never a throw ([0,3] along axis 1 -> empty [0])", () => {
  const shape: number[] = [0, 3];
  const w = WNDArray.fromArray(core, shape, []);
  try {
    const got = w.argmax(1);
    try {
      assertShapeEqual([0], got.shape as readonly number[], "size-0 output shape");
      assert.strictEqual(got.toArray().length, 0, "size-0 output must be empty");
      // Cross-surface: the naive surface agrees, shape and emptiness alike.
      const nd = NDArray.fromArray(shape, []).argmax(1);
      assertShapeEqual(nd.shape, got.shape as readonly number[], "size-0 output shape must match NDArray.argmax");
      assert.strictEqual(nd.data.length, 0, "NDArray size-0 output must be empty too");
    } finally {
      got.dispose();
    }
  } finally {
    w.dispose();
  }
});

test("resident argmax: an empty full reduction throws, in BOTH full-reduction forms", () => {
  const shape: number[] = [0];
  const w = WNDArray.fromArray(core, shape, []);
  try {
    assert.throws(() => w.argmax(), /argmax: attempt to get argmax of an empty array/);
    assert.throws(() => w.argmax(undefined), /argmax: attempt to get argmax of an empty array/);
    assert.throws(() => w.argmax(undefined, true), /argmax: attempt to get argmax of an empty array/);
  } finally {
    w.dispose();
  }
});

test("resident argmax: a zero-length AXIS throws (unlike sum/mean, which are well-defined there)", () => {
  const shape: number[] = [2, 0, 3];
  const w = WNDArray.fromArray(core, shape, []);
  try {
    assert.throws(() => w.argmax(1), /argmax: attempt to get argmax of an empty array/);
    assert.throws(() => w.argmax(-2), /argmax: attempt to get argmax of an empty array/);
    // Sanity that this really is argmax-specific: `sum` over the same axis is fine.
    const summed = w.sum(1);
    try {
      assertShapeEqual([2, 3], summed.shape as readonly number[], "sum over the same size-0 axis stays well-defined");
    } finally {
      summed.dispose();
    }
  } finally {
    w.dispose();
  }
});

// --- cross-surface MESSAGE parity, word-for-word (T4, M3) -----------------
// Same technique as the item/stack stem tests above: a plain, non-`const`
// shape variable widens past the compile-time guard so the call actually
// reaches the runtime throw under test.

test("cross-surface message parity: argmax empty-array stem (full reduction), word-for-word (T4, M3)", () => {
  const shape: number[] = [0];
  const nd = NDArray.fromArray(shape, []);
  const wnd = WNDArray.fromArray(core, shape, []);
  try {
    const ndMsg = throwMessage(() => nd.argmax());
    const wndMsg = throwMessage(() => wnd.argmax());
    assert.strictEqual(ndMsg, "argmax: attempt to get argmax of an empty array", "sanity: exact expected wording");
    assert.strictEqual(wndMsg, ndMsg, "argmax empty-array stem must be word-for-word identical across surfaces");
  } finally {
    wnd.dispose();
  }
});

test("cross-surface message parity: argmax empty-array stem (zero-length axis), word-for-word (T4, M3)", () => {
  const shape: number[] = [2, 0, 3];
  const nd = NDArray.fromArray(shape, []);
  const wnd = WNDArray.fromArray(core, shape, []);
  try {
    const ndMsg = throwMessage(() => nd.argmax(1));
    const wndMsg = throwMessage(() => wnd.argmax(1));
    assert.strictEqual(ndMsg, "argmax: attempt to get argmax of an empty array", "sanity: exact expected wording");
    assert.strictEqual(wndMsg, ndMsg, "argmax zero-length-axis stem must be word-for-word identical across surfaces");
  } finally {
    wnd.dispose();
  }
});

test("cross-surface message parity: argmax out-of-range axis stem, word-for-word (T4, M3)", () => {
  const shape: number[] = [2, 3];
  const nd = NDArray.fromArray(shape, [1, 2, 3, 4, 5, 6]);
  const wnd = WNDArray.fromArray(core, shape, [1, 2, 3, 4, 5, 6]);
  try {
    const ndMsg = throwMessage(() => nd.argmax(5));
    const wndMsg = throwMessage(() => wnd.argmax(5));
    assert.strictEqual(ndMsg, "reduce: axis 5 is out of range for shape [2,3] (rank 2)", "sanity: exact expected wording");
    assert.strictEqual(wndMsg, ndMsg, "argmax out-of-range-axis stem must be word-for-word identical across surfaces");
    // The negative-axis normalization is shared too.
    assert.strictEqual(throwMessage(() => wnd.argmax(-3)), throwMessage(() => nd.argmax(-3)), "negative out-of-range axis stem must match too");
  } finally {
    wnd.dispose();
  }
});

// --- D1(3): oracle-FREE cross-surface SHAPE pin ---------------------------
// No `keepDimsShape`, no `argmaxRuntime` — just "the two surfaces agree".
// Blind to a SHARED `keepDimsShape` bug by construction; the structural
// invariants above and the existing `keepdims.test.ts` pins cover that.
{
  const rng = makeRng(0x414d41585f584653n); // "AMAX_XFS"
  for (let c = 0; c < 24; c++) {
    const shape = genShape(rng, 1, 4);
    const data = genData(rng, shape);
    const rank = shape.length;
    const positiveAxis = rng.nextInt(0, rank - 1);
    const axis = rng.nextBool() ? positiveAxis - rank : positiveAxis;
    const keepdims = rng.nextBool();

    test(`cross-surface shape pin (oracle-free) case ${c}: shape=[${shape.join(",")}] axis=${axis} keepdims=${keepdims}`, () => {
      const nd = NDArray.fromArray(shape, data).argmax(axis, keepdims);
      const w = WNDArray.fromArray(core, shape, Array.from(data));
      try {
        const got = w.argmax(axis, keepdims);
        try {
          assert.deepStrictEqual(
            [...(got.shape as readonly number[])],
            [...nd.shape],
            `argmax shape must be identical across surfaces (shape=[${shape.join(",")}] axis=${axis} keepdims=${keepdims})`,
          );
          assert.deepStrictEqual(Array.from(got.toArray()), Array.from(nd.data), "…and so must the indices");
        } finally {
          got.dispose();
        }
      } finally {
        w.dispose();
      }
    });
  }
}

// =============================================================================
// D7 (Verify-Runde addendum, docs/wasm-parity-argmax-spec.md): a call large
// enough to actually trigger `memory.grow` STRICTLY DURING `argmax(0)` —
// not during the preceding `fromArray()`. Baustein B built this case itself
// against the real artifact ([4, 2000000], growth confirmed to happen
// during the argmax(0) call, not before) and measured 0/2,000,000
// deviations: the "never cache memory.buffer or its views" discipline
// (project house rule; shared memory fails SILENTLY on a stale reference —
// wrong length, no detach) held, but there was no committed regression test
// for it, despite `stack` already having one (above, "a large call that
// actually triggers memory.grow mid-loop…"). This is the same regression
// class, re-dimensioned much smaller for speed while preserving the exact
// distinguishing property.
//
// Uses its OWN fresh core (not this file's shared `core`) so the result is
// deterministic regardless of how much the shared core has already grown
// from earlier tests in this file. A disposable "prime" allocation is made
// first to establish a small, page-exact free block that the real input
// then reuses (proving fromArray needs no growth); disposing it does NOT
// shrink `core.memory.buffer` (WASM memory only ever grows), so the freed
// block remains reusable capacity. `argmax(0)`'s own scratch + output
// allocations then need MORE than what's left, forcing a fresh grow. Sizes
// verified empirically against the real artifact (throwaway probe script,
// not committed) to reproduce this before/after signature deterministically
// across repeated runs, with headroom inside the verified working range
// (a prime/input ratio of 1.0005-1.10 all reproduced identically) rather
// than sitting at its edge.
// =============================================================================

test("argmax(axis): a call large enough to trigger memory.grow STRICTLY DURING the call (not during the preceding fromArray) stays correct (D7 Verify-Runde addendum)", async () => {
  const freshCore = await initCore();

  // Prime: allocate-then-free a buffer bigger than the real input, to leave
  // an exactly-reusable free block behind.
  const primeElems = 205000;
  const prime = WNDArray.fromArray(freshCore, [primeElems], new Float64Array(primeElems));
  prime.dispose();

  const rows = 2;
  const cols = 100000;
  const n = rows * cols;
  const inputData = new Float64Array(n);
  for (let i = 0; i < n; i++) inputData[i] = ((i * 37) % 997) - 500; // real varied signed values

  const beforeFrom = freshCore.memory.buffer.byteLength;
  const a = WNDArray.fromArray(freshCore, [rows, cols], inputData);
  const afterFrom = freshCore.memory.buffer.byteLength;
  assert.strictEqual(
    afterFrom,
    beforeFrom,
    `precondition: fromArray() must NOT need to grow memory here (before=${beforeFrom}, after=${afterFrom}) — ` +
      `otherwise this case would not isolate argmax's OWN growth from the receiver's`,
  );

  try {
    const beforeArgmax = freshCore.memory.buffer.byteLength;
    const got = a.argmax(0);
    const afterArgmax = freshCore.memory.buffer.byteLength;
    try {
      assert.notStrictEqual(
        afterArgmax,
        beforeArgmax,
        `precondition: this case must actually trigger memory.grow STRICTLY DURING the argmax() call ` +
          `(before=${beforeArgmax}, after=${afterArgmax}) — otherwise the regression class would not be exercised`,
      );
      const ref = argmaxRuntime([rows, cols], inputData, 0);
      assertShapeEqual(ref.shape, got.shape as readonly number[], "argmax memory.grow case shape");
      assertIndexBitsIdentical(
        ref.data,
        got.toArray(),
        "argmax memory.grow case data — the result must reflect the POST-growth buffer, not a stale/detached reference cached across the grow",
      );
    } finally {
      got.dispose();
    }
  } finally {
    a.dispose();
  }
});

// --- T4 / Arbeitsregel 2: real-tsc diagnostic pin for `WNDArray.argmax` ---
// The house rule for overload sets is to pin the diagnostic's CONTENT, not
// merely the existence of an error — and `argmax` is a three-overload set
// whose guard-carrying candidate must be declared LAST to own the diagnostic.
// The `@ts-expect-error` pins in `spike/tests/ndarray.test-d.ts` prove the
// call is rejected; this proves WHERE (the axis argument's exact column) and
// WITH WHAT WORDING, and that the three well-formed calls beside it resolve
// cleanly. Runs the real compiler on a throwaway fixture OUTSIDE the repo, so
// the deliberately-broken code never joins any type corpus (and costs the
// `check:diag` pins nothing). Mirrors special-values.test.ts's own S1 probe.

import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

test("diagnostic quality (T4, WASM parity S4): an out-of-range literal axis is rejected AT the axis argument, with the shape-naming reduce stem", () => {
  const dir = mkdtempSync(join(tmpdir(), "numtype-argmax-diag-pin-"));
  try {
    const residentPath = fileURLToPath(new URL("../src/wasm/resident.ts", import.meta.url).href);
    const ambientPath = fileURLToPath(new URL("../src/ambient.d.ts", import.meta.url).href);
    const repoRoot = fileURLToPath(new URL("../..", import.meta.url).href);
    // Line 3 is the ONLY bad call. `BAD_LINE`/`BAD_COL` are derived from the
    // fixture text below, never hand-copied from an observed tsc run.
    const badCall = `a.argmax(5); // deliberate: axis 5 is out of range for rank 3`;
    const BAD_LINE = 3;
    const BAD_COL = badCall.indexOf("5") + 1; // 1-based column of the axis argument
    writeFileSync(
      join(dir, "probe.ts"),
      `import type { WNDArray } from ${JSON.stringify(residentPath)};\n` +
        `declare const a: WNDArray<[2, 3, 4]>;\n` +
        `${badCall}\n` +
        `a.argmax(1, true);   // must stay clean\n` +
        `a.argmax();          // niladic -> number, must stay clean\n` +
        `a.argmax(undefined, true); // full reduction with keepdims, must stay clean\n`,
    );
    writeFileSync(
      join(dir, "tsconfig.json"),
      JSON.stringify({
        compilerOptions: {
          strict: true,
          target: "ES2022",
          module: "ESNext",
          moduleResolution: "bundler",
          noEmit: true,
          allowImportingTsExtensions: true,
          skipLibCheck: true,
          noUncheckedIndexedAccess: true,
          exactOptionalPropertyTypes: true,
        },
        include: ["probe.ts", ambientPath],
      }),
    );
    const res = spawnSync("pnpm", ["exec", "tsc", "--noEmit", "-p", dir], { cwd: repoRoot, encoding: "utf8" });
    const out = `${res.stdout ?? ""}\n${res.stderr ?? ""}`;
    assert.notStrictEqual(res.status, 0, `fixture must fail to compile:\n${out}`);
    const probeErrors = out.split("\n").filter((l) => l.includes("probe.ts(") && l.includes("error TS"));
    assert.strictEqual(probeErrors.length, 1, `expected exactly ONE fixture error (the three well-formed argmax calls must resolve cleanly):\n${out}`);
    assert.ok(
      out.includes(`reduce: axis 5 is out of range for shape [2,3,4] (rank 3)`),
      `the shape-naming reduce stem must survive overload resolution on WNDArray.argmax:\n${out}`,
    );
    // The rejection sits AT the axis ARGUMENT, not at the receiver or the call.
    assert.ok(
      (probeErrors[0] ?? "").includes(`probe.ts(${BAD_LINE},${BAD_COL})`),
      `the error must be reported at line ${BAD_LINE}, column ${BAD_COL} (the axis argument itself):\n${probeErrors[0]}`,
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
