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
import { assertDataBitIdentical, assertShapeEqual, wideSpecs } from "./assert-helpers.ts";
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
        const view = w.slice(...wideSpecs({ step: 2 }, null)); // O(1) view, non-natural strides
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
        const view = w.slice(...wideSpecs({ start: 2 })); // O(1) view, offset 6, natural strides
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
          const view = t.slice(...wideSpecs(null, { start: 1 }, null));
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
        const view = w.slice(...wideSpecs({ step: 2 }, null));
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
        const view = w.slice(...wideSpecs({ start: 2 }));
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
          const view = t.slice(...wideSpecs(null, { start: 1 }, null));
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
        const view = w.slice(...wideSpecs({ step: 2 }, null)); // O(1) view, non-natural strides
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
        const view = w.slice(...wideSpecs({ start: 2 })); // O(1) view, offset 6, natural strides
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
          const view = t.slice(...wideSpecs(null, { start: 1 }, null));
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

// =============================================================================
// WASM parity S5 (docs/wasm-parity-topk-spec.md, D7): `WNDArray.topk` vs the
// TypeScript reference `topkRuntime`, bit-for-bit — the M1 proof for the LAST
// slice of the S0-S5 campaign.
//
// Oracle methodology: `topkRuntime` over the RECEIVER's own logical
// shape/data (`toArray()`, read BEFORE the topk call), so a view is compared
// against exactly the vector it logically is. Two things make this slice's
// oracle discipline different from `argmax`'s:
//
//  1. `topk` returns real DATA VALUES, not just indices, so the comparison is
//     over raw 64-bit patterns read straight out of the result buffers —
//     never `===` (which conflates NaN payloads and `+0`/`-0`) and never via
//     `bitsOf`'s `new Float64Array([x])` round trip (which a V8 JIT
//     canonicalization quirk can corrupt for NaN payloads; the same reason
//     `bitsAtBuf` exists above).
//  2. `topk`'s order has THREE tie levels (NaN-vs-NaN by index; equal values
//     by index; `+0`/`-0` treated as equal). Randomized f64 draws from a
//     continuous distribution hit exact ties with probability 0, so they are
//     PROVABLY BLIND to the whole tiebreak-bug class. The constructed
//     tie raster below is therefore mandatory, not decoration — including a
//     boundary group made of several NaNs, which is a DIFFERENT comparator
//     branch (`aNaN && bNaN`) from a numeric tie (`a === b`).
//
// The tests deliberately pin only the OBSERVABLE result. `topk`'s order is a
// strict total order on the distinct indices `0..n-1`, so exactly one answer
// is correct and the kernel's selection ALGORITHM is free (kernel doc,
// crates/core/src/kernels/topk.rs). Pinning heap internals would block a
// future legitimate optimization for no correctness gain.
// =============================================================================

import { topkRuntime } from "../src/runtime.ts";

/** Raw 64-bit read out of a `Float64Array`'s OWN backing buffer, with no
 * intermediate array-literal construction — the payload-preserving read
 * (see `bitsAtBuf` above for the JIT quirk that makes this necessary). */
function bitsAtArr(a: Float64Array, i: number): bigint {
  return new DataView(a.buffer, a.byteOffset, a.byteLength).getBigUint64(i * 8, true);
}

function assertF64BitsIdentical(expected: Float64Array, actual: Float64Array, ctx: string): void {
  assert.strictEqual(actual.length, expected.length, `${ctx}: result length mismatch, expected ${expected.length} got ${actual.length}`);
  for (let i = 0; i < expected.length; i++) {
    const e = bitsAtArr(expected, i);
    const a = bitsAtArr(actual, i);
    assert.strictEqual(a, e, `${ctx}: bits differ at ${i}: reference=${expected[i]} (0x${e.toString(16)}) wasm=${actual[i]} (0x${a.toString(16)})`);
  }
}

/** Runs `recv.topk(k)` and asserts BOTH outputs bit-identical to
 * `topkRuntime` over the receiver's own logical content, plus the
 * `values[i] === data[indices[i]]` cross-consistency the op's own contract
 * claims. Disposes both result handles; the caller owns `recv`. */
function assertTopkMatches(recv: AnyWNDArray, k: number, ctx: string): void {
  const shape = recv.shape as readonly number[];
  const data = recv.toArray(); // logical content, BEFORE the topk call
  const ref = topkRuntime(shape, data, k);
  const got = recv.topk(k as never);
  try {
    assertShapeEqual([k], got.values.shape as readonly number[], `${ctx}: values shape`);
    assertShapeEqual([k], got.indices.shape as readonly number[], `${ctx}: indices shape`);
    const gotValues = got.values.toArray();
    const gotIndices = got.indices.toArray();
    assertF64BitsIdentical(ref.values, gotValues, `${ctx}: values`);
    assertF64BitsIdentical(ref.indices, gotIndices, `${ctx}: indices`);
    // Cross-consistency against the INPUT vector, byte-exact: the returned
    // index really addresses the returned value, in the view's LOGICAL
    // flattening (a memory-offset confusion would break this even when both
    // outputs were internally self-consistent).
    for (let i = 0; i < gotIndices.length; i++) {
      const idx = gotIndices[i] ?? 0;
      assert.ok(Number.isInteger(idx) && idx >= 0 && idx < data.length, `${ctx}: indices[${i}]=${idx} is not a valid logical index into a length-${data.length} vector`);
      assert.strictEqual(bitsAtArr(gotValues, i), bitsAtArr(data, idx), `${ctx}: values[${i}] must be data[indices[${i}]] byte-exactly`);
    }
    // The two result handles are INDEPENDENT buffers, never one aliased twice
    // (D5). Only meaningful for k > 0: at k = 0 both are the `nt_alloc(0)`
    // ptr-0 sentinel — a zero-byte non-allocation that is never read and
    // whose `nt_free(0, 0)` is a no-op, so there is nothing to alias.
    if (k > 0) {
      assert.notStrictEqual(got.values.describe().ptr, got.indices.describe().ptr, `${ctx}: values and indices must not share a buffer`);
    } else {
      assert.strictEqual(got.values.describe().ptr, 0, `${ctx}: a k=0 values handle must carry the ptr-0 sentinel`);
      assert.strictEqual(got.indices.describe().ptr, 0, `${ctx}: a k=0 indices handle must carry the ptr-0 sentinel`);
    }
  } finally {
    got.values.dispose();
    got.indices.dispose();
  }
}

/** The mandated k raster for a length-`n` receiver: 0, 1, n/2, n (deduped,
 * and 1 dropped when the vector is empty). */
function kRaster(n: number): number[] {
  const raw = n === 0 ? [0] : [0, 1, Math.floor(n / 2), n];
  return [...new Set(raw)];
}

// --- topk: contiguous receivers, randomized, over the whole k raster ------
{
  const rng = makeRng(0x5245535f544f504bn); // "RES_TOPK"
  for (let c = 0; c < CASE_COUNT; c++) {
    const n = rng.nextInt(0, 12);
    const shape = [n];
    const data = genData(rng, shape);
    test(`resident topk contiguous case ${c}: n=${n}`, () => {
      const a = WNDArray.fromArray(core, shape, Array.from(data));
      try {
        for (const k of kRaster(n)) {
          assertTopkMatches(a, k, `resident topk contiguous case ${c} n=${n} k=${k}`);
        }
      } finally {
        a.dispose();
      }
    });
  }
}

// --- topk on VIEWS (Arbeitsregel 12 / spec D7: the four view classes) -----
// At rank 1 the interesting receivers are: stride > 1 (a step slice), offset
// > 0 (a window), a ROW of a transposed matrix (stride = the base's row
// count, offset = the row index), and a composed transpose+slice view with
// both. For `topk` this is load-bearing beyond coverage: the returned indices
// must be indices into the VIEW's logical flattening, and a memory-offset
// confusion would corrupt the VALUES too (kernel doc).

{
  // Fixed, hand-chosen data (no PRNG here): each view class is a structural
  // claim, so the receivers must be deterministic and inspectable.
  const BASE = Array.from({ length: 24 }, (_, i) => ((i * 7) % 19) - 9);

  // (a) step slice: stride 2, offset 0
  for (const k of [0, 1, 3, 6]) {
    test(`resident topk on step-sliced view (stride 2, offset 0): k=${k}`, () => {
      const w = WNDArray.fromArray(core, [12], BASE.slice(0, 12));
      try {
        const view = w.slice(...wideSpecs({ step: 2 })); // [6], stride 2
        try {
          assert.deepStrictEqual([...(view.shape as readonly number[])], [6], "precondition: the step slice is a length-6 rank-1 view");
          assert.notStrictEqual(view.describe().strides[0], 1, "precondition: the view must be genuinely non-contiguous (stride != 1)");
          assertTopkMatches(view, k, `resident topk step-sliced view k=${k}`);
        } finally {
          view.dispose();
        }
      } finally {
        w.dispose();
      }
    });
  }

  // (b) offset window: stride 1, offset > 0
  for (const k of [0, 1, 4, 8]) {
    test(`resident topk on offset window (stride 1, offset 4): k=${k}`, () => {
      const w = WNDArray.fromArray(core, [12], BASE.slice(0, 12));
      try {
        const view = w.slice(...wideSpecs({ start: 4 })); // [8], offset 4
        try {
          assert.deepStrictEqual([...(view.shape as readonly number[])], [8], "precondition: the window is a length-8 rank-1 view");
          assert.notStrictEqual(view.describe().offset, 0, "precondition: the window must have a nonzero offset");
          assertTopkMatches(view, k, `resident topk offset window k=${k}`);
        } finally {
          view.dispose();
        }
      } finally {
        w.dispose();
      }
    });
  }

  // (c) a ROW of a TRANSPOSED matrix: stride = the base's column count of the
  // pre-transpose layout, offset = the row index. Both non-trivial at once.
  for (const row of [0, 2, 5]) {
    for (const k of [0, 2, 4]) {
      test(`resident topk on a transposed matrix row (row ${row}): k=${k}`, () => {
        const w = WNDArray.fromArray(core, [4, 6], BASE);
        try {
          const t = w.transpose(); // [6,4], strides [1,6]
          try {
            const view = t.slice(...wideSpecs(row, null)) as AnyWNDArray; // rank-1, stride 6, offset `row`
            try {
              assert.deepStrictEqual([...(view.shape as readonly number[])], [4], "precondition: a transposed row is a length-4 rank-1 view");
              assert.strictEqual(view.describe().strides[0], 6, "precondition: the row's stride must be the base's row length");
              assertTopkMatches(view, k, `resident topk transposed row ${row} k=${k}`);
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

  // (d) composed: transpose -> row -> step slice. Non-natural stride AND a
  // nonzero offset AND a second slicing step on top.
  for (const k of [0, 1, 2]) {
    test(`resident topk on a composed transpose+row+step view: k=${k}`, () => {
      const w = WNDArray.fromArray(core, [4, 6], BASE);
      try {
        const t = w.transpose(); // [6,4]
        try {
          const row = t.slice(...wideSpecs(3, null)) as AnyWNDArray; // [4], stride 6, offset 3
          try {
            const view = row.slice(...wideSpecs({ step: 2 })) as AnyWNDArray; // [2], stride 12, offset 3
            try {
              assert.deepStrictEqual([...(view.shape as readonly number[])], [2], "precondition: the composed view is length 2");
              assert.strictEqual(view.describe().strides[0], 12, "precondition: composed stride");
              assert.strictEqual(view.describe().offset, 3, "precondition: composed offset");
              assertTopkMatches(view, k, `resident topk composed view k=${k}`);
            } finally {
              view.dispose();
            }
          } finally {
            row.dispose();
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

// --- topk: randomized IEEE special-value raster, contiguous AND view ------
// `topk`'s order is DEFINED on these values, so this block is load-bearing
// for M1. Still not a substitute for the constructed tie raster below:
// `genDataSpecial` draws ties only by luck, never by construction.
{
  const rng = makeRng(0x544f504b5f535056n); // "TOPK_SPV"
  for (let c = 0; c < 60; c++) {
    const n = rng.nextInt(1, 10);
    const data = genDataSpecial(rng, [n]);
    const asView = rng.nextBool();
    const k = rng.nextInt(0, n);
    test(`resident topk special values case ${c}: n=${n} k=${k} view=${asView}`, () => {
      const ctx = `resident topk special case ${c} n=${n} k=${k} view=${asView}`;
      if (!asView) {
        const a = WNDArray.fromArray(core, [n], data);
        try {
          assertTopkMatches(a, k, ctx);
        } finally {
          a.dispose();
        }
        return;
      }
      // Embed the vector as every 3rd element of a longer buffer, so the
      // receiver is a genuine stride-3 view with the SAME logical content.
      const padded = new Float64Array(n * 3);
      for (let i = 0; i < n * 3; i++) padded[i] = 12345.5;
      for (let i = 0; i < n; i++) padded[i * 3] = data[i] ?? 0;
      const base = WNDArray.fromArray(core, [n * 3], padded);
      try {
        const view = base.slice(...wideSpecs({ step: 3 })) as AnyWNDArray;
        try {
          assertTopkMatches(view, k, ctx);
        } finally {
          view.dispose();
        }
      } finally {
        base.dispose();
      }
    });
  }
}

// --- topk: CONSTRUCTED tie raster (spec D7, the mandatory part) -----------
// Every case here puts the k boundary INSIDE a tie group, where nothing but
// the index tiebreak decides the answer. `expectedIndices` is a HAND
// reference, not `topkRuntime`'s output — so these cases would catch a bug
// shared by both surfaces, which the differential alone cannot.
{
  const NAN = Number.NaN;
  const TIE_CASES: readonly { name: string; data: number[]; k: number; expectedIndices: number[] }[] = [
    // Numeric tie group, boundary in the middle of it.
    { name: "all-equal, k splits the group", data: [5, 5, 5, 5, 5], k: 3, expectedIndices: [0, 1, 2] },
    { name: "numeric tie group under a unique max", data: [9, 5, 5, 5, 1], k: 3, expectedIndices: [0, 1, 2] },
    { name: "numeric tie group straddling the boundary", data: [1, 7, 2, 7, 7, 0], k: 2, expectedIndices: [1, 3] },
    // NaN tie group — a DIFFERENT comparator branch (aNaN && bNaN).
    { name: "all-NaN, k splits the NaN group", data: [NAN, NAN, NAN, NAN], k: 2, expectedIndices: [0, 1] },
    { name: "several NaNs ahead of everything, boundary inside the NaN group", data: [1, NAN, 2, NAN, 3, NAN], k: 2, expectedIndices: [1, 3] },
    { name: "NaN group exactly filled, then the largest real value", data: [NAN, 7, 7, NAN, 7], k: 3, expectedIndices: [0, 3, 1] },
    // +0 / -0 are EQUAL under plain >/<, so only the index decides — and the
    // returned VALUES keep their own signs.
    { name: "mixed +0/-0 tie group, boundary inside it", data: [-0, 0, -0, 0, -1], k: 2, expectedIndices: [0, 1] },
    { name: "+0/-0 tie above a negative, whole group kept", data: [0, -0, -0, -5], k: 3, expectedIndices: [0, 1, 2] },
    // All three tie levels in one vector.
    { name: "NaN + numeric + signed-zero ties together", data: [3, NAN, 3, -0, 0, NAN, 3], k: 5, expectedIndices: [1, 5, 0, 2, 6] },
    // Infinities alongside ties.
    { name: "+Inf duplicated, boundary inside the +Inf group", data: [Infinity, 1, Infinity, Infinity], k: 2, expectedIndices: [0, 2] },
    { name: "-Inf duplicated at the bottom, k = n", data: [-Infinity, 2, -Infinity], k: 3, expectedIndices: [1, 0, 2] },
  ];

  for (const tc of TIE_CASES) {
    test(`resident topk constructed tie raster: ${tc.name} (k=${tc.k})`, () => {
      const shape = [tc.data.length];
      const a = WNDArray.fromArray(core, shape, tc.data);
      try {
        // (1) hand reference — catches a bug SHARED by both surfaces.
        const got = a.topk(tc.k as never);
        try {
          assert.deepStrictEqual(Array.from(got.indices.toArray()), tc.expectedIndices, `${tc.name}: indices must match the hand reference`);
          const gotValues = got.values.toArray();
          for (let i = 0; i < tc.expectedIndices.length; i++) {
            const src = tc.expectedIndices[i] ?? 0;
            const expectedBits = bitsAtArr(Float64Array.from(tc.data), src);
            assert.strictEqual(bitsAtArr(gotValues, i), expectedBits, `${tc.name}: values[${i}] must be data[${src}] byte-exactly (sign of zero and NaN payload included)`);
          }
        } finally {
          got.values.dispose();
          got.indices.dispose();
        }
        // (2) …and the full differential against topkRuntime, over the whole
        // k raster, so the boundary is crossed from both sides too.
        for (const k of kRaster(tc.data.length)) {
          assertTopkMatches(a, k, `tie raster ${tc.name} k=${k}`);
        }
      } finally {
        a.dispose();
      }
    });
  }
}

// --- topk: NON-CANONICAL NaN PAYLOAD, byte-exact (the M1 risk point) ------
// `topk` is the first op of the campaign that returns real DATA VALUES rather
// than indices, so this is where the NaN-payload caveat actually bites: an
// f64 load/store through the kernel must preserve a payload (only arithmetic
// may canonicalize), and `values[i] = data[indices[i]]` must stay a plain
// element copy. Its own named test, never a side effect of another.
test("resident topk: an EXACT non-canonical NaN payload survives byte-identically through the kernel (M1 risk point, spec T3)", () => {
  const data = new Float64Array(6);
  data[0] = 1;
  data[2] = 1e308;
  data[3] = Number.POSITIVE_INFINITY;
  data[5] = -7;
  // Two DIFFERENT non-canonical payloads, so the test also proves they are
  // not conflated with each other or with the canonical NaN.
  new DataView(data.buffer).setBigUint64(1 * 8, 0x7ff8_0000_cafe_baben, true);
  new DataView(data.buffer).setBigUint64(4 * 8, 0x7ff0_0000_dead_beefn, true);
  assert.ok(Number.isNaN(data[1] ?? 0) && Number.isNaN(data[4] ?? 0), "precondition: both constructed elements must actually be NaN");
  assert.notStrictEqual(bitsAtArr(data, 1), 0x7ff8_0000_0000_0000n, "precondition: payload 1 must be non-canonical");
  assert.notStrictEqual(bitsAtArr(data, 4), 0x7ff8_0000_0000_0000n, "precondition: payload 2 must be non-canonical");

  const shape: number[] = [6];
  const w = WNDArray.fromArray(core, shape, data); // Float64Array source -> memcpy
  try {
    // The payloads really made it into WASM memory unchanged.
    const d = w.describe();
    assert.strictEqual(bitsAtBuf(core.memory.buffer, d.ptr, 1), 0x7ff8_0000_cafe_baben, "payload 1 must survive the copy INTO wasm memory");
    assert.strictEqual(bitsAtBuf(core.memory.buffer, d.ptr, 4), 0x7ff0_0000_dead_beefn, "payload 2 must survive the copy INTO wasm memory");

    const got = w.topk(6 as never);
    try {
      const values = got.values.toArray();
      const indices = got.indices.toArray();
      // NaNs sort first, among themselves by ascending index.
      assert.deepStrictEqual(Array.from(indices), [1, 4, 3, 2, 0, 5], "NaNs first (by ascending index), then +Inf, 1e308, 1, -7");
      assert.strictEqual(bitsAtArr(values, 0), 0x7ff8_0000_cafe_baben, "values[0] must carry payload 1 EXACTLY — no canonicalization, no substitution");
      assert.strictEqual(bitsAtArr(values, 1), 0x7ff0_0000_dead_beefn, "values[1] must carry payload 2 EXACTLY");
      // …and the TS reference agrees bit-for-bit, which is the M1 claim.
      const ref = topkRuntime(shape, data, 6);
      assertF64BitsIdentical(ref.values, values, "NaN-payload case: values vs topkRuntime");
      assertF64BitsIdentical(ref.indices, indices, "NaN-payload case: indices vs topkRuntime");
    } finally {
      got.values.dispose();
      got.indices.dispose();
    }
  } finally {
    w.dispose();
  }
});

test("resident topk: a non-canonical NaN payload survives through a STRIDED view too (the kernel's own read path)", () => {
  // Payload sits at a nonzero offset behind a stride, i.e. exactly the read
  // `data[offset + i * strides[0]]` the kernel performs.
  const padded = new Float64Array(9);
  for (let i = 0; i < 9; i++) padded[i] = i;
  new DataView(padded.buffer).setBigUint64(7 * 8, 0x7ff8_0000_0bad_f00dn, true);
  const shape: number[] = [9];
  const w = WNDArray.fromArray(core, shape, padded);
  try {
    const view = w.slice({ start: 1, step: 3 }) as AnyWNDArray; // logical [1, 4, 7]
    try {
      assert.deepStrictEqual([...(view.shape as readonly number[])], [3], "precondition: length-3 strided view");
      const got = view.topk(3 as never);
      try {
        assert.deepStrictEqual(Array.from(got.indices.toArray()), [2, 1, 0], "the NaN (logical index 2) first, then 4, then 1");
        assert.strictEqual(bitsAtArr(got.values.toArray(), 0), 0x7ff8_0000_0bad_f00dn, "the payload must survive the STRIDED read path byte-exactly");
      } finally {
        got.values.dispose();
        got.indices.dispose();
      }
    } finally {
      view.dispose();
    }
  } finally {
    w.dispose();
  }
});

// --- topk: k = 0 / k = n / n = 0 edges ------------------------------------

test("resident topk: k = 0 yields two valid, empty [0] handles — never a throw, even on an empty vector", () => {
  const shape: number[] = [4];
  const w = WNDArray.fromArray(core, shape, [3, 1, 4, 1]);
  try {
    const got = w.topk(0 as never);
    try {
      assertShapeEqual([0], got.values.shape as readonly number[], "k=0 values shape");
      assertShapeEqual([0], got.indices.shape as readonly number[], "k=0 indices shape");
      assert.strictEqual(got.values.toArray().length, 0);
      assert.strictEqual(got.indices.toArray().length, 0);
      assert.strictEqual(got.values.disposed, false, "a k=0 handle is a live handle, not a disposed one");
    } finally {
      got.values.dispose();
      got.indices.dispose();
    }
  } finally {
    w.dispose();
  }

  const emptyShape: number[] = [0];
  const e = WNDArray.fromArray(core, emptyShape, []);
  try {
    const got = e.topk(0 as never);
    try {
      assertShapeEqual([0], got.values.shape as readonly number[], "empty receiver, k=0 values shape");
      assertShapeEqual([0], got.indices.shape as readonly number[], "empty receiver, k=0 indices shape");
    } finally {
      got.values.dispose();
      got.indices.dispose();
    }
  } finally {
    e.dispose();
  }
});

test("resident topk: k = n returns the WHOLE vector in sorted order, bit-identical to topkRuntime", () => {
  const shape: number[] = [7];
  const data = [2, -5, 9, 0, 9, -5, 3];
  const w = WNDArray.fromArray(core, shape, data);
  try {
    assertTopkMatches(w, 7, "k = n");
    const got = w.topk(7 as never);
    try {
      assert.deepStrictEqual(Array.from(got.values.toArray()), [9, 9, 3, 2, 0, -5, -5], "k=n values, descending");
      assert.deepStrictEqual(Array.from(got.indices.toArray()), [2, 4, 6, 0, 3, 1, 5], "k=n indices, ties by ascending index");
    } finally {
      got.values.dispose();
      got.indices.dispose();
    }
  } finally {
    w.dispose();
  }
});

// --- topk: the two result handles are genuinely independent ---------------

test("resident topk: values and indices are INDEPENDENT resident buffers — disposing one leaves the other fully usable", () => {
  const shape: number[] = [5];
  const w = WNDArray.fromArray(core, shape, [1, 5, 3, 2, 4]);
  const got = w.topk(3 as never);
  w.dispose(); // the receiver goes first: both results are fresh, not views
  assert.strictEqual(got.values.disposed, false);
  assert.strictEqual(got.indices.disposed, false);
  got.values.dispose();
  assert.strictEqual(got.indices.disposed, false, "disposing values must not dispose indices");
  assert.deepStrictEqual(Array.from(got.indices.toArray()), [1, 4, 2], "indices stay readable after values is disposed");
  got.indices.dispose();
});

// --- cross-surface MESSAGE parity, word-for-word (T4, M3) -----------------
// Same technique as the argmax/item/stack stem tests above: a plain,
// non-`const` shape variable widens past the compile-time guard so the call
// actually reaches the runtime throw under test.

test("cross-surface message parity: topk rank stem, word-for-word (T4, M3)", () => {
  const shape: number[] = [2, 3];
  const nd = NDArray.fromArray(shape, [1, 2, 3, 4, 5, 6]);
  const wnd = WNDArray.fromArray(core, shape, [1, 2, 3, 4, 5, 6]);
  try {
    const ndMsg = throwMessage(() => nd.topk(2 as never));
    const wndMsg = throwMessage(() => wnd.topk(2 as never));
    assert.strictEqual(ndMsg, "topk: expected a 1-D vector (got shape [2,3])", "sanity: exact expected wording");
    assert.strictEqual(wndMsg, ndMsg, "topk rank stem must be word-for-word identical across surfaces");
  } finally {
    wnd.dispose();
  }
});

test("cross-surface message parity: topk invalid-k stems (negative AND dot-form), word-for-word (T4, M3)", () => {
  const shape: number[] = [3];
  const nd = NDArray.fromArray(shape, [1, 2, 3]);
  const wnd = WNDArray.fromArray(core, shape, [1, 2, 3]);
  try {
    assert.strictEqual(throwMessage(() => nd.topk(-1 as never)), "topk: k must be a non-negative integer (got -1)", "sanity: exact expected wording");
    assert.strictEqual(throwMessage(() => wnd.topk(-1 as never)), throwMessage(() => nd.topk(-1 as never)), "negative-k stem must match across surfaces");
    assert.strictEqual(throwMessage(() => nd.topk(1.5 as never)), "topk: k must be a non-negative integer (got 1.5)", "sanity: exact expected wording");
    assert.strictEqual(throwMessage(() => wnd.topk(1.5 as never)), throwMessage(() => nd.topk(1.5 as never)), "dot-form-k stem must match across surfaces");
    assert.strictEqual(throwMessage(() => wnd.topk(Number.NaN as never)), throwMessage(() => nd.topk(Number.NaN as never)), "NaN-k stem must match across surfaces");
    // Verify-round finding (S5): `k = Infinity` is only reachable dynamically
    // (`TopkCheck` rejects it statically), but `Number.isInteger(Infinity)`
    // is `false`, so it must hit the SAME invalid-k stem as NaN/dot-form/
    // negative — never fall through to the bounds check.
    assert.strictEqual(throwMessage(() => nd.topk(Infinity as never)), "topk: k must be a non-negative integer (got Infinity)", "sanity: exact expected wording");
    assert.strictEqual(throwMessage(() => wnd.topk(Infinity as never)), throwMessage(() => nd.topk(Infinity as never)), "Infinity-k stem must match across surfaces");
  } finally {
    wnd.dispose();
  }
});

test("cross-surface message parity: topk bounds stem, word-for-word (T4, M3)", () => {
  const shape: number[] = [3];
  const nd = NDArray.fromArray(shape, [1, 2, 3]);
  const wnd = WNDArray.fromArray(core, shape, [1, 2, 3]);
  try {
    const ndMsg = throwMessage(() => nd.topk(4 as never));
    const wndMsg = throwMessage(() => wnd.topk(4 as never));
    assert.strictEqual(ndMsg, "topk: k=4 exceeds the vector length 3", "sanity: exact expected wording");
    assert.strictEqual(wndMsg, ndMsg, "topk bounds stem must be word-for-word identical across surfaces");
  } finally {
    wnd.dispose();
  }
});

test("resident topk: the VALIDATION ORDER is pinned — a call violating two conditions at once reports the SAME one on both surfaces (D4)", () => {
  // rank AND k are both wrong: rank is checked first, so the rank stem wins.
  const shape2d: number[] = [2, 3];
  const nd2 = NDArray.fromArray(shape2d, [1, 2, 3, 4, 5, 6]);
  const w2 = WNDArray.fromArray(core, shape2d, [1, 2, 3, 4, 5, 6]);
  try {
    const wndMsg = throwMessage(() => w2.topk(-1 as never));
    assert.strictEqual(wndMsg, "topk: expected a 1-D vector (got shape [2,3])", "rank is checked BEFORE k's own validity");
    assert.strictEqual(wndMsg, throwMessage(() => nd2.topk(-1 as never)), "…and both surfaces agree on which one wins");
    const wndMsg2 = throwMessage(() => w2.topk(99 as never));
    assert.strictEqual(wndMsg2, "topk: expected a 1-D vector (got shape [2,3])", "rank is checked BEFORE the bounds check too");
    assert.strictEqual(wndMsg2, throwMessage(() => nd2.topk(99 as never)), "…and both surfaces agree");
  } finally {
    w2.dispose();
  }

  // k's own validity AND the bounds are both violated (-1 on a length-3
  // vector is negative, and would also be "out of bounds" if compared
  // numerically): the invalid-k stem wins, never the bounds stem.
  const shape1d: number[] = [3];
  const nd1 = NDArray.fromArray(shape1d, [1, 2, 3]);
  const w1 = WNDArray.fromArray(core, shape1d, [1, 2, 3]);
  try {
    const msg = throwMessage(() => w1.topk(3.5 as never));
    assert.strictEqual(msg, "topk: k must be a non-negative integer (got 3.5)", "k's own validity is checked BEFORE the length bound (3.5 > 3 would also be out of bounds)");
    assert.strictEqual(msg, throwMessage(() => nd1.topk(3.5 as never)), "…and both surfaces agree");
  } finally {
    w1.dispose();
  }
});

// --- oracle-FREE cross-surface pin ----------------------------------------
// No `topkRuntime` anywhere: just "the two surfaces agree", shapes included.
{
  const rng = makeRng(0x544f504b5f584653n); // "TOPK_XFS"
  for (let c = 0; c < 24; c++) {
    const n = rng.nextInt(1, 10);
    const data = genData(rng, [n]);
    const k = rng.nextInt(0, n);
    test(`topk cross-surface pin (oracle-free) case ${c}: n=${n} k=${k}`, () => {
      const nd = NDArray.fromArray([n], data).topk(k as never);
      const w = WNDArray.fromArray(core, [n], Array.from(data));
      try {
        const got = w.topk(k as never);
        try {
          assert.deepStrictEqual([...(got.values.shape as readonly number[])], [...nd.values.shape], "values shape must be identical across surfaces");
          assert.deepStrictEqual([...(got.indices.shape as readonly number[])], [...nd.indices.shape], "indices shape must be identical across surfaces");
          assertF64BitsIdentical(nd.values.data, got.values.toArray(), `cross-surface values case ${c}`);
          assertF64BitsIdentical(nd.indices.data, got.indices.toArray(), `cross-surface indices case ${c}`);
        } finally {
          got.values.dispose();
          got.indices.dispose();
        }
      } finally {
        w.dispose();
      }
    });
  }
}

// --- memory.grow DURING the call (the "never cache memory.buffer" rule) ---
// Same regression class `stack` and `argmax` already pin above: `topk`
// allocates FOUR buffers per call (two scratch + two outputs), any of which
// can trigger a grow, so a stale cached view would be caught here.
test("topk: a call large enough to trigger memory.grow STRICTLY DURING the call (not during the preceding fromArray) stays correct", async () => {
  const freshCore = await initCore();

  // Prime: allocate-then-free a block bigger than the real input, so the
  // input itself needs no growth (WASM memory never shrinks, so the freed
  // block stays reusable capacity).
  const primeElems = 205000;
  const prime = WNDArray.fromArray(freshCore, [primeElems], new Float64Array(primeElems));
  prime.dispose();

  const n = 200000;
  const inputData = new Float64Array(n);
  for (let i = 0; i < n; i++) inputData[i] = ((i * 37) % 997) - 500;

  const beforeFrom = freshCore.memory.buffer.byteLength;
  const shape: number[] = [n];
  const a = WNDArray.fromArray(freshCore, shape, inputData);
  const afterFrom = freshCore.memory.buffer.byteLength;
  assert.strictEqual(afterFrom, beforeFrom, `precondition: fromArray() must NOT need to grow memory here (before=${beforeFrom}, after=${afterFrom})`);

  try {
    const beforeTopk = freshCore.memory.buffer.byteLength;
    const got = a.topk(50000 as never);
    const afterTopk = freshCore.memory.buffer.byteLength;
    try {
      assert.notStrictEqual(afterTopk, beforeTopk, `precondition: this case must actually trigger memory.grow STRICTLY DURING the topk() call (before=${beforeTopk}, after=${afterTopk})`);
      const ref = topkRuntime(shape, inputData, 50000);
      assertF64BitsIdentical(ref.values, got.values.toArray(), "topk memory.grow case values — the result must reflect the POST-growth buffer");
      assertF64BitsIdentical(ref.indices, got.indices.toArray(), "topk memory.grow case indices — the result must reflect the POST-growth buffer");
    } finally {
      got.values.dispose();
      got.indices.dispose();
    }
  } finally {
    a.dispose();
  }
});

// --- T4 / Arbeitsregel 2: real-tsc diagnostic pin for `WNDArray.topk` -----
// `topk` has a SINGLE signature (no overload set), so Arbeitsregel 2's
// "declare the guard carrier last" does not apply — but the house rule to pin
// the diagnostic's CONTENT and POSITION does. The `@ts-expect-error` pins in
// `spike/tests/ndarray.test-d.ts` prove the calls are rejected; this proves
// WHERE (the `k` argument's exact column, including for a RECEIVER-rank
// problem, the DotCheck precedent) and WITH WHAT WORDING. Runs the real
// compiler on a throwaway fixture OUTSIDE the repo, so the deliberately
// broken code never joins any type corpus.
test("diagnostic quality (T4, WASM parity S5): topk's three error classes are all rejected AT the k argument, with the pinned stems", () => {
  const dir = mkdtempSync(join(tmpdir(), "numtype-topk-diag-pin-"));
  try {
    const residentPath = fileURLToPath(new URL("../src/wasm/resident.ts", import.meta.url).href);
    const ambientPath = fileURLToPath(new URL("../src/ambient.d.ts", import.meta.url).href);
    const repoRoot = fileURLToPath(new URL("../..", import.meta.url).href);
    // Three bad calls, one per error class. Columns are DERIVED from the
    // fixture text below, never hand-copied from an observed tsc run.
    const badBounds = `v.topk(6);    // deliberate: k=6 exceeds the vector length 5`;
    const badNegative = `v.topk(-1);  // deliberate: k must be non-negative`;
    const badRank = `m.topk(2);    // deliberate: rank-2 receiver, reported AT the k argument`;
    const BOUNDS_LINE = 4;
    const NEGATIVE_LINE = 5;
    const RANK_LINE = 6;
    const BOUNDS_COL = badBounds.indexOf("6") + 1;
    const NEGATIVE_COL = badNegative.indexOf("-1") + 1;
    const RANK_COL = badRank.indexOf("2") + 1;
    writeFileSync(
      join(dir, "probe.ts"),
      `import type { WNDArray } from ${JSON.stringify(residentPath)};\n` +
        `declare const v: WNDArray<[5]>;\n` +
        `declare const m: WNDArray<[2, 3]>;\n` +
        `${badBounds}\n` +
        `${badNegative}\n` +
        `${badRank}\n` +
        `v.topk(3);   // must stay clean\n` +
        `v.topk(0);   // k=0 boundary, must stay clean\n` +
        `v.topk(5);   // k=D boundary, must stay clean\n`,
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
    assert.strictEqual(probeErrors.length, 3, `expected exactly THREE fixture errors (the three well-formed topk calls must resolve cleanly):\n${out}`);
    assert.ok(out.includes(`topk: k=6 exceeds the vector length 5`), `the bounds stem must survive to the diagnostic:\n${out}`);
    assert.ok(out.includes(`topk: k must be a non-negative integer (got -1)`), `the invalid-k stem must survive to the diagnostic:\n${out}`);
    assert.ok(out.includes(`topk: expected a 1-D vector (got shape [2,3])`), `the rank stem must survive to the diagnostic:\n${out}`);
    // Every rejection sits AT the `k` ARGUMENT — including the rank one,
    // whose actual problem is with the RECEIVER (DotCheck precedent).
    for (const [line, col, what] of [
      [BOUNDS_LINE, BOUNDS_COL, "bounds"],
      [NEGATIVE_LINE, NEGATIVE_COL, "negative k"],
      [RANK_LINE, RANK_COL, "receiver rank"],
    ] as const) {
      assert.ok(
        probeErrors.some((l) => l.includes(`probe.ts(${line},${col})`)),
        `the ${what} error must be reported at line ${line}, column ${col} (the k argument itself):\n${probeErrors.join("\n")}`,
      );
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// --- wideSpecs: the shared widening helper itself (Verify-B finding) -------
// `wideSpecs` is pure plumbing — it exists only to make `Specs` infer as an
// array instead of a tuple — but 32 call sites now route through it, and a
// bug INSIDE it is invisible at 22 of them: wherever the naive reference and
// the resident candidate both build their view through the same helper with
// the same arguments, a helper bug cancels out symmetrically and both sides
// stay bit-identical while having silently lost the slicing. Measured: with
// `return []` (drop every spec) only 23 cases across ~10 sites fail. These
// two assertions close all 22 blind sites at once, so the helper can never
// be the silent failure.
test("wideSpecs returns its arguments unchanged, in order", () => {
  const specs = wideSpecs(1, null, { start: 1, stop: 7, step: 3 }, { step: 2 });
  assert.deepStrictEqual([...specs], [1, null, { start: 1, stop: 7, step: 3 }, { step: 2 }],
    "wideSpecs must be an identity on its spec list — order, count and each spec's own shape");
});

test("wideSpecs on an empty spec list stays empty", () => {
  assert.deepStrictEqual([...wideSpecs()], [], "wideSpecs() must not invent a spec");
});
