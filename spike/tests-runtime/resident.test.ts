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
import { elementwiseBinary, matmulRuntime, sumRuntime, transposeRuntime } from "../src/runtime.ts";
import { initCore } from "../src/wasm/loader.ts";
import { WNDArray } from "../src/wasm/resident.ts";
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
