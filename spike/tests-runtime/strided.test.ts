/**
 * Kern 03 differential tests: ops on strided VIEWS (transpose without
 * materialization) must be bit-identical to the naive TS reference computed
 * on the materialized equivalent. The existing resident suite already
 * regression-covers the strided entry points under contiguous metadata
 * (resident.ts routes everything through them); THIS file is what proves
 * the view semantics — permuted strides, shared buffers, logical-order
 * accumulation — against `runtime.ts`.
 *
 * View construction trick used throughout: to get a resident view whose
 * LOGICAL content equals `refData` under `shape`, build a base of the
 * reversed shape holding the TRANSPOSED data, then `.transpose()` it —
 * transposition is an involution, so the view reads back as `refData`.
 * (That involution itself is pinned by the `toArray`/round-trip group
 * below, so the op tests don't rest on an unverified assumption.)
 *
 * Same seeded-PRNG methodology as the other suites; wired into
 * `pnpm test:resident` (explicit file list in package.json — the known
 * footgun; see CLAUDE.md).
 */
import assert from "node:assert";
import { test } from "node:test";
import { NDArray } from "../src/ndarray.ts";
import { computeStrides, elementwiseBinary, matmulRuntime, sumRuntime, transposeRuntime } from "../src/runtime.ts";
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

interface Operand {
  readonly arr: AnyWNDArray;
  /** Every handle that must be disposed when the case is done (for a view:
   * the base AND the view). */
  readonly owners: readonly AnyWNDArray[];
}

/** A contiguous resident array holding `refData` under `shape`. */
function makeContiguous(shape: readonly number[], refData: Float64Array): Operand {
  const arr = WNDArray.fromArray(core, shape, refData);
  return { arr, owners: [arr] };
}

/** A transpose VIEW whose logical shape is `shape` and logical content is
 * `refData` (see file header for the involution trick). */
function makeView(shape: readonly number[], refData: Float64Array): Operand {
  const baseShape = [...shape].reverse();
  const baseData = transposeRuntime(shape, refData).data;
  const base = WNDArray.fromArray(core, baseShape, baseData);
  const view = base.transpose();
  return { arr: view, owners: [base, view] };
}

function makeOperand(asView: boolean, shape: readonly number[], refData: Float64Array): Operand {
  return asView ? makeView(shape, refData) : makeContiguous(shape, refData);
}

function disposeAll(...operands: Operand[]): void {
  for (const op of operands) for (const h of op.owners) h.dispose();
}

// --- toArray / round-trip: a view reads back its logical content ----------
// This group pins the involution the other groups' view construction relies
// on, plus the strides metadata itself.
{
  const rng = makeRng(0x5354525f544f4152n); // "STR_TOAR"
  for (let c = 0; c < 60; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);

    test(`strided toArray case ${c}: view of shape=[${shape.join(",")}] reads back logically`, () => {
      const v = makeView(shape, data);
      try {
        const ctx = `strided toArray case ${c} shape=[${shape.join(",")}]`;
        // D-V2.3 fallout: `v.arr: AnyWNDArray = WNDArray<any>`, so `.shape` is
        // `Readonly<any>` — doesn't structurally collapse to `any` (TS quirk),
        // so no longer matches `readonly number[]` (TS2740). Cast only;
        // runtime value unaffected (see demo.ts's `assertResidentAgrees` for
        // the full explanation).
        assertShapeEqual(shape, v.arr.shape as readonly number[], ctx);
        assertDataBitIdentical(data, v.arr.toArray(), ctx);
        // strides metadata: reversed natural strides of the reversed shape
        const baseShape = [...shape].reverse();
        const expectedStrides = computeStrides(baseShape).reverse();
        assert.deepStrictEqual([...v.arr.strides], expectedStrides, `${ctx}: view strides`);
      } finally {
        disposeAll(v);
      }
    });
  }
}

test("strides observability: fresh arrays are natural row-major, views are permuted", () => {
  const a = WNDArray.fromArray(core, [2, 3, 4], new Array(24).fill(0));
  const t = a.transpose();
  const tt = t.transpose();
  try {
    assert.deepStrictEqual([...a.strides], [12, 4, 1]);
    assert.deepStrictEqual([...t.strides], [1, 4, 12]);
    // double transpose: contiguous-strided view again, same metadata as base
    assert.deepStrictEqual([...tt.shape], [2, 3, 4]);
    assert.deepStrictEqual([...tt.strides], [12, 4, 1]);
  } finally {
    tt.dispose();
    t.dispose();
    a.dispose();
  }
});

// B3 (verify-round closure, docs/phase-d-vorarbeiten-v2-ergebnisse.md,
// 2026-07-13): the sibling naive-backend group. D-V2.1 harmonized
// `NDArray.strides()` (a method) into a `get strides()` GETTER — the WNDArray
// group above pins the resident FIELD's values, but nothing in the existing
// corpus pinned the actual VALUES the new `NDArray` getter returns (the
// spec's own fallout section only checked "0 call sites break", never the
// getter's output). Expected values below are HARD LITERALS, independently
// hand-computed from the row-major stride formula (each axis's stride is the
// product of every dim strictly to its right; a rank-1 shape's sole axis has
// nothing to its right, so its stride is always 1) — NOT derived by calling
// `computeStrides` (which would make this circular against the very function
// the getter delegates to, `ndarray.ts`'s `computeStrides(this.shape)`).
test("NDArray.strides getter: values match hand-computed row-major strides (rank 1/2/3)", () => {
  const rank1 = NDArray.zeros([5]);
  assert.deepStrictEqual([...rank1.strides], [1]); // sole axis: always stride 1

  const rank2 = NDArray.zeros([3, 4]);
  assert.deepStrictEqual([...rank2.strides], [4, 1]); // axis 0: 4 (= dim 1's size); axis 1: 1

  const rank3 = NDArray.zeros([2, 3, 4]);
  assert.deepStrictEqual([...rank3.strides], [12, 4, 1]); // axis 0: 3*4; axis 1: 4; axis 2: 1

  // getter freshness (documented in the NDArrayView doc comment, ndarray.ts):
  // recomputed on every access, not cached — a fresh array each read but the
  // same VALUES every time for an unchanged shape.
  assert.notStrictEqual(rank3.strides, rank3.strides); // fresh array identity each access
  assert.deepStrictEqual([...rank3.strides], [...rank3.strides]); // same values every access
});

test("view-of-view: double transpose is bit-identical to the base", () => {
  const rng = makeRng(0x5354525f56495732n); // "STR_VIW2"
  const shape = [3, 4, 5];
  const data = genData(rng, shape);
  const a = WNDArray.fromArray(core, shape, data);
  const tt = a.transpose().transpose();
  try {
    assertShapeEqual(shape, tt.shape, "double transpose shape");
    assertDataBitIdentical(data, tt.toArray(), "double transpose data");
  } finally {
    // the intermediate view handle from the chained call is unreachable —
    // its buffer reference is released by the GC backstop eventually; base
    // and tt are disposed here (the buffer itself outlives until then,
    // which is exactly the refcount contract).
    tt.dispose();
    a.dispose();
  }
});

// --- add: views on either/both operand positions, vs naive reference ------
{
  const rng = makeRng(0x5354525f4144445fn); // "STR_ADD_"
  for (let c = 0; c < CASE_COUNT; c++) {
    const { aShape, bShape } = genBroadcastShapes(rng);
    const aData = genData(rng, aShape);
    const bData = genData(rng, bShape);
    // at least one operand is a view (both-contiguous is the resident suite)
    let aView = rng.nextBool();
    let bView = rng.nextBool();
    if (!aView && !bView) {
      if (rng.nextBool()) aView = true;
      else bView = true;
    }

    test(`strided add case ${c}: a=[${aShape.join(",")}]${aView ? "ᵛ" : ""} b=[${bShape.join(",")}]${bView ? "ᵛ" : ""}`, () => {
      const ref = elementwiseBinary(aShape, aData, bShape, bData, (x, y) => x + y);
      const a = makeOperand(aView, aShape, aData);
      const b = makeOperand(bView, bShape, bData);
      try {
        const got = a.arr.add(b.arr);
        try {
          const ctx = `strided add case ${c} a=[${aShape.join(",")}] b=[${bShape.join(",")}] aView=${aView} bView=${bView}`;
          assertShapeEqual(ref.shape, got.shape, ctx);
          assertDataBitIdentical(ref.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        disposeAll(a, b);
      }
    });
  }
}

// --- matmul: views on either/both operand positions ------------------------
{
  const rng = makeRng(0x5354525f4d4d554cn); // "STR_MMUL"
  for (let c = 0; c < CASE_COUNT; c++) {
    const { aShape, bShape } = genMatmulShapes(rng);
    const aData = genData(rng, aShape);
    const bData = genData(rng, bShape);
    let aView = rng.nextBool();
    let bView = rng.nextBool();
    if (!aView && !bView) {
      if (rng.nextBool()) aView = true;
      else bView = true;
    }

    test(`strided matmul case ${c}: a=[${aShape.join(",")}]${aView ? "ᵛ" : ""} b=[${bShape.join(",")}]${bView ? "ᵛ" : ""}`, () => {
      const ref = matmulRuntime(aShape, aData, bShape, bData);
      const a = makeOperand(aView, aShape, aData);
      const b = makeOperand(bView, bShape, bData);
      try {
        const got = a.arr.matmul(b.arr);
        try {
          const ctx = `strided matmul case ${c} a=[${aShape.join(",")}] b=[${bShape.join(",")}] aView=${aView} bView=${bView}`;
          assertShapeEqual(ref.shape, got.shape, ctx);
          assertDataBitIdentical(ref.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        disposeAll(a, b);
      }
    });
  }
}

// --- sum (full): THE order-sensitivity case ---------------------------------
// A transposed view's logical row-major order differs from its memory
// order; float addition is order-sensitive, so a kernel summing in memory
// order would produce different BITS here. This group is the reason
// `nt_sum_all_strided` walks logical order (see its Rust doc comment).
{
  const rng = makeRng(0x5354525f53554d5fn); // "STR_SUM_"
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);

    test(`strided sum_all case ${c}: view of shape=[${shape.join(",")}]`, () => {
      const ref = sumRuntime(shape, data, undefined);
      const v = makeView(shape, data);
      try {
        const got = v.arr.sum();
        try {
          const ctx = `strided sum_all case ${c} shape=[${shape.join(",")}]`;
          assertShapeEqual(ref.shape, got.shape, ctx);
          assertDataBitIdentical(ref.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        disposeAll(v);
      }
    });
  }
}

// --- sum (axis) on views ----------------------------------------------------
{
  const rng = makeRng(0x5354525f41585f5fn); // "STR_AX__"
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 1, 4);
    const data = genData(rng, shape);
    const rank = shape.length;
    const positiveAxis = rng.nextInt(0, rank - 1);
    const axis = rng.nextBool() ? positiveAxis - rank : positiveAxis;

    test(`strided sum_axis case ${c}: view of shape=[${shape.join(",")}] axis=${axis}`, () => {
      const ref = sumRuntime(shape, data, axis);
      const v = makeView(shape, data);
      try {
        const got = v.arr.sum(axis);
        try {
          const ctx = `strided sum_axis case ${c} shape=[${shape.join(",")}] axis=${axis}`;
          assertShapeEqual(ref.shape, got.shape, ctx);
          assertDataBitIdentical(ref.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        disposeAll(v);
      }
    });
  }
}

// --- transpose of a view (composition beyond double-transpose) -------------
// A view's transpose must again match the reference chain exactly (shape
// reversal + logical content transposition), not just land back at base.
{
  const rng = makeRng(0x5354525f5452414en); // "STR_TRAN"
  for (let c = 0; c < 60; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);

    test(`strided transpose-of-view case ${c}: shape=[${shape.join(",")}]`, () => {
      const ref = transposeRuntime(shape, data);
      const v = makeView(shape, data);
      try {
        const got = v.arr.transpose();
        try {
          const ctx = `strided transpose-of-view case ${c} shape=[${shape.join(",")}]`;
          assertShapeEqual(ref.shape, got.shape, ctx);
          assertDataBitIdentical(ref.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        disposeAll(v);
      }
    });
  }
}

// --- contiguous(): materialization ------------------------------------------
{
  const rng = makeRng(0x5354525f434f4e54n); // "STR_CONT"
  for (let c = 0; c < 60; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);

    test(`strided contiguous case ${c}: shape=[${shape.join(",")}]`, () => {
      const v = makeView(shape, data);
      const cCopy = v.arr.contiguous();
      // The copy is independently owned: dispose base AND view first, then
      // read the copy — it must still hold the full logical content, with
      // natural row-major strides.
      disposeAll(v);
      try {
        const ctx = `strided contiguous case ${c} shape=[${shape.join(",")}]`;
        // D-V2.3 fallout (same as above): cCopy: AnyWNDArray -> Readonly<any>.
        assertShapeEqual(shape, cCopy.shape as readonly number[], ctx);
        assert.deepStrictEqual([...cCopy.strides], computeStrides(shape), `${ctx}: natural strides`);
        assertDataBitIdentical(data, cCopy.toArray(), ctx);
      } finally {
        cCopy.dispose();
      }
    });
  }
}

test("contiguous() on an already-contiguous array is an independent copy (deliberate always-copy)", () => {
  const a = WNDArray.fromArray(core, [2, 3], [1, 2, 3, 4, 5, 6]);
  const c = a.contiguous();
  a.dispose();
  try {
    assert.deepStrictEqual(Array.from(c.toArray()), [1, 2, 3, 4, 5, 6]);
  } finally {
    c.dispose();
  }
});

// --- edge shapes -------------------------------------------------------------
test("size-0 view: ops and reads behave like the reference", () => {
  const shape: number[] = [0, 3];
  const v = makeView(shape, new Float64Array(0));
  try {
    // D-V2.3 fallout (same as above): v.arr: AnyWNDArray -> Readonly<any>.
    assertShapeEqual([0, 3], v.arr.shape as readonly number[], "size-0 view shape");
    assert.deepStrictEqual(Array.from(v.arr.toArray()), []);
    const s = v.arr.sum();
    try {
      assert.deepStrictEqual(Array.from(s.toArray()), [0]);
    } finally {
      s.dispose();
    }
  } finally {
    disposeAll(v);
  }
});

test("rank-0 view: transpose of a scalar stays a working scalar handle", () => {
  const a = WNDArray.fromArray(core, [], [42]);
  const t = a.transpose();
  try {
    assertShapeEqual([], t.shape, "rank-0 view shape");
    assert.deepStrictEqual(Array.from(t.toArray()), [42]);
  } finally {
    t.dispose();
    a.dispose();
  }
});

// --- ABI boundary: status 4 surfaces across the TS boundary -----------------
// resident.ts itself can never construct invalid strides today (only
// computeStrides + reversals exist at the TS surface), so this exercises the
// entry point directly: the Rust-side StridesOutOfBounds validation must be
// visible as status 4 to a TS caller (verification finding: the status-4
// path was previously pinned only in cargo tests, never across the ABI).
test("raw ABI: out-of-bounds strides return status 4 across the boundary", () => {
  // All allocations BEFORE any view derivation (memory rule: an alloc may
  // grow memory and detach earlier views).
  const shapePtr = core.nt_alloc(2 * 4);
  const stridesPtr = core.nt_alloc(2 * 4);
  const dataPtr = core.nt_alloc(6 * 8);
  const outPtr = core.nt_alloc(6 * 8);
  try {
    new Uint32Array(core.memory.buffer, shapePtr, 2).set([2, 3]);
    // max reach 1*3 + 2*2 = 7 >= data_len 6 -> StridesOutOfBounds
    new Uint32Array(core.memory.buffer, stridesPtr, 2).set([3, 2]);
    new Float64Array(core.memory.buffer, dataPtr, 6).fill(1);
    const status = core.nt_materialize(shapePtr, 2, stridesPtr, 0, dataPtr, 6, outPtr, 6);
    assert.strictEqual(status, 4, "StridesOutOfBounds must surface as status 4 across the ABI");
  } finally {
    core.nt_free(outPtr, 6 * 8);
    core.nt_free(dataPtr, 6 * 8);
    core.nt_free(stridesPtr, 2 * 4);
    core.nt_free(shapePtr, 2 * 4);
  }
});

// --- ABI boundary: defense-in-depth prevalidation (rank/len, status 2/3) ----
// abi.rs hardening finding: the strided/blocked entry points used to build
// Rust slices from caller-declared rank/len BEFORE any validation ran, so a
// garbage rank or length could make read_slice/read_slice_mut construct an
// invalid slice (immediate UB on wasm32's 32-bit isize, no dereference
// required). Rank and every (ptr, len) pair are now prevalidated first —
// this exercises that path across the real ABI boundary (through the
// compiled .wasm, not just the native cargo tests), confirming a garbage
// caller value returns the documented status instead of trapping.
test("raw ABI: garbage rank on a strided entry point returns status 2, no trap", () => {
  // rank = 2**31 is a plausible corrupted/garbage value (well above
  // MAX_RANK) that still fits a JS number cleanly; every pointer/len below
  // is otherwise a legitimate zero-length pair, isolating the rank check.
  const status = core.nt_add_strided(0, 2 ** 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  assert.strictEqual(status, 2, "RankTooLarge must surface as status 2 across the ABI, without trapping");
});

test("raw ABI: garbage data length on a strided entry point returns status 3, no trap", () => {
  // rank = 0 (valid) but a_data_len = 2**32 - 1: the implied byte region
  // (len * 8) vastly exceeds isize::MAX on wasm32, so prevalidation must
  // reject it before any slice is constructed over data_ptr = 0.
  const status = core.nt_add_strided(0, 0, 0, 0, 0, 2 ** 32 - 1, 0, 0, 0, 0, 0, 0, 0, 0);
  assert.strictEqual(status, 3, "SizeOverflow must surface as status 3 across the ABI, without trapping");
});

// --- chains through views, staying resident ---------------------------------
// (Aᵛ @ B).T -> sum(0), all resident, views never materialized, compared
// bit-identically against the same chain on the naive reference.
{
  const rng = makeRng(0x5354525f4348414en); // "STR_CHAN"
  for (let c = 0; c < 60; c++) {
    const n = rng.nextInt(1, 6);
    const shape = [n, n];
    const aData = genData(rng, shape);
    const bData = genData(rng, shape);

    test(`strided chain case ${c}: viewᵀ->matmul->transpose->sum(0), n=${n}`, () => {
      // reference: Aᵀ @ B, then transpose, then sum axis 0
      const refA = transposeRuntime(shape, aData); // Aᵀ
      const refC = matmulRuntime(refA.shape, refA.data, shape, bData);
      const refT = transposeRuntime(refC.shape, refC.data);
      const refS = sumRuntime(refT.shape, refT.data, 0);

      // resident: A stays as stored; Aᵀ is an O(1) view; the chain's own
      // transpose is ALSO a view; only sum's output materializes anything.
      const a = WNDArray.fromArray(core, shape, aData);
      const b = WNDArray.fromArray(core, shape, bData);
      const aT = a.transpose();
      const rC = aT.matmul(b);
      const rT = rC.transpose();
      const rS = rT.sum(0);
      try {
        const ctx = `strided chain case ${c} n=${n}`;
        assertShapeEqual(refS.shape, rS.shape, ctx);
        assertDataBitIdentical(refS.data, rS.toArray(), ctx);
      } finally {
        rS.dispose();
        rT.dispose();
        rC.dispose();
        aT.dispose();
        b.dispose();
        a.dispose();
      }
    });
  }
}
