/**
 * Op-Scheibe W1 (docs/op-w1-argmax-topk-spec.md): `argmax`/`topk` reference
 * tests. D1: NDArray-only, no WASM/`WNDArray` counterpart for this slice —
 * so, unlike every other op's suite in this directory, there is no second
 * backend to diff against. Coverage instead combines:
 *  - hand-computed FIXTURES for the pinned semantic edge cases (D4's total
 *    order: NaN-maximal, first-index-wins ties, `+0`/`-0`), each written so
 *    a plausible mutant (`>=` for `>`, last-index-wins, NaN-not-maximal)
 *    would flip the asserted index/value (T4's "Mutations-Nachweisbarkeit"
 *    as a property of the fixture, not just a value-class check);
 *  - message-stem word-equality pins, including a SELF-VERIFYING
 *    cross-check against `sumRuntime`'s own out-of-range-axis throw (D4:
 *    `argmaxRuntime`'s axis stem is claimed WORD-FOR-WORD identical to
 *    `sumRuntime`'s — this file catches both, once each, off the SAME
 *    (shape, axis) input, and asserts the two message STRINGS are equal,
 *    rather than hand-retyping the expected text twice);
 *  - broad randomized cross-checks against `bruteArgmax`/`bruteTopk`,
 *    independently-written references (different code shape than
 *    `runtime.ts`'s loops) exercised over many random shapes/axes/k with
 *    NaN injection (`genDataSpecial`);
 *  - structural keepdims invariants, mirroring `keepdims.test.ts`'s own
 *    non-circular technique (derive the keepdims claim from rank/product
 *    invariants plus the non-keepdims reference, never from
 *    `keepDimsShape` itself);
 *  - transposed/sliced receivers, to explicitly pin the "JS `NDArray` is
 *    always materialized-contiguous" assumption `argmax`/`topk` lean on
 *    (D6) — expected data is derived independently via `transposeRuntime`/
 *    `sliceRuntime` directly, not by re-calling `.transpose()`/`.slice()`.
 */
import assert from "node:assert";
import { test } from "node:test";
import { NDArray } from "../src/ndarray.ts";
import {
  argmaxRuntime,
  keepDimsShape,
  normalizeSliceSpecs,
  product,
  sliceRuntime,
  sumRuntime,
  topkRuntime,
  transposeRuntime,
  type SliceSpec,
} from "../src/runtime.ts";
import { assertShapeEqual, bitsOf } from "./assert-helpers.ts";
import { genData, genDataSpecial, makeRng, SPECIAL_VALUES, type Rng } from "./prng.ts";

function genShape(rng: Rng, minRank: number, maxRank: number): number[] {
  const rank = rng.nextInt(minRank, maxRank);
  return Array.from({ length: rank }, () => rng.nextInt(1, 6));
}

/** Scalar bit-identity assertion (same standard as `assertDataBitIdentical`,
 * for a lone `number`). Replicated locally rather than imported, matching
 * `vector.test.ts`'s/`special-values.test.ts`'s own precedent of an
 * unexported per-file copy. */
function assertScalarBitIdentical(expected: number, actual: number, context: string): void {
  assert.ok(
    Object.is(expected, actual),
    `${context}: expected ${expected} (0x${bitsOf(expected).toString(16)}), got ${actual} (0x${bitsOf(actual).toString(16)})`,
  );
}

/** Independent (from `argmaxRuntime`) brute-force reference for D4's total
 * order: NaN counts as maximal; the first index wins every tie (value
 * equality, `0`/`-0` included, via plain `>`, and NaN-vs-NaN). */
function bruteArgmax(values: readonly number[]): number {
  let bestIdx = 0;
  let bestVal = values[0] ?? 0;
  for (let i = 1; i < values.length; i++) {
    const v = values[i] ?? 0;
    if ((Number.isNaN(v) && !Number.isNaN(bestVal)) || v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/** Independent (from `topkRuntime`) brute-force reference: sort `(value,
 * index)` pairs by D4's total order (NaN first — ascending index among
 * themselves — then descending value, ties ascending index) and take the
 * first `k`. */
function bruteTopk(values: readonly number[], k: number): { values: number[]; indices: number[] } {
  const order = values.map((_, i) => i);
  order.sort((i, j) => {
    const a = values[i] ?? 0;
    const b = values[j] ?? 0;
    const aNaN = Number.isNaN(a);
    const bNaN = Number.isNaN(b);
    if (aNaN && bNaN) return i - j;
    if (aNaN) return -1;
    if (bNaN) return 1;
    if (a > b) return -1;
    if (a < b) return 1;
    return i - j;
  });
  const top = order.slice(0, k);
  return { values: top.map((i) => values[i] ?? 0), indices: top };
}

// =============================================================================
// argmax: niladic form (flat, row-major)
// =============================================================================

test("argmax(): basic fixture picks the flat row-major max index", () => {
  const nd = NDArray.fromArray([8], [3, 1, 4, 1, 5, 9, 2, 6]);
  assert.strictEqual(nd.argmax(), 5);
});

test("argmax(): value ties (non-NaN) -> FIRST index wins, even when the tie is not the leading element", () => {
  const nd = NDArray.fromArray([4], [7, 3, 7, 1]);
  assert.strictEqual(nd.argmax(), 0);
});

test("argmax(): +0/-0 tie -> FIRST index wins (plain `>`, never Object.is)", () => {
  assert.strictEqual(NDArray.fromArray([2], [-0, 0]).argmax(), 0);
  assert.strictEqual(NDArray.fromArray([2], [0, -0]).argmax(), 0);
});

test("argmax(): a single NaN beats every real number, even ones that come after it", () => {
  const nd = NDArray.fromArray([3], [1, Number.NaN, 3]);
  assert.strictEqual(nd.argmax(), 1);
});

test("argmax(): among multiple NaNs, the FIRST index wins (NaN-vs-NaN tie)", () => {
  const nd = NDArray.fromArray([5], [1, Number.NaN, 3, Number.NaN, 2]);
  assert.strictEqual(nd.argmax(), 1);
});

test("argmax(): rank-0 (single-element) receiver -> index 0", () => {
  const nd = NDArray.fromArray([], [42]);
  assert.strictEqual(nd.argmax(), 0);
});

test("argmax(): empty receiver throws the pinned empty-array stem", () => {
  const nd = NDArray.fromArray([0], []);
  assert.throws(() => nd.argmax(), /^Error: argmax: attempt to get argmax of an empty array$/);
});

test("argmax(): randomized cross-check against an independent brute-force reference, incl. NaN injection", () => {
  const rng = makeRng(0x41524d5f464c4154n); // "ARM_FLAT"
  for (let c = 0; c < 150; c++) {
    const shape = genShape(rng, 1, 4);
    const data = genDataSpecial(rng, shape, 0.25);
    const expected = bruteArgmax(Array.from(data));
    const got = NDArray.fromArray(shape, data).argmax();
    assert.strictEqual(got, expected, `case ${c} shape=[${shape.join(",")}]`);
  }
});

// =============================================================================
// argmax(axis): per-axis reduction
// =============================================================================

test("argmax(axis): rank-2 fixture, axis 0 and axis 1", () => {
  // [[3, 9, 1],
  //  [4, 2, 8]]
  const nd = NDArray.fromArray([2, 3], [3, 9, 1, 4, 2, 8]);
  const byCol = nd.argmax(0); // per-column argmax over rows -> shape [3]
  assert.deepStrictEqual(Array.from(byCol.data), [1, 0, 1]);
  assertShapeEqual([3], byCol.shape, "argmax(0) shape");

  const byRow = nd.argmax(1); // per-row argmax over columns -> shape [2]
  assert.deepStrictEqual(Array.from(byRow.data), [1, 2]);
  assertShapeEqual([2], byRow.shape, "argmax(1) shape");
});

test("argmax(axis): negative axis normalizes exactly like a positive one", () => {
  const nd = NDArray.fromArray([2, 3], [3, 9, 1, 4, 2, 8]);
  const viaNeg = nd.argmax(-1);
  const viaPos = nd.argmax(1);
  assert.deepStrictEqual(Array.from(viaNeg.data), Array.from(viaPos.data));
  assertShapeEqual(viaPos.shape, viaNeg.shape, "argmax(-1) vs argmax(1)");
});

test("argmax(axis): rank-3 fixture", () => {
  // (0,0,:)=[1,8]->1  (0,1,:)=[3,4]->1  (1,0,:)=[5,2]->0  (1,1,:)=[7,0]->0
  const nd = NDArray.fromArray([2, 2, 2], [1, 8, 3, 4, 5, 2, 7, 0]);
  const got = nd.argmax(2);
  assert.deepStrictEqual(Array.from(got.data), [1, 1, 0, 0]);
});

test("argmax(axis): out-of-range axis throws a message BYTE-IDENTICAL to sumRuntime's own throw for the same (shape, axis)", () => {
  const shape = [2, 3];
  const data = new Float64Array(6);
  let sumMsg: string | undefined;
  try {
    sumRuntime(shape, data, 5);
  } catch (e) {
    sumMsg = (e as Error).message;
  }
  let argmaxMsg: string | undefined;
  try {
    argmaxRuntime(shape, data, 5);
  } catch (e) {
    argmaxMsg = (e as Error).message;
  }
  assert.ok(sumMsg !== undefined, "sumRuntime must throw for axis 5 on rank-2 shape");
  assert.ok(argmaxMsg !== undefined, "argmaxRuntime must throw for axis 5 on rank-2 shape");
  assert.strictEqual(argmaxMsg, sumMsg, "argmaxRuntime's out-of-range message must be word-for-word identical to sumRuntime's");
  assert.strictEqual(argmaxMsg, "reduce: axis 5 is out of range for shape [2,3] (rank 2)");
});

test("argmax(axis): a size-0 axis throws the empty-array stem (unlike sum, which returns 0)", () => {
  const nd = NDArray.fromArray([2, 0, 3], []);
  assert.throws(() => nd.argmax(1), /^Error: argmax: attempt to get argmax of an empty array$/);
  // Contrast: sum over the SAME zero-length axis is well-defined (0), never throws.
  const summed = nd.sum(1);
  assertShapeEqual([2, 3], summed.shape, "sum(1) over a size-0 axis stays well-defined");
  assert.ok(
    Array.from(summed.data).every((v) => v === 0),
    "sum over a size-0 axis must be all zeros",
  );
});

test("argmax(axis, keepdims): shape AND data identity vs the non-keepdims form (structural, non-circular)", () => {
  const rng = makeRng(0x41524d5f4b4544n); // "ARM_KED"
  for (let c = 0; c < 60; c++) {
    const shape = genShape(rng, 1, 4);
    const data = genData(rng, shape);
    const rank = shape.length;
    const positiveAxis = rng.nextInt(0, rank - 1);
    const axis = rng.nextBool() ? positiveAxis - rank : positiveAxis;

    const nonKeep = NDArray.fromArray(shape, data).argmax(axis);
    const keep = NDArray.fromArray(shape, data).argmax(axis, true);
    const ctx = `case ${c} shape=[${shape.join(",")}] axis=${axis}`;

    // Rank is always preserved.
    assert.strictEqual(keep.shape.length, shape.length, `${ctx}: keepdims must preserve rank`);
    // product invariant: element count never changes.
    assert.strictEqual(product(keep.shape), product(nonKeep.shape), `${ctx}: product must be unchanged`);
    const norm = axis < 0 ? shape.length + axis : axis;
    assert.strictEqual(keep.shape[norm], 1, `${ctx}: reduced axis ${norm} must be size 1`);
    const removed = [...keep.shape.slice(0, norm), ...keep.shape.slice(norm + 1)];
    assertShapeEqual(nonKeep.shape, removed, `${ctx}: keepShape minus reduced axis == non-keep shape`);
    // Data is byte-identical between keepdims and non-keepdims (pure shape metadata).
    assert.deepStrictEqual(Array.from(keep.data), Array.from(nonKeep.data), `${ctx}: keepdims data must equal non-keepdims data`);
  }
});

test("argmax(undefined, true): full-reduction keepdims -> all-ones shape, same flat index as niladic argmax()", () => {
  const nd3 = NDArray.fromArray([2, 3], [3, 9, 1, 4, 2, 8]);
  const kept = nd3.argmax(undefined, true);
  assertShapeEqual([1, 1], kept.shape, "argmax(undefined, true) shape");
  assert.strictEqual(kept.data[0], nd3.argmax());
});

test("argmax(axis): randomized per-axis cross-check against an independent brute-force reference over each output slice, incl. NaN injection", () => {
  const rng = makeRng(0x41524d5f415849n); // "ARM_AXI"
  for (let c = 0; c < 150; c++) {
    const shape = genShape(rng, 1, 4);
    const data = genDataSpecial(rng, shape, 0.2);
    const rank = shape.length;
    const positiveAxis = rng.nextInt(0, rank - 1);
    const axis = rng.nextBool() ? positiveAxis - rank : positiveAxis;
    const nd = NDArray.fromArray(shape, data);
    const got = nd.argmax(axis);

    // Independently re-derive the expected per-slice argmax: for every
    // output flat position, gather the underlying axis's raw values (via a
    // local, from-scratch index walk — not `runtime.ts`'s own helpers) and
    // run bruteArgmax over them.
    const normAxis = axis < 0 ? rank + axis : axis;
    const outShape = [...shape.slice(0, normAxis), ...shape.slice(normAxis + 1)];
    const outSize = product(outShape);
    for (let outFlat = 0; outFlat < outSize; outFlat++) {
      // Decode outFlat into per-axis (non-reduced) indices via a local, from-scratch walk.
      const outStrides = new Array<number>(outShape.length).fill(0);
      {
        let acc = 1;
        for (let i = outShape.length - 1; i >= 0; i--) {
          outStrides[i] = acc;
          acc *= outShape[i] ?? 1;
        }
      }
      const idx = outShape.map((d, i) => Math.floor(outFlat / (outStrides[i] ?? 1)) % d);
      const axisDim = shape[normAxis] ?? 1;
      const slice: number[] = [];
      for (let a = 0; a < axisDim; a++) {
        const full = [...idx.slice(0, normAxis), a, ...idx.slice(normAxis)];
        // row-major flat index for `full` against `shape`
        let flat = 0;
        let strideAcc = 1;
        for (let i = shape.length - 1; i >= 0; i--) {
          flat += (full[i] ?? 0) * strideAcc;
          strideAcc *= shape[i] ?? 1;
        }
        slice.push(data[flat] ?? 0);
      }
      const expected = bruteArgmax(slice);
      assert.strictEqual(got.data[outFlat], expected, `case ${c} shape=[${shape.join(",")}] axis=${axis} outFlat=${outFlat}`);
    }
  }
});

// =============================================================================
// argmax on transposed/sliced receivers (D6: materialization assumption)
// =============================================================================

test("argmax() on a transposed receiver reads the LOGICAL (post-transpose) order", () => {
  const base = NDArray.fromArray([3, 4], Array.from({ length: 12 }, (_, i) => (i === 5 ? 999 : i)));
  const transposed = base.transpose();
  const expectedLogicalData = transposeRuntime([3, 4], base.data).data; // independent derivation
  assert.strictEqual(transposed.argmax(), bruteArgmax(Array.from(expectedLogicalData)));
});

test("argmax(axis) on a sliced receiver reduces over the LOGICAL (post-slice) shape/data", () => {
  const base = NDArray.fromArray([3, 4], Array.from({ length: 12 }, (_, i) => i * 2 + 1));
  const specs: SliceSpec[] = [{ start: 1 }, null]; // rows 1.. -> [2,4]
  const view = base.slice(...specs);
  const norm = normalizeSliceSpecs([3, 4], specs);
  const expected = sliceRuntime([3, 4], base.data, norm); // independent derivation

  const got0 = view.argmax(0);
  const refVia = argmaxRuntime(expected.shape, expected.data, 0);
  assert.deepStrictEqual(Array.from(got0.data), Array.from(refVia.data));
});

// =============================================================================
// topk: fixtures
// =============================================================================

test("topk(0): empty result, valid (never throws) even on an empty (D=0) vector", () => {
  const nd = NDArray.fromArray([5], [3, 1, 4, 1, 5]);
  const { values, indices } = nd.topk(0);
  assertShapeEqual([0], values.shape, "topk(0) values shape");
  assertShapeEqual([0], indices.shape, "topk(0) indices shape");
  assert.strictEqual(values.data.length, 0);
  assert.strictEqual(indices.data.length, 0);

  const empty = NDArray.fromArray([0], []);
  const emptyResult = empty.topk(0);
  assert.strictEqual(emptyResult.values.data.length, 0);
});

test("topk(D): the whole vector, sorted descending (NaN-free fixture)", () => {
  const nd = NDArray.fromArray([5], [3, 1, 4, 1, 5]);
  const { values, indices } = nd.topk(5);
  assert.deepStrictEqual(Array.from(values.data), [5, 4, 3, 1, 1]);
  assert.deepStrictEqual(Array.from(indices.data), [4, 2, 0, 1, 3]); // ties (the two 1s) broken by ascending index
});

test("topk: k > length throws the pinned bounds stem", () => {
  const nd = NDArray.fromArray([3], [1, 2, 3]);
  assert.throws(() => nd.topk(4 as unknown as number), /^Error: topk: k=4 exceeds the vector length 3$/);
});

test("topk: negative k throws the pinned invalid-k stem", () => {
  const nd = NDArray.fromArray([3], [1, 2, 3]);
  assert.throws(() => nd.topk(-1 as unknown as number), /^Error: topk: k must be a non-negative integer \(got -1\)$/);
});

test("topk: non-integer k throws the pinned invalid-k stem", () => {
  const nd = NDArray.fromArray([3], [1, 2, 3]);
  assert.throws(() => nd.topk(1.5 as unknown as number), /^Error: topk: k must be a non-negative integer \(got 1\.5\)$/);
});

test("topk: rank != 1 receiver throws the pinned rank stem", () => {
  const nd = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]) as unknown as NDArray<[3]>;
  assert.throws(() => nd.topk(2), /^Error: topk: expected a 1-D vector \(got shape \[2,3\]\)$/);
});

test("topk: NaN entries sort FIRST, ascending index among themselves, ahead of every real number", () => {
  const nd = NDArray.fromArray([5], [1, Number.NaN, 3, Number.NaN, 2]);
  const { values, indices } = nd.topk(5);
  assert.strictEqual(Number.isNaN(values.data[0]), true);
  assert.strictEqual(Number.isNaN(values.data[1]), true);
  assert.deepStrictEqual(Array.from(indices.data), [1, 3, 2, 4, 0]); // NaNs (asc index) first, then 3,2,1 descending
  assert.deepStrictEqual(Array.from(values.data).slice(2), [3, 2, 1]);
});

test("topk: value ties break by ASCENDING index (not just the leading duplicate)", () => {
  const nd = NDArray.fromArray([5], [5, 3, 5, 1, 5]);
  const { values, indices } = nd.topk(3);
  assert.deepStrictEqual(Array.from(values.data), [5, 5, 5]);
  assert.deepStrictEqual(Array.from(indices.data), [0, 2, 4]);
});

// `bitsOf` (assert-helpers.ts) round-trips its argument through `new
// Float64Array([x])` — an ARRAY-LITERAL construction that, empirically
// (bisected while writing this test), is not reliably payload-preserving
// for a deliberately NON-canonical NaN under some V8 JIT tiers (repeated
// calls of `bitsOf` earlier in the SAME test occasionally flip a later call
// to the canonical 0x7FF8000000000000 payload — a JIT/engine quirk, not a
// `topkRuntime` bug: reading the SAME bits directly out of the Float64Array
// that already holds them, via a `DataView` over its own backing buffer
// with no intermediate literal, is deterministic). Used ONLY for this one
// exact-payload assertion; every other bit-identity check in this file
// stays on `bitsOf`/`Object.is`, matching house convention.
function bitsAt(arr: Float64Array, i: number): bigint {
  const view = new DataView(arr.buffer, arr.byteOffset, arr.byteLength);
  return view.getBigUint64(i * 8, true);
}

test("topk: values[i] === data[indices[i]] byte-exact, including an EXACT NaN payload (non-canonical bit pattern)", () => {
  // Construct a NaN with a deliberately non-canonical payload (not the
  // typical 0x7FF8000000000000 "quiet NaN") via a raw bit pattern.
  const buf = new ArrayBuffer(8);
  const dv = new DataView(buf);
  dv.setBigUint64(0, 0x7ff800000000deadn);
  const weirdNaN = dv.getFloat64(0);
  const weirdBits = dv.getBigUint64(0);
  assert.ok(Number.isNaN(weirdNaN), "constructed value must actually be NaN");

  const nd = NDArray.fromArray([4], [1, weirdNaN, 3, 2]);
  const { values, indices } = nd.topk(4);
  for (let i = 0; i < indices.data.length; i++) {
    const idx = indices.data[i] ?? 0;
    assertScalarBitIdentical(nd.data[idx] ?? 0, values.data[i] ?? 0, `topk values[${i}] vs data[indices[${i}]]`);
  }
  // The weird-NaN's exact bit payload survives through topk (not silently
  // canonicalized to a "plain" NaN, e.g. via arithmetic).
  const weirdIdxInIndices = Array.from(indices.data).indexOf(1);
  assert.strictEqual(bitsAt(values.data, weirdIdxInIndices), weirdBits);
});

test("topk: randomized cross-check against an independent brute-force reference, incl. NaN injection", () => {
  const rng = makeRng(0x544f504b5f524546n); // "TOPK_REF"
  for (let c = 0; c < 150; c++) {
    const n = rng.nextInt(0, 24);
    const data = genDataSpecial(rng, [n], 0.25);
    const k = rng.nextInt(0, n);
    const expected = bruteTopk(Array.from(data), k);
    const got = NDArray.fromArray([n], data).topk(k);
    assert.deepStrictEqual(Array.from(got.indices.data), expected.indices, `case ${c} n=${n} k=${k} indices`);
    for (let i = 0; i < k; i++) {
      assertScalarBitIdentical(expected.values[i] ?? 0, got.values.data[i] ?? 0, `case ${c} n=${n} k=${k} values[${i}]`);
    }
  }
});

// =============================================================================
// topk on a sliced receiver (D6: materialization assumption)
// =============================================================================

test("topk(k) on a sliced 1-D receiver reduces over the LOGICAL (post-slice) data", () => {
  const base = NDArray.fromArray([10], [9, 2, 7, 1, 8, 3, 6, 0, 5, 4]);
  const specs: SliceSpec[] = [{ start: 2, stop: 7 }]; // -> [7,1,8,3,6]
  const view = base.slice(...specs);
  const norm = normalizeSliceSpecs([10], specs);
  const expected = sliceRuntime([10], base.data, norm); // independent derivation

  const got = view.topk(3);
  const ref = topkRuntime(expected.shape, expected.data, 3);
  assert.deepStrictEqual(Array.from(got.values.data), Array.from(ref.values));
  assert.deepStrictEqual(Array.from(got.indices.data), Array.from(ref.indices));
});

// =============================================================================
// Non-vacuity: `genDataSpecial`'s NaN draws actually exercise the argmax/topk
// NaN paths above (guards against a generator regression silently vacuuming
// the randomized NaN-injected cases).
// =============================================================================

test("non-vacuity: SPECIAL_VALUES includes NaN, and genDataSpecial with specialProb=1 always includes it", () => {
  assert.ok(SPECIAL_VALUES.some((v) => Number.isNaN(v)), "SPECIAL_VALUES must include NaN");
  const rng = makeRng(0x4e414e5f4e4f4e56n); // "NAN_NONV"
  const data = genDataSpecial(rng, [200], 1);
  assert.ok(
    Array.from(data).some((v) => Number.isNaN(v)),
    "genDataSpecial(specialProb=1) over 200 draws must produce at least one NaN",
  );
});
