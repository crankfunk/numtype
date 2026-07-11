/**
 * Differential + fixture + parity tests (Kern 07): `dot`, `norm`,
 * `cosineSimilarity`. Resident WASM (`WNDArray`) must be bit-identical to
 * the naive TS reference (`dotRuntime`/`normSqRuntime`) — see
 * docs/kern-07-elementwise-vector-spec.md.
 *
 * `dot`/`cosineSimilarity` return a plain `number` (not an `NDArray`), so
 * "bit-identical" here means `Object.is` on the two scalars directly (same
 * standard as `assertDataBitIdentical`'s per-element check, just without
 * the `Float64Array` wrapper).
 */
import assert from "node:assert";
import { test } from "node:test";
import { NDArray } from "../src/ndarray.ts";
import { dotRuntime, normalizeSliceSpecs, normSqRuntime, sliceRuntime, transposeRuntime, type SliceSpec } from "../src/runtime.ts";
import { initCore } from "../src/wasm/loader.ts";
import { WNDArray, type AnyWNDArray } from "../src/wasm/resident.ts";
import { assertShapeEqual } from "./assert-helpers.ts";
import { genData, makeRng, type Rng } from "./prng.ts";

const core = await initCore();

function genShape(rng: Rng, minRank: number, maxRank: number): number[] {
  const rank = rng.nextInt(minRank, maxRank);
  return Array.from({ length: rank }, () => rng.nextInt(1, 8));
}

/** Scalar bit-identity assertion, `Object.is`-based (same standard as
 * `assertDataBitIdentical`, just for a lone `number` result rather than a
 * `Float64Array`). */
function assertScalarBitIdentical(expected: number, actual: number, context: string): void {
  assert.ok(Object.is(expected, actual), `${context}: expected ${expected}, got ${actual}`);
}

// =============================================================================
// dot: >=100 seeded 1-D pairs, lengths 0..64 (incl. 0 and 1 explicitly)
// =============================================================================
{
  const rng = makeRng(0x444f545f4b4e3037n); // "DOT_KN07"
  const CASE_COUNT = 110;
  for (let c = 0; c < CASE_COUNT; c++) {
    const n = rng.nextInt(0, 64);
    const aData = genData(rng, [n]);
    const bData = genData(rng, [n]);

    test(`dot case ${c}: n=${n}`, () => {
      const ref = dotRuntime([n], aData, [n], bData);
      const a = WNDArray.fromArray(core, [n], Array.from(aData));
      const b = WNDArray.fromArray(core, [n], Array.from(bData));
      try {
        const got = a.dot(b);
        assertScalarBitIdentical(ref, got, `dot case ${c} n=${n}`);
      } finally {
        a.dispose();
        b.dispose();
      }
    });
  }
}

test("dot fixture: n=0 (empty vectors) -> 0 on both surfaces", () => {
  const naive = NDArray.fromArray([0], []).dot(NDArray.fromArray([0], []));
  assert.strictEqual(naive, 0);
  const a = WNDArray.fromArray(core, [0], []);
  const b = WNDArray.fromArray(core, [0], []);
  try {
    assert.strictEqual(a.dot(b), 0);
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("dot fixture: n=1 on both surfaces", () => {
  const naive = NDArray.fromArray([1], [3]).dot(NDArray.fromArray([1], [4]));
  assert.strictEqual(naive, 12);
  const a = WNDArray.fromArray(core, [1], [3]);
  const b = WNDArray.fromArray(core, [1], [4]);
  try {
    assert.strictEqual(a.dot(b), 12);
  } finally {
    a.dispose();
    b.dispose();
  }
});

// --- dot on strided operands: sliced windows + step-2 views ----------------
// Reference computed over the MATERIALIZED (sliced) data via the same
// `normalizeSliceSpecs`/`sliceRuntime` machinery `WNDArray.slice` itself is
// built on (Kern 05) — proves the resident dot kernel reads the view's
// logical content correctly, not just its raw buffer.
{
  const rng = makeRng(0x444f545f5354524cn); // "DOT_STRL"-ish
  const STRIDED_CASE_COUNT = 30;
  for (let c = 0; c < STRIDED_CASE_COUNT; c++) {
    const baseLen = rng.nextInt(4, 40);
    const aBaseData = genData(rng, [baseLen]);
    const bBaseData = genData(rng, [baseLen]);
    const useStep2 = rng.nextBool();
    const specs: SliceSpec[] = useStep2
      ? [{ step: 2 }]
      : [{ start: rng.nextInt(0, Math.floor(baseLen / 2)), stop: rng.nextInt(Math.floor(baseLen / 2) + 1, baseLen) }];

    test(`dot strided-view case ${c}: baseLen=${baseLen} spec=${JSON.stringify(specs[0])}`, () => {
      const norm = normalizeSliceSpecs([baseLen], specs);
      const aSliced = sliceRuntime([baseLen], aBaseData, norm);
      const bSliced = sliceRuntime([baseLen], bBaseData, norm);
      const ref = dotRuntime(aSliced.shape, aSliced.data, bSliced.shape, bSliced.data);

      const aBase = WNDArray.fromArray(core, [baseLen], aBaseData);
      const bBase = WNDArray.fromArray(core, [baseLen], bBaseData);
      const aView = aBase.slice(...specs);
      const bView = bBase.slice(...specs);
      try {
        const got = aView.dot(bView);
        assertScalarBitIdentical(ref, got, `dot strided-view case ${c}`);
      } finally {
        aView.dispose();
        bView.dispose();
        aBase.dispose();
        bBase.dispose();
      }
    });
  }
}

// =============================================================================
// norm: >=50 seeded cases, ranks 0..4, both surfaces, incl. transposed/sliced
// views; plus the absorption-pattern logical-order pin.
// =============================================================================
{
  const rng = makeRng(0x4e4f524d5f4b4e30n); // "NORM_KN0"
  const CASE_COUNT = 60;
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);

    test(`norm case ${c}: shape=[${shape.join(",")}]`, () => {
      const ref = Math.sqrt(normSqRuntime(data));
      const naive = NDArray.fromArray(shape, data).norm();
      assertScalarBitIdentical(ref, naive, `norm case ${c} naive`);

      const a = WNDArray.fromArray(core, shape, data);
      try {
        assertScalarBitIdentical(ref, a.norm(), `norm case ${c} resident`);
      } finally {
        a.dispose();
      }
    });
  }
}

// --- norm on transposed views -----------------------------------------------
{
  const rng = makeRng(0x4e4f524d5f545241n); // "NORM_TRA"
  const CASE_COUNT = 20;
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);

    test(`norm transposed-view case ${c}: shape=[${shape.join(",")}]`, () => {
      const ref = Math.sqrt(normSqRuntime(data));
      const baseShape = [...shape].reverse();
      const baseData = transposeRuntime(shape, data).data;
      const base = WNDArray.fromArray(core, baseShape, baseData);
      const view = base.transpose();
      try {
        assertScalarBitIdentical(ref, view.norm(), `norm transposed-view case ${c}`);
      } finally {
        view.dispose();
        base.dispose();
      }
    });
  }
}

// --- norm on sliced views ---------------------------------------------------
{
  const rng = makeRng(0x4e4f524d5f534c4en); // "NORM_SLN"-ish
  const CASE_COUNT = 15;
  for (let c = 0; c < CASE_COUNT; c++) {
    const baseLen = rng.nextInt(4, 40);
    const baseData = genData(rng, [baseLen]);
    const specs: SliceSpec[] = [{ start: rng.nextInt(0, Math.floor(baseLen / 2)), stop: rng.nextInt(Math.floor(baseLen / 2) + 1, baseLen) }];

    test(`norm sliced-view case ${c}: baseLen=${baseLen}`, () => {
      const norm = normalizeSliceSpecs([baseLen], specs);
      const sliced = sliceRuntime([baseLen], baseData, norm);
      const ref = Math.sqrt(normSqRuntime(sliced.data));

      const base = WNDArray.fromArray(core, [baseLen], baseData);
      const view = base.slice(...specs);
      try {
        assertScalarBitIdentical(ref, view.norm(), `norm sliced-view case ${c}`);
      } finally {
        view.dispose();
        base.dispose();
      }
    });
  }
}

/** The order-sensitivity pin (adapted for squares — see this phase's
 * implementation notes/cargo test `norm_sq_strided_transposed_view_uses_
 * logical_order` for the derivation and the exact values): these integer
 * values were found by exhaustive random search specifically screening for
 * `memory-order sum-of-squares != logical-order sum-of-squares`, then
 * independently re-verified via the cargo test AND directly in Node. Since
 * `norm_sq` only ever accumulates NON-NEGATIVE terms, the classic
 * large-plus-large-negative CANCELLATION trick (the ORIGINAL
 * `sum_all_strided` absorption test) does not transfer — squaring destroys
 * sign — so this is plain floating-point-addition non-associativity, not a
 * hand-derivable cancellation story.
 *
 * Exercised via the RAW ABI directly (like `strided.test.ts`'s own "raw
 * ABI" group) rather than through `WNDArray`'s public view surface:
 * `WNDArray` only ever constructs views via `transpose()`/`slice()`, whose
 * strides are always DERIVED from a real shape — there is no public way to
 * hand it these exact ARBITRARY strides directly over a hand-picked buffer
 * (the `makeView` "materialize-then-transpose-back" trick used elsewhere in
 * this suite reconstructs a *different* buffer permutation than the one
 * these magic numbers were verified against, so it would not exercise the
 * same order pair — using the raw ABI keeps this test anchored to the
 * EXACT permutation the cargo test already proves diverges). */
test("norm: nt_norm_sq_strided logical-order pin (raw ABI, mirrors the cargo test), with non-vacuity", () => {
  const data = [60590962, -54635165, -9, 85265572, -6, -7];
  // All allocations BEFORE any view derivation (memory rule).
  const shapePtr = core.nt_alloc(2 * 4);
  const stridesPtr = core.nt_alloc(2 * 4);
  const dataPtr = core.nt_alloc(6 * 8);
  const outPtr = core.nt_alloc(8);
  try {
    new Uint32Array(core.memory.buffer, shapePtr, 2).set([3, 2]);
    new Uint32Array(core.memory.buffer, stridesPtr, 2).set([1, 3]); // transpose-of-[2,3] view over `data`
    new Float64Array(core.memory.buffer, dataPtr, 6).set(data);

    const status = core.nt_norm_sq_strided(shapePtr, 2, stridesPtr, 0, dataPtr, 6, outPtr);
    assert.strictEqual(status, 0, "nt_norm_sq_strided must succeed for a valid strided view");
    const logicalOrderNormSq = new Float64Array(core.memory.buffer, outPtr, 1)[0] ?? 0;
    const logicalOrderNorm = Math.sqrt(logicalOrderNormSq);

    // The naive TS reference over the SAME data in the SAME logical order:
    // reading `data` via strides [1,3] visits offsets 0,3,1,4,2,5 — build
    // that permutation directly and sum its squares via `normSqRuntime`.
    const logicalPermutation = [0, 3, 1, 4, 2, 5].map((i) => data[i] ?? 0);
    const naiveLogicalOrderNormSq = normSqRuntime(new Float64Array(logicalPermutation));
    assertScalarBitIdentical(naiveLogicalOrderNormSq, logicalOrderNormSq, "norm logical-order pin (naive vs strided kernel)");

    // Non-vacuity: accumulating squares in the buffer's OWN flat/memory
    // order (ignoring the view's strides entirely) must differ in bits from
    // the logical-order result — proving this test pins the ORDER, not just
    // the value.
    const memoryOrderNormSq = normSqRuntime(new Float64Array(data));
    const memoryOrderNorm = Math.sqrt(memoryOrderNormSq);
    assert.ok(
      !Object.is(memoryOrderNorm, logicalOrderNorm),
      `memory-order (${memoryOrderNorm}) and logical-order (${logicalOrderNorm}) norm must differ here (non-vacuity)`,
    );
  } finally {
    core.nt_free(outPtr, 8);
    core.nt_free(dataPtr, 6 * 8);
    core.nt_free(stridesPtr, 2 * 4);
    core.nt_free(shapePtr, 2 * 4);
  }
});

// =============================================================================
// cosineSimilarity: >=50 seeded cases + exact fixtures (parallel/antiparallel/
// orthogonal/zero-vector), both surfaces.
// =============================================================================
{
  const rng = makeRng(0x434f535f4b4e3037n); // "COS_KN07"
  const CASE_COUNT = 60;
  for (let c = 0; c < CASE_COUNT; c++) {
    const n = rng.nextInt(1, 20);
    const aData = genData(rng, [n]);
    const bData = genData(rng, [n]);

    test(`cosineSimilarity case ${c}: n=${n}`, () => {
      const num = dotRuntime([n], aData, [n], bData);
      const den = Math.sqrt(normSqRuntime(aData)) * Math.sqrt(normSqRuntime(bData));
      const ref = num / den;

      const naive = NDArray.fromArray([n], aData).cosineSimilarity(NDArray.fromArray([n], bData));
      assertScalarBitIdentical(ref, naive, `cosineSimilarity case ${c} naive`);

      const a = WNDArray.fromArray(core, [n], Array.from(aData));
      const b = WNDArray.fromArray(core, [n], Array.from(bData));
      try {
        assertScalarBitIdentical(ref, a.cosineSimilarity(b), `cosineSimilarity case ${c} resident`);
      } finally {
        a.dispose();
        b.dispose();
      }
    });
  }
}

// Exact fixtures, constructed on the 3-4-5 right triangle so every norm is a
// perfect integer and the division is exact — both surfaces.
test("cosineSimilarity fixture: parallel vectors -> exactly 1", () => {
  const naive = NDArray.fromArray([2], [3, 4]).cosineSimilarity(NDArray.fromArray([2], [6, 8]));
  assert.strictEqual(naive, 1);
  const a = WNDArray.fromArray(core, [2], [3, 4]);
  const b = WNDArray.fromArray(core, [2], [6, 8]);
  try {
    assert.strictEqual(a.cosineSimilarity(b), 1);
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("cosineSimilarity fixture: antiparallel vectors -> exactly -1", () => {
  const naive = NDArray.fromArray([2], [3, 4]).cosineSimilarity(NDArray.fromArray([2], [-6, -8]));
  assert.strictEqual(naive, -1);
  const a = WNDArray.fromArray(core, [2], [3, 4]);
  const b = WNDArray.fromArray(core, [2], [-6, -8]);
  try {
    assert.strictEqual(a.cosineSimilarity(b), -1);
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("cosineSimilarity fixture: orthogonal vectors -> exactly 0", () => {
  const naive = NDArray.fromArray([2], [3, 4]).cosineSimilarity(NDArray.fromArray([2], [4, -3]));
  assert.strictEqual(naive, 0);
  const a = WNDArray.fromArray(core, [2], [3, 4]);
  const b = WNDArray.fromArray(core, [2], [4, -3]);
  try {
    assert.strictEqual(a.cosineSimilarity(b), 0);
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("cosineSimilarity fixture: a zero vector -> NaN (pure IEEE, no epsilon guard)", () => {
  const naive = NDArray.fromArray([2], [0, 0]).cosineSimilarity(NDArray.fromArray([2], [3, 4]));
  assert.ok(Number.isNaN(naive));
  const a = WNDArray.fromArray(core, [2], [0, 0]);
  const b = WNDArray.fromArray(core, [2], [3, 4]);
  try {
    assert.ok(Number.isNaN(a.cosineSimilarity(b)));
  } finally {
    a.dispose();
    b.dispose();
  }
});

// =============================================================================
// Parity pins
// =============================================================================

test("parity: dotRuntime(a,b) bits == naive a.matmul(b) rank-0 result bits (dot === matmul [1,k]x[k,1])", () => {
  const rng = makeRng(0x50415249545f4d4dn); // "PARIT_MM"-ish
  for (let c = 0; c < 20; c++) {
    const n = rng.nextInt(1, 16);
    const aData = genData(rng, [n]);
    const bData = genData(rng, [n]);
    const dot = dotRuntime([n], aData, [n], bData);
    const matmulResult = NDArray.fromArray([n], aData).matmul(NDArray.fromArray([n], bData));
    assertShapeEqual([], matmulResult.shape, `parity dot/matmul case ${c}`);
    assertScalarBitIdentical(dot, matmulResult.data[0] ?? Number.NaN, `parity dot/matmul case ${c}`);
  }
});

test("parity: dotRuntime(a,a) bits == normSqRuntime(a) bits", () => {
  const rng = makeRng(0x50415249545f4e53n); // "PARIT_NS"-ish
  for (let c = 0; c < 20; c++) {
    const n = rng.nextInt(0, 16);
    const aData = genData(rng, [n]);
    const dotAA = dotRuntime([n], aData, [n], aData);
    const ns = normSqRuntime(aData);
    assertScalarBitIdentical(dotAA, ns, `parity dot(a,a)/normSq case ${c}`);
  }
});

// =============================================================================
// Error paths, both surfaces, exact pinned messages. Widened types
// (Spike-03 lesson): a concrete shape violating the guard would be a
// COMPILE error at the call site, so the OFFENDING operand's static type is
// deliberately widened to a dynamic rank (`number[]`) via `as unknown as`,
// letting `DotCheck` degrade to "no claim" so the call compiles — the
// runtime shape is unchanged, so `assertVectorPair` still throws for real.
// =============================================================================

test("dot: naive NDArray throws for a rank!=1 FIRST operand, naming the shape", () => {
  const a = NDArray.zeros([2, 3]) as unknown as NDArray<number[]>;
  const b = NDArray.zeros([3]);
  assert.throws(() => a.dot(b), /dot: expected a 1-D vector as the first operand \(got shape \[2,3\]\)/);
});

test("dot: naive NDArray throws for a rank!=1 SECOND operand, naming the shape", () => {
  const a = NDArray.zeros([3]);
  const b = NDArray.zeros([2, 3]) as unknown as NDArray<number[]>;
  assert.throws(() => a.dot(b), /dot: expected a 1-D vector as the second operand \(got shape \[2,3\]\)/);
});

test("dot: naive NDArray throws for a length mismatch, naming both lengths", () => {
  const a = NDArray.zeros([3]);
  const b = NDArray.zeros([4]) as unknown as NDArray<number[]>;
  assert.throws(() => a.dot(b), /dot: vector lengths 3 and 4 do not match/);
});

test("dot: resident WNDArray throws for a rank!=1 FIRST operand, naming the shape", () => {
  const a = WNDArray.zeros(core, [2, 3]) as unknown as AnyWNDArray;
  const b = WNDArray.zeros(core, [3]);
  try {
    assert.throws(() => a.dot(b), /dot: expected a 1-D vector as the first operand \(got shape \[2,3\]\)/);
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("dot: resident WNDArray throws for a rank!=1 SECOND operand, naming the shape", () => {
  const a = WNDArray.zeros(core, [3]);
  const b = WNDArray.zeros(core, [2, 3]) as unknown as AnyWNDArray;
  try {
    assert.throws(() => a.dot(b), /dot: expected a 1-D vector as the second operand \(got shape \[2,3\]\)/);
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("dot: resident WNDArray throws for a length mismatch, naming both lengths", () => {
  const a = WNDArray.zeros(core, [3]);
  const b = WNDArray.zeros(core, [4]) as unknown as AnyWNDArray;
  try {
    assert.throws(() => a.dot(b), /dot: vector lengths 3 and 4 do not match/);
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("cosineSimilarity: naive NDArray throws for a rank!=1 FIRST operand, naming the shape", () => {
  const a = NDArray.zeros([2, 3]) as unknown as NDArray<number[]>;
  const b = NDArray.zeros([3]);
  assert.throws(() => a.cosineSimilarity(b), /cosineSimilarity: expected a 1-D vector as the first operand \(got shape \[2,3\]\)/);
});

test("cosineSimilarity: naive NDArray throws for a rank!=1 SECOND operand, naming the shape", () => {
  const a = NDArray.zeros([3]);
  const b = NDArray.zeros([2, 3]) as unknown as NDArray<number[]>;
  assert.throws(() => a.cosineSimilarity(b), /cosineSimilarity: expected a 1-D vector as the second operand \(got shape \[2,3\]\)/);
});

test("cosineSimilarity: naive NDArray throws for a length mismatch, naming both lengths", () => {
  const a = NDArray.zeros([3]);
  const b = NDArray.zeros([4]) as unknown as NDArray<number[]>;
  assert.throws(() => a.cosineSimilarity(b), /cosineSimilarity: vector lengths 3 and 4 do not match/);
});

test("cosineSimilarity: resident WNDArray throws for a rank!=1 FIRST operand, naming the shape", () => {
  const a = WNDArray.zeros(core, [2, 3]) as unknown as AnyWNDArray;
  const b = WNDArray.zeros(core, [3]);
  try {
    assert.throws(() => a.cosineSimilarity(b), /cosineSimilarity: expected a 1-D vector as the first operand \(got shape \[2,3\]\)/);
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("cosineSimilarity: resident WNDArray throws for a rank!=1 SECOND operand, naming the shape", () => {
  const a = WNDArray.zeros(core, [3]);
  const b = WNDArray.zeros(core, [2, 3]) as unknown as AnyWNDArray;
  try {
    assert.throws(() => a.cosineSimilarity(b), /cosineSimilarity: expected a 1-D vector as the second operand \(got shape \[2,3\]\)/);
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("cosineSimilarity: resident WNDArray throws for a length mismatch, naming both lengths", () => {
  const a = WNDArray.zeros(core, [3]);
  const b = WNDArray.zeros(core, [4]) as unknown as AnyWNDArray;
  try {
    assert.throws(() => a.cosineSimilarity(b), /cosineSimilarity: vector lengths 3 and 4 do not match/);
  } finally {
    a.dispose();
    b.dispose();
  }
});

// --- disposed-handle throws: the three new WNDArray vector ops + one
// elementwise op (`sub`), matching the house pattern in
// resident-lifecycle.test.ts. -----------------------------------------------
test("use-after-dispose: dot/norm/cosineSimilarity/sub throw naming the op when `this` is disposed", () => {
  const a = WNDArray.fromArray(core, [3], [1, 2, 3]);
  const b = WNDArray.fromArray(core, [3], [4, 5, 6]);
  a.dispose();
  try {
    assert.throws(() => a.dot(b), /WNDArray\.dot:.*disposed/);
    assert.throws(() => a.norm(), /WNDArray\.norm:.*disposed/);
    assert.throws(() => a.cosineSimilarity(b), /WNDArray\.cosineSimilarity:.*disposed/);
    assert.throws(() => a.sub(b), /WNDArray\.sub:.*disposed/);
  } finally {
    b.dispose();
  }
});

test("use-after-dispose: dot/cosineSimilarity/sub throw naming the op when `other` is disposed", () => {
  const a = WNDArray.fromArray(core, [3], [1, 2, 3]);
  const b = WNDArray.fromArray(core, [3], [4, 5, 6]);
  b.dispose();
  try {
    assert.throws(() => a.dot(b), /WNDArray\.dot:.*disposed/);
    assert.throws(() => a.cosineSimilarity(b), /WNDArray\.cosineSimilarity:.*disposed/);
    assert.throws(() => a.sub(b), /WNDArray\.sub:.*disposed/);
  } finally {
    a.dispose();
  }
});
