/**
 * Kern 05 slicing tests: the fixture table (spec-pinned unit tests directly
 * against the shared `normalizeSliceSpecs`), then the differential —
 * `WNDArray.slice(...)` (an O(1) view) vs `NDArray.slice(...)` (a fresh
 * copy) must be bit-identical, including composition with Kern-03 transpose
 * views and ops (add/matmul/sum) through the strided kernels / blocked
 * matmul with nonzero offsets for the first time from TS — plus the
 * lifecycle contract (shared buffer, refcounted, same mechanism as
 * `transpose()`) and the error paths (index OOB, step <= 0, too many specs).
 *
 * `normalizeSliceSpecs` is shared, byte-for-byte, by both backends (see its
 * doc comment in runtime.ts) — a deliberate, documented differential blind
 * spot: this file's differential exercises *data movement* divergence
 * (copy vs view), not spec-parsing divergence, which the fixture table
 * above pins directly instead.
 *
 * Same seeded-PRNG methodology as strided.test.ts; wired into
 * `pnpm test:resident` (explicit file list in package.json).
 */
import assert from "node:assert";
import { test } from "node:test";
import { NDArray } from "../src/ndarray.ts";
import {
  computeStrides,
  elementwiseBinary,
  matmulRuntime,
  normalizeSliceSpecs,
  sliceRuntime,
  type SliceSpec,
  sumRuntime,
  transposeRuntime,
} from "../src/runtime.ts";
import { initCore } from "../src/wasm/loader.ts";
import { getResidentFreeCount, WNDArray } from "../src/wasm/resident.ts";
import { assertDataBitIdentical, assertShapeEqual } from "./assert-helpers.ts";
import { genData, makeRng, type Rng } from "./prng.ts";

const core = await initCore();

// --- generators -------------------------------------------------------------

function genShape(rng: Rng, minRank: number, maxRank: number): number[] {
  const rank = rng.nextInt(minRank, maxRank);
  return Array.from({ length: rank }, () => rng.nextInt(1, 10)); // dims 1..10, per spec
}

/** One random axis spec: index (always in-bounds, to keep the MAIN
 * differential throw-free — the dedicated error-path tests below cover
 * out-of-bounds indices explicitly), `null`, or a range object with a mix of
 * present/omitted `start`/`stop`/`step` and deliberately out-of-range values
 * (exercises clamping) and steps 1..3. */
function genSpec(rng: Rng, d: number, opts: { allowIndex?: boolean } = {}): SliceSpec {
  const allowIndex = (opts.allowIndex ?? true) && d > 0; // a size-0 axis has no valid index (fixture table: int on d=0 always throws)
  const kind = allowIndex ? rng.nextInt(0, 2) : rng.nextInt(1, 2); // 0=index,1=null,2=range
  if (kind === 0) return rng.nextInt(-d, d - 1);
  if (kind === 1) return null;
  const spec: { start?: number; stop?: number; step?: number } = {};
  if (rng.nextBool()) spec.start = rng.nextInt(-2 * d, 2 * d); // deliberately out-of-range sometimes
  if (rng.nextBool()) spec.stop = rng.nextInt(-2 * d, 2 * d);
  if (rng.nextBool()) spec.step = rng.nextInt(1, 3);
  return spec;
}

/** A spec vector for a random PREFIX of `shape`'s axes (0..rank specs) —
 * trailing axes beyond that are implicitly taken in full. */
function genSpecs(rng: Rng, shape: readonly number[]): SliceSpec[] {
  const numSpecs = rng.nextInt(0, shape.length);
  return Array.from({ length: numSpecs }, (_, axis) => genSpec(rng, shape[axis] ?? 1));
}

/** Compose the shared normalizer + the naive gather in one call (test-file
 * local convenience; `runtime.ts`'s public surface stays exactly the two
 * functions the spec calls for). */
function naiveSlice(shape: readonly number[], data: Float64Array, specs: readonly SliceSpec[]): { shape: number[]; data: Float64Array } {
  return sliceRuntime(shape, data, normalizeSliceSpecs(shape, specs));
}

// =============================================================================
// Fixture table (spec-pinned, verbatim) — direct unit tests of the shared
// normalizer, since the differential below deliberately does NOT cover
// spec-parsing itself (both backends share it).
// =============================================================================

test("normalizeSliceSpecs fixtures: d=5", () => {
  const d5 = [5];
  assert.deepStrictEqual(normalizeSliceSpecs(d5, [{ start: 1, stop: 4 }]), [{ kind: "range", start: 1, dim: 3, step: 1 }]);
  assert.deepStrictEqual(normalizeSliceSpecs(d5, [{ step: 2 }]), [{ kind: "range", start: 0, dim: 3, step: 2 }]); // indices 0,2,4
  assert.deepStrictEqual(normalizeSliceSpecs(d5, [{ start: 1, step: 2 }]), [{ kind: "range", start: 1, dim: 2, step: 2 }]); // indices 1,3
  assert.deepStrictEqual(normalizeSliceSpecs(d5, [{ start: -2 }]), [{ kind: "range", start: 3, dim: 2, step: 1 }]);
  assert.deepStrictEqual(normalizeSliceSpecs(d5, [{ stop: -1 }]), [{ kind: "range", start: 0, dim: 4, step: 1 }]);
  assert.deepStrictEqual(normalizeSliceSpecs(d5, [{ start: 10 }]), [{ kind: "range", start: 5, dim: 0, step: 1 }]);
  assert.deepStrictEqual(normalizeSliceSpecs(d5, [{ start: 3, stop: 2 }]), [{ kind: "range", start: 3, dim: 0, step: 1 }]);
  assert.deepStrictEqual(normalizeSliceSpecs(d5, [-1]), [{ kind: "index", i: 4 }]);
  assert.throws(() => normalizeSliceSpecs(d5, [5]), /out of bounds/);
  assert.throws(() => normalizeSliceSpecs(d5, [-6]), /out of bounds/);
});

test("normalizeSliceSpecs fixtures: d=0", () => {
  const d0 = [0];
  assert.deepStrictEqual(normalizeSliceSpecs(d0, [{}]), [{ kind: "range", start: 0, dim: 0, step: 1 }]);
  assert.throws(() => normalizeSliceSpecs(d0, [0]), /out of bounds/);
});

// =============================================================================
// Core differential: WNDArray.slice(...).toArray() vs NDArray.slice(...).data
// =============================================================================
{
  const rng = makeRng(0x534c4943455f4331n); // "SLICE_C1"
  const CASE_COUNT = 150;
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);
    const specs = genSpecs(rng, shape);

    test(`slice differential case ${c}: shape=[${shape.join(",")}] specs=${JSON.stringify(specs)}`, () => {
      const nd = NDArray.fromArray(shape, data).slice(...specs);

      const base = WNDArray.fromArray(core, shape, data);
      const w = base.slice(...specs);
      try {
        const ctx = `slice differential case ${c} shape=[${shape.join(",")}] specs=${JSON.stringify(specs)}`;
        assertShapeEqual(nd.shape, w.shape, ctx);
        assertDataBitIdentical(nd.data, w.toArray(), ctx);
      } finally {
        w.dispose();
        base.dispose();
      }
    });
  }
}

// =============================================================================
// Ops on sliced operands: add / matmul / sum, vs naive ops on the naive
// slices — exercises nonzero offsets through every strided kernel and the
// blocked matmul for the first time from TS.
// =============================================================================

// --- add: same base shape, SAME spec vector applied to both operands (so
// both slice to identical shapes deterministically; offsets/data differ) ---
{
  const rng = makeRng(0x534c4943455f4144n); // "SLICE_AD"
  const CASE_COUNT = 100;
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const specs = genSpecs(rng, shape);
    const aData = genData(rng, shape);
    const bData = genData(rng, shape);

    test(`slice+add case ${c}: shape=[${shape.join(",")}] specs=${JSON.stringify(specs)}`, () => {
      const refA = naiveSlice(shape, aData, specs);
      const refB = naiveSlice(shape, bData, specs);
      const ref = elementwiseBinary(refA.shape, refA.data, refB.shape, refB.data, (x, y) => x + y);

      const a = WNDArray.fromArray(core, shape, aData);
      const b = WNDArray.fromArray(core, shape, bData);
      const aS = a.slice(...specs);
      const bS = b.slice(...specs);
      try {
        const got = aS.add(bS);
        try {
          const ctx = `slice+add case ${c} shape=[${shape.join(",")}]`;
          assertShapeEqual(ref.shape, got.shape, ctx);
          assertDataBitIdentical(ref.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        aS.dispose();
        bS.dispose();
        a.dispose();
        b.dispose();
      }
    });
  }
}

// --- matmul: A=[m,k] sliced on m only, B=[k,n] sliced on n only, SAME spec
// shared on the contraction axis k for both — guarantees matching inner
// dims post-slice by construction while still exercising nonzero offsets on
// the non-contracted axes (and, when the shared k-spec is a range, a
// non-unit stride on the contracted axis too). The shared k-spec never
// drops the axis on its own (would still work via 1-D promotion, but is
// avoided so BOTH m and k, or BOTH n and k, never drop simultaneously,
// which would otherwise occasionally produce a rank-0 operand matmul
// legitimately rejects). ---
{
  const rng = makeRng(0x534c4943455f4d4dn); // "SLICE_MM"
  const CASE_COUNT = 100;
  for (let c = 0; c < CASE_COUNT; c++) {
    const m = rng.nextInt(1, 8);
    const k = rng.nextInt(1, 8);
    const n = rng.nextInt(1, 8);
    const specForM = genSpec(rng, m);
    const specForK = genSpec(rng, k, { allowIndex: false });
    const specForN = genSpec(rng, n);
    const specsA = [specForM, specForK];
    const specsB = [specForK, specForN];
    const aData = genData(rng, [m, k]);
    const bData = genData(rng, [k, n]);

    test(`slice+matmul case ${c}: m=${m} k=${k} n=${n}`, () => {
      const refA = naiveSlice([m, k], aData, specsA);
      const refB = naiveSlice([k, n], bData, specsB);
      const ref = matmulRuntime(refA.shape, refA.data, refB.shape, refB.data);

      const a = WNDArray.fromArray(core, [m, k], aData);
      const b = WNDArray.fromArray(core, [k, n], bData);
      const aS = a.slice(...specsA);
      const bS = b.slice(...specsB);
      try {
        const got = aS.matmul(bS);
        try {
          const ctx = `slice+matmul case ${c} m=${m} k=${k} n=${n}`;
          assertShapeEqual(ref.shape, got.shape, ctx);
          assertDataBitIdentical(ref.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        aS.dispose();
        bS.dispose();
        a.dispose();
        b.dispose();
      }
    });
  }
}

// --- sum (full) on a sliced operand ---
{
  const rng = makeRng(0x534c4943455f5355n); // "SLICE_SU"
  const CASE_COUNT = 80;
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const specs = genSpecs(rng, shape);
    const data = genData(rng, shape);

    test(`slice+sum(all) case ${c}: shape=[${shape.join(",")}]`, () => {
      const ref = naiveSlice(shape, data, specs);
      const refSum = sumRuntime(ref.shape, ref.data, undefined);

      const base = WNDArray.fromArray(core, shape, data);
      const sliced = base.slice(...specs);
      try {
        const got = sliced.sum();
        try {
          const ctx = `slice+sum(all) case ${c} shape=[${shape.join(",")}]`;
          assertShapeEqual(refSum.shape, got.shape, ctx);
          assertDataBitIdentical(refSum.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        sliced.dispose();
        base.dispose();
      }
    });
  }
}

// --- sum (axis) on a sliced operand: axis 0 is always `null` (kept), so the
// sliced result always has rank >= 1 and axis 0 is always valid to reduce ---
{
  const rng = makeRng(0x534c4943455f4158n); // "SLICE_AX"
  const CASE_COUNT = 80;
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 1, 4);
    const rest = genSpecs(rng, shape.slice(1)); // specs for axes 1..rank-1
    const specs: SliceSpec[] = [null, ...rest];
    const data = genData(rng, shape);

    test(`slice+sum(axis) case ${c}: shape=[${shape.join(",")}]`, () => {
      const ref = naiveSlice(shape, data, specs);
      const refSum = sumRuntime(ref.shape, ref.data, 0);

      const base = WNDArray.fromArray(core, shape, data);
      const sliced = base.slice(...specs);
      try {
        const got = sliced.sum(0);
        try {
          const ctx = `slice+sum(axis) case ${c} shape=[${shape.join(",")}]`;
          assertShapeEqual(refSum.shape, got.shape, ctx);
          assertDataBitIdentical(refSum.data, got.toArray(), ctx);
        } finally {
          got.dispose();
        }
      } finally {
        sliced.dispose();
        base.dispose();
      }
    });
  }
}

// =============================================================================
// Composition: slice-of-transpose, transpose-of-slice, slice-of-slice.
// =============================================================================

{
  const rng = makeRng(0x534c4943455f5431n); // "SLICE_T1" (slice-of-transpose)
  const CASE_COUNT = 60;
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);
    const transposedShape = [...shape].reverse();
    const specs = genSpecs(rng, transposedShape);

    test(`slice-of-transpose case ${c}: shape=[${shape.join(",")}]`, () => {
      const refT = transposeRuntime(shape, data);
      const ref = naiveSlice(refT.shape, refT.data, specs);

      const base = WNDArray.fromArray(core, shape, data);
      const t = base.transpose();
      const sliced = t.slice(...specs);
      try {
        const ctx = `slice-of-transpose case ${c} shape=[${shape.join(",")}]`;
        assertShapeEqual(ref.shape, sliced.shape, ctx);
        assertDataBitIdentical(ref.data, sliced.toArray(), ctx);
      } finally {
        sliced.dispose();
        t.dispose();
        base.dispose();
      }
    });
  }
}

{
  const rng = makeRng(0x534c4943455f5432n); // "SLICE_T2" (transpose-of-slice)
  const CASE_COUNT = 60;
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);
    const specs = genSpecs(rng, shape);

    test(`transpose-of-slice case ${c}: shape=[${shape.join(",")}]`, () => {
      const refS = naiveSlice(shape, data, specs);
      const ref = transposeRuntime(refS.shape, refS.data);

      const base = WNDArray.fromArray(core, shape, data);
      const sliced = base.slice(...specs);
      const t = sliced.transpose();
      try {
        const ctx = `transpose-of-slice case ${c} shape=[${shape.join(",")}]`;
        assertShapeEqual(ref.shape, t.shape, ctx);
        assertDataBitIdentical(ref.data, t.toArray(), ctx);
      } finally {
        t.dispose();
        sliced.dispose();
        base.dispose();
      }
    });
  }
}

{
  const rng = makeRng(0x534c4943455f5333n); // "SLICE_S3" (slice-of-slice)
  const CASE_COUNT = 60;
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);
    const specs1 = genSpecs(rng, shape);

    test(`slice-of-slice case ${c}: shape=[${shape.join(",")}]`, () => {
      const refS1 = naiveSlice(shape, data, specs1);
      const specs2 = genSpecs(rng, refS1.shape);
      const refS2 = naiveSlice(refS1.shape, refS1.data, specs2);

      const base = WNDArray.fromArray(core, shape, data);
      const s1 = base.slice(...specs1);
      const s2 = s1.slice(...specs2);
      try {
        const ctx = `slice-of-slice case ${c} shape=[${shape.join(",")}]`;
        assertShapeEqual(refS2.shape, s2.shape, ctx);
        assertDataBitIdentical(refS2.data, s2.toArray(), ctx);
      } finally {
        s2.dispose();
        s1.dispose();
        base.dispose();
      }
    });
  }
}

// =============================================================================
// Lifecycle: slice views share the buffer, same refcounted mechanism as
// transpose() (mirrors strided-lifecycle.test.ts idioms exactly).
// =============================================================================

// shape [4,3], values 1..12 row-major: [[1,2,3],[4,5,6],[7,8,9],[10,11,12]].
// .slice(1, {start:1}): axis0 index 1 -> row [4,5,6]; axis1 range
// start=1,stop=3(default),step=1 -> dim 2 -> [5,6]. Hand-verified.
const LIFECYCLE_SHAPE = [4, 3];
const LIFECYCLE_DATA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

test("slice refcount: disposing the base keeps a live slice view fully usable; free happens on the LAST release", () => {
  const base = WNDArray.fromArray(core, LIFECYCLE_SHAPE, LIFECYCLE_DATA);
  const view = base.slice(1, { start: 1 });
  const before = getResidentFreeCount();

  base.dispose();
  assert.strictEqual(base.disposed, true);
  assert.strictEqual(view.disposed, false);
  assert.strictEqual(getResidentFreeCount(), before, "buffer must NOT be freed while a slice view lives");
  assert.deepStrictEqual(Array.from(view.toArray()), [5, 6]);

  view.dispose();
  assert.strictEqual(getResidentFreeCount(), before + 1, "last release frees the shared buffer exactly once");
});

test("slice refcount: disposing the view first keeps the base usable (symmetric order)", () => {
  const base = WNDArray.fromArray(core, LIFECYCLE_SHAPE, LIFECYCLE_DATA);
  const view = base.slice(1, { start: 1 });
  const before = getResidentFreeCount();

  view.dispose();
  assert.strictEqual(view.disposed, true);
  assert.strictEqual(base.disposed, false);
  assert.strictEqual(getResidentFreeCount(), before);
  assert.deepStrictEqual(Array.from(base.toArray()), LIFECYCLE_DATA);

  base.dispose();
  assert.strictEqual(getResidentFreeCount(), before + 1);
});

test("slice-of-slice (view-of-view): frees only after all three handles release", () => {
  const base = WNDArray.fromArray(core, LIFECYCLE_SHAPE, LIFECYCLE_DATA);
  const s1 = base.slice(null, { start: 1 }); // shape [4,2]: cols [2,3],[5,6],[8,9],[11,12]
  const s2 = s1.slice(1); // shape [2]: row index1 of s1 -> [5,6]
  const before = getResidentFreeCount();

  base.dispose();
  s1.dispose();
  assert.strictEqual(getResidentFreeCount(), before, "two of three released — buffer still live");
  assert.deepStrictEqual(Array.from(s2.toArray()), [5, 6]);

  s2.dispose();
  assert.strictEqual(getResidentFreeCount(), before + 1);
});

test("slice-of-transpose (view-of-view): frees only after both handles release", () => {
  const base = WNDArray.fromArray(core, LIFECYCLE_SHAPE, LIFECYCLE_DATA); // shape [4,3]
  const t = base.transpose(); // shape [3,4]
  const s = t.slice(1); // row index1 of transpose -> column 1 of base -> [2,5,8,11]
  const before = getResidentFreeCount();

  base.dispose();
  t.dispose();
  assert.strictEqual(getResidentFreeCount(), before, "base+transpose released — slice view still holds a reference");
  assert.deepStrictEqual(Array.from(s.toArray()), [2, 5, 8, 11]);

  s.dispose();
  assert.strictEqual(getResidentFreeCount(), before + 1);
});

test("double-dispose of a slice view releases its reference only once", () => {
  const base = WNDArray.fromArray(core, LIFECYCLE_SHAPE, LIFECYCLE_DATA);
  const view = base.slice(1, { start: 1 });
  const before = getResidentFreeCount();

  view.dispose();
  view.dispose(); // no-op: must NOT decrement again (would free under the live base)
  assert.strictEqual(getResidentFreeCount(), before);
  assert.deepStrictEqual(Array.from(base.toArray()), LIFECYCLE_DATA, "base unaffected by double-disposed view");

  base.dispose();
  assert.strictEqual(getResidentFreeCount(), before + 1);
});

test("use-after-dispose: a disposed slice view throws per-handle; creating a view of a disposed handle also throws", () => {
  const base = WNDArray.fromArray(core, LIFECYCLE_SHAPE, LIFECYCLE_DATA);
  const view = base.slice(1, { start: 1 });
  view.dispose();
  assert.throws(() => view.toArray(), /WNDArray\.toArray:.*disposed/);
  assert.throws(() => view.slice(0), /WNDArray\.slice:.*disposed/);
  assert.throws(() => view.transpose(), /WNDArray\.transpose:.*disposed/);
  base.dispose();
  assert.throws(() => base.slice(1), /WNDArray\.slice:.*disposed/);
});

test("op outputs never alias operands: result of an op on a slice view has its own buffer", () => {
  const base = WNDArray.fromArray(core, LIFECYCLE_SHAPE, LIFECYCLE_DATA);
  const view = base.slice(1, { start: 1 }); // [5,6]
  const zeros = WNDArray.zeros(core, [2]);
  const out = view.add(zeros); // out holds the view's logical content [5,6]
  zeros.dispose();
  const before = getResidentFreeCount();
  base.dispose();
  view.dispose();
  assert.strictEqual(getResidentFreeCount(), before + 1);
  assert.deepStrictEqual(Array.from(out.toArray()), [5, 6]);
  out.dispose();
});

test("slice view strides/offset observability: matches the hand-computed offset/stride algebra", () => {
  const base = WNDArray.fromArray(core, LIFECYCLE_SHAPE, LIFECYCLE_DATA); // strides [3,1]
  const view = base.slice(1, { start: 1 }); // offset = 1*3 + 1*1 = 4; stride = [1*1] = [1]
  try {
    assert.deepStrictEqual([...view.shape], [2]);
    assert.deepStrictEqual([...view.strides], [1]);
  } finally {
    view.dispose();
    base.dispose();
  }
});

// =============================================================================
// Error cases: index OOB, step <= 0, too many specs — through BOTH class
// methods (not just the normalizer fixtures above).
// =============================================================================

test("error: integer index out of bounds throws through NDArray.slice and WNDArray.slice", () => {
  // Since Spike 03 (docs/spike-03-index-bounds-spec.md), a LITERAL OOB index
  // against a literal dim is already a COMPILE error — which is exactly what
  // this test's original literal calls started triggering. The runtime
  // backstop this test pins is the path for indices the type layer cannot
  // prove anything about, so the indices are deliberately WIDENED to
  // `number` here (the gradual escape hatch); the compile-time twin of this
  // test lives in spike/tests/slice.test-d.ts.
  const nd = NDArray.fromArray([5], [1, 2, 3, 4, 5]);
  assert.throws(() => nd.slice(5 as number), /out of bounds/);
  assert.throws(() => nd.slice(-6 as number), /out of bounds/);

  const base = WNDArray.fromArray(core, [5], [1, 2, 3, 4, 5]);
  try {
    assert.throws(() => base.slice(5 as number), /out of bounds/);
    assert.throws(() => base.slice(-6 as number), /out of bounds/);
  } finally {
    base.dispose();
  }
});

// Spike 03 runtime parity: the exact boundary the TYPE layer draws
// (docs/spike-03-index-bounds-spec.md, acceptance criterion 2) must match
// the runtime's behavior on both sides — every statically-REJECTED case
// throws at runtime (checked via widened indices, since the literal forms no
// longer compile), every statically-ACCEPTED boundary case succeeds as a
// literal call.
test("Spike 03 parity: runtime throws exactly where the type layer rejects, succeeds exactly where it accepts", () => {
  const nd = NDArray.fromArray([5], [1, 2, 3, 4, 5]);

  // Statically accepted boundary cases — these compile as literals AND must
  // not throw: i = d-1 (last valid), i = -d (normalizes to index 0).
  assert.strictEqual(nd.slice(4).data[0], 5);
  assert.strictEqual(nd.slice(-5).data[0], 1);

  // Statically rejected cases — the same values, widened, must throw: i = d,
  // i = -(d+1) (one past each end).
  assert.throws(() => nd.slice(5 as number), /out of bounds/);
  assert.throws(() => nd.slice(-6 as number), /out of bounds/);

  // d = 0: every integer index is statically rejected; runtime agrees.
  const empty = NDArray.fromArray([0], []);
  assert.throws(() => empty.slice(0 as number), /out of bounds/);
  assert.throws(() => empty.slice(-1 as number), /out of bounds/);

  // Values the type layer makes NO claim about (non-integer literal) still
  // hit the runtime's own error, unchanged.
  assert.throws(() => nd.slice(1.5), /not an integer/);
});

test("error: step <= 0 throws through both backends", () => {
  // Since Spike 06 (docs/spike-06-range-literals-spec.md), a LITERAL
  // provably-invalid step (plain-digit 0, negative, or dot-form
  // non-integer) is already a COMPILE error — exactly what this test's
  // original literal `{ step: 0 }`/`{ step: -1 }` calls started triggering.
  // The runtime backstop this test pins is the path for steps the type
  // layer cannot prove anything about, so the values are deliberately
  // WIDENED to `number` here (the gradual escape hatch, same pattern Spike
  // 03 uses for OOB indices); the compile-time twin lives in
  // spike/tests/slice.test-d.ts's Spike-06 section.
  const nd = NDArray.fromArray([5], [1, 2, 3, 4, 5]);
  assert.throws(() => nd.slice({ step: 0 as number }), /step/);
  assert.throws(() => nd.slice({ step: -1 as number }), /step/);

  const base = WNDArray.fromArray(core, [5], [1, 2, 3, 4, 5]);
  try {
    assert.throws(() => base.slice({ step: 0 as number }), /step/);
  } finally {
    base.dispose();
  }
});

test("error: non-integer index/start/stop/step throw through both backends (verify finding: paths were untested)", () => {
  // `start`/`stop` non-integer literals stay OUT of Spike 06's guard scope
  // (deferred as a symmetric follow-on, FOLLOWUPS) so they need no
  // widening; a literal non-integer `step` (dot-form, e.g. `1.5`) IS now a
  // compile error (Spike 06), so it's widened the same way as above.
  const nd = NDArray.fromArray([4], [1, 2, 3, 4]);
  assert.throws(() => nd.slice(1.5), /not an integer/);
  assert.throws(() => nd.slice({ start: 1.5 }), /not an integer/);
  assert.throws(() => nd.slice({ stop: 2.5 }), /not an integer/);
  assert.throws(() => nd.slice({ step: 1.5 as number }), /step 1.5.*invalid/);
  const base = WNDArray.fromArray(core, [4], [1, 2, 3, 4]);
  try {
    assert.throws(() => base.slice(1.5), /not an integer/);
    assert.throws(() => base.slice({ start: 1.5 }), /not an integer/);
    assert.throws(() => base.slice({ step: 1.5 as number }), /step 1.5.*invalid/);
    // a failing slice leaves the handle fully valid
    assert.deepStrictEqual(Array.from(base.toArray()), [1, 2, 3, 4]);
  } finally {
    base.dispose();
  }
});

// Spike 06 runtime parity (acceptance criterion 3): the exact statically-
// rejected step forms (0, negative, non-integer dot-form) throw at runtime
// via the widened calls above; a statically-ACCEPTED literal step (>= 2)
// must succeed, both as a literal call (computes a literal dim, pinned at
// the type level in slice.test-d.ts) and produce the runtime-correct data.
test("Spike 06 parity: a statically-accepted literal step succeeds through both backends", () => {
  const nd = NDArray.fromArray([7], [1, 2, 3, 4, 5, 6, 7]);
  const sliced = nd.slice({ start: 1, stop: 7, step: 3 }); // indices 1,4 -> [2,5]
  assert.deepStrictEqual(Array.from(sliced.data), [2, 5]);

  const base = WNDArray.fromArray(core, [7], [1, 2, 3, 4, 5, 6, 7]);
  try {
    const view = base.slice({ start: 1, stop: 7, step: 3 });
    try {
      assert.deepStrictEqual(Array.from(view.toArray()), [2, 5]);
    } finally {
      view.dispose();
    }
  } finally {
    base.dispose();
  }
});

// --- broadcasting between DIFFERENTLY-sliced operands + batch-broadcast
// matmul with nonzero offsets (verify finding: the main op differentials
// deliberately use same-spec pairs, which never reaches this combination —
// structurally new in Kern 05, since transpose views never produce a
// nonzero offset) ------------------------------------------------------------
{
  const rng = makeRng(0x534c4943455f4243n); // "SLICE_BC"
  for (let c = 0; c < 20; c++) {
    // A: [m+pad, k+pad] sliced with independent nonzero starts -> [m, k] view
    // with a nonzero offset; B: [k+pad] sliced -> [k] view, ALSO offset —
    // broadcast-add across differently-shaped, differently-sliced operands.
    const m = rng.nextInt(1, 5);
    const k = rng.nextInt(1, 5);
    const padA = rng.nextInt(1, 3);
    const padB = rng.nextInt(1, 3);
    const aBaseShape = [m + padA, k + padA];
    const bBaseShape = [k + padB];
    const aData = genData(rng, aBaseShape);
    const bData = genData(rng, bBaseShape);
    const aSpecs: SliceSpec[] = [{ start: padA }, { start: padA }];
    const bSpecs: SliceSpec[] = [{ start: padB }];

    test(`slice broadcast-add case ${c}: [${aBaseShape.join(",")}]${JSON.stringify(aSpecs)} + [${bBaseShape.join(",")}]${JSON.stringify(bSpecs)}`, () => {
      const ndA = naiveSlice(aBaseShape, aData, aSpecs);
      const ndB = naiveSlice(bBaseShape, bData, bSpecs);
      const ref = elementwiseBinary(ndA.shape, ndA.data, ndB.shape, ndB.data, (x, y) => x + y);

      const baseA = WNDArray.fromArray(core, aBaseShape, aData);
      const baseB = WNDArray.fromArray(core, bBaseShape, bData);
      const vA = baseA.slice(...aSpecs);
      const vB = baseB.slice(...bSpecs);
      const got = vA.add(vB);
      try {
        const ctx = `slice broadcast-add case ${c}`;
        assertShapeEqual(ref.shape, got.shape, ctx);
        assertDataBitIdentical(ref.data, got.toArray(), ctx);
      } finally {
        got.dispose();
        vB.dispose();
        vA.dispose();
        baseB.dispose();
        baseA.dispose();
      }
    });
  }

  for (let c = 0; c < 10; c++) {
    // Batch-broadcast matmul with a nonzero-offset sliced batch operand:
    // A base [b+pad, m, k] sliced {start: pad} -> [b, m, k] view (offset != 0),
    // B [k, n] contiguous rank-2 -> batch-broadcast to [b, m, n].
    const b = rng.nextInt(1, 3);
    const m = rng.nextInt(1, 4);
    const k = rng.nextInt(1, 4);
    const n = rng.nextInt(1, 4);
    const pad = rng.nextInt(1, 2);
    const aBaseShape = [b + pad, m, k];
    const bShape = [k, n];
    const aData = genData(rng, aBaseShape);
    const bData = genData(rng, bShape);
    const aSpecs: SliceSpec[] = [{ start: pad }];

    test(`slice batch-broadcast matmul case ${c}: [${aBaseShape.join(",")}]${JSON.stringify(aSpecs)} @ [${bShape.join(",")}]`, () => {
      const ndA = naiveSlice(aBaseShape, aData, aSpecs);
      const ref = matmulRuntime(ndA.shape, ndA.data, bShape, bData);

      const baseA = WNDArray.fromArray(core, aBaseShape, aData);
      const vA = baseA.slice(...aSpecs);
      const wB = WNDArray.fromArray(core, bShape, bData);
      const got = vA.matmul(wB);
      try {
        const ctx = `slice batch-broadcast matmul case ${c}`;
        assertShapeEqual(ref.shape, got.shape, ctx);
        assertDataBitIdentical(ref.data, got.toArray(), ctx);
      } finally {
        got.dispose();
        wB.dispose();
        vA.dispose();
        baseA.dispose();
      }
    });
  }
}

test("error: too many specs throws through both backends, naming rank and spec count", () => {
  // Dynamic (non-literal-tuple) shape: too-many-specs is a COMPILE error for
  // a literal-rank receiver (SliceSpecsGuard, see slice.test-d.ts) — this
  // test specifically exercises the runtime backstop `normalizeSliceSpecs`
  // provides for gradual/dynamic-rank callers the type layer can't check
  // statically (same technique strided-lifecycle.test.ts uses).
  const shape: number[] = [2, 3];
  const nd = NDArray.fromArray(shape, [1, 2, 3, 4, 5, 6]);
  assert.throws(() => nd.slice(0, 0, 0), /3 specs given for rank 2/);

  const base = WNDArray.fromArray(core, shape, [1, 2, 3, 4, 5, 6]);
  try {
    assert.throws(() => base.slice(0, 0, 0), /3 specs given for rank 2/);
  } finally {
    base.dispose();
  }
});

// =============================================================================
// Edge shapes: rank-0, a dim-0 axis, zero specs.
// =============================================================================

test("rank-0 slice: zero specs on a scalar is a no-op copy/view", () => {
  const nd = NDArray.fromArray([], [42]);
  const ndSliced = nd.slice();
  assert.deepStrictEqual([...ndSliced.shape], []);
  assert.deepStrictEqual(Array.from(ndSliced.data), [42]);

  const base = WNDArray.fromArray(core, [], [42]);
  const sliced = base.slice();
  try {
    assert.deepStrictEqual([...sliced.shape], []);
    assert.deepStrictEqual(Array.from(sliced.toArray()), [42]);
  } finally {
    sliced.dispose();
    base.dispose();
  }
});

test("dim-0 axis: an empty range spec on a size-0 axis stays empty, never throws", () => {
  const shape = [0, 3];
  const nd = NDArray.fromArray(shape, []);
  const ndSliced = nd.slice({});
  assert.deepStrictEqual([...ndSliced.shape], [0, 3]);
  assert.deepStrictEqual(Array.from(ndSliced.data), []);

  const base = WNDArray.fromArray(core, shape, []);
  const sliced = base.slice({});
  try {
    assert.deepStrictEqual([...sliced.shape], [0, 3]);
    assert.deepStrictEqual(Array.from(sliced.toArray()), []);
  } finally {
    sliced.dispose();
    base.dispose();
  }
});

test("empty result: a range spec with start>=stop produces a size-0 axis, not a throw", () => {
  const nd = NDArray.fromArray([5], [1, 2, 3, 4, 5]);
  const ndSliced = nd.slice({ start: 3, stop: 2 });
  assert.deepStrictEqual([...ndSliced.shape], [0]);
  assert.deepStrictEqual(Array.from(ndSliced.data), []);

  const base = WNDArray.fromArray(core, [5], [1, 2, 3, 4, 5]);
  const sliced = base.slice({ start: 3, stop: 2 });
  try {
    assert.deepStrictEqual([...sliced.shape], [0]);
    assert.deepStrictEqual(Array.from(sliced.toArray()), []);
  } finally {
    sliced.dispose();
    base.dispose();
  }
});

test("zero specs: full shape/data preserved (differs from `null`-per-axis only in the type layer)", () => {
  const shape = [2, 3];
  const data = [1, 2, 3, 4, 5, 6];
  const nd = NDArray.fromArray(shape, data);
  const ndSliced = nd.slice();
  assert.deepStrictEqual([...ndSliced.shape], shape);
  assert.deepStrictEqual(Array.from(ndSliced.data), data);

  const base = WNDArray.fromArray(core, shape, data);
  const sliced = base.slice();
  try {
    assert.deepStrictEqual([...sliced.shape], shape);
    assert.deepStrictEqual(Array.from(sliced.toArray()), data);
    assert.deepStrictEqual([...sliced.strides], computeStrides(shape));
  } finally {
    sliced.dispose();
    base.dispose();
  }
});
