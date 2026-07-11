/**
 * Kern 08 differential + fixture + error-path tests: `reshape`/`flatten`
 * (docs/kern-08-reshape-flatten-spec.md). Naive `NDArray` (always a fresh
 * copy) vs resident `WNDArray` (view-if-contiguous, else materialize-copy)
 * must be bit-identical, including view-routing pins (buffer sharing via
 * `describe().ptr`, refcount lifecycle) and materialize-routing over
 * non-contiguous (transposed/sliced) views.
 *
 * No cargo tests / no mutation proof (spec: "no new kernels — the
 * differential content is routing/lifecycle, covered by the ptr/refcount
 * pins"): `reshape`/`flatten` are pure metadata ops on the naive surface
 * (a straight `Float64Array` copy) and route through the EXISTING
 * `nt_materialize` entry point on the resident surface — zero new WASM
 * entry points, zero new Rust code.
 */
import assert from "node:assert";
import { test } from "node:test";
import { NDArray } from "../src/ndarray.ts";
import { computeStrides, transposeRuntime, type SliceSpec } from "../src/runtime.ts";
import { initCore } from "../src/wasm/loader.ts";
import { getResidentFreeCount, WNDArray } from "../src/wasm/resident.ts";
import { assertDataBitIdentical, assertShapeEqual } from "./assert-helpers.ts";
import { genData, makeRng, type Rng } from "./prng.ts";

const core = await initCore();

// =============================================================================
// Seeded generators: a shape + a "regrouping" of the same element product
// into a different rank/shape (added/removed 1-dims included).
// =============================================================================

function genShape(rng: Rng, minRank: number, maxRank: number): number[] {
  const rank = rng.nextInt(minRank, maxRank);
  return Array.from({ length: rank }, () => rng.nextInt(1, 6));
}

/** Given `size`, build a random alternative shape with the SAME element
 * product, by peeling small factors off `size` one at a time (largest
 * factor tried first among a small fixed set, falling back to appending the
 * whole remainder) — deliberately simple, hand-verifiable factorization,
 * plus a random sprinkling of extra `1`-dims (added/removed 1-dims per the
 * spec's test-plan wording) and (for size 0) a dedicated branch, since 0
 * has no multiplicative factorization. */
function genRegroupedShape(rng: Rng, size: number): number[] {
  if (size === 0) {
    // Any shape containing at least one 0 dim has product 0 — regroup into
    // a fresh combination of a 0 dim plus a few other small dims.
    const rank = rng.nextInt(1, 3);
    const shape = Array.from({ length: rank }, () => rng.nextInt(1, 5));
    shape[rng.nextInt(0, rank - 1)] = 0;
    return shape;
  }
  const FACTORS = [5, 4, 3, 2];
  let remaining = size;
  const dims: number[] = [];
  for (const f of FACTORS) {
    while (remaining % f === 0 && remaining > f) {
      dims.push(f);
      remaining /= f;
    }
  }
  dims.push(remaining);
  // Randomly interleave a few extra `1` dims (never changes the product).
  const extras = rng.nextInt(0, 2);
  for (let i = 0; i < extras; i++) {
    dims.splice(rng.nextInt(0, dims.length), 0, 1);
  }
  // Randomly shuffle order (still same product, different regrouping/rank
  // "shape" of the factors — not a true random permutation, just enough
  // reordering to avoid always landing back at the same canonical form).
  if (rng.nextBool() && dims.length > 1) {
    const i = rng.nextInt(0, dims.length - 1);
    const j = rng.nextInt(0, dims.length - 1);
    const tmp = dims[i]!;
    dims[i] = dims[j]!;
    dims[j] = tmp;
  }
  return dims;
}

// =============================================================================
// 1. Naive <-> resident parity: reshape, >= 60 seeded cases, ranks 0-4.
// =============================================================================
{
  const rng = makeRng(0x52455348415f4e30n); // "RESHA_N0"-ish
  const CASE_COUNT = 70;
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const size = shape.reduce((a, d) => a * d, 1);
    const newShape = genRegroupedShape(rng, size);
    const data = genData(rng, shape);

    test(`reshape parity case ${c}: [${shape.join(",")}] -> [${newShape.join(",")}]`, () => {
      const nd = NDArray.fromArray(shape, data).reshape(newShape as number[]);

      const base = WNDArray.fromArray(core, shape, data);
      const w = base.reshape(newShape as number[]);
      try {
        const ctx = `reshape parity case ${c} [${shape.join(",")}] -> [${newShape.join(",")}]`;
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
// 2. Naive <-> resident parity: flatten, >= 60 seeded cases, ranks 0-4.
// =============================================================================
{
  const rng = makeRng(0x464c41545f4e3038n); // "FLAT_N08"-ish
  const CASE_COUNT = 70;
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);
    const size = shape.reduce((a, d) => a * d, 1);

    test(`flatten parity case ${c}: shape=[${shape.join(",")}]`, () => {
      const nd = NDArray.fromArray(shape, data).flatten();

      const base = WNDArray.fromArray(core, shape, data);
      const w = base.flatten();
      try {
        const ctx = `flatten parity case ${c} shape=[${shape.join(",")}]`;
        assertShapeEqual([size], nd.shape, ctx);
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
// 3. View-routing pins (resident): contiguous reshape/flatten shares the
// buffer; non-contiguous (transposed/sliced) routes through materialize.
// =============================================================================

test("reshape view-routing: a contiguous handle's reshape shares the buffer (same ptr), refcount +1", () => {
  const base = WNDArray.fromArray(core, [4, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  const beforeFree = getResidentFreeCount();
  const r = base.reshape([2, 6]);
  try {
    assert.strictEqual(r.describe().ptr, base.describe().ptr, "contiguous reshape must be a VIEW: same ptr");
    assert.deepStrictEqual([...r.shape], [2, 6]);
    assert.deepStrictEqual([...r.strides], computeStrides([2, 6]));
    assert.deepStrictEqual(Array.from(r.toArray()), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

    base.dispose();
    assert.strictEqual(base.disposed, true);
    assert.strictEqual(r.disposed, false);
    assert.strictEqual(getResidentFreeCount(), beforeFree, "buffer must NOT be freed while the reshape view lives");
    assert.deepStrictEqual(Array.from(r.toArray()), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "base dispose leaves the view fully usable");
  } finally {
    r.dispose();
  }
  assert.strictEqual(getResidentFreeCount(), beforeFree + 1, "last release frees the shared buffer exactly once");
});

test("flatten view-routing: a contiguous handle's flatten shares the buffer (same ptr), refcount +1", () => {
  const base = WNDArray.fromArray(core, [2, 3], [1, 2, 3, 4, 5, 6]);
  const beforeFree = getResidentFreeCount();
  const f = base.flatten();
  try {
    assert.strictEqual(f.describe().ptr, base.describe().ptr, "contiguous flatten must be a VIEW: same ptr");
    assert.deepStrictEqual([...f.shape], [6]);
    base.dispose();
    assert.strictEqual(getResidentFreeCount(), beforeFree, "buffer must NOT be freed while the flatten view lives");
    assert.deepStrictEqual(Array.from(f.toArray()), [1, 2, 3, 4, 5, 6]);
  } finally {
    f.dispose();
  }
  assert.strictEqual(getResidentFreeCount(), beforeFree + 1);
});

test("reshape view-routing: reshaping a VIEW of a view still shares the ORIGINAL buffer", () => {
  const base = WNDArray.fromArray(core, [4, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  const view = base.slice(null, { start: 0 }); // full-axis range slice: still contiguous [4,3]
  const beforeFree = getResidentFreeCount();
  const r = view.reshape([12]);
  try {
    assert.strictEqual(r.describe().ptr, base.describe().ptr, "reshape of a contiguous view shares the ORIGINAL buffer");
    assert.deepStrictEqual(Array.from(r.toArray()), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  } finally {
    r.dispose();
    view.dispose();
    base.dispose();
  }
  assert.strictEqual(getResidentFreeCount(), beforeFree + 1, "three handles onto one buffer: exactly one free");
});

test("reshape materialize-routing: a transposed (non-contiguous) view materializes a FRESH buffer", () => {
  // shape [2,3] values 1..6: [[1,2,3],[4,5,6]]. Transposed: [[1,4],[2,5],[3,6]]
  // logical order -> flatten/reshape must read that LOGICAL order, not the
  // underlying memory order.
  const base = WNDArray.fromArray(core, [2, 3], [1, 2, 3, 4, 5, 6]);
  const t = base.transpose(); // shape [3,2], non-contiguous
  const beforeFree = getResidentFreeCount();
  const r = t.reshape([6]);
  try {
    assert.notStrictEqual(r.describe().ptr, base.describe().ptr, "non-contiguous reshape must MATERIALIZE: fresh ptr");
    assert.deepStrictEqual(Array.from(r.toArray()), [1, 4, 2, 5, 3, 6], "materialized in the view's LOGICAL row-major order");

    const refNaive = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]).transpose().reshape([6]);
    assertDataBitIdentical(refNaive.data, r.toArray(), "materialize-routing matches the naive reference computed over the materialized logical order");
  } finally {
    r.dispose();
    t.dispose();
    base.dispose();
  }
  assert.strictEqual(getResidentFreeCount(), beforeFree + 2, "base+transpose buffer freed once, materialized output buffer freed once");
});

test("flatten materialize-routing: a step-2 sliced (non-contiguous) view materializes a FRESH buffer", () => {
  const base = WNDArray.fromArray(core, [8], [1, 2, 3, 4, 5, 6, 7, 8]);
  const sliced = base.slice({ step: 2 }); // shape [4], non-contiguous (stride 2)
  const beforeFree = getResidentFreeCount();
  const f = sliced.flatten();
  try {
    assert.notStrictEqual(f.describe().ptr, base.describe().ptr, "non-contiguous flatten must MATERIALIZE: fresh ptr");
    assert.deepStrictEqual(Array.from(f.toArray()), [1, 3, 5, 7]);
  } finally {
    f.dispose();
    sliced.dispose();
    base.dispose();
  }
  assert.strictEqual(getResidentFreeCount(), beforeFree + 2);
});

// =============================================================================
// 4. Edge cases: rank-0 <-> [1]/[1,1]; size-0; identity reshape.
// =============================================================================

test("edge: rank-0 reshapes to [1] and [1,1], both surfaces", () => {
  const nd = NDArray.fromArray([], [42]);
  assert.deepStrictEqual([...nd.reshape([1]).shape], [1]);
  assert.deepStrictEqual(Array.from(nd.reshape([1]).data), [42]);
  assert.deepStrictEqual([...nd.reshape([1, 1]).shape], [1, 1]);
  assert.deepStrictEqual(Array.from(nd.reshape([1, 1]).data), [42]);

  const base = WNDArray.fromArray(core, [], [42]);
  try {
    const r1 = base.reshape([1]);
    try {
      assert.deepStrictEqual([...r1.shape], [1]);
      assert.deepStrictEqual(Array.from(r1.toArray()), [42]);
    } finally {
      r1.dispose();
    }
    const r2 = base.reshape([1, 1]);
    try {
      assert.deepStrictEqual([...r2.shape], [1, 1]);
      assert.deepStrictEqual(Array.from(r2.toArray()), [42]);
    } finally {
      r2.dispose();
    }
  } finally {
    base.dispose();
  }
});

test("edge: [1] and [1,1] reshape back to rank-0, both surfaces", () => {
  const nd1 = NDArray.fromArray([1], [7]);
  assert.deepStrictEqual([...nd1.reshape([]).shape], []);
  assert.deepStrictEqual(Array.from(nd1.reshape([]).data), [7]);

  const base = WNDArray.fromArray(core, [1, 1], [9]);
  try {
    const r = base.reshape([]);
    try {
      assert.deepStrictEqual([...r.shape], []);
      assert.deepStrictEqual(Array.from(r.toArray()), [9]);
    } finally {
      r.dispose();
    }
  } finally {
    base.dispose();
  }
});

test("edge: size-0 reshapes [0,3] -> [0] and [3,0] -> [0,5], both surfaces", () => {
  const nd1 = NDArray.fromArray([0, 3], []);
  assert.deepStrictEqual([...nd1.reshape([0]).shape], [0]);
  assert.deepStrictEqual(Array.from(nd1.reshape([0]).data), []);

  const nd2 = NDArray.fromArray([3, 0], []);
  assert.deepStrictEqual([...nd2.reshape([0, 5]).shape], [0, 5]);

  const base = WNDArray.fromArray(core, [0, 3], []);
  try {
    const r = base.reshape([0]);
    try {
      assert.deepStrictEqual([...r.shape], [0]);
      assert.deepStrictEqual(Array.from(r.toArray()), []);
    } finally {
      r.dispose();
    }
  } finally {
    base.dispose();
  }
});

test("edge: identity reshape (same shape) is valid and produces the same logical content, both surfaces", () => {
  const nd = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
  const r = nd.reshape([2, 3]);
  assert.deepStrictEqual([...r.shape], [2, 3]);
  assert.deepStrictEqual(Array.from(r.data), [1, 2, 3, 4, 5, 6]);
  // naive reshape always COPIES: the data buffer must be a distinct object.
  assert.notStrictEqual(r.data, nd.data);

  const base = WNDArray.fromArray(core, [2, 3], [1, 2, 3, 4, 5, 6]);
  try {
    const rb = base.reshape([2, 3]);
    try {
      assert.strictEqual(rb.describe().ptr, base.describe().ptr, "identity reshape of a contiguous handle is still routed as a VIEW");
      assert.deepStrictEqual(Array.from(rb.toArray()), [1, 2, 3, 4, 5, 6]);
    } finally {
      rb.dispose();
    }
  } finally {
    base.dispose();
  }
});

test("edge: flatten of rank-0 -> [1], of size-0 -> [0], both surfaces", () => {
  const ndRank0 = NDArray.fromArray([], [5]);
  assert.deepStrictEqual([...ndRank0.flatten().shape], [1]);
  assert.deepStrictEqual(Array.from(ndRank0.flatten().data), [5]);

  const ndSize0 = NDArray.fromArray([0, 4], []);
  assert.deepStrictEqual([...ndSize0.flatten().shape], [0]);

  const baseRank0 = WNDArray.fromArray(core, [], [5]);
  try {
    const f = baseRank0.flatten();
    try {
      assert.deepStrictEqual([...f.shape], [1]);
      assert.deepStrictEqual(Array.from(f.toArray()), [5]);
    } finally {
      f.dispose();
    }
  } finally {
    baseRank0.dispose();
  }

  const baseSize0 = WNDArray.fromArray(core, [0, 4], []);
  try {
    const f = baseSize0.flatten();
    try {
      assert.deepStrictEqual([...f.shape], [0]);
      assert.deepStrictEqual(Array.from(f.toArray()), []);
    } finally {
      f.dispose();
    }
  } finally {
    baseSize0.dispose();
  }
});

// =============================================================================
// 5. Error paths, EXACT pinned messages, BOTH surfaces (widened types,
// Spike-03 lesson: a literal violation would be a COMPILE error, so the
// runtime-backstop path needs a deliberately widened static type).
// =============================================================================

test("error: negative dim in new shape throws message 1, naming the dim and the full new shape, both surfaces", () => {
  const nd = NDArray.fromArray([6], [1, 2, 3, 4, 5, 6]);
  assert.throws(
    () => nd.reshape([-1, 6] as number[]),
    /reshape: invalid dimension -1 in shape \[-1,6\] \(dims must be non-negative integers\)/,
  );

  const base = WNDArray.fromArray(core, [6], [1, 2, 3, 4, 5, 6]);
  try {
    assert.throws(
      () => base.reshape([-1, 6] as number[]),
      /reshape: invalid dimension -1 in shape \[-1,6\] \(dims must be non-negative integers\)/,
    );
  } finally {
    base.dispose();
  }
});

test("error: non-integer dim in new shape throws message 1, both surfaces", () => {
  const nd = NDArray.fromArray([6], [1, 2, 3, 4, 5, 6]);
  assert.throws(
    () => nd.reshape([1.5, 4] as number[]),
    /reshape: invalid dimension 1\.5 in shape \[1\.5,4\] \(dims must be non-negative integers\)/,
  );

  const base = WNDArray.fromArray(core, [6], [1, 2, 3, 4, 5, 6]);
  try {
    assert.throws(
      () => base.reshape([1.5, 4] as number[]),
      /reshape: invalid dimension 1\.5 in shape \[1\.5,4\] \(dims must be non-negative integers\)/,
    );
  } finally {
    base.dispose();
  }
});

test("error: a `-1` new-shape dim documents the deferral (message 1, not silently accepted/inferred), both surfaces", () => {
  // -1 dim inference (NumPy's reshape(-1, k)) is explicitly OUT OF SCOPE
  // (FOLLOWUPS) — a literal -1 dim is just "a negative dim" here, and
  // throws message 1 exactly like any other negative dim.
  const nd = NDArray.fromArray([12], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  assert.throws(
    () => nd.reshape([-1, 4] as number[]),
    /reshape: invalid dimension -1 in shape \[-1,4\] \(dims must be non-negative integers\)/,
  );
  const base = WNDArray.fromArray(core, [12], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  try {
    assert.throws(
      () => base.reshape([-1, 4] as number[]),
      /reshape: invalid dimension -1 in shape \[-1,4\] \(dims must be non-negative integers\)/,
    );
  } finally {
    base.dispose();
  }
});

test("error: product mismatch throws message 2, naming the old size and new shape, both surfaces", () => {
  const nd = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
  assert.throws(() => nd.reshape([4, 2] as number[]), /reshape: cannot reshape array of size 6 into shape \[4,2\]/);

  const base = WNDArray.fromArray(core, [2, 3], [1, 2, 3, 4, 5, 6]);
  try {
    assert.throws(() => base.reshape([4, 2] as number[]), /reshape: cannot reshape array of size 6 into shape \[4,2\]/);
  } finally {
    base.dispose();
  }
});

test("error: dim-validity-before-product ORDER pinned (a shape violating both reports message 1)", () => {
  // [2,3] (size 6) -> [-1, 100]: BOTH violated (invalid dim AND, if -1 were
  // somehow taken at face value, a product of -100 != 6 anyway) — message 1
  // must win, per the runtime's own check order (dim validity first).
  const nd = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
  assert.throws(
    () => nd.reshape([-1, 100] as number[]),
    /reshape: invalid dimension -1 in shape \[-1,100\] \(dims must be non-negative integers\)/,
  );
});

test("error: disposed-handle throws for both new WNDArray methods, naming the op", () => {
  const base = WNDArray.fromArray(core, [2, 3], [1, 2, 3, 4, 5, 6]);
  base.dispose();
  assert.throws(() => base.reshape([3, 2]), /WNDArray\.reshape:.*disposed/);
  assert.throws(() => base.flatten(), /WNDArray\.flatten:.*disposed/);
});

// =============================================================================
// 6. Composition sanity: reshape/flatten after slice/transpose still reads
// the correct logical content (beyond the dedicated routing pins above).
// =============================================================================

{
  const rng = makeRng(0x52455348415f434fn); // "RESHA_CO"-ish
  const CASE_COUNT = 30;
  for (let c = 0; c < CASE_COUNT; c++) {
    const baseShape = genShape(rng, 1, 3);
    const data = genData(rng, baseShape);
    const specs: SliceSpec[] = baseShape.map((d) => (d > 1 ? { start: 0, stop: d } : null));

    test(`reshape-of-slice composition case ${c}: shape=[${baseShape.join(",")}]`, () => {
      const size = baseShape.reduce((a, d) => a * d, 1);

      const refNaive = NDArray.fromArray(baseShape, data)
        .slice(...specs)
        .reshape([size] as number[]);

      const base = WNDArray.fromArray(core, baseShape, data);
      const sliced = base.slice(...specs);
      const reshaped = sliced.reshape([size] as number[]);
      try {
        const ctx = `reshape-of-slice composition case ${c} shape=[${baseShape.join(",")}]`;
        assertShapeEqual(refNaive.shape, reshaped.shape, ctx);
        assertDataBitIdentical(refNaive.data, reshaped.toArray(), ctx);
      } finally {
        reshaped.dispose();
        sliced.dispose();
        base.dispose();
      }
    });
  }
}

test("flatten-of-transpose composition: reads the LOGICAL (post-transpose) order, both surfaces", () => {
  const rng = makeRng(0x464c41545f545241n); // "FLAT_TRA"
  const shape = [3, 4];
  const data = genData(rng, shape);
  const refT = transposeRuntime(shape, data);

  const refNaive = NDArray.fromArray(refT.shape, refT.data).flatten();

  const base = WNDArray.fromArray(core, shape, data);
  const t = base.transpose();
  const f = t.flatten();
  try {
    assertShapeEqual(refNaive.shape, f.shape, "flatten-of-transpose shape");
    assertDataBitIdentical(refNaive.data, f.toArray(), "flatten-of-transpose data");
  } finally {
    f.dispose();
    t.dispose();
    base.dispose();
  }
});
