/**
 * Op-Scheibe W2 (docs/op-w2-scalar-mean-spec.md): scalar-overload
 * (`add`/`sub`/`mul`/`div`) and `mean` reference tests. D1: NDArray-only, no
 * WASM/`WNDArray` counterpart for this slice (same disclosed surface
 * asymmetry as W1's argmax/topk) — so, like `argmax-topk.test.ts`, there is
 * no second backend to diff against. Coverage instead combines:
 *  - an explicit, hand-verified matrix (op × rank 0/1/2 × special values —
 *    NaN/±Infinity scalar, NaN embedded in the array, ±0) asserted via the
 *    native IEEE operator itself as the independent "ground truth" (the
 *    same house convention `add.test.ts`'s own `(x, y) => x + y` reference
 *    already uses — the arithmetic IS the spec, D3's own contract is
 *    "elementwise `data[i] op s`" verbatim);
 *  - a randomized differential against the PRE-EXISTING `[1]`-broadcast path
 *    (`elementwiseBinary` with a `[1]`-shaped operand) for rank >= 1 — a
 *    genuinely different code path (full broadcast/stride machinery) than
 *    the new scalar loop, so this is a real equivalence proof, not a
 *    tautology (D3's mandated differential);
 *  - an explicit rank-0 contrast against the OLD `[1]`-wrap workaround,
 *    pinning D2's own motivating claim (`[]` stays `[]`, the workaround
 *    would have produced `[1]`);
 *  - for `mean`: a NON-VACUOUS `sum/n` vs `sum*(1/n)` discriminator (D5) —
 *    both formulas are computed and asserted to actually differ before
 *    pinning which one `meanRuntime` uses, at both the full-reduction and
 *    per-axis granularity;
 *  - a randomized cross-check against an independently-written "sum-then-
 *    divide" brute-force reference (different code shape than
 *    `runtime.ts`'s `sumRuntime`/`meanRuntime`, mirroring W1's
 *    `bruteArgmax`/`bruteTopk` precedent);
 *  - a self-verifying word-for-word stem-equality probe against
 *    `sumRuntime`'s own out-of-range-axis throw (same technique as W1's
 *    `argmaxRuntime` cross-check).
 *
 * Op-Scheibe W3 (docs/op-w3-sqrt-spec.md): a clearly-marked block APPENDED at
 * the end of this file (D4) covers `NDArray.sqrt()`/`sqrtRuntime` — same
 * house convention as the scalar/mean sections above (no new file; this file
 * is already registered in test:core). Coverage: the D2 IEEE-edge matrix
 * (-0/NaN/negative/Infinity/subnormal-bits/size-0), rank 0/1/2 + shape
 * preservation, transposed/sliced receivers, a randomized bit-differential
 * against a direct `Math.sqrt` loop, and the F1-closure retro-proof (the
 * `mul -> sum(1) -> sqrt -> reshape -> div` chain byte-identical to the old
 * hand-loop formulation from examples/rag-demo/main.ts).
 *
 * Op-Scheibe W4 (docs/op-w4-stack-spec.md): another clearly-marked block
 * APPENDED at the end of this file (D5) covers `NDArray.stack(rows)`/
 * `stackRuntime` — same no-new-file house convention. Coverage: word-for-
 * word stem pins for all three throws (empty/rank!=1/length-mismatch),
 * reached BOTH via `stackRuntime` directly and through the public
 * `NDArray.stack` API (dynamic-rank rows built via plain, non-`const`
 * shape variables — the same "widen past the compile-time guard" technique
 * `mean(5)`'s own out-of-range-axis pin above already uses); 1/2/3-row
 * successful cases; D=0 rows; a non-canonical-NaN-payload byte-exactness
 * fixture (`bitsOf`, mirroring special-values.test.ts's own transpose
 * fixture — stack is a pure movement op too); the F5 closure retro-proof
 * (examples/rag-demo/embedding.ts's `embedMatrix` row-major
 * `Float64Array#set`-at-offset algorithm REBUILT locally, not imported —
 * that package is a separate npm-registry-consuming example, deliberately
 * outside the spike/ compilation graph — proven byte-identical to
 * `NDArray.stack`'s own output); a large-N smoke case; and an aliasing-
 * isolation/buffer-freshness pin (the W3-lesson: the result is a fresh
 * buffer, rows stay unmutated).
 *
 * Op-Scheibe W5 (docs/op-w5-item-spec.md): another clearly-marked block
 * APPENDED at the end of this file (D5) covers `NDArray.item(...indices)`/
 * `itemRuntime` — same no-new-file house convention (FOLLOWUPS now tracks
 * splitting this file, which has grown into the W2-W5 collection point, as
 * its own mini-slice with an empty-then-fill protocol). Coverage: rank
 * 0/1/2/3 full-indexing reads; NumPy-parity negative indices; all three
 * throw stems (arity/not-integer/out-of-bounds) word-for-word, reached both
 * via `itemRuntime` directly and through the public `NDArray.item` API;
 * size-0-dim OOB (every index is out of bounds, no special case needed);
 * NaN/-0 byte-exact pass-through (`bitsOf`, same fixture style as
 * special-values.test.ts/W4's stack fixture); a 200+-case randomized
 * flat-index differential against independently-computed row-major offset
 * arithmetic over `.data`; and transposed/sliced receivers (real strided,
 * a distinct materialized data layout; itemRuntime has a single
 * unconditional computeStrides+offset path, no fast-path split).
 */
import assert from "node:assert";
import { test } from "node:test";
import { NDArray } from "../src/ndarray.ts";
import { computeStrides, elementwiseBinary, itemRuntime, meanRuntime, sumRuntime } from "../src/runtime.ts";
import { assertDataBitIdentical, assertShapeEqual, wideSpecs } from "./assert-helpers.ts";
import { genData, genDataSpecial, makeRng, nextF64Special, SPECIAL_VALUES, type Rng } from "./prng.ts";

type ScalarOp = "add" | "sub" | "mul" | "div";
const OPS: readonly ScalarOp[] = ["add", "sub", "mul", "div"];

/** The native IEEE 754 operator per op — the SAME closures `add.test.ts`/
 * `sub`/`mul`/`div`'s own `elementwiseBinary` call sites already use as
 * their pinned reference (ndarray.ts, `(x, y) => x + y` etc.). D3's own
 * contract ("elementwise `data[i] op s`") IS this operator applied in
 * order, so this is the correct independent ground truth, not a
 * tautological copy of `scalarElementwiseRuntime`'s implementation. */
const NATIVE_OPS: Record<ScalarOp, (a: number, b: number) => number> = {
  add: (a, b) => a + b,
  sub: (a, b) => a - b,
  mul: (a, b) => a * b,
  div: (a, b) => a / b,
};

/** Dispatch to the right `NDArray` scalar-overload method — a plain switch
 * (TS-safe; `nd[op](s)` would need an unsound indexed-access cast instead). */
function callScalar<S extends readonly number[]>(nd: NDArray<S>, op: ScalarOp, s: number): NDArray<S> {
  switch (op) {
    case "add":
      return nd.add(s);
    case "sub":
      return nd.sub(s);
    case "mul":
      return nd.mul(s);
    case "div":
      return nd.div(s);
  }
}

/** Independent reference for a scalar op via the PRE-EXISTING binary
 * broadcast path: wrap `s` as a `[1]`-shaped operand and run it through
 * `elementwiseBinary` — genuinely different code (full broadcast/stride
 * alignment machinery) than `scalarElementwiseRuntime`'s straight loop, so
 * matching it is a real differential, not a copy of the code under test. */
function scalarViaWrap(op: ScalarOp, shape: readonly number[], data: Float64Array, s: number): { shape: number[]; data: Float64Array } {
  return elementwiseBinary(shape, data, [1], Float64Array.from([s]), NATIVE_OPS[op]);
}

function genShape(rng: Rng, minRank: number, maxRank: number): number[] {
  const rank = rng.nextInt(minRank, maxRank);
  return Array.from({ length: rank }, () => rng.nextInt(1, 6));
}

// =============================================================================
// Section 1: explicit op x rank(0/1/2) x special-value matrix (D7). Each
// data fixture already embeds a NaN and a +0/-0 element (rank >= 1) so a
// single grouped test proves "NaN in the array" alongside every scalar
// variant; the scalar list separately covers NaN/+Inf/-Inf/+0/-0 plus two
// ordinary finite scalars (order-sensitivity: sub/div would flip sign/value
// under an operand-order bug, which a plain add/mul mutant wouldn't catch).
// =============================================================================

const MATRIX_SCALARS: readonly number[] = [
  Number.NaN,
  Number.POSITIVE_INFINITY,
  Number.NEGATIVE_INFINITY,
  0,
  -0,
  2,
  -3.5,
];

const RANK_FIXTURES: readonly { rank: 0 | 1 | 2; shape: number[]; data: number[] }[] = [
  { rank: 0, shape: [], data: [7] },
  { rank: 1, shape: [5], data: [1, -2.5, Number.NaN, 0, -0] },
  { rank: 2, shape: [2, 3], data: [1, -1, 2, -2, 0.5, -0.5] },
];

for (const op of OPS) {
  for (const fixture of RANK_FIXTURES) {
    test(`${op}(s): explicit rank=${fixture.rank} x special-scalar matrix (data embeds NaN/+0/-0 at rank>=1)`, () => {
      for (const s of MATRIX_SCALARS) {
        const nd = NDArray.fromArray(fixture.shape, fixture.data);
        const actual = callScalar(nd, op, s);
        assertShapeEqual(fixture.shape, actual.shape, `${op} rank=${fixture.rank} s=${s}: shape must be preserved exactly`);
        const expected = Float64Array.from(fixture.data.map((v) => NATIVE_OPS[op](v, s)));
        assertDataBitIdentical(expected, actual.data, `${op} rank=${fixture.rank} s=${s}`);
      }
    });
  }
}

// =============================================================================
// Section 2: [1]-wrap byte-exact equivalence (D3 mandatory differential),
// rank >= 1, randomized shapes/data/scalars with special-value injection.
// =============================================================================

{
  const rng = makeRng(0x5343414c5f57524150n); // "SCAL_WRAP" (truncated to fit)
  const CASE_COUNT = 160;
  for (let c = 0; c < CASE_COUNT; c++) {
    const op = OPS[c % OPS.length]!;
    const shape = genShape(rng, 1, 3);
    const data = genDataSpecial(rng, shape, 0.25);
    const s = nextF64Special(rng, 0.3);

    test(`${op}(s) case ${c}: [1]-wrap byte-exact equivalence, shape=[${shape.join(",")}] s=${s}`, () => {
      const nd = NDArray.fromArray(shape, data);
      const actual = callScalar(nd, op, s);
      const expected = scalarViaWrap(op, shape, data, s);
      assertShapeEqual(shape, actual.shape, `case ${c}: shape must equal the ORIGINAL shape (D2: no [1]-broadcast growth)`);
      assertShapeEqual(expected.shape, actual.shape, `case ${c}: shape vs [1]-wrap-path shape`);
      assertDataBitIdentical(expected.data, actual.data, `case ${c} ${op}(${s})`);
    });
  }
}

// =============================================================================
// Section 3: rank-0 shape preservation, explicit contrast against the OLD
// [1]-wrap workaround (D2's own motivating claim, pinned as a real run).
// =============================================================================

test("div(s) at rank 0: shape stays [] — contrast against the OLD [1]-wrap workaround, which would have produced [1]", () => {
  const nd = NDArray.fromArray([], [10]);
  const divided = nd.div(2);
  assertShapeEqual([], divided.shape, "rank-0 div(2) shape");
  assert.strictEqual(divided.data[0], 5);

  // The pre-W2 workaround: `x.div(fromArray([1], [s]))`. Still compiles and
  // still works (D3), but changes rank 0 -> rank 1 — exactly the NumPy-false
  // behavior D2 rejected the new scalar overload's shape semantics against.
  const viaWrapWorkaround = nd.div(NDArray.fromArray([1], [2]));
  assertShapeEqual([1], viaWrapWorkaround.shape, "rank-0 [1]-wrap workaround shape (contrast — NOT what the scalar overload does)");
  assert.strictEqual(viaWrapWorkaround.data[0], 5);
});

test("add/sub/mul(s) at rank 0: shape stays [] for every op, not just div", () => {
  for (const op of OPS) {
    const nd = NDArray.fromArray([], [3]);
    const result = callScalar(nd, op, 4);
    assertShapeEqual([], result.shape, `rank-0 ${op}(4) shape`);
    assert.strictEqual(result.data[0], NATIVE_OPS[op](3, 4), `rank-0 ${op}(4) value`);
  }
});

// =============================================================================
// Section 4: mean — non-vacuous sum/n vs sum*(1/n) discriminator, full
// reduction (D5). n=49, sum=5: 5/49 !== 5*(1/49) in f64 (verified below
// before the pin is even asserted, so the pin can never be vacuous).
// =============================================================================

test("mean(): sum/n vs sum*(1/n) discriminator is non-vacuous (n=49, sum=5), and meanRuntime picks sum/n", () => {
  const n = 49;
  const sum = 5;
  const viaDiv = sum / n; // the D5-pinned formula
  const viaMul = sum * (1 / n); // the REJECTED formula
  assert.notStrictEqual(viaDiv, viaMul, "precondition: the two formulas must actually differ in f64, else this pin proves nothing");

  const data = new Float64Array(n); // 48 zeros + one `5`, ascending-order sum is exactly 5 (no rounding in the summation itself)
  data[0] = sum;
  const summedCheck = sumRuntime([n], data, undefined);
  assert.strictEqual(summedCheck.data[0], sum, "grounding check: the hand-constructed array really does sum to exactly 5");

  const nd = NDArray.fromArray([n], data);
  const meanResult = nd.mean();
  assertShapeEqual([], meanResult.shape, "mean() niladic shape");
  assert.ok(Object.is(meanResult.data[0], viaDiv), `mean() must equal sum/n = ${viaDiv}, got ${meanResult.data[0]}`);
  assert.notStrictEqual(meanResult.data[0], viaMul, "mean() must NOT equal the rejected sum*(1/n) formula");

  // Direct runtime-function-level pin too (not just through the class).
  const direct = meanRuntime([n], data, undefined);
  assert.ok(Object.is(direct.data[0], viaDiv));
});

// =============================================================================
// Section 5: mean — the SAME discriminator at axis granularity (Baustein-0
// addendum evidence: shape [4,49], axis=1, row sums [5,9,1,2] — rows 0/1
// (sums 5,9) discriminate, rows 2/3 (sums 1,2) don't; "2/4 abweichende
// Elemente" is a spec-warned, not-every-example-discriminates fact, proven
// here rather than assumed. Also covers negative axis + keepdims.
// =============================================================================

test("mean(axis): sum/n vs sum*(1/n) discriminator at axis granularity — 2 of 4 rows diverge, 2 don't (both proven, not assumed)", () => {
  const n = 49;
  const rowSums = [5, 9, 1, 2];
  const discriminates = rowSums.map((s) => s / n !== s * (1 / n));
  assert.deepStrictEqual(discriminates, [true, true, false, false], "precondition: exactly rows 0,1 must discriminate and rows 2,3 must not");

  const data = new Float64Array(4 * n);
  for (let r = 0; r < 4; r++) data[r * n] = rowSums[r]!; // rest of each row stays 0

  const nd = NDArray.fromArray([4, n], data);
  const meaned = nd.mean(1);
  assertShapeEqual([4], meaned.shape, "mean(1) shape");
  for (let r = 0; r < 4; r++) {
    const expected = rowSums[r]! / n;
    assert.ok(Object.is(meaned.data[r], expected), `row ${r}: expected sum/n = ${expected}, got ${meaned.data[r]}`);
    if (discriminates[r]) {
      assert.notStrictEqual(meaned.data[r], rowSums[r]! * (1 / n), `row ${r} was supposed to discriminate but matched the rejected formula`);
    }
  }

  // Negative axis normalizes exactly like the positive equivalent (mirrors
  // argmax's/sum's own negative-axis pin).
  const viaNeg = nd.mean(-1);
  assert.deepStrictEqual(Array.from(viaNeg.data), Array.from(meaned.data));
  assertShapeEqual(meaned.shape, viaNeg.shape, "mean(-1) vs mean(1) shape");

  // keepdims: shape gains a size-1 axis, data is byte-identical to non-keepdims.
  const kept = nd.mean(1, true);
  assertShapeEqual([4, 1], kept.shape, "mean(1, true) shape");
  assert.deepStrictEqual(Array.from(kept.data), Array.from(meaned.data), "keepdims data must be byte-identical to non-keepdims");

  const keptNeg = nd.mean(-1, true);
  assertShapeEqual([4, 1], keptNeg.shape, "mean(-1, true) shape");
  assert.deepStrictEqual(Array.from(keptNeg.data), Array.from(meaned.data));
});

// =============================================================================
// Section 6: mean — randomized cross-check against an independently-written
// "sum-then-divide" brute-force reference (different code shape than
// sumRuntime/meanRuntime), across ranks/axes, with special values injected.
// =============================================================================

function bruteMeanAll(data: Float64Array): number {
  let total = 0;
  for (let i = 0; i < data.length; i++) total += data[i] ?? 0;
  return total / data.length;
}

/** Independent per-axis mean: walks the shape by hand (own stride/unravel
 * arithmetic, not runtime.ts's `computeStrides`/`unravel`) and divides each
 * slice's sum by the axis dim. */
function bruteMeanAxis(shape: readonly number[], data: Float64Array, axis: number): { shape: number[]; data: Float64Array } {
  const rank = shape.length;
  const normAxis = axis < 0 ? rank + axis : axis;
  const outShape = [...shape.slice(0, normAxis), ...shape.slice(normAxis + 1)];
  const outStrides = new Array<number>(outShape.length).fill(0);
  {
    let acc = 1;
    for (let i = outShape.length - 1; i >= 0; i--) {
      outStrides[i] = acc;
      acc *= outShape[i] ?? 1;
    }
  }
  const strides = new Array<number>(rank).fill(0);
  {
    let acc = 1;
    for (let i = rank - 1; i >= 0; i--) {
      strides[i] = acc;
      acc *= shape[i] ?? 1;
    }
  }
  const axisDim = shape[normAxis] ?? 1;
  const totalOut = outShape.reduce((a, d) => a * d, 1);
  const out = new Float64Array(totalOut);
  for (let outFlat = 0; outFlat < totalOut; outFlat++) {
    const idx = outShape.map((d, i) => Math.floor(outFlat / (outStrides[i] ?? 1)) % d);
    let sum = 0;
    for (let a = 0; a < axisDim; a++) {
      const full = [...idx.slice(0, normAxis), a, ...idx.slice(normAxis)];
      let flat = 0;
      let strideAcc = 1;
      for (let i = shape.length - 1; i >= 0; i--) {
        flat += (full[i] ?? 0) * strideAcc;
        strideAcc *= shape[i] ?? 1;
      }
      sum += data[flat] ?? 0;
    }
    out[outFlat] = sum / axisDim;
  }
  return { shape: outShape, data: out };
}

{
  const rng = makeRng(0x4d45414e5f415849n); // "MEAN_AXI"
  const CASE_COUNT = 150;
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 1, 4);
    const data = genDataSpecial(rng, shape, 0.15);
    const rank = shape.length;
    const positiveAxis = rng.nextInt(0, rank - 1);
    const axis = rng.nextBool() ? positiveAxis - rank : positiveAxis;

    test(`mean(axis) case ${c}: cross-check against an independent brute-force sum/n reference, shape=[${shape.join(",")}] axis=${axis}`, () => {
      const got = NDArray.fromArray(shape, data).mean(axis);
      const expected = bruteMeanAxis(shape, data, axis);
      assertShapeEqual(expected.shape, got.shape, `case ${c}`);
      for (let i = 0; i < expected.data.length; i++) {
        assert.ok(Object.is(expected.data[i] ?? 0, got.data[i] ?? 0), `case ${c} flat=${i}: expected ${expected.data[i]}, got ${got.data[i]}`);
      }
    });
  }

  const rngAll = makeRng(0x4d45414e5f414c4cn); // "MEAN_ALL"
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rngAll, 0, 4);
    const data = genDataSpecial(rngAll, shape, 0.15);
    test(`mean() case ${c}: cross-check against an independent brute-force sum/n reference (full reduction), shape=[${shape.join(",")}]`, () => {
      const got = NDArray.fromArray(shape, data).mean();
      const expected = data.length === 0 ? Number.NaN : bruteMeanAll(data);
      assertShapeEqual([], got.shape, `case ${c}`);
      assert.ok(Object.is(expected, got.data[0] ?? 0), `case ${c}: expected ${expected}, got ${got.data[0]}`);
    });
  }
}

// Sanity smoke: a trivially hand-verifiable case, independent of the
// randomized/discriminator machinery above.
test("mean(): plain smoke case, sum=10 n=4 -> 2.5 (no rounding ambiguity)", () => {
  const nd = NDArray.fromArray([4], [1, 2, 3, 4]);
  assert.strictEqual(nd.mean().data[0], 2.5);
  assert.strictEqual(nd.mean().data[0], (nd.sum().data[0] ?? 0) / 4);
});

// =============================================================================
// Section 7: mean-of-empty -> NaN, both reduction paths (D5: never a throw,
// unlike argmax on the same inputs).
// =============================================================================

test("mean(): full reduction of an empty (size-0) receiver is NaN (0/0), never throws", () => {
  const nd = NDArray.fromArray([0], []);
  const meaned = nd.mean();
  assertShapeEqual([], meaned.shape, "mean() of empty shape");
  assert.ok(Number.isNaN(meaned.data[0]), `expected NaN, got ${meaned.data[0]}`);

  const direct = meanRuntime([0], new Float64Array(0), undefined);
  assert.ok(Number.isNaN(direct.data[0] ?? Number.NaN));
});

test("mean(axis): a size-0 axis is NaN for every output element (0/0), never throws — contrast with argmax, which throws on the same shape", () => {
  const nd = NDArray.fromArray([2, 0, 3], []);
  const meaned = nd.mean(1);
  assertShapeEqual([2, 3], meaned.shape, "mean(1) over a size-0 axis stays well-defined (shape)");
  assert.ok(
    Array.from(meaned.data).every((v) => Number.isNaN(v)),
    "mean over a size-0 axis must be all-NaN",
  );
  assert.throws(() => nd.argmax(1), /^Error: argmax: attempt to get argmax of an empty array$/, "contrast: argmax DOES throw on the same input");

  const direct = meanRuntime([2, 0, 3], new Float64Array(0), 1);
  assertShapeEqual([2, 3], direct.shape, "meanRuntime([2,0,3], [], 1) shape");
  assert.ok(Array.from(direct.data).every((v) => Number.isNaN(v)));
});

// =============================================================================
// Section 8: stem word-equality — meanRuntime's out-of-range-axis throw is
// entirely sumRuntime's own (D5: no separate check), so the message must be
// WORD-FOR-WORD identical, caught here off the SAME (shape, axis) input.
// =============================================================================

test("mean(axis): out-of-range axis throws a message BYTE-IDENTICAL to sumRuntime's own throw for the same (shape, axis)", () => {
  const shape = [2, 3];
  const data = new Float64Array(6);
  let sumMsg: string | undefined;
  try {
    sumRuntime(shape, data, 5);
  } catch (e) {
    sumMsg = (e as Error).message;
  }
  let meanMsg: string | undefined;
  try {
    meanRuntime(shape, data, 5);
  } catch (e) {
    meanMsg = (e as Error).message;
  }
  assert.ok(sumMsg !== undefined, "sumRuntime must throw for axis 5 on rank-2 shape");
  assert.ok(meanMsg !== undefined, "meanRuntime must throw for axis 5 on rank-2 shape");
  assert.strictEqual(meanMsg, sumMsg, "meanRuntime's out-of-range message must be word-for-word identical to sumRuntime's");
  assert.strictEqual(meanMsg, "reduce: axis 5 is out of range for shape [2,3] (rank 2)");

  // Same pin through the public class API.
  const nd = NDArray.fromArray(shape, data);
  assert.throws(() => nd.mean(5), /^Error: reduce: axis 5 is out of range for shape \[2,3\] \(rank 2\)$/);
});

// =============================================================================
// Section 9: mean(undefined, true) — full-reduction keepdims, matches the
// niladic mean()'s value under an all-ones shape (D7's explicit ask; mean
// does NOT need argmax's arguments.length discrimination — D4 — since,
// unlike argmax, EVERY mean overload returns NDArray<...>, never a bare
// number, so there is no niladic-vs-1-arg branch to confuse).
// =============================================================================

test("mean(undefined, true): full-reduction keepdims -> all-ones shape, same value as niladic mean()", () => {
  const nd = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
  const kept = nd.mean(undefined, true);
  assertShapeEqual([1, 1], kept.shape, "mean(undefined, true) shape");
  assert.ok(Object.is(kept.data[0], nd.mean().data[0]));
});

// =============================================================================
// Non-vacuity: genDataSpecial's NaN/Infinity draws actually exercise the
// scalar-op/mean special-value paths above (guards against a generator
// regression silently vacuuming the randomized special-value cases).
// =============================================================================

test("non-vacuity: nextF64Special/genDataSpecial with high specialProb reliably produce NaN and Infinity", () => {
  const rng = makeRng(0x5343414c5f4e4f4e56n); // "SCAL_NONV" (truncated to fit)
  let sawNaN = false;
  let sawInf = false;
  for (let i = 0; i < 200; i++) {
    const v = nextF64Special(rng, 1);
    if (Number.isNaN(v)) sawNaN = true;
    if (!Number.isFinite(v) && !Number.isNaN(v)) sawInf = true;
  }
  assert.ok(sawNaN, "200 draws at specialProb=1 must include at least one NaN");
  assert.ok(sawInf, "200 draws at specialProb=1 must include at least one Infinity");
  assert.ok(SPECIAL_VALUES.some((v) => Number.isNaN(v)));
  const rng2 = makeRng(0x5343414c5f47454e00n);
  const data = genDataSpecial(rng2, [200], 1);
  assert.ok(Array.from(data).some((v) => Number.isNaN(v)));
});

// --- diagnostic QUALITY pin (Verify-B finding F1, W2 verify round) ---------
// The W2 overload conversion of add/sub/mul/div made the DECLARATION ORDER
// of the two overloads load-bearing: on a failed overload set, tsc surfaces
// the error of the LAST candidate. With the generic Guard-carrying overload
// declared last, a plain broadcast mismatch shows the shape-naming
// `__shapeError` message (nested under the unavoidable TS2769 header); with
// the order flipped, the message VANISHES behind the scalar decoy ("not
// assignable to parameter of type 'number'") — and every `@ts-expect-error`
// pin stays green, because those only assert "some error here", never the
// message content. This test closes that gap: it runs the real compiler on
// a throwaway fixture (OUTSIDE the repo, so the deliberately-broken code
// never joins any type corpus) and asserts the message CONTENT.
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

test("diagnostic quality (F1 pin): broadcast mismatch surfaces the shape-naming message through the overload set", () => {
  const dir = mkdtempSync(join(tmpdir(), "numtype-diag-pin-"));
  try {
    const ndarrayPath = fileURLToPath(new URL("../src/ndarray.ts", import.meta.url).href);
    const ambientPath = fileURLToPath(new URL("../src/ambient.d.ts", import.meta.url).href);
    const repoRoot = fileURLToPath(new URL("../..", import.meta.url).href);
    writeFileSync(
      join(dir, "probe.ts"),
      `import { NDArray } from ${JSON.stringify(ndarrayPath)};\n` +
        `const a = NDArray.zeros([2, 3]);\n` +
        `const b = NDArray.zeros([4]);\n` +
        `a.add(b); // deliberate mismatch — must surface the broadcast message\n` +
        `a.div(2); // scalar overload — must stay clean\n`,
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
    assert.ok(
      out.includes("cannot broadcast shapes [2,3] and [4]"),
      `the shape-naming broadcast message must survive overload resolution (F1 regression):\n${out}`,
    );
    const probeErrors = out.split("\n").filter((l) => l.includes("probe.ts(") && l.includes("error TS"));
    assert.strictEqual(
      probeErrors.length,
      1,
      `expected exactly ONE fixture error (the bad add; div(2) must resolve cleanly to the scalar overload):\n${out}`,
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// =============================================================================
// Op-Scheibe W3 (docs/op-w3-sqrt-spec.md): NDArray.sqrt() / sqrtRuntime.
// APPENDED block (D4) — see the file-header comment above for the coverage
// summary. Imports needed only by this block are added right here, not
// hoisted to the top, to keep the append visually self-contained.
// =============================================================================

import { sqrtRuntime } from "../src/runtime.ts";

// --- Section W3.1: explicit D2 IEEE-edge matrix -----------------------------

test("sqrt(): D2 IEEE-edge matrix — -0, NaN, negative->NaN, +/-Infinity, size-0", () => {
  // sqrt(-0) === -0 (Object.is-distinguished IEEE edge, not +0).
  const negZero = NDArray.fromArray([1], [-0]);
  assert.ok(Object.is(negZero.sqrt().data[0], -0), `sqrt(-0) must be -0, got ${negZero.sqrt().data[0]}`);

  // sqrt(NaN) -> NaN.
  const nanArr = NDArray.fromArray([1], [Number.NaN]);
  assert.ok(Number.isNaN(nanArr.sqrt().data[0]), "sqrt(NaN) must be NaN");

  // sqrt(-x) -> NaN for finite x > 0.
  const negatives = NDArray.fromArray([4], [-1, -2.5, -Number.MAX_VALUE, -Number.MIN_VALUE]);
  for (const v of Array.from(negatives.sqrt().data)) {
    assert.ok(Number.isNaN(v), `sqrt of a negative finite must be NaN, got ${v}`);
  }

  // sqrt(+Infinity) -> +Infinity; sqrt(-Infinity) -> NaN (IEEE: no real root).
  const infs = NDArray.fromArray([2], [Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]);
  const infResult = infs.sqrt();
  assert.strictEqual(infResult.data[0], Number.POSITIVE_INFINITY, "sqrt(+Infinity) must be +Infinity");
  assert.ok(Number.isNaN(infResult.data[1]), "sqrt(-Infinity) must be NaN");

  // size-0 -> empty output, no throw.
  const empty = NDArray.fromArray([0], []);
  const emptySqrt = empty.sqrt();
  assertShapeEqual([0], emptySqrt.shape, "sqrt() of a size-0 receiver: shape");
  assert.strictEqual(emptySqrt.data.length, 0, "sqrt() of a size-0 receiver: data length");

  // Direct runtime-function-level pin too (not just through the class).
  assert.strictEqual(sqrtRuntime(new Float64Array(0)).length, 0);
  assert.ok(Object.is(sqrtRuntime(Float64Array.from([-0]))[0], -0));
});

test("sqrt(): subnormal inputs pass through Math.sqrt exactly — bit-compared (Object.is + explicit bit pattern) against a direct Math.sqrt reference", () => {
  const subnormals = [Number.MIN_VALUE, Number.MIN_VALUE * 4, -Number.MIN_VALUE, -Number.MIN_VALUE * 4];
  const nd = NDArray.fromArray([subnormals.length], subnormals);
  const actual = nd.sqrt();
  for (let i = 0; i < subnormals.length; i++) {
    const expected = Math.sqrt(subnormals[i] ?? 0);
    assert.ok(Object.is(expected, actual.data[i]), `subnormal ${subnormals[i]}: expected ${expected}, got ${actual.data[i]}`);
    // Explicit bit-pattern comparison via DataView (D2's own wording):
    // the op's definition IS Math.sqrt, so this proves faithful pass-through
    // of the exact IEEE-754 bit pattern, not independent mathematics.
    const expectedBuf = new DataView(new ArrayBuffer(8));
    expectedBuf.setFloat64(0, expected);
    const actualBuf = new DataView(new ArrayBuffer(8));
    actualBuf.setFloat64(0, actual.data[i] ?? Number.NaN);
    assert.strictEqual(actualBuf.getBigUint64(0), expectedBuf.getBigUint64(0), `subnormal ${subnormals[i]}: bit pattern mismatch`);
  }
});

// --- Section W3.2: rank 0/1/2 + shape preservation, transposed/sliced receivers ---

test("sqrt(): shape preservation at rank 0/1/2, exact values", () => {
  const r0 = NDArray.fromArray([], [16]);
  assertShapeEqual([], r0.sqrt().shape, "sqrt() rank-0 shape");
  assert.strictEqual(r0.sqrt().data[0], 4);

  const r1 = NDArray.fromArray([4], [0, 1, 4, 9]);
  assertShapeEqual([4], r1.sqrt().shape, "sqrt() rank-1 shape");
  assert.deepStrictEqual(Array.from(r1.sqrt().data), [0, 1, 2, 3]);

  const r2 = NDArray.fromArray([2, 2], [1, 4, 9, 16]);
  assertShapeEqual([2, 2], r2.sqrt().shape, "sqrt() rank-2 shape");
  assert.deepStrictEqual(Array.from(r2.sqrt().data), [1, 2, 3, 4]);
});

test("sqrt(): transposed and sliced receivers stay correct (non-contiguous-in-origin data, contiguous NDArray.data by construction)", () => {
  const base = NDArray.fromArray([2, 3], [1, 4, 9, 16, 25, 36]);
  const transposed = base.transpose(); // NDArray<[3, 2]>, fresh copy per transposeRuntime
  assertShapeEqual([3, 2], transposed.shape, "transpose() shape");
  assert.deepStrictEqual(Array.from(transposed.sqrt().data), Array.from(transposed.data).map((v) => Math.sqrt(v)));

  const sliced = base.slice(...wideSpecs(1)); // NDArray<[3]>, row 1: [16, 25, 36]
  assertShapeEqual([3], sliced.shape, "slice(1) shape");
  assert.deepStrictEqual(Array.from(sliced.sqrt().data), [4, 5, 6]);
});

// --- Section W3.3: randomized bit-differential against a direct Math.sqrt loop ---

{
  const rng = makeRng(0x53515254205733n); // "SQRT W3"
  const CASE_COUNT = 220; // >= 200 per spec D4
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genDataSpecial(rng, shape, 0.3);

    test(`sqrt() case ${c}: bit-differential against a direct Math.sqrt loop, shape=[${shape.join(",")}]`, () => {
      const nd = NDArray.fromArray(shape, data);
      const actual = nd.sqrt();
      const expected = Float64Array.from(data, (v) => Math.sqrt(v));
      assertShapeEqual(shape, actual.shape, `case ${c}: shape must be preserved exactly`);
      assertDataBitIdentical(expected, actual.data, `case ${c} sqrt`);
    });
  }
}

// --- Section W3.4: F1-closure retro-proof — the chain
// `mul -> sum(1) -> sqrt -> reshape -> div` byte-identical to the OLD
// hand-loop formulation from examples/rag-demo/main.ts (Friction F1). ------

test("sqrt(): F1 closure — m.mul(m).sum(1).sqrt() byte-identical to the old hand-loop (Math.sqrt over .data + fromArray)", () => {
  const rng = makeRng(0x46315f434c4f5345n); // "F1_CLOSE"
  const N = 6;
  const D = 5;
  const data = genDataSpecial(rng, [N, D], 0.1).map((v) => (Number.isNaN(v) || !Number.isFinite(v) ? rng.nextF64() : v));
  const m = NDArray.fromArray([N, D], data);

  // New chain (W3): fully inside NDArray.
  const viaChain = m.mul(m).sum(1).sqrt();
  assertShapeEqual([N], viaChain.shape, "chain shape");

  // OLD hand-loop, verbatim shape (examples/rag-demo/main.ts lines ~46-61,
  // pre-W3): mul -> sum(1) -> drop to .data -> Math.sqrt by hand -> fromArray.
  const squared = m.mul(m);
  const sumSquares = squared.sum(1);
  const rowNormsData = new Float64Array(sumSquares.data.length);
  for (let i = 0; i < rowNormsData.length; i++) rowNormsData[i] = Math.sqrt(sumSquares.data[i] ?? 0);
  const viaHandLoop = NDArray.fromArray([N], rowNormsData);

  assertShapeEqual(viaHandLoop.shape, viaChain.shape, "F1 closure: chain vs hand-loop shape");
  assertDataBitIdentical(viaHandLoop.data, viaChain.data, "F1 closure: chain vs hand-loop data");
});

test("sqrt(): F1 closure, full L2 normalization — m.div(m.mul(m).sum(1).sqrt().reshape([N,1])) byte-identical to the rag-demo hand-loop formulation", () => {
  const rng = makeRng(0x4c324e4f524d0000n); // "L2NORM"
  const N = 5;
  const D = 4;
  // Keep values away from 0 to avoid a genuinely-zero row norm (which would
  // make BOTH formulations agree on NaN anyway, but strictly positive rows
  // are the representative, non-degenerate case this closure proof targets).
  const data = genData(rng, [N, D]).map((v) => Math.abs(v) + 0.1);
  const m = NDArray.fromArray([N, D], data);

  // New chain (W3): the exact rag-demo-intended NumPy idiom, fully in NDArray.
  const viaChain = m.div(m.mul(m).sum(1).sqrt().reshape([N, 1]));
  assertShapeEqual([N, D], viaChain.shape, "L2-normalize chain shape");

  // OLD hand-loop formulation (examples/rag-demo/main.ts lines ~46-65,
  // pre-W3, verbatim structure): mul -> sum(1) -> hand Math.sqrt loop ->
  // fromArray -> reshape([N,1]) -> div.
  const squared = m.mul(m);
  const sumSquares = squared.sum(1);
  const rowNormsData = new Float64Array(sumSquares.data.length);
  for (let i = 0; i < rowNormsData.length; i++) rowNormsData[i] = Math.sqrt(sumSquares.data[i] ?? 0);
  const rowNorms = NDArray.fromArray([N], rowNormsData);
  const rowNormsCol = rowNorms.reshape([N, 1]);
  const viaHandLoop = m.div(rowNormsCol);

  assertShapeEqual(viaHandLoop.shape, viaChain.shape, "L2-normalize closure: chain vs hand-loop shape");
  assertDataBitIdentical(viaHandLoop.data, viaChain.data, "L2-normalize closure: chain vs hand-loop data");
});

// --- Section W3.5: non-vacuity smoke -----------------------------------

test("sqrt(): plain smoke case, sqrt(2) at rank 0", () => {
  const nd = NDArray.fromArray([], [2]);
  assert.strictEqual(nd.sqrt().data[0], Math.SQRT2);
});

// --- W3 verify-round closures (Verify-B findings F1 + F2) ------------------

test("sqrt: returns a FRESH buffer and never mutates the receiver (aliasing isolation, Verify-B F1)", () => {
  const receiver = NDArray.fromArray([2, 2], [4, 9, 16, 25]);
  const before = Float64Array.from(receiver.data);
  const result = receiver.sqrt();
  assert.notStrictEqual(result.data, receiver.data, "sqrt must allocate a new buffer, never alias the receiver's");
  assertDataBitIdentical(receiver.data, before, "receiver data must be untouched by sqrt");
  assertDataBitIdentical(result.data, Float64Array.from([2, 3, 4, 5]), "result carries the roots");
});

test("sqrt: LARGEST subnormal passes through bit-exactly (Verify-B F2)", () => {
  // 0x000fffffffffffff — the largest subnormal, directly below the normal
  // boundary; the smallest ones are already pinned above, this closes the
  // other end of the subnormal range.
  const buf = new ArrayBuffer(8);
  const dv = new DataView(buf);
  dv.setBigUint64(0, 0x000fffffffffffffn, false);
  const largestSubnormal = dv.getFloat64(0, false);
  const arr = NDArray.fromArray([1], [largestSubnormal]).sqrt();
  const expected = Math.sqrt(largestSubnormal);
  assertDataBitIdentical(arr.data, Float64Array.from([expected]), "largest subnormal must round-trip through sqrtRuntime bit-exactly");
});

// =============================================================================
// Op-Scheibe W4 (docs/op-w4-stack-spec.md): NDArray.stack(rows) / stackRuntime.
// APPENDED block (D5) — see the file-header comment above for the coverage
// summary. Imports needed only by this block are added right here, not
// hoisted to the top, to keep the append visually self-contained (W3's own
// convention).
// =============================================================================

import { bitsOf } from "./assert-helpers.ts";
import { stackRuntime } from "../src/runtime.ts";

// --- Section W4.1: stem-equality pins (real throws vs. expected strings, D3) ---

test("stack(): all three stems are word-for-word pinned via stackRuntime directly", () => {
  assert.throws(() => stackRuntime([]), /^Error: stack: expected at least one row$/);

  assert.throws(
    () =>
      stackRuntime([
        { shape: [2, 3], data: new Float64Array(6) },
        { shape: [3], data: new Float64Array(3) },
      ]),
    /^Error: stack: expected 1-D rows \(got shape \[2,3\] at index 0\)$/,
  );

  assert.throws(
    () =>
      stackRuntime([
        { shape: [3], data: new Float64Array(3) },
        { shape: [4], data: new Float64Array(4) },
      ]),
    /^Error: stack: row length mismatch \(expected 3, got 4 at index 1\)$/,
  );
});

test("stack(): the same three stems are reachable through the public NDArray.stack API via dynamic-rank rows (the mean(5)-style widen-past-the-guard technique — no unsafe cast needed)", () => {
  // Empty: a genuinely dynamic-length runtime ARRAY (never a `[]` TUPLE
  // LITERAL, which is a compile-time rejection, F3 — proven separately by
  // the @ts-expect-error pin in ndarray.test-d.ts). Built via an explicitly
  // annotated `NDArray<number[]>[]` variable so `NDArray.stack` sees a real
  // (non-tuple) array type and the D2 runtime backstop is reachable through
  // the actual class method.
  const emptyRows: NDArray<number[]>[] = [];
  assert.throws(() => NDArray.stack(emptyRows), /^Error: stack: expected at least one row$/);

  // Rank != 1: two DYNAMIC-RANK rows (built from plain, non-`const` shape
  // variables — `mean(5)`'s own out-of-range-axis pin above uses the exact
  // same trick), one of them actually rank 2 at runtime.
  const rank2Shape = [2, 3];
  const rank1Shape = [3];
  const badRank0 = NDArray.fromArray(rank2Shape, [1, 2, 3, 4, 5, 6]);
  const badRank1 = NDArray.fromArray(rank1Shape, [1, 2, 3]);
  assert.throws(() => NDArray.stack([badRank0, badRank1]), /^Error: stack: expected 1-D rows \(got shape \[2,3\] at index 0\)$/);

  // Length mismatch: two dynamic-rank rows with genuinely different lengths.
  const lenAShape = [3];
  const lenBShape = [4];
  const goodA = NDArray.fromArray(lenAShape, [1, 2, 3]);
  const goodB = NDArray.fromArray(lenBShape, [1, 2, 3, 4]);
  assert.throws(() => NDArray.stack([goodA, goodB]), /^Error: stack: row length mismatch \(expected 3, got 4 at index 1\)$/);
});

// --- Section W4.2: successful 1/2/3-row cases + D=0 -------------------------

test("stack(): 1/2/3-row successful cases — shape + values, row-major order preserved", () => {
  const r1 = NDArray.fromArray([3], [1, 2, 3]);
  const single = NDArray.stack([r1]);
  assertShapeEqual([1, 3], single.shape, "1-row stack shape");
  assert.deepStrictEqual(Array.from(single.data), [1, 2, 3]);

  const a = NDArray.fromArray([3], [1, 2, 3]);
  const b = NDArray.fromArray([3], [4, 5, 6]);
  const two = NDArray.stack([a, b]);
  assertShapeEqual([2, 3], two.shape, "2-row stack shape");
  assert.deepStrictEqual(Array.from(two.data), [1, 2, 3, 4, 5, 6]);

  const c = NDArray.fromArray([3], [7, 8, 9]);
  const three = NDArray.stack([a, b, c]);
  assertShapeEqual([3, 3], three.shape, "3-row stack shape");
  assert.deepStrictEqual(Array.from(three.data), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
});

test("stack(): D=0 rows are valid — [[],[]] stacks to shape [2, 0], empty data", () => {
  const z1 = NDArray.fromArray([0], []);
  const z2 = NDArray.fromArray([0], []);
  const stacked = NDArray.stack([z1, z2]);
  assertShapeEqual([2, 0], stacked.shape, "D=0 stack shape");
  assert.strictEqual(stacked.data.length, 0);

  const direct = stackRuntime([
    { shape: [0], data: new Float64Array(0) },
    { shape: [0], data: new Float64Array(0) },
  ]);
  assertShapeEqual([2, 0], direct.shape, "stackRuntime D=0 shape");
  assert.strictEqual(direct.data.length, 0);
});

// --- Section W4.3: NaN-payload byte-exactness (movement op) -----------------
//
// Mirrors special-values.test.ts's own transpose fixture: a NaN with a
// NON-canonical payload must survive stack's pure row-major copy BYTE-EXACT,
// not merely "still NaN" — checked via `bitsOf`, not `Object.is`/`isNaN`
// (which treat every NaN as equal regardless of payload).

function nonCanonicalNaNW4(): number {
  const bits = new BigUint64Array([0x7ff8_0000_cafe_baben]);
  return new Float64Array(bits.buffer)[0] ?? Number.NaN;
}

test("stack(): a non-canonical NaN payload survives the row-major copy byte-exact (bitsOf, not Object.is)", () => {
  const nan = nonCanonicalNaNW4();
  assert.ok(Number.isNaN(nan), "sanity: constructed value must actually be NaN");
  const nanBits = bitsOf(nan);
  assert.strictEqual(nanBits, 0x7ff8_0000_cafe_baben, "sanity: constructed NaN must carry the intended non-canonical payload bits");

  const row0 = NDArray.fromArray([3], [nan, 1, 2]);
  const row1 = NDArray.fromArray([3], [3, 4, 5]);
  const stacked = NDArray.stack([row0, row1]);
  assert.strictEqual(bitsOf(stacked.data[0] ?? 0), nanBits, "stack must preserve the exact NaN payload at flat index 0");

  const direct = stackRuntime([
    { shape: [3], data: Float64Array.from([nan, 1, 2]) },
    { shape: [3], data: Float64Array.from([3, 4, 5]) },
  ]);
  assert.strictEqual(bitsOf(direct.data[0] ?? 0), nanBits, "stackRuntime must preserve the exact NaN payload at flat index 0");
});

// --- Section W4.4: F5 closure retro-proof ------------------------------------
//
// examples/rag-demo/embedding.ts's `embedMatrix` (the F5 friction,
// docs/dogfooding-rag-ergebnisse.md) is REBUILT locally, not imported — that
// package is a separate npm-registry-consuming example with its own
// package.json/tsconfig, deliberately outside the spike/ compilation graph
// (importing it would pull an extra file into `check:diag`'s corpus, adding
// order-noise the D6 measurement discipline forbids). The rebuild is
// `embedMatrix`'s own algorithm verbatim: `Float64Array#set` at each row's
// offset.

function rebuiltEmbedMatrix(rows: readonly Float64Array[], dims: number): Float64Array {
  const flat = new Float64Array(rows.length * dims);
  rows.forEach((row, i) => {
    flat.set(row, i * dims);
  });
  return flat;
}

test("stack(): F5 closure — byte-identical to embedMatrix's own row-major Float64Array#set-at-offset algorithm, rebuilt locally", () => {
  const rng = makeRng(0x53544143_4b5f4635n); // "STACK_F5"
  const dims = 12;
  const rowCount = 7;
  const rowsData: Float64Array[] = [];
  for (let i = 0; i < rowCount; i++) rowsData.push(genData(rng, [dims]));

  const viaRebuiltHelper = rebuiltEmbedMatrix(rowsData, dims);
  const ndRows = rowsData.map((d) => NDArray.fromArray([dims], d));
  const viaStack = NDArray.stack(ndRows);

  assertShapeEqual([rowCount, dims], viaStack.shape, "F5 closure: stack shape must be [rowCount, dims]");
  assertDataBitIdentical(viaRebuiltHelper, viaStack.data, "F5 closure: stack data must be byte-identical to the rebuilt embedMatrix algorithm");
});

// --- Section W4.5: large-N smoke ---------------------------------------------

test("stack(): large-N smoke — 5000 rows of dimension 8, shape + spot-checked values", () => {
  const rowCount = 5000;
  const dims = 8;
  const rng = makeRng(0x4c415247455f4e0an); // "LARGE_N"
  const ndRows: NDArray<[8]>[] = [];
  const expectedRows: Float64Array[] = [];
  for (let i = 0; i < rowCount; i++) {
    const d = genData(rng, [dims]);
    expectedRows.push(d);
    ndRows.push(NDArray.fromArray([dims], d));
  }
  const stacked = NDArray.stack(ndRows);
  assertShapeEqual([rowCount, dims], stacked.shape, "large-N stack shape");
  for (const idx of [0, Math.floor(rowCount / 2), rowCount - 1]) {
    const expectedRow = expectedRows[idx] ?? new Float64Array(dims);
    for (let j = 0; j < dims; j++) {
      assert.strictEqual(stacked.data[idx * dims + j], expectedRow[j], `row ${idx}, col ${j} mismatch`);
    }
  }
});

// --- Section W4.6: aliasing isolation / buffer freshness (W3 lesson) --------

test("stack: result.data is a FRESH buffer, never aliasing any row's own data — rows stay unmutated (aliasing isolation, W3 lesson)", () => {
  const a = NDArray.fromArray([3], [1, 2, 3]);
  const b = NDArray.fromArray([3], [4, 5, 6]);
  const beforeA = Float64Array.from(a.data);
  const beforeB = Float64Array.from(b.data);
  const stacked = NDArray.stack([a, b]);

  assert.notStrictEqual(stacked.data, a.data, "stacked.data must not alias row a's data");
  assert.notStrictEqual(stacked.data, b.data, "stacked.data must not alias row b's data");
  assertDataBitIdentical(a.data, beforeA, "row a's data must be untouched by stack");
  assertDataBitIdentical(b.data, beforeB, "row b's data must be untouched by stack");

  stacked.data[0] = 999;
  assert.notStrictEqual(a.data[0], 999, "mutating the stack result must not affect row a's data (no shared buffer)");
});

// =============================================================================
// Op-Scheibe W5 (docs/op-w5-item-spec.md): NDArray.item(...indices) /
// itemRuntime. See this file's own header comment for the coverage summary.
// =============================================================================

// --- Section W5.1: rank 0/1/2/3 valid full-indexing reads -------------------

test("item(): rank 0 — item() (zero arguments) reads the sole element", () => {
  const scalar = NDArray.fromArray([], [7]);
  assert.strictEqual(scalar.item(), 7, "rank-0 item() must read the sole element");
});

test("item(): rank 1 — item(i) matches .data[i]", () => {
  const v = NDArray.fromArray([4], [10, 20, 30, 40]);
  for (let i = 0; i < 4; i++) {
    assert.strictEqual(v.item(i), v.data[i], `rank-1 item(${i}) must equal .data[${i}]`);
  }
});

test("item(): rank 2 — item(i, j) matches row-major flat offset", () => {
  const m = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 3; j++) {
      assert.strictEqual(m.item(i, j), m.data[i * 3 + j], `item(${i},${j}) must equal .data[${i * 3 + j}]`);
    }
  }
});

test("item(): rank 3 — item(i, j, k) matches row-major flat offset", () => {
  const rng = makeRng(0x4954454d5f523300n); // "ITEM_R3"
  const shape = [2, 3, 4];
  const data = genData(rng, shape);
  const t = NDArray.fromArray(shape, data);
  const strides = computeStrides(shape);
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 4; k++) {
        const flat = i * (strides[0] ?? 0) + j * (strides[1] ?? 0) + k * (strides[2] ?? 0);
        assert.strictEqual(t.item(i, j, k), t.data[flat], `item(${i},${j},${k}) must equal .data[${flat}]`);
      }
    }
  }
});

// --- Section W5.2: negative indices (NumPy parity) ---------------------------

test("item(): negative indices normalize NumPy-style (i < 0 -> i + d)", () => {
  const m = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
  assert.strictEqual(m.item(-1, -1), m.item(1, 2), "item(-1,-1) must equal item(1,2)");
  assert.strictEqual(m.item(-2, -3), m.item(0, 0), "item(-2,-3) must equal item(0,0)");
  assert.strictEqual(m.item(-1, 0), m.item(1, 0), "item(-1,0) must equal item(1,0)");
  assert.strictEqual(m.item(0, -1), m.item(0, 2), "item(0,-1) must equal item(0,2)");
});

// --- Section W5.3: throw stems, word-for-word, direct + public API ----------

test("itemRuntime(): arity stem — expected N indices (got M)", () => {
  const shape = [2, 3];
  const data = Float64Array.from([1, 2, 3, 4, 5, 6]);
  assert.throws(() => itemRuntime(shape, data, [0]), /^Error: item: expected 2 indices \(got 1\)$/, "under-arity stem");
  assert.throws(() => itemRuntime(shape, data, [0, 0, 0]), /^Error: item: expected 2 indices \(got 3\)$/, "over-arity stem");
  assert.throws(() => itemRuntime([], data, [0]), /^Error: item: expected 0 indices \(got 1\)$/, "rank-0 over-arity stem");
});

test("item(): public API arity throws reach itemRuntime's own stem (dynamic-rank widened receiver)", () => {
  // Widen past the compile-time guard: a plain (non-const, non-literal-typed)
  // variable receiver has a dynamic `number[]` shape at the type layer, the
  // same "widen past the guard" technique mean(5)'s own out-of-range-axis
  // pin (W2 section above) already uses.
  const shape: number[] = [2, 3];
  const m = NDArray.fromArray(shape, [1, 2, 3, 4, 5, 6]) as unknown as NDArray<number[]>;
  assert.throws(
    () => (m as unknown as { item: (...i: number[]) => number }).item(0),
    /^Error: item: expected 2 indices \(got 1\)$/,
    "public item() arity throw must reach itemRuntime's own stem",
  );
});

test("itemRuntime(): not-an-integer stem — index X for axis A is not an integer", () => {
  const shape = [2, 3];
  const data = Float64Array.from([1, 2, 3, 4, 5, 6]);
  assert.throws(() => itemRuntime(shape, data, [0.5, 0]), /^Error: item: index 0\.5 for axis 0 is not an integer$/, "axis-0 not-integer stem");
  assert.throws(() => itemRuntime(shape, data, [0, 1.5]), /^Error: item: index 1\.5 for axis 1 is not an integer$/, "axis-1 not-integer stem");
});

test("item(): public API not-an-integer throw reaches itemRuntime's own stem (dynamic-index widened receiver)", () => {
  const m = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
  const badIndex: number = 0.5; // widened past the literal-index compile-time guard
  assert.throws(
    () => m.item(badIndex, 0),
    /^Error: item: index 0\.5 for axis 0 is not an integer$/,
    "public item() not-integer throw must reach itemRuntime's own stem",
  );
});

test("itemRuntime(): out-of-bounds stem — index X out of bounds for axis A with dim D", () => {
  const shape = [2, 3];
  const data = Float64Array.from([1, 2, 3, 4, 5, 6]);
  assert.throws(() => itemRuntime(shape, data, [2, 0]), /^Error: item: index 2 is out of bounds for axis 0 with dim 2$/, "axis-0 positive OOB stem");
  assert.throws(() => itemRuntime(shape, data, [0, 3]), /^Error: item: index 3 is out of bounds for axis 1 with dim 3$/, "axis-1 positive OOB stem");
  assert.throws(() => itemRuntime(shape, data, [-3, 0]), /^Error: item: index -3 is out of bounds for axis 0 with dim 2$/, "axis-0 negative OOB stem");
  assert.throws(() => itemRuntime(shape, data, [0, -4]), /^Error: item: index -4 is out of bounds for axis 1 with dim 3$/, "axis-1 negative OOB stem");
});

test("item(): public API out-of-bounds throw reaches itemRuntime's own stem (dynamic-index widened receiver)", () => {
  const m = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
  const badIndex: number = 2; // widened past the literal-index compile-time guard
  assert.throws(
    () => m.item(badIndex, 0),
    /^Error: item: index 2 is out of bounds for axis 0 with dim 2$/,
    "public item() out-of-bounds throw must reach itemRuntime's own stem",
  );
});

// --- Section W5.4: size-0-dim OOB — every index is unreachable ---------------

test("itemRuntime(): size-0 dim — EVERY index is out of bounds, no special case needed", () => {
  const shape = [2, 0, 3];
  const data = new Float64Array(0);
  for (const idx of [-1, 0, 1]) {
    assert.throws(
      () => itemRuntime(shape, data, [0, idx, 0]),
      /^Error: item: index -?\d+ is out of bounds for axis 1 with dim 0$/,
      `size-0 axis must reject index ${idx}`,
    );
  }
});

// --- Section W5.5: NaN / -0 byte-exact pass-through --------------------------

test("item(): NaN payload bits and -0 pass through exactly (direct read, no arithmetic)", () => {
  const nan = Float64Array.from([Number.NaN])[0] ?? Number.NaN;
  const nanBits = bitsOf(nan);
  const negZero = -0;
  const m = NDArray.fromArray([3], [nan, negZero, 1]);
  assert.strictEqual(bitsOf(m.item(0)), nanBits, "item(0) must preserve the exact NaN payload bits");
  assert.strictEqual(Object.is(m.item(1), -0), true, "item(1) must be Object.is-distinguished -0, not +0");
  assert.strictEqual(m.item(2), 1, "item(2) sanity check");
});

// --- Section W5.6: 200+-case randomized flat-index differential -------------

test("item(): 200+ randomized cases match independently-computed row-major flat-offset arithmetic", () => {
  const rng = makeRng(0x4954454d5f444946n); // "ITEM_DIF"
  let cases = 0;
  const shapes: readonly (readonly number[])[] = [[], [5], [2, 3], [4, 2, 3], [2, 2, 2, 2]];
  for (const shape of shapes) {
    const data = genData(rng, shape);
    const nd = NDArray.fromArray(shape, data);
    const strides = computeStrides(shape);
    if (shape.some((d) => d === 0)) continue; // size-0 axis: always OOB, covered separately above
    for (let trial = 0; trial < 50; trial++) {
      const useNegative = rng.nextBool();
      const idxNums = shape.map((d) => {
        const positive = rng.nextInt(0, d - 1);
        return useNegative && rng.nextBool() ? positive - d : positive;
      });
      let expectedFlat = 0;
      for (let axis = 0; axis < shape.length; axis++) {
        const raw = idxNums[axis] ?? 0;
        const d = shape[axis] ?? 0;
        const normalized = raw < 0 ? raw + d : raw;
        expectedFlat += normalized * (strides[axis] ?? 0);
      }
      const expected = data[expectedFlat] ?? Number.NaN;
      const actual = nd.item(...idxNums);
      assert.strictEqual(actual, expected, `case ${cases} shape=[${shape.join(",")}] idx=[${idxNums.join(",")}]`);
      cases++;
    }
  }
  assert.ok(cases >= 200, `must run at least 200 differential cases (ran ${cases})`);
});

// --- Section W5.7: transposed / sliced receivers (real strided reads) -------

test("item(): transposed receiver — reads agree across a distinct materialized layout", () => {
  const m = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
  const t = m.transpose(); // shape [3, 2]
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 2; j++) {
      assert.strictEqual(t.item(i, j), m.item(j, i), `transposed item(${i},${j}) must equal original item(${j},${i})`);
    }
  }
});

test("item(): sliced receiver — item reads into the freshly-copied slice buffer, not the parent's", () => {
  const m = NDArray.fromArray([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const row = m.slice(...wideSpecs(1)); // shape [3]: [4, 5, 6]
  assert.strictEqual(row.item(0), 4, "sliced row item(0)");
  assert.strictEqual(row.item(1), 5, "sliced row item(1)");
  assert.strictEqual(row.item(2), 6, "sliced row item(2)");
  assert.strictEqual(row.item(-1), 6, "sliced row item(-1)");
});

// =============================================================================
// dt1 (docs/dtype-dt1-spec.md): dtype core on `NDArray` — storage, creation,
// conversion, and the dtype-neutral movement ops. Same no-new-file house
// convention as W2-W5 above (NDArray-only, no WNDArray/WASM counterpart in
// this slice — D9: the naive TS reference carries every dtype first, kernels
// follow per dtype in later slices, M1 v5 kernel-less-reference tracking in
// FOLLOWUPS). Appended at the end of this file.
// =============================================================================
import { type AnyNDArray, type NestedBoolValue, type NestedValue } from "../src/ndarray.ts";
import {
  astypeConvert,
  BOOL_ARITHMETIC_MESSAGE,
  convertToDType,
  lockedOpMessage,
  normalizeSliceSpecs,
  onesData,
  sameKindArray,
  sliceDtyped,
  transposeDtyped,
  type DType,
  zerosData,
} from "../src/runtime.ts";
import { copySameKindArray } from "../src/ndarray.ts";
import { naturalStrides } from "./assert-helpers.ts";

const DTYPES: readonly DType[] = ["float64", "float32", "int32", "bool"];
const DATA_CTOR: Record<DType, Float64ArrayConstructor | Float32ArrayConstructor | Int32ArrayConstructor | Uint8ArrayConstructor> = {
  float64: Float64Array,
  float32: Float32Array,
  int32: Int32Array,
  bool: Uint8Array,
};

/** A RegExp matching `lockedOpMessage(op, dtype)` verbatim — built from the
 * SAME function `runtime.ts` exports (never re-derived by hand), so this
 * test can never silently drift from the actual thrown message. */
function lockedMsgRegex(op: string, dtype: DType): RegExp {
  return new RegExp(lockedOpMessage(op, dtype).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
}

/** dt2 (P4): a RegExp matching `BOOL_ARITHMETIC_MESSAGE` verbatim — built
 * from the SAME constant `runtime.ts` exports (never re-derived by hand),
 * for the PERMANENT bool-arithmetic rejection (every op, both overloads),
 * as opposed to `lockedMsgRegex`'s dt1 TRANSITIONAL message above (still
 * used by the ops that stay locked until dt5: argmax/topk/sqrt/stack). */
const boolMsgRegex = new RegExp(BOOL_ARITHMETIC_MESSAGE.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));

// --- Section dt1.1: construction + round trip per dtype ---------------------

test("NDArray.zeros/ones: every dtype allocates the right typed-array class, filled correctly", () => {
  for (const dtype of DTYPES) {
    const z = NDArray.zeros([2, 3], dtype);
    const o = NDArray.ones([2, 3], dtype);
    assert.strictEqual(z.dtype, dtype, `zeros dtype tag [${dtype}]`);
    assert.strictEqual(o.dtype, dtype, `ones dtype tag [${dtype}]`);
    assert.ok(z.data instanceof DATA_CTOR[dtype], `zeros[${dtype}] data must be ${DATA_CTOR[dtype].name}`);
    assert.ok(o.data instanceof DATA_CTOR[dtype], `ones[${dtype}] data must be ${DATA_CTOR[dtype].name}`);
    assert.deepStrictEqual([...z.shape], [2, 3], `zeros[${dtype}] shape`);
    for (let i = 0; i < 6; i++) {
      assert.strictEqual(z.data[i], 0, `zeros[${dtype}] element ${i}`);
      assert.strictEqual(o.data[i], 1, `ones[${dtype}] element ${i}`);
    }
  }
});

test("NDArray.zeros/ones: no dtype argument defaults to float64 (unchanged pre-dtype behavior)", () => {
  const z = NDArray.zeros([2, 2]);
  const o = NDArray.ones([2, 2]);
  assert.strictEqual(z.dtype, "float64");
  assert.strictEqual(o.dtype, "float64");
  assert.ok(z.data instanceof Float64Array);
  assert.ok(o.data instanceof Float64Array);
});

test("NDArray.fromArray({ dtype }): round-trips through toArray/item/toNestedArray/toJSON for every dtype", () => {
  const values = [0, 1, 1, 0, 1, 0]; // bool-safe (0/1 only) so the SAME array round-trips through every dtype, including bool's strict fromArray validation
  for (const dtype of DTYPES) {
    const nd = NDArray.fromArray([2, 3], values, { dtype });
    assert.strictEqual(nd.dtype, dtype, `[${dtype}] dtype tag`);
    assert.ok(nd.data instanceof DATA_CTOR[dtype], `[${dtype}] data class`);
    assert.ok(nd.toArray() instanceof DATA_CTOR[dtype], `[${dtype}] toArray() class`);
    assert.strictEqual(nd.toArray(), nd.data, "toArray() is the same backing store as .data (D2 alias)");

    const nested = nd.toNestedArray();
    const flatFromNested = (nested as unknown[]).flatMap((row) => row as unknown[]);
    if (dtype === "bool") {
      assert.deepStrictEqual(flatFromNested, values.map((v) => v !== 0), `[bool] toNestedArray leaves must be boolean, v!==0`);
      assert.strictEqual(typeof (flatFromNested[0] as unknown), "boolean", "[bool] leaf typeof boolean");
      assert.strictEqual(nd.item(0, 0), false, "[bool] item(0,0) is boolean false");
      assert.strictEqual(nd.item(0, 1), true, "[bool] item(0,1) is boolean true");
      assert.strictEqual(typeof nd.item(0, 0), "boolean", "[bool] item() return typeof boolean");
    } else {
      assert.deepStrictEqual(flatFromNested, values, `[${dtype}] toNestedArray leaves must be numeric, unchanged`);
      assert.strictEqual(typeof nd.item(0, 0), "number", `[${dtype}] item() return typeof number`);
    }

    const json = nd.toJSON();
    assert.deepStrictEqual(json.shape, [2, 3], `[${dtype}] toJSON shape`);
    if (dtype === "bool") {
      assert.deepStrictEqual(json.data, values.map((v) => v !== 0), "[bool] toJSON data must be boolean[]");
    } else {
      assert.deepStrictEqual(json.data, values, `[${dtype}] toJSON data must be number[], unchanged`);
    }
  }
});

test("NDArray.fromArray: Float32Array/Int32Array sources infer their dtype without an explicit option", () => {
  const f32 = NDArray.fromArray([3], new Float32Array([1.5, 2.5, 3.5]));
  const i32 = NDArray.fromArray([3], new Int32Array([1, -2, 3]));
  assert.strictEqual(f32.dtype, "float32");
  assert.ok(f32.data instanceof Float32Array);
  assert.strictEqual(i32.dtype, "int32");
  assert.ok(i32.data instanceof Int32Array);
});

test("NDArray.fromArray: a bare Uint8Array without an explicit dtype throws at runtime (D3 ambiguity backstop)", () => {
  // The type layer already rejects this at compile time (no overload accepts
  // a bare Uint8Array without `{ dtype }` — pinned in ndarray.test-d.ts); this
  // is the RUNTIME backstop for a caller that bypasses the type layer (M2).
  const bypass = NDArray.fromArray as unknown as (shape: readonly number[], values: Uint8Array) => unknown;
  assert.throws(() => bypass([3], new Uint8Array([1, 0, 1])), /Uint8Array source requires an explicit \{ dtype \} option/);
});

test("NDArray.fromArray({ dtype: 'int32' }): rejects non-integer or out-of-range values", () => {
  assert.throws(() => NDArray.fromArray([2], [1.5, 2], { dtype: "int32" }), /value 1\.5 at index 0 is not a valid int32/);
  assert.throws(() => NDArray.fromArray([2], [2147483648, 0], { dtype: "int32" }), /value 2147483648 at index 0 is not a valid int32/);
  assert.strictEqual(NDArray.fromArray([2], [2147483647, -2147483648], { dtype: "int32" }).data[0], 2147483647, "int32 max in range");
});

test("NDArray.fromArray({ dtype: 'bool' }): rejects any value other than exactly 0 or 1", () => {
  assert.throws(() => NDArray.fromArray([2], [1, 2], { dtype: "bool" }), /value 2 at index 1 is not a valid bool/);
  assert.throws(() => NDArray.fromArray([2], [0.5, 0], { dtype: "bool" }), /value 0\.5 at index 0 is not a valid bool/);
});

// --- Section dt1.2: astype conversion rules (D3, prototype probe edges) -----

test("astype('float32'): converts via Math.fround (correctly-rounded contract)", () => {
  const nd = NDArray.fromArray([2], [0.1, 1 / 3]);
  const f32 = nd.astype("float32");
  assert.strictEqual(f32.dtype, "float32");
  assert.ok(f32.data instanceof Float32Array);
  assert.strictEqual(f32.data[0], Math.fround(0.1));
  assert.strictEqual(f32.data[1], Math.fround(1 / 3));
});

test("astype('int32'): truncates toward zero, throws on NaN/Infinity/out-of-range", () => {
  const nd = NDArray.fromArray([4], [2.9, -2.9, 0.5, -0.5]);
  const i32 = nd.astype("int32");
  assert.deepStrictEqual([...i32.data], [2, -2, 0, 0], "truncation toward zero, not Math.round/floor");

  assert.throws(() => NDArray.fromArray([1], [Number.NaN]).astype("int32"), /is not finite \(NaN\/Infinity cannot convert to int32\)/);
  assert.throws(() => NDArray.fromArray([1], [Number.POSITIVE_INFINITY]).astype("int32"), /is not finite/);
  assert.throws(() => NDArray.fromArray([1], [2147483648]).astype("int32"), /is out of int32 range/);
  assert.throws(() => NDArray.fromArray([1], [-2147483649]).astype("int32"), /is out of int32 range/);
  // Edges that must NOT throw:
  assert.strictEqual(NDArray.fromArray([1], [2147483647.9]).astype("int32").data[0], 2147483647, "in-range truncation at the max edge");
  assert.strictEqual(NDArray.fromArray([1], [-2147483648]).astype("int32").data[0], -2147483648, "exact min edge");
});

test("astype('bool'): x !== 0, and NaN converts to true (determinism pin, D3/D6 precedent — never value-dependent in a way that contradicts this)", () => {
  const nd = NDArray.fromArray([4], [0, 1, -3.5, Number.NaN]);
  const b = nd.astype("bool");
  assert.strictEqual(b.dtype, "bool");
  assert.ok(b.data instanceof Uint8Array);
  assert.deepStrictEqual([...b.data], [0, 1, 1, 1], "0 -> false, everything else including NaN -> true");
  assert.strictEqual(b.item(3), true, "NaN astype(bool) -> true, read back through item() as boolean true");
});

test("astype from bool: plain 0/1 passthrough to every numeric dtype", () => {
  const boolArr = NDArray.fromArray([2], [1, 0], { dtype: "bool" });
  for (const target of ["float64", "float32", "int32"] as const) {
    const out = boolArr.astype(target);
    assert.deepStrictEqual([...out.data], [1, 0], `bool -> ${target} passthrough`);
  }
});

test("astype: always returns a fresh copy, even when the target equals the source dtype", () => {
  const nd = NDArray.fromArray([2], [1, 2], { dtype: "int32" });
  const same = nd.astype("int32");
  assert.notStrictEqual(same.data, nd.data, "astype must never alias, even for a no-op conversion");
  assert.deepStrictEqual([...same.data], [1, 2]);
});

// --- Section dt1.3: dtype-neutral movement ops (K3, D5 "D unverändert") ----
// Every case ASSERTS ITS CLASS directly after construction (rule 12): exact
// dtype tag, exact backing typed-array constructor, exact shape, exact
// (freshly-computed, always-natural since NDArray never aliases) strides.

test("transpose(): dtype-neutral for every dtype — class, shape, strides, and values all preserved/correct", () => {
  const shape = [2, 3];
  const values = [1, 0, 1, 0, 1, 0]; // bool-safe (0/1 only), same rationale as the round-trip test above
  const oracle = transposeDtyped(shape, new Float64Array(values));
  for (const dtype of DTYPES) {
    const nd = NDArray.fromArray(shape, values, { dtype });
    const t = nd.transpose();
    assert.strictEqual(t.dtype, dtype, `[${dtype}] transpose dtype tag preserved`);
    assert.ok(t.data instanceof DATA_CTOR[dtype], `[${dtype}] transpose data class preserved`);
    assert.deepStrictEqual([...t.shape], oracle.shape, `[${dtype}] transpose shape matches the oracle`);
    assert.deepStrictEqual([...t.strides], naturalStrides(oracle.shape), `[${dtype}] transpose strides are natural row-major (fresh copy)`);
    const expectedValues = dtype === "bool" ? [...oracle.data].map((v) => (v !== 0 ? 1 : 0)) : [...oracle.data];
    assert.deepStrictEqual([...t.data], expectedValues, `[${dtype}] transpose values match the float64 oracle (converted)`);
  }
});

test("slice(): dtype-neutral for every dtype — class, shape, strides, and values all preserved/correct", () => {
  const shape = [3, 3];
  const values = [1, 0, 1, 0, 1, 0, 1, 0, 1]; // bool-safe (0/1 only), same rationale as the round-trip test above
  const normSpecs = normalizeSliceSpecs(shape, [{ start: 0, stop: 2 }]);
  const oracle = sliceDtyped(shape, new Float64Array(values), normSpecs);
  for (const dtype of DTYPES) {
    const nd = NDArray.fromArray(shape, values, { dtype });
    const s = nd.slice(...wideSpecs({ start: 0, stop: 2 }));
    assert.strictEqual(s.dtype, dtype, `[${dtype}] slice dtype tag preserved`);
    assert.ok(s.data instanceof DATA_CTOR[dtype], `[${dtype}] slice data class preserved`);
    assert.deepStrictEqual([...s.shape], oracle.shape, `[${dtype}] slice shape matches the oracle`);
    assert.deepStrictEqual([...s.strides], naturalStrides(oracle.shape), `[${dtype}] slice strides are natural row-major (fresh copy)`);
    const expectedValues = dtype === "bool" ? [...oracle.data].map((v) => (v !== 0 ? 1 : 0)) : [...oracle.data];
    assert.deepStrictEqual([...s.data], expectedValues, `[${dtype}] slice values match the float64 oracle (converted)`);
  }
});

test("reshape()/flatten(): dtype-neutral inline copy for every dtype — class, shape, and values preserved", () => {
  const shape = [2, 3];
  const values = [1, 0, 1, 0, 1, 0];
  for (const dtype of DTYPES) {
    const nd = NDArray.fromArray(shape, values, { dtype });

    const reshaped = nd.reshape([3, 2]);
    assert.strictEqual(reshaped.dtype, dtype, `[${dtype}] reshape dtype tag preserved`);
    assert.ok(reshaped.data instanceof DATA_CTOR[dtype], `[${dtype}] reshape data class preserved`);
    assert.deepStrictEqual([...reshaped.shape], [3, 2], `[${dtype}] reshape shape`);
    assert.deepStrictEqual([...reshaped.data], values, `[${dtype}] reshape values unchanged (row-major preserved)`);
    assert.notStrictEqual(reshaped.data, nd.data, `[${dtype}] reshape must copy, never alias`);

    const flat = nd.flatten();
    assert.strictEqual(flat.dtype, dtype, `[${dtype}] flatten dtype tag preserved`);
    assert.ok(flat.data instanceof DATA_CTOR[dtype], `[${dtype}] flatten data class preserved`);
    assert.deepStrictEqual([...flat.shape], [6], `[${dtype}] flatten shape`);
    assert.deepStrictEqual([...flat.data], values, `[${dtype}] flatten values unchanged`);
    assert.notStrictEqual(flat.data, nd.data, `[${dtype}] flatten must copy, never alias`);
  }
});

// --- Section dt1.4: K4 locks (O2(a)) — every non-float64 receiver/argument --
// throws the word-identical runtime message for every op dt1 does not yet
// implement outside float64. `lockedOpMessage` is imported directly from
// runtime.ts (not re-derived) so this test can never silently drift from the
// actual thrown message.

const NON_FLOAT64: readonly DType[] = ["float32", "int32", "bool"];

// --- Section dt2.1 (dt2 P3/P4, docs/dtype-dt2-spec.md A2 point 1): the dt1
// scalar-form LOCK is gone for float32/int32 — add/sub/mul/div now compute,
// keeping the receiver's own dtype (D6/E4, div always widens to float — D5).
// bool stays PERMANENTLY locked (P4): the dauerhafte "use astype()" message
// (`BOOL_ARITHMETIC_MESSAGE`), never dt1's transitional `lockedOpMessage`.

test("add/sub/mul/div: scalar form now COMPUTES for float32/int32 (dt2 P3), bool stays PERMANENTLY locked (dt2 P4)", () => {
  const f32 = NDArray.fromArray([2], [2, 4], { dtype: "float32" });
  assert.strictEqual(f32.add(1).dtype, "float32");
  assert.deepStrictEqual([...f32.add(1).data], [3, 5]);
  assert.strictEqual(f32.sub(1).dtype, "float32");
  assert.deepStrictEqual([...f32.sub(1).data], [1, 3]);
  assert.strictEqual(f32.mul(2).dtype, "float32");
  assert.deepStrictEqual([...f32.mul(2).data], [4, 8]);
  assert.strictEqual(f32.div(2).dtype, "float32");
  assert.deepStrictEqual([...f32.div(2).data], [1, 2]);

  const i32 = NDArray.fromArray([2], [10, 20], { dtype: "int32" });
  assert.strictEqual(i32.add(1).dtype, "int32");
  assert.deepStrictEqual([...i32.add(1).data], [11, 21]);
  assert.strictEqual(i32.sub(1).dtype, "int32");
  assert.deepStrictEqual([...i32.sub(1).data], [9, 19]);
  assert.strictEqual(i32.mul(2).dtype, "int32");
  assert.deepStrictEqual([...i32.mul(2).data], [20, 40]);
  // div on int32 ALWAYS widens to float64 (D5), even given an integer scalar.
  assert.strictEqual(i32.div(2).dtype, "float64");
  assert.deepStrictEqual([...i32.div(2).data], [5, 10]);

  // bool: PERMANENT-lock test (A2 point 1) — every op, scalar form, the
  // dauerhafte "use astype()" message, not dt1's transitional lock.
  const boolArr = NDArray.fromArray([2], [1, 0], { dtype: "bool" });
  for (const op of ["add", "sub", "mul", "div"] as const) {
    assert.throws(() => (boolArr as unknown as { [k: string]: (n: number) => unknown })[op]!(1), boolMsgRegex, `${op} scalar on [bool] must stay permanently locked`);
  }
});

test("add/sub/mul/div: array form now COMPUTES across float64/float32/int32 per Promote (dt2 P1/P2), bool stays PERMANENTLY locked as receiver OR argument (dt2 P4)", () => {
  const f64 = NDArray.fromArray([2], [1, 2]);
  const f32 = NDArray.fromArray([2], [1, 2], { dtype: "float32" });
  const i32 = NDArray.fromArray([2], [1, 2], { dtype: "int32" });

  // Mixed-dtype combos: result dtype follows Promote (D4), never a throw.
  assert.strictEqual(f64.add(f32).dtype, "float64");
  assert.strictEqual(f64.add(i32).dtype, "float64");
  assert.strictEqual(f32.add(f32).dtype, "float32");
  assert.strictEqual(i32.add(i32).dtype, "int32");
  assert.strictEqual(i32.div(i32).dtype, "float64"); // D5: div always widens to float

  // bool: PERMANENT-lock test (A2 point 2) — as RECEIVER or ARGUMENT, every op.
  const boolArr = NDArray.fromArray([2], [1, 0], { dtype: "bool" });
  for (const op of ["add", "sub", "mul", "div"] as const) {
    assert.throws(
      () => (boolArr as unknown as { [k: string]: (n: unknown) => unknown })[op]!(f64),
      boolMsgRegex,
      `${op}: bool RECEIVER must stay permanently locked`,
    );
    assert.throws(
      () => (f64 as unknown as { [k: string]: (n: unknown) => unknown })[op]!(boolArr),
      boolMsgRegex,
      `${op}: bool ARGUMENT must stay permanently locked`,
    );
  }
});

// dt3 A2 points 1+2: matmul/dot/cosineSimilarity and sum/mean are no longer
// locked for float32/int32 (O1/E3) — the full dtype matrix lives in the dt3
// section appended at the end of this file. What stays here is the bool
// lock for the two-operand ops (permanent `BOOL_ARITHMETIC_MESSAGE`, as
// receiver or argument), and the "sum/mean accept bool" counterpart.
test("matmul/dot/cosineSimilarity: bool as receiver or argument throws BOOL_ARITHMETIC_MESSAGE (permanent lock, dt3 R3)", () => {
  const mat64 = NDArray.fromArray([2, 2], [1, 2, 3, 4]);
  const vec64 = NDArray.fromArray([2], [1, 2]);
  const matBool = NDArray.fromArray([2, 2], [1, 0, 0, 1], { dtype: "bool" });
  const vecBool = NDArray.fromArray([2], [1, 0], { dtype: "bool" });
  assert.throws(() => matBool.matmul(mat64 as unknown as Parameters<typeof matBool.matmul>[0]), boolMsgRegex, "matmul bool RECEIVER");
  assert.throws(() => mat64.matmul(matBool as unknown as Parameters<typeof mat64.matmul>[0]), boolMsgRegex, "matmul bool ARGUMENT");
  assert.throws(() => vecBool.dot(vec64 as unknown as Parameters<typeof vecBool.dot>[0]), boolMsgRegex, "dot bool RECEIVER");
  assert.throws(() => vec64.dot(vecBool as unknown as Parameters<typeof vec64.dot>[0]), boolMsgRegex, "dot bool ARGUMENT");
  assert.throws(
    () => vecBool.cosineSimilarity(vec64 as unknown as Parameters<typeof vecBool.cosineSimilarity>[0]),
    boolMsgRegex,
    "cosineSimilarity bool RECEIVER",
  );
  assert.throws(
    () => vec64.cosineSimilarity(vecBool as unknown as Parameters<typeof vec64.cosineSimilarity>[0]),
    boolMsgRegex,
    "cosineSimilarity bool ARGUMENT",
  );
});

test("sum/mean: no lock for any dtype, bool included — both the 0-arg and axis-bearing forms compute (dt3 R2)", () => {
  for (const dtype of DTYPES) {
    const nd = NDArray.fromArray([2, 2], [1, 0, 1, 0], { dtype });
    assert.deepStrictEqual([...nd.sum().data], [2], `sum() [${dtype}]`);
    assert.deepStrictEqual([...nd.sum(0).data], [2, 0], `sum(0) [${dtype}]`);
    assert.deepStrictEqual([...nd.mean().data], [0.5], `mean() [${dtype}]`);
    assert.deepStrictEqual([...nd.mean(0).data], [1, 0], `mean(0) [${dtype}]`);
  }
});

test("argmax/topk: both the 0-arg (argmax) and argument-bearing forms throw the locked message", () => {
  for (const dtype of NON_FLOAT64) {
    const nd = NDArray.fromArray([3], [1, 0, 1], { dtype });
    assert.throws(() => nd.argmax(), lockedMsgRegex("argmax", dtype), `argmax() [${dtype}]`);
    assert.throws(() => nd.argmax(0), lockedMsgRegex("argmax", dtype), `argmax(0) [${dtype}]`);
    assert.throws(() => nd.topk(2), lockedMsgRegex("topk", dtype), `topk(2) [${dtype}]`);
  }
});

// dt3 A2 point 3: `norm` left this test (it computes for float32/int32 now, and
// its bool rejection is its own permanent runtime-only test in the dt3 section
// at the end of this file); `sqrt` stays locked until dt5.
test("sqrt: niladic locked op throws for every non-float64 receiver (no compile-time claim possible, M2 disclosed gap)", () => {
  for (const dtype of NON_FLOAT64) {
    const nd = NDArray.fromArray([3], [1, 0, 1], { dtype });
    assert.throws(() => nd.sqrt(), lockedMsgRegex("sqrt", dtype), `sqrt() [${dtype}]`);
  }
});

test("stack(): a non-float64 row throws the locked message at runtime (M3 v8 disclosed exception: the EDITOR shows a native structural diagnostic instead until dt5, but the runtime message is word-identical to every other locked op)", () => {
  const stackAny = NDArray.stack as unknown as (rows: readonly AnyNDArray[]) => unknown;
  for (const dtype of NON_FLOAT64) {
    const row = NDArray.fromArray([3], [1, 0, 1], { dtype });
    assert.throws(() => stackAny([row]), lockedMsgRegex("stack", dtype), `stack() [${dtype}]`);
  }
});

// --- Section dt1.5: K5 — AnyNDArray fix + two DIFFERENT non-float64 dtypes -

test("AnyNDArray spans every dtype (K5 fix): a non-float64 array assigned through the top type still round-trips its real dtype at runtime", () => {
  const i32: AnyNDArray = NDArray.fromArray([2], [1, 2], { dtype: "int32" });
  assert.strictEqual(i32.dtype, "int32", "AnyNDArray must not silently narrow to float64");
  const nested: NestedValue | NestedBoolValue = i32.toNestedArray();
  assert.deepStrictEqual(nested, [1, 2]);
});

// dt3 A2 point 4: the "a locked two-operand op with TWO DIFFERENT non-float64
// dtypes rejects on the RECEIVER's dtype first" test is GONE. After dt3 no
// locked two-operand op remains, and the bool-arithmetic message names no
// dtype, so there is no receiver-first order left to prove.

// --- Section dt1.6 (F2 post-review fix; dt3 A2 point 5): the compile-time
// lock text equals `lockedOpMessage` verbatim (cross-layer parity) ----------
//
// `DTypeLockPair` — whose receiver-first branch order this section used to
// prove with a swapped-branch mutant — was deleted by dt3 R5 (no users left),
// so the mutant half is gone with its subject. The cross-layer half moves to
// `argmax(0)`, which stays `DTypeLock`-locked until dt5 (and will move or end
// there): the compile-time message text must equal `lockedOpMessage(op,
// dtype)` — the same function the runtime tests above import from
// runtime.ts, never a hand-typed duplicate. Real-compiler-on-a-throwaway-
// fixture pattern (spawnSync tsc against an out-of-repo dir, so the
// deliberately non-float64 fixture code never joins any type corpus).
test("DTypeLock (F2 pin, moved to argmax(0) by dt3): the compile-time lock text matches lockedOpMessage exactly (cross-layer parity)", () => {
  const dir = mkdtempSync(join(tmpdir(), "numtype-dtypelock-pin-"));
  try {
    const ndarrayPath = fileURLToPath(new URL("../src/ndarray.ts", import.meta.url).href);
    const ambientPath = fileURLToPath(new URL("../src/ambient.d.ts", import.meta.url).href);
    const repoRoot = fileURLToPath(new URL("../..", import.meta.url).href);
    writeFileSync(
      join(dir, "probe.ts"),
      `import { NDArray } from ${JSON.stringify(ndarrayPath)};\n` +
        `const i32Mat = NDArray.fromArray([2, 2], [1, 0, 0, 1], { dtype: "int32" });\n` +
        `i32Mat.argmax(0); // must name int32 (the RECEIVER)\n`,
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
    assert.notStrictEqual(res.status, 0, `fixture must fail to compile (a genuine dtype lock):\n${out}`);

    const probeErrors = out.split("\n").filter((l) => l.includes("probe.ts(") && l.includes("error TS"));
    assert.strictEqual(probeErrors.length, 1, `expected exactly ONE fixture error (the locked argmax(0) call):\n${out}`);

    // tsc prints the message as the CONTENT of a quoted string-literal type,
    // so its own embedded `"` chars come out backslash-escaped;
    // `JSON.stringify(...).slice(1, -1)` reproduces that exact escaping
    // instead of hand-duplicating it.
    const escaped = (op: string, dtype: DType) => JSON.stringify(lockedOpMessage(op, dtype)).slice(1, -1);
    assert.ok(
      out.includes(escaped("argmax", "int32")),
      `i32Mat.argmax(0) must surface lockedOpMessage("argmax", "int32") verbatim:\n${out}`,
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// --- Section dt1.7 (F1 post-review fix): every DType switch is exhaustive,
// and never silently mislabels an unrecognized value as float64 -------------
//
// Before this fix, `zerosData`/`onesData`/`convertToDType`/`astypeConvert`
// were `switch (dtype)` statements over the closed `DType` union with NO
// `default` arm — an invalid dtype string smuggled in past the type layer
// (e.g. `"invalid" as DType`, or a value read from JSON/an external
// boundary) fell through with no return value, so `data` silently became
// `undefined` instead of throwing (an M2 violation: confidently wrong,
// not an honest failure). Likewise `sameKindArray`/`copySameKindArray`
// (K3's typed-array-preserving helpers for transpose/slice/reshape/flatten)
// used to fall back to `new Float64Array(...)` for ANY unrecognized backing
// store instead of throwing — silently mislabeling e.g. a foreign typed
// array as float64 data.

test("zerosData/onesData/convertToDType/astypeConvert: throw a clear message naming the invalid dtype, never return undefined data", () => {
  const bad = "invalid" as unknown as DType;
  assert.throws(() => zerosData(bad, 3), /zerosData: invalid dtype "invalid"/, "zerosData must throw, not silently return undefined data");
  assert.throws(() => onesData(bad, 3), /onesData: invalid dtype "invalid"/, "onesData must throw, not silently return undefined data");
  assert.throws(
    () => convertToDType(bad, [1, 2, 3]),
    /convertToDType: invalid dtype "invalid"/,
    "convertToDType must throw, not silently return undefined data",
  );
  assert.throws(
    () => astypeConvert(bad, new Float64Array([1, 2, 3])),
    /astypeConvert: invalid dtype "invalid"/,
    "astypeConvert must throw, not silently return undefined data",
  );
});

test("sameKindArray/copySameKindArray: allocate/copy the correct typed-array class for every real DType, and throw (never fall back to Float64Array) for an unrecognized input", () => {
  // Positive cases: every real DType's backing store round-trips through
  // its OWN class, including float64 itself (the pre-fix code path for
  // float64 relied on the same silent "everything unmatched is float64"
  // fallback branch this fix removes — this proves float64 is still
  // handled correctly by its own explicit check, not by accident).
  const f64 = new Float64Array([1, 2, 3]);
  const f32 = new Float32Array([1, 2, 3]);
  const i32 = new Int32Array([1, 2, 3]);
  const b8 = new Uint8Array([1, 0, 1]);
  assert.ok(sameKindArray(f64, 3) instanceof Float64Array, "sameKindArray(float64) allocates Float64Array");
  assert.ok(sameKindArray(f32, 3) instanceof Float32Array, "sameKindArray(float32) allocates Float32Array");
  assert.ok(sameKindArray(i32, 3) instanceof Int32Array, "sameKindArray(int32) allocates Int32Array");
  assert.ok(sameKindArray(b8, 3) instanceof Uint8Array, "sameKindArray(bool) allocates Uint8Array");
  assert.ok(copySameKindArray(f64) instanceof Float64Array, "copySameKindArray(float64) copies into Float64Array");
  assert.ok(copySameKindArray(f32) instanceof Float32Array, "copySameKindArray(float32) copies into Float32Array");
  assert.ok(copySameKindArray(i32) instanceof Int32Array, "copySameKindArray(int32) copies into Int32Array");
  assert.ok(copySameKindArray(b8) instanceof Uint8Array, "copySameKindArray(bool) copies into Uint8Array");
  assert.deepStrictEqual([...(copySameKindArray(i32) as Int32Array)], [1, 2, 3], "copySameKindArray must copy values, not just class");
  assert.notStrictEqual(copySameKindArray(i32), i32, "copySameKindArray must never alias the source buffer");

  // Negative case: an unrecognized typed array (bypassing the type layer,
  // e.g. a BigInt64Array — never a legal DataOfRuntime) must THROW, not
  // silently fall back to allocating/copying as Float64Array.
  const foreign = new BigInt64Array([1n, 2n, 3n]) as unknown as ReturnType<typeof zerosData>;
  assert.throws(
    () => sameKindArray(foreign, 3),
    /sameKindArray: unrecognized typed array/,
    "sameKindArray must throw for an unrecognized backing store, never silently allocate Float64Array",
  );
  assert.throws(
    () => copySameKindArray(foreign),
    /copySameKindArray: unrecognized typed array/,
    "copySameKindArray must throw for an unrecognized backing store, never silently copy as Float64Array",
  );
});

// =============================================================================
// Section dt2 (docs/dtype-dt2-spec.md, P5): Promotion + elementwise
// arithmetic for float64/float32/int32 (add/sub/mul/div); bool permanently
// locked (covered above, A2 tests). Same no-new-file house convention as
// W2-W5/dt1 above. Imports needed only by this block are added right here.
// =============================================================================
import { promoteDType, promoteDTypeDiv, type NumericDType } from "../src/runtime.ts";

const NUMERIC_DTYPES: readonly NumericDType[] = ["float64", "float32", "int32"];

/** Dispatch any of the four binary ops on two NDArrays — a plain switch over
 * a runtime-unsafe cast (same pattern `callScalar` above and the dt1 lock
 * tests already use for `[k: string]`-indexed dispatch), needed here because
 * the tests below exercise EVERY dtype combination generically. */
function callBinary(a: AnyNDArray, op: ScalarOp, b: unknown): AnyNDArray {
  return (a as unknown as { [k: string]: (x: unknown) => AnyNDArray })[op]!(b);
}

// --- P5: the promotion table, against the ONE source (runtime.ts), by hand -

/** D4: the promotion table from the spec, written out BY HAND as an
 * independent oracle (never derived from `PROMOTE_NUMERIC` itself) — the
 * point is to prove the SHIPPED single-source table actually matches the
 * SPEC's own D4 table, not merely that the implementation is internally
 * consistent with itself. */
const EXPECTED_PROMOTE: Record<NumericDType, Record<NumericDType, NumericDType>> = {
  float64: { float64: "float64", float32: "float64", int32: "float64" },
  float32: { float64: "float64", float32: "float32", int32: "float64" },
  int32: { float64: "float64", float32: "float64", int32: "int32" },
};

test("promoteDType: the full 3x3 numeric promotion table matches the spec's D4 table (hand-written oracle)", () => {
  for (const a of NUMERIC_DTYPES) {
    for (const b of NUMERIC_DTYPES) {
      assert.strictEqual(promoteDType(a, b), EXPECTED_PROMOTE[a]![b], `promoteDType(${a}, ${b})`);
    }
  }
});

test("promoteDTypeDiv: ALWAYS floating-point — float32⊕float32 stays float32, every other numeric combo (incl int32⊕int32) widens to float64 (D5)", () => {
  for (const a of NUMERIC_DTYPES) {
    for (const b of NUMERIC_DTYPES) {
      const expected = a === "float32" && b === "float32" ? "float32" : "float64";
      assert.strictEqual(promoteDTypeDiv(a, b), expected, `promoteDTypeDiv(${a}, ${b})`);
    }
  }
});

test("promoteDType/promoteDTypeDiv: bool rejects with BOOL_ARITHMETIC_MESSAGE regardless of position or the other operand's dtype", () => {
  const ALL: readonly DType[] = ["float64", "float32", "int32", "bool"];
  for (const other of ALL) {
    assert.throws(() => promoteDType("bool", other), boolMsgRegex, `promoteDType(bool, ${other})`);
    assert.throws(() => promoteDType(other, "bool"), boolMsgRegex, `promoteDType(${other}, bool)`);
    assert.throws(() => promoteDTypeDiv("bool", other), boolMsgRegex, `promoteDTypeDiv(bool, ${other})`);
    assert.throws(() => promoteDTypeDiv(other, "bool"), boolMsgRegex, `promoteDTypeDiv(${other}, bool)`);
  }
});

test("add/sub/mul/div through the public NDArray API: every numeric dtype combination's RESULT dtype matches promoteDType/promoteDTypeDiv exactly", () => {
  for (const a of NUMERIC_DTYPES) {
    for (const b of NUMERIC_DTYPES) {
      const x = NDArray.fromArray([2], [4, 8], { dtype: a });
      const y = NDArray.fromArray([2], [2, 2], { dtype: b });
      for (const op of ["add", "sub", "mul"] as const) {
        assert.strictEqual(callBinary(x, op, y).dtype, promoteDType(a, b), `${op}(${a}, ${b})`);
      }
      assert.strictEqual(callBinary(x, "div", y).dtype, promoteDTypeDiv(a, b), `div(${a}, ${b})`);
    }
  }
});

// --- P5: float32⊕float32 bit-identity vs an INDEPENDENT reference ----------
//
// The reference computes in plain JS `number` (double) space and stores the
// result into a FRESH `Float32Array` — genuinely different machinery than
// the implementation's explicit `Math.fround` call (an implicit truncation
// on typed-array store vs an explicit function call), even though D8's own
// correctness argument (Figueroa 1995, doubly-rounding-is-harmless) says
// both must agree bit-for-bit.

test("add/sub/mul/div (float32⊕float32): bit-identical to an independent Float32Array-store reference (D8)", () => {
  const rng = makeRng(0xf32n);
  for (let trial = 0; trial < 200; trial++) {
    const shape = genShape(rng, 1, 3);
    const size = shape.reduce((acc, d) => acc * d, 1);
    const aRaw = genDataSpecial(rng, shape);
    const bRaw = genDataSpecial(rng, shape);
    const a = NDArray.fromArray(shape, [...aRaw], { dtype: "float32" });
    const b = NDArray.fromArray(shape, [...bRaw], { dtype: "float32" });
    for (const op of OPS) {
      const actual = callBinary(a, op, b);
      assert.strictEqual(actual.dtype, "float32", `${op} float32⊕float32 trial ${trial}: result dtype`);
      const aData = a.data as unknown as Float32Array;
      const bData = b.data as unknown as Float32Array;
      const expected = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        expected[i] = NATIVE_OPS[op](aData[i] ?? 0, bData[i] ?? 0); // implicit fround on store
      }
      assertDataBitIdentical(expected as unknown as Float64Array, actual.data as unknown as Float64Array, `${op} float32⊕float32 trial ${trial}`);
    }
  }
});

test("add/sub/mul (float32 SCALAR): same fround-on-store bit-identity check, for the scalar overload (not just array⊕array)", () => {
  const rng = makeRng(0xf32502n);
  for (let trial = 0; trial < 100; trial++) {
    const shape = genShape(rng, 1, 3);
    const size = shape.reduce((acc, d) => acc * d, 1);
    const aRaw = genDataSpecial(rng, shape);
    const a = NDArray.fromArray(shape, [...aRaw], { dtype: "float32" });
    const sRaw = nextF64Special(rng);
    for (const op of ["add", "sub", "mul"] as const) {
      const actual = callBinary(a, op, sRaw);
      assert.strictEqual(actual.dtype, "float32", `${op} float32 scalar trial ${trial}: result dtype`);
      const aData = a.data as unknown as Float32Array;
      const expected = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        expected[i] = NATIVE_OPS[op](aData[i] ?? 0, sRaw);
      }
      assertDataBitIdentical(expected as unknown as Float64Array, actual.data as unknown as Float64Array, `${op} float32 scalar trial ${trial}`);
    }
  }
});

// --- P5: int32 two's-complement wrap vs BigInt.asIntN(32, …) ---------------

test("add/sub/mul (int32⊕int32): two's-complement wrap matches BigInt.asIntN(32, …) over random pairs across the full int32 range", () => {
  const rng = makeRng(0x1132n);
  for (let trial = 0; trial < 500; trial++) {
    const av = rng.nextInt(-2147483648, 2147483647);
    const bv = rng.nextInt(-2147483648, 2147483647);
    const a = NDArray.fromArray([1], [av], { dtype: "int32" });
    const b = NDArray.fromArray([1], [bv], { dtype: "int32" });
    const bigA = BigInt(av);
    const bigB = BigInt(bv);
    for (const op of ["add", "sub", "mul"] as const) {
      const actual = callBinary(a, op, b);
      assert.strictEqual(actual.dtype, "int32", `${op}(${av}, ${bv}) result dtype`);
      const bigRaw = op === "add" ? bigA + bigB : op === "sub" ? bigA - bigB : bigA * bigB;
      const expected = Number(BigInt.asIntN(32, bigRaw));
      assert.strictEqual((actual.data as unknown as Int32Array)[0], expected, `${op}(${av}, ${bv})`);
    }
  }
});

test("mul (int32⊕int32): LARGE products beyond 2^53 wrap correctly — Math.imul, never (a*b)|0 (dt2 spec B1/v1.1)", () => {
  // Both cases VERIFIED (this session) to make the naive `(a*b)|0` form
  // actually diverge from the correct wrap — otherwise they would not be
  // adversarial. The first is the spec's own worked example.
  const largeCases: readonly (readonly [number, number])[] = [
    [2147483647, 2147483647],
    [1234567891, 987654321],
  ];
  for (const [av, bv] of largeCases) {
    const naive = (av * bv) | 0;
    const expected = Number(BigInt.asIntN(32, BigInt(av) * BigInt(bv)));
    assert.notStrictEqual(naive, expected, `non-vacuity: (${av}*${bv})|0 must diverge from the correct wrap, else this case proves nothing`);

    const a = NDArray.fromArray([1], [av], { dtype: "int32" });
    const b = NDArray.fromArray([1], [bv], { dtype: "int32" });
    const actual = a.mul(b);
    assert.strictEqual(actual.dtype, "int32");
    assert.strictEqual(actual.data[0], expected, `mul(${av}, ${bv}) must use Math.imul, not (a*b)|0`);

    // Scalar form too: a.mul(bv) — same wrap requirement, same divergence.
    const actualScalar = a.mul(bv);
    assert.strictEqual(actualScalar.data[0], expected, `scalar mul(${av}, ${bv}) must use Math.imul, not (a*b)|0`);
  }
});

// --- P5: broadcasting across MIXED dtypes -----------------------------------

test("add/sub/mul/div: broadcasting works across MIXED dtypes — shape AND values correct, result dtype follows Promote/PromoteDiv", () => {
  const f32 = NDArray.fromArray([2, 1], [1, 2], { dtype: "float32" });
  const i32 = NDArray.fromArray([1, 3], [10, 20, 30], { dtype: "int32" });

  const added = f32.add(i32);
  assert.strictEqual(added.dtype, "float64"); // Promote(float32, int32) = float64
  assert.deepStrictEqual([...added.shape], [2, 3]);
  assert.deepStrictEqual([...added.data], [11, 21, 31, 12, 22, 32]);

  const divided = i32.div(f32); // receiver int32, argument float32 -> PromoteDiv -> float64
  assert.strictEqual(divided.dtype, "float64");
  assert.deepStrictEqual([...divided.shape], [2, 3]);
  assert.deepStrictEqual([...divided.data], [10, 20, 30, 5, 10, 15]);
});

// --- P5: D6 scalar-rule edges (2.0, -0, 1e3, 2.5, wide number) --------------

test("add (int32 scalar): D6 edge values — 2.0/-0/1e3 compute correctly (dot-form-but-integral / exponent-form), a wide-number 2.5 throws with the exact D6 message at runtime", () => {
  const i32 = NDArray.fromArray([2], [10, 20], { dtype: "int32" });
  assert.deepStrictEqual([...i32.add(2.0).data], [12, 22], "2.0 normalizes to the integer literal 2 (TS drops the trailing .0) -- must compute");
  assert.deepStrictEqual([...i32.add(-0).data], [10, 20], "-0 is integral (Number.isInteger(-0) === true) -- must compute");
  assert.deepStrictEqual([...i32.add(1e3).data], [1010, 1020], "1e3 is an exponent-form integer -- must compute (no static claim either way)");

  const wideNonInt: number = 2.5;
  assert.throws(
    () => i32.add(wideNonInt),
    /add: int32 scalar 2\.5 is not a valid integer operand \(must be an integer in \[-2147483648, 2147483647\]\)/,
    "a wide-number 2.5 must throw at runtime with the exact D6 message",
  );

  // Out-of-int32-range integer scalar: D6 says "Number.isInteger + Bereich"
  // (range) -- an integer OUTSIDE [-2^31, 2^31-1] must also throw.
  assert.throws(
    () => i32.add(3000000000),
    /add: int32 scalar 3000000000 is not a valid integer operand/,
    "an out-of-range integer scalar must throw (D6's range half of the check)",
  );
});

// G5 fix (post-verify coverage gap): the test above proves the range check
// exists (via a value far outside int32, 3000000000) but never pins the
// EXACT boundary -- a ±1 mutant on either bound comparison in
// `scalarArithTyped` (`s < -2147483648` / `s > 2147483647`) survived
// `pnpm check`/`test:core` with no test failing (verified this session by
// re-applying the mutant, backup copy + diff proof). This test closes that
// gap: both edges (INT32_MIN/INT32_MAX) must be ACCEPTED, and one step past
// either edge must be REJECTED.
test("G5 parity pin: int32 scalar range boundary is EXACTLY [-2147483648, 2147483647] — both edges accepted, one step past either edge rejected", () => {
  const i32 = NDArray.fromArray([1], [0], { dtype: "int32" });

  // In-range boundary values themselves: must compute (D6's range check is
  // inclusive on both ends).
  assert.deepStrictEqual([...i32.add(2147483647).data], [2147483647], "INT32_MAX (2147483647) must be accepted as a valid int32 scalar");
  assert.deepStrictEqual([...i32.add(-2147483648).data], [-2147483648], "INT32_MIN (-2147483648) must be accepted as a valid int32 scalar");

  // One step past either edge: must reject, with the exact G4-fixed message
  // (word-identical to the type level, pinned separately above).
  assert.throws(
    () => i32.add(2147483648),
    /add: int32 scalar 2147483648 is not a valid integer operand \(must be an integer in \[-2147483648, 2147483647\]\)/,
    "2147483648 (INT32_MAX + 1) must be rejected -- a `s > 2147483647` -> `s >= 2147483647` mutant would silently accept this",
  );
  assert.throws(
    () => i32.add(-2147483649),
    /add: int32 scalar -2147483649 is not a valid integer operand \(must be an integer in \[-2147483648, 2147483647\]\)/,
    "-2147483649 (INT32_MIN - 1) must be rejected -- a `s < -2147483648` -> `s <= -2147483648` mutant would silently accept this",
  );
});

test("div (int32 scalar): NO integer restriction at all — a fractional scalar computes, widening to float64 (D5, unlike add/sub/mul's D6 rule)", () => {
  const i32 = NDArray.fromArray([2], [10, 20], { dtype: "int32" });
  const r = i32.div(2.5);
  assert.strictEqual(r.dtype, "float64");
  assert.deepStrictEqual([...r.data], [4, 8]);
});

// --- P5: diagnostic CONTENTS (M3 v9 named exception), via real tsc ---------
//
// The scalar overload's OWN `ShapeError` message is masked by TS overload
// resolution (it attributes a total mismatch to the LAST-declared overload,
// the array form) — a NAMED M3 exception (A3, docs/dtype-dt2-spec.md): the
// call is still a genuine compile error (M2 holds), but the TEXT shown is
// the array overload's generic TS2769, not the scalar guard's own message.
// The ARRAY form's own rejection (bool operand) is NOT masked and shows its
// own message directly. Positions/text below were MEASURED against real
// tsc this session (Arbeitsregel 13/CLAUDE.md "GEMESSEN, nicht gelesen"),
// not guessed.

test("dt2 diagnostic CONTENTS (M3 v9): i32.add(2.5) and boolArr.add(1) show the masked generic TS2769; boolArr.add(f64) shows its OWN BOOL_ARITHMETIC_MESSAGE directly", () => {
  const dir = mkdtempSync(join(tmpdir(), "numtype-dt2-diag-pin-"));
  try {
    const ndarrayPath = fileURLToPath(new URL("../src/ndarray.ts", import.meta.url).href);
    const ambientPath = fileURLToPath(new URL("../src/ambient.d.ts", import.meta.url).href);
    const repoRoot = fileURLToPath(new URL("../..", import.meta.url).href);
    writeFileSync(
      join(dir, "probe.ts"),
      `import { NDArray } from ${JSON.stringify(ndarrayPath)};\n` +
        `const i32 = NDArray.fromArray([2], [1, 2], { dtype: "int32" });\n` +
        `const boolArr = NDArray.fromArray([2], [1, 0], { dtype: "bool" });\n` +
        `const f64 = NDArray.fromArray([2], [1, 2]);\n` +
        `i32.add(2.5);\n` +
        `boolArr.add(1);\n` +
        `boolArr.add(f64);\n`,
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
    assert.strictEqual(probeErrors.length, 3, `expected exactly THREE fixture errors:\n${out}`);

    // i32.add(2.5) at probe.ts line 5, col 9 -- masked generic TS2769.
    assert.ok(out.includes("probe.ts(5,9): error TS2769"), `i32.add(2.5) must error at (5,9):\n${out}`);
    // boolArr.add(1) at line 6, col 13 -- masked generic TS2769, SAME text.
    assert.ok(out.includes("probe.ts(6,13): error TS2769"), `boolArr.add(1) must error at (6,13):\n${out}`);
    assert.ok(
      out.includes("Argument of type 'number' is not assignable to parameter of type 'NDArray<Shape, DType>'"),
      `both masked scalar-form calls must show the generic last-overload message:\n${out}`,
    );
    // boolArr.add(f64) at line 7, col 13 -- OWN message surfaces (array form
    // is the LAST overload, so its own Guard rejection is not masked).
    assert.ok(out.includes("probe.ts(7,13): error TS2769"), `boolArr.add(f64) must error at (7,13):\n${out}`);
    const escapedBoolMsg = JSON.stringify(BOOL_ARITHMETIC_MESSAGE).slice(1, -1);
    assert.ok(
      out.includes(escapedBoolMsg),
      `boolArr.add(f64) must surface BOOL_ARITHMETIC_MESSAGE verbatim (own message, not masked):\n${out}`,
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// --- G4 fix (post-3b-verify M3 finding): the int32 non-integer scalar
// message must be WORD-IDENTICAL between the type level (`ArithScalarOperand`'s
// own `ShapeError` text, ndarray.ts) and the runtime (`scalarArithTyped`'s
// thrown message) — they had DRIFTED (the type-level text omitted the int32
// range this function's message states), an M3 violation invisible in the
// "dt2 diagnostic CONTENTS" test above only because that call-site form is
// MASKED by the M3 v9 named exception (it shows the array overload's
// generic TS2769, never this type's own text). This test bypasses the
// masking entirely: `ArithScalarOperand` is checked DIRECTLY (not through
// overload resolution), via the same real-tsc-on-an-out-of-repo-fixture
// pattern the pins above use, and neither side's message is hand-typed —
// the runtime side comes from an ACTUAL thrown error, the type-level side
// from ACTUAL tsc output — so this can never silently pass by both sides
// independently drifting to the same wrong (hand-copied) string.

test("G4 parity pin: int32 non-integer scalar message is word-identical between ArithScalarOperand's ShapeError (type level) and scalarArithTyped's thrown message (runtime)", () => {
  const i32 = NDArray.fromArray([1], [1], { dtype: "int32" });
  // Widened to `number` (same technique the D6 edge-values test above uses
  // for its own `wideNonInt`): `2.5` as a literal is a PROVEN non-integer
  // dot-form scalar (D6), which `ArithScalarOperand` rejects AT COMPILE
  // TIME -- exactly the mechanism this test is about, but not what THIS
  // particular call needs to exercise (this call is deliberately triggering
  // the RUNTIME throw to read its message; the type-level side is checked
  // separately, directly, below via the exported `ArithScalarOperand`).
  const nonIntegerScalar: number = 2.5;
  // Captured via `assert.throws`'s predicate-function form (this project's
  // `node:assert` shim, ambient.d.ts, is a minimal scoped surface — no
  // `.fail`/`.match` — so the message is captured as a side effect of the
  // predicate instead, same `throws(fn, predicate, message)` shape every
  // other assertion in this file already uses).
  let runtimeMessage = "";
  assert.throws(
    () => i32.add(nonIntegerScalar),
    (err: unknown) => {
      runtimeMessage = (err as Error).message;
      return true;
    },
    "i32.add(2.5) must throw",
  );
  // Sanity: didn't accidentally catch the wrong error / an empty message.
  assert.ok(
    /^add: int32 scalar 2\.5 is not a valid integer operand \(must be an integer in \[-2147483648, 2147483647\]\)$/.test(runtimeMessage),
    `sanity: the actual runtime message must have the expected shape before comparing the type level against it, got: ${runtimeMessage}`,
  );

  const dir = mkdtempSync(join(tmpdir(), "numtype-g4-parity-pin-"));
  try {
    const ndarrayPath = fileURLToPath(new URL("../src/ndarray.ts", import.meta.url).href);
    const ambientPath = fileURLToPath(new URL("../src/ambient.d.ts", import.meta.url).href);
    const repoRoot = fileURLToPath(new URL("../..", import.meta.url).href);
    writeFileSync(
      join(dir, "probe.ts"),
      `import type { ArithScalarOperand } from ${JSON.stringify(ndarrayPath)};\n` +
        // A DIRECT type-level check (never through a method call, so the M3
        // v9 overload-resolution masking never applies here) — the deliberate
        // property-mismatch below forces tsc to print `Probe`'s FULLY
        // resolved shape, including the literal `__shapeError` message text,
        // in its own diagnostic.
        `type Probe = ArithScalarOperand<"int32", 2.5, "add">;\n` +
        `declare const p: Probe;\n` +
        `const marker: { __wontMatch: true } = p; // deliberate mismatch: reveals Probe's exact resolved type\n` +
        `void marker;\n`,
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
    assert.notStrictEqual(res.status, 0, `fixture must fail to compile (deliberate mismatch):\n${out}`);

    const probeErrors = out.split("\n").filter((l) => l.includes("probe.ts(") && l.includes("error TS"));
    assert.strictEqual(probeErrors.length, 1, `expected exactly ONE fixture error (the deliberate mismatch):\n${out}`);

    // tsc prints the message as the content of a quoted string-literal type
    // (its own `"`/`[`/`]` chars come out escaped) — `JSON.stringify(...).
    // slice(1, -1)` reproduces that escaping exactly, same technique the
    // DTypeLockPair/F1 pins above use, so neither side of this comparison is
    // ever hand-typed.
    const escapedRuntimeMessage = JSON.stringify(runtimeMessage).slice(1, -1);
    assert.ok(
      out.includes(escapedRuntimeMessage),
      `ArithScalarOperand<"int32", 2.5, "add">'s ShapeError text must be WORD-IDENTICAL (M3) to scalarArithTyped's actual thrown message:\n${out}`,
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});


// =============================================================================
// dt3 (docs/dtype-dt3-spec.md): reductions on every dtype — `sum`/`mean`
// (E3: ReduceDType), `matmul`/`dot`/`cosineSimilarity` (O1: PromoteDiv) and
// `norm` (R4). Same no-new-file house convention (appended at the end of this
// file; the Dateiset stays fixed). NDArray-only — no WNDArray/WASM counterpart
// (D9: dt6+), so the oracles are (a) the PRE-dt3 float64 runtime functions
// (`sumRuntime`/`meanRuntime`/`matmulRuntime`/`dotRuntime`/`normSqRuntime`,
// byte-unchanged) for every float64-compute path, (b) an INDEPENDENT float32
// reference that rounds by STORING into a `Float32Array` cell (never
// `Math.fround`, the implementation's own device) for every float32 path, and
// (c) a BigInt oracle that rounds every step to float64 for int32 magnitudes
// beyond 2^53. Receivers in the loops below are typed through the structural
// `LooseNd` view (below) so the loops do not pay the generic-overload type
// machinery per call; the real typed surface is pinned by the type-level
// tests (spike/tests/ndarray.test-d.ts) and by the typed calls in the
// replaced A2 tests above.
// =============================================================================
import { dotRuntime, matmulRuntime, normSqRuntime, PROMOTE_DIV, REDUCE_DTYPE, type DataOfRuntime } from "../src/runtime.ts";

/** The structural slice of `NDArray` the dt3 loops use (runtime semantics
 * only — the compile-time claims are pinned in ndarray.test-d.ts). */
interface LooseNd {
  readonly dtype: DType;
  readonly shape: readonly number[];
  readonly strides: readonly number[];
  readonly data: DataOfRuntime;
  sum(axis?: number, keepdims?: boolean): LooseNd;
  mean(axis?: number, keepdims?: boolean): LooseNd;
  matmul(other: LooseNd): LooseNd;
  dot(other: LooseNd): number;
  norm(): number;
  cosineSimilarity(other: LooseNd): number;
}
const loose = (nd: unknown): LooseNd => nd as LooseNd;

/** Build an array of `dtype` from plain numbers (shape variables are plain
 * `number[]`, so no literal-shape machinery runs at these call sites). */
function mkNd(shape: readonly number[], values: ArrayLike<number>, dtype: DType): LooseNd {
  return loose(NDArray.fromArray(shape as number[], Array.from(values), { dtype }));
}

const f32Cell = new Float32Array(1);
const f32Bits = new Uint32Array(f32Cell.buffer);
/** float32 bit pattern of a number, via a cell store (independent of `Math.fround`). */
function bits32(x: number): number {
  f32Cell[0] = x;
  return f32Bits[0] ?? 0;
}

/** Bit-exact element comparison: float32 by 32-bit pattern, everything else
 * via `Object.is` (distinguishes ±0); any NaN equals any NaN (payloads are
 * implementation-defined, same stance as `assertDataBitIdentical`). */
function assertSameValues(expected: ArrayLike<number>, actual: ArrayLike<number>, f32: boolean, context: string): void {
  assert.strictEqual(actual.length, expected.length, `${context}: length`);
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i] ?? 0;
    const a = actual[i] ?? 0;
    // A float32 result must itself BE a float32 value (a number-returning op like dot/norm that
    // skipped its final float32 rounding would otherwise hide behind the cell store in bits32).
    if (f32) assert.ok(Number.isNaN(a) || Object.is(Math.fround(a), a), `${context}: element ${i}: ${a} is not exactly a float32 value`);
    const same = (Number.isNaN(e) && Number.isNaN(a)) || (f32 ? bits32(e) === bits32(a) : Object.is(e, a));
    assert.ok(same, `${context}: element ${i}: expected ${e} (${f32 ? "0x" + bits32(e).toString(16) : ""}) got ${a} (${f32 ? "0x" + bits32(a).toString(16) : ""})`);
  }
}

const REDUCE_EXPECTED = { float64: "float64", float32: "float32", int32: "float64", bool: "float64" } as const;
const PROMOTE_DIV_EXPECTED = {
  float64: { float64: "float64", float32: "float64", int32: "float64" },
  float32: { float64: "float64", float32: "float32", int32: "float64" },
  int32: { float64: "float64", float32: "float64", int32: "float64" },
} as const;
const CLASS_OF = (d: DType) => (d === "float32" ? Float32Array : Float64Array);

const F32_SPECIALS: readonly number[] = [
  0, -0, 1, -1, 0.1, -0.1, 3.4028234663852886e38, -3.4028234663852886e38, 1e-45, -1e-45, 1.1754943508222875e-38, 1e-40, Infinity, -Infinity, Number.NaN,
  16777216, 16777217, 1e30, -1e30, 1 / 3,
];

/** Random float32 payload. "finite": moderate and scaled values only (sums
 * stay informative). "wild": ~40% IEEE specials (±0/±Inf/NaN/subnormals/
 * overflow-prone magnitudes) plus raw random bit patterns. */
function genF32(rng: Rng, n: number, mode: "finite" | "wild"): Float32Array {
  const out = new Float32Array(n);
  const view = new Uint32Array(1);
  const asF32 = new Float32Array(view.buffer);
  for (let i = 0; i < n; i++) {
    const roll = rng.nextInt(0, 99);
    if (mode === "wild" && roll < 40) out[i] = F32_SPECIALS[rng.nextInt(0, F32_SPECIALS.length - 1)] ?? 0;
    else if (mode === "wild" && roll >= 90) {
      view[0] = Number(rng.nextU64() & 0xffffffffn);
      out[i] = asF32[0] ?? 0;
    } else if (roll < 70) out[i] = rng.nextF64();
    else out[i] = rng.nextF64() * 2 ** rng.nextInt(-40, 40);
  }
  return out;
}

const prod = (xs: readonly number[]): number => xs.reduce((a, b) => a * b, 1);

/** INDEPENDENT float32 sum: walks the input in ascending flat order and adds
 * each element into its output cell (`out[o] = out[o] + x` — the typed-array
 * STORE is the float32 rounding), so a lane's additions occur in ascending
 * axis order without any per-lane gather code (a different algorithm shape
 * than `sumF32Runtime`'s base-offset loop). */
function refSumF32(shape: readonly number[], data: Float32Array, axis: number | undefined): { shape: number[]; data: Float32Array } {
  if (axis === undefined) {
    const acc = new Float32Array(1);
    for (let i = 0; i < data.length; i++) acc[0] = (acc[0] ?? 0) + (data[i] ?? 0);
    return { shape: [], data: acc };
  }
  const rank = shape.length;
  const ax = axis < 0 ? rank + axis : axis;
  const outShape = shape.filter((_, d) => d !== ax);
  const out = new Float32Array(prod(outShape));
  const coord = new Array<number>(rank).fill(0);
  for (let flat = 0; flat < data.length; flat++) {
    let o = 0;
    for (let d = 0; d < rank; d++) if (d !== ax) o = o * (shape[d] ?? 1) + (coord[d] ?? 0);
    out[o] = (out[o] ?? 0) + (data[flat] ?? 0);
    for (let d = rank - 1; d >= 0; d--) {
      coord[d] = (coord[d] ?? 0) + 1;
      if ((coord[d] ?? 0) < (shape[d] ?? 1)) break;
      coord[d] = 0;
    }
  }
  return { shape: outShape, data: out };
}

function refMeanF32(shape: readonly number[], data: Float32Array, axis: number | undefined): { shape: number[]; data: Float32Array } {
  const s = refSumF32(shape, data, axis);
  const n = axis === undefined ? data.length : (shape[axis < 0 ? shape.length + axis : axis] ?? 1);
  const out = new Float32Array(s.data.length);
  for (let i = 0; i < out.length; i++) out[i] = (s.data[i] ?? 0) / n; // f64 division by the exact integer n, then the float32 store
  return { shape: s.shape, data: out };
}

function refDotF32(a: Float32Array, b: Float32Array): number {
  const acc = new Float32Array(1);
  const p = new Float32Array(1);
  for (let i = 0; i < a.length; i++) {
    p[0] = (a[i] ?? 0) * (b[i] ?? 0);
    acc[0] = (acc[0] ?? 0) + (p[0] ?? 0);
  }
  return acc[0] ?? 0;
}
function refNsqF32(a: Float32Array): number {
  return refDotF32(a, a);
}
function refNormF32(a: Float32Array): number {
  const r = new Float32Array(1);
  r[0] = Math.sqrt(refNsqF32(a));
  return r[0] ?? 0;
}
function refCosF32(a: Float32Array, b: Float32Array): number {
  const sa = new Float32Array(1);
  const sb = new Float32Array(1);
  sa[0] = Math.sqrt(refNsqF32(a));
  sb[0] = Math.sqrt(refNsqF32(b));
  const den = new Float32Array(1);
  den[0] = (sa[0] ?? 0) * (sb[0] ?? 0);
  const r = new Float32Array(1);
  r[0] = refDotF32(a, b) / (den[0] ?? 0);
  return r[0] ?? 0;
}

/** INDEPENDENT float32 N-d matmul: 1-D promotion, batch broadcast via explicit
 * coordinate arithmetic, per-element `acc = acc + (a*b)` with a float32 store
 * at each step. */
function refMatmulF32(aShapeIn: readonly number[], A: Float32Array, bShapeIn: readonly number[], B: Float32Array): { shape: number[]; data: Float32Array } {
  const aP = aShapeIn.length === 1;
  const bP = bShapeIn.length === 1;
  const as = aP ? [1, aShapeIn[0] ?? 1] : [...aShapeIn];
  const bs = bP ? [bShapeIn[0] ?? 1, 1] : [...bShapeIn];
  const m = as[as.length - 2] ?? 1;
  const k = as[as.length - 1] ?? 1;
  const n = bs[bs.length - 1] ?? 1;
  const ba = as.slice(0, -2);
  const bb = bs.slice(0, -2);
  const r = Math.max(ba.length, bb.length);
  const bo: number[] = [];
  for (let i = 0; i < r; i++) bo.push(Math.max(ba[i - (r - ba.length)] ?? 1, bb[i - (r - bb.length)] ?? 1));
  const out = new Float32Array(prod(bo) * m * n);
  const acc = new Float32Array(1);
  const p = new Float32Array(1);
  const batchOffset = (coord: readonly number[], batchShape: readonly number[], matSize: number): number => {
    let off = 0;
    let stride = matSize;
    for (let j = batchShape.length - 1; j >= 0; j--) {
      const c = coord[j + (r - batchShape.length)] ?? 0;
      off += (batchShape[j] === 1 ? 0 : c) * stride;
      stride *= batchShape[j] ?? 1;
    }
    return off;
  };
  for (let bi = 0; bi < prod(bo); bi++) {
    const coord: number[] = [];
    let rest = bi;
    for (let j = r - 1; j >= 0; j--) {
      coord[j] = rest % (bo[j] ?? 1);
      rest = Math.floor(rest / (bo[j] ?? 1));
    }
    const aOff = batchOffset(coord, ba, m * k);
    const bOff = batchOffset(coord, bb, k * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        acc[0] = 0;
        for (let q = 0; q < k; q++) {
          p[0] = (A[aOff + i * k + q] ?? 0) * (B[bOff + q * n + j] ?? 0);
          acc[0] = (acc[0] ?? 0) + (p[0] ?? 0);
        }
        out[bi * m * n + i * n + j] = acc[0] ?? 0;
      }
    }
  }
  let shape = [...bo, m, n];
  if (aP) shape = [...shape.slice(0, r), ...shape.slice(r + 1)];
  if (bP) shape = shape.slice(0, -1);
  return { shape, data: out };
}

/** BigInt oracle: round-to-nearest-even to float64 at EVERY step (product,
 * then sum) — `Number(bigint)` is exactly that rounding. */
function bigDot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let acc = 0;
  for (let i = 0; i < a.length; i++) {
    const p = Number(BigInt(a[i] ?? 0) * BigInt(b[i] ?? 0));
    acc = Number(BigInt(acc) + BigInt(p));
  }
  return acc;
}
function bigSum(a: ArrayLike<number>): number {
  let acc = 0;
  for (let i = 0; i < a.length; i++) acc = Number(BigInt(acc) + BigInt(a[i] ?? 0));
  return acc;
}

test("dt3 tables: REDUCE_DTYPE and PROMOTE_DIV equal the spec's explicit tables (type and runtime read them as ONE source)", () => {
  assert.deepStrictEqual({ ...REDUCE_DTYPE }, { ...REDUCE_EXPECTED });
  assert.deepStrictEqual(JSON.parse(JSON.stringify(PROMOTE_DIV)), PROMOTE_DIV_EXPECTED);
});

test("dt3 sum/mean: result dtype, data class and shape for every receiver dtype x (0-arg | axis | negative axis | keepdims)", () => {
  type Form = { axis: number | undefined; keep: boolean | undefined; shape: number[] };
  const inShape = [2, 3, 2];
  const forms: Form[] = [
    { axis: undefined, keep: undefined, shape: [] },
    { axis: 0, keep: undefined, shape: [3, 2] },
    { axis: 1, keep: undefined, shape: [2, 2] },
    { axis: -1, keep: undefined, shape: [2, 3] },
    { axis: 0, keep: true, shape: [1, 3, 2] },
    { axis: -2, keep: true, shape: [2, 1, 2] },
    { axis: undefined, keep: true, shape: [1, 1, 1] },
    { axis: 1, keep: false, shape: [2, 2] },
  ];
  for (const dtype of DTYPES) {
    const nd = mkNd(inShape, [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1], dtype);
    for (const f of forms) {
      const callSum = nd.sum(f.axis, f.keep);
      const callMean = nd.mean(f.axis, f.keep);
      for (const [name, r] of [["sum", callSum], ["mean", callMean]] as const) {
        const ctx = `${name}(${f.axis}, ${f.keep}) [${dtype}]`;
        assert.strictEqual(r.dtype, REDUCE_EXPECTED[dtype], `${ctx}: dtype`);
        assert.ok(r.data instanceof CLASS_OF(REDUCE_EXPECTED[dtype]) && r.data.constructor === CLASS_OF(REDUCE_EXPECTED[dtype]), `${ctx}: data class`);
        assert.deepStrictEqual([...r.shape], f.shape, `${ctx}: shape`);
        assert.strictEqual(r.data.length, prod(f.shape), `${ctx}: data length matches shape`);
      }
    }
  }
});

test("dt3 sum/mean float64 path: bit-identical to the pre-dt3 sumRuntime/meanRuntime oracles — also for exactly-widened int32/bool receivers (float32 has its own accumulator, next test)", () => {
  const rng = makeRng(0xd731n);
  for (let t = 0; t < 120; t++) {
    const shape = genShape(rng, 1, 3);
    const size = prod(shape);
    const axes: (number | undefined)[] = [undefined, ...shape.flatMap((_, i) => [i, i - shape.length])];
    const f64 = genDataSpecial(rng, shape);
    const i32 = Int32Array.from({ length: size }, () => rng.nextInt(-2147483648, 2147483647));
    const bool = Uint8Array.from({ length: size }, () => (rng.nextBool() ? 1 : 0));
    const cases: [DType, ArrayLike<number>, Float64Array][] = [
      ["float64", f64, f64],
      ["int32", i32, Float64Array.from(i32)],
      ["bool", bool, Float64Array.from(bool)],
    ];
    for (const [dtype, values, widened] of cases) {
      const nd = mkNd(shape, values, dtype);
      for (const axis of axes) {
        const sOracle = sumRuntime(shape, widened, axis);
        const mOracle = meanRuntime(shape, widened, axis);
        const s = axis === undefined ? nd.sum() : nd.sum(axis);
        const m = axis === undefined ? nd.mean() : nd.mean(axis);
        assert.strictEqual(s.dtype, "float64");
        assert.strictEqual(m.dtype, "float64");
        assertSameValues(sOracle.data, s.data, false, `sum axis=${axis} [${dtype}] shape=[${shape}]`);
        assertSameValues(mOracle.data, m.data, false, `mean axis=${axis} [${dtype}] shape=[${shape}]`);
        assert.deepStrictEqual([...s.shape], sOracle.shape);
      }
    }
  }
});

test("dt3 sum/mean float32: bit-identical to an INDEPENDENT Float32Array-store reference (finite and wild payloads, every axis, ranks 1-3)", () => {
  const rng = makeRng(0xf32511n);
  let checked = 0;
  for (const mode of ["finite", "wild"] as const) {
    for (let t = 0; t < 250; t++) {
      const shape = genShape(rng, 1, 3);
      const data = genF32(rng, prod(shape), mode);
      const nd = mkNd(shape, data, "float32");
      const axes: (number | undefined)[] = [undefined, ...shape.flatMap((_, i) => [i, i - shape.length])];
      for (const axis of axes) {
        const ctx = `[${mode}] shape=[${shape}] axis=${axis}`;
        const rs = refSumF32(shape, data, axis);
        const rm = refMeanF32(shape, data, axis);
        const s = axis === undefined ? nd.sum() : nd.sum(axis);
        const m = axis === undefined ? nd.mean() : nd.mean(axis);
        assert.ok(s.data instanceof Float32Array && m.data instanceof Float32Array, `${ctx}: Float32Array result`);
        assert.deepStrictEqual([...s.shape], rs.shape, `${ctx}: shape`);
        assertSameValues(rs.data, s.data, true, `sum ${ctx}`);
        assertSameValues(rm.data, m.data, true, `mean ${ctx}`);
        checked++;
      }
    }
  }
  assert.ok(checked > 1500, `exercised ${checked} float32 sum/mean cases`);
});

test("dt3 sum/mean float32: ±0 seed, subnormals, f32 overflow to Infinity that float64 would not give, and the 2^24 saturation (n = 2^24+8 ones: sum 16,777,216, mean 0.99999952)", () => {
  // Seed is +0: a sum of negative zeros is +0 (seed + -0 = +0), like the f64 path.
  assert.ok(Object.is(mkNd([2], [-0, -0], "float32").sum().data[0], 0), "f32 sum([-0,-0]) is +0");
  // Subnormals accumulate exactly: 1e-45 stores as the smallest float32 subnormal 2^-149, three of them sum to 3 * 2^-149.
  assert.strictEqual(mkNd([3], [1e-45, 1e-45, 1e-45], "float32").sum().data[0], 3 * 2 ** -149);
  // Overflow: 3e38 + 3e38 exceeds float32 max -> Infinity; the same values in float64 do not overflow.
  const big = [3e38, 3e38];
  assert.strictEqual(mkNd([2], big, "float32").sum().data[0], Infinity);
  assert.ok(Number.isFinite(mkNd([2], big, "float64").sum().data[0] ?? NaN));
  // 2^24 + 8 ones: an ascending float32 accumulator saturates at 2^24 (documented behavior, not pairwise).
  const n = 2 ** 24 + 8;
  const ones = new Float32Array(n).fill(1);
  const nd = loose(NDArray.fromArray([n], ones));
  assert.strictEqual(nd.sum().data[0], 16777216);
  assert.strictEqual(nd.mean().data[0], new Float32Array([16777216 / n])[0]);
  assert.ok(Math.abs((nd.mean().data[0] ?? 0) - 0.99999952) < 1e-8, "mean is the saturated 0.99999952");
  // n = 2^24 + 9 is NOT a float32 (it would round to 2^24 + 8): the divisor must be the EXACT integer n, not fround(n).
  // Non-vacuity first: the two divisors really give different float32 means for this sum.
  const n9 = 2 ** 24 + 9;
  assert.notStrictEqual(Math.fround(16777216 / n9), Math.fround(16777216 / Math.fround(n9)), "the exact-n and fround(n) divisors must differ here");
  const nd9 = loose(NDArray.fromArray([n9], new Float32Array(n9).fill(1)));
  assert.strictEqual(nd9.sum().data[0], 16777216);
  assert.strictEqual(nd9.mean().data[0], new Float32Array([16777216 / n9])[0], "mean divides by the exact n");
});

test("dt3 sum int32: exact in float64 without wrap beyond 2^31 (BigInt oracle, per-step float64 rounding)", () => {
  const i32 = mkNd([3], [2147483647, 2147483647, 2147483647], "int32");
  assert.strictEqual(i32.sum().data[0], 6442450941, "sum exceeds 2^31 and is NOT wrapped");
  assert.strictEqual(i32.mean().data[0], 2147483647);
  const rng = makeRng(0x1325n);
  for (let t = 0; t < 60; t++) {
    const n = rng.nextInt(1, 40);
    const vals = Int32Array.from({ length: n }, () => rng.nextInt(-2147483648, 2147483647));
    const nd = mkNd([n], vals, "int32");
    assert.strictEqual(nd.sum().data[0], bigSum(vals), `int32 sum n=${n}`);
  }
});

test("dt3 sum/mean on bool: sum COUNTS true, mean is the PROPORTION (0-arg, axis, keepdims, size-0)", () => {
  const b = mkNd([2, 3], [1, 0, 1, 1, 1, 0], "bool");
  assert.strictEqual(b.sum().dtype, "float64");
  assert.deepStrictEqual([...b.sum().data], [4]);
  assert.deepStrictEqual([...b.mean().data], [4 / 6]);
  assert.deepStrictEqual([...b.sum(0).data], [2, 1, 1]);
  assert.deepStrictEqual([...b.sum(1).data], [2, 2]);
  assert.deepStrictEqual([...b.mean(1).data], [2 / 3, 2 / 3]);
  assert.deepStrictEqual([...b.sum(1, true).shape], [2, 1]);
  const none = mkNd([0], [], "bool");
  assert.deepStrictEqual([...none.sum().data], [0]);
  assert.ok(Number.isNaN(none.mean().data[0]));
});

test("dt3 matmul: result dtype and data class for all nine numeric dtype pairs = PROMOTE_DIV (float32 only for float32 x float32)", () => {
  for (const da of NUMERIC_DTYPES) {
    for (const db of NUMERIC_DTYPES) {
      const a = mkNd([2, 3], [1, 2, 3, 4, 5, 6], da);
      const b = mkNd([3, 2], [1, 0, 2, 1, 0, 3], db);
      const r = a.matmul(b);
      const expected = PROMOTE_DIV_EXPECTED[da][db];
      assert.strictEqual(r.dtype, expected, `matmul [${da}]x[${db}] dtype`);
      assert.ok(r.data.constructor === CLASS_OF(expected), `matmul [${da}]x[${db}] data class`);
      assert.deepStrictEqual([...r.shape], [2, 2]);
      assert.deepStrictEqual([...r.data], [5, 11, 14, 23], `matmul [${da}]x[${db}] values`);
    }
  }
  // int32 x int32 widens to float64 (NOT int32, unlike add/sub/mul) — O1.
  assert.strictEqual(mkNd([1, 1], [2], "int32").matmul(mkNd([1, 1], [3], "int32")).dtype, "float64");
});

test("dt3 matmul float32 x float32: bit-identical to an INDEPENDENT Float32Array-store reference (2-D, batch broadcast, 1-D promotion both sides, finite and wild payloads)", () => {
  const rng = makeRng(0x3a71n);
  const shapePairs: [number[], number[]][] = [
    [[2, 3], [3, 4]],
    [[1, 5], [5, 1]],
    [[4, 2, 3], [3, 2]],
    [[2, 3], [5, 3, 4]],
    [[2, 1, 2, 3], [3, 3, 2]],
    [[2, 2, 3], [2, 3, 4]],
    [[3], [3, 4]],
    [[2, 3], [3]],
    [[3, 2, 3], [3]],
    [[3], [4, 3, 2]],
    [[3], [3]],
  ];
  let checked = 0;
  for (const mode of ["finite", "wild"] as const) {
    for (const [sa, sb] of shapePairs) {
      for (let t = 0; t < 12; t++) {
        const A = genF32(rng, prod(sa), mode);
        const B = genF32(rng, prod(sb), mode);
        const r = mkNd(sa, A, "float32").matmul(mkNd(sb, B, "float32"));
        const ref = refMatmulF32(sa, A, sb, B);
        const ctx = `[${mode}] ${JSON.stringify(sa)} @ ${JSON.stringify(sb)}`;
        assert.strictEqual(r.dtype, "float32", ctx);
        assert.ok(r.data instanceof Float32Array, `${ctx}: Float32Array`);
        assert.deepStrictEqual([...r.shape], ref.shape, `${ctx}: shape`);
        assertSameValues(ref.data, r.data, true, ctx);
        checked++;
      }
    }
  }
  assert.ok(checked >= 130);
});

test("dt3 matmul float64-compute pairs (float64 / int32 / mixed / float32 x float64): bit-identical to the pre-dt3 matmulRuntime on exactly-widened operands — batch-broadcast and 1-D promotion in BOTH directions", () => {
  const rng = makeRng(0x7712n);
  const shapePairs: [number[], number[]][] = [
    [[2, 3], [3, 4]],
    [[4, 2, 3], [3, 2]],
    [[2, 1, 2, 3], [3, 3, 2]],
    [[3], [3, 4]],
    [[2, 3], [3]],
    [[3], [4, 3, 2]],
    [[5, 3], [3]],
  ];
  const pairs: [DType, DType][] = [
    ["float64", "float64"],
    ["int32", "int32"],
    ["float64", "int32"],
    ["int32", "float64"],
    ["float32", "float64"],
    ["float64", "float32"],
    ["float32", "int32"],
    ["int32", "float32"],
  ];
  const gen = (d: DType, n: number): DataOfRuntime =>
    d === "int32" ? Int32Array.from({ length: n }, () => rng.nextInt(-1000, 1000)) : d === "float32" ? genF32(rng, n, "finite") : genDataSpecial(rng, [n]);
  for (const [sa, sb] of shapePairs) {
    for (const [da, db] of pairs) {
      const a = gen(da, prod(sa));
      const b = gen(db, prod(sb));
      const r = mkNd(sa, a, da).matmul(mkNd(sb, b, db));
      const oracle = matmulRuntime(sa, Float64Array.from(a), sb, Float64Array.from(b));
      const ctx = `[${da}]${JSON.stringify(sa)} @ [${db}]${JSON.stringify(sb)}`;
      assert.strictEqual(r.dtype, "float64", ctx);
      assert.deepStrictEqual([...r.shape], oracle.shape, `${ctx}: shape`);
      assertSameValues(oracle.data, r.data, false, ctx);
    }
  }
});

test("dt3 dot/norm/cosineSimilarity float32 x float32: bit-identical to an INDEPENDENT Float32Array-store reference (finite and wild payloads), result is a float32-valued number", () => {
  const rng = makeRng(0xc05151n);
  for (const mode of ["finite", "wild"] as const) {
    for (let t = 0; t < 300; t++) {
      const n = rng.nextInt(0, 40);
      const a = genF32(rng, n, mode);
      const b = genF32(rng, n, mode);
      const na = mkNd([n], a, "float32");
      const nb = mkNd([n], b, "float32");
      const ctx = `[${mode}] n=${n}`;
      assertSameValues([refDotF32(a, b)], [na.dot(nb)], true, `dot ${ctx}`);
      assertSameValues([refNormF32(a)], [na.norm()], true, `norm ${ctx}`);
      assertSameValues([refCosF32(a, b)], [na.cosineSimilarity(nb)], true, `cosineSimilarity ${ctx}`);
      assert.strictEqual(typeof na.dot(nb), "number");
    }
  }
});

test("dt3 dot/norm/cosineSimilarity float64-compute pairs: bit-identical to the pre-dt3 formulas (dotRuntime/normSqRuntime) on exactly-widened operands", () => {
  const rng = makeRng(0xd07d07n);
  const pairs: [DType, DType][] = [
    ["float64", "float64"],
    ["int32", "int32"],
    ["float64", "int32"],
    ["int32", "float64"],
    ["float32", "float64"],
    ["float64", "float32"],
    ["float32", "int32"],
    ["int32", "float32"],
  ];
  const gen = (d: DType, n: number): DataOfRuntime =>
    d === "int32" ? Int32Array.from({ length: n }, () => rng.nextInt(-100000, 100000)) : d === "float32" ? genF32(rng, n, "finite") : genData(rng, [n]);
  for (const [da, db] of pairs) {
    for (let t = 0; t < 30; t++) {
      const n = rng.nextInt(0, 12);
      const a = gen(da, n);
      const b = gen(db, n);
      const a64 = Float64Array.from(a);
      const b64 = Float64Array.from(b);
      const na = mkNd([n], a, da);
      const nb = mkNd([n], b, db);
      const ctx = `[${da}]x[${db}] n=${n}`;
      const num = dotRuntime([n], a64, [n], b64);
      assertSameValues([num], [na.dot(nb)], false, `dot ${ctx}`);
      const den = Math.sqrt(normSqRuntime(a64)) * Math.sqrt(normSqRuntime(b64));
      assertSameValues([num / den], [na.cosineSimilarity(nb)], false, `cosineSimilarity ${ctx}`);
    }
    // norm depends on the receiver only (int32 / float64 widen exactly; float32 has its own test above).
    if (da !== "float32") {
      const n = 9;
      const a = gen(da, n);
      assertSameValues([Math.sqrt(normSqRuntime(Float64Array.from(a)))], [mkNd([n], a, da).norm()], false, `norm [${da}]`);
    }
  }
});

test("dt3 int32 magnitudes beyond 2^53 are ROUNDED in float64, never wrapped: matmul/dot/norm/cosineSimilarity vs a BigInt oracle that rounds each step (product, then sum)", () => {
  const rng = makeRng(0xb161n);
  const I32 = [-2147483648, 2147483647, 2147483646, -2147483647, 65536, -65536, 94906265, 1];
  const gen = (n: number) => Int32Array.from({ length: n }, () => (rng.nextInt(0, 99) < 70 ? (I32[rng.nextInt(0, I32.length - 1)] ?? 0) : rng.nextInt(-2147483648, 2147483647)));
  // Anchor: one product > 2^53 where wrap and rounding differ (2147483647^2 = 4611686014132420609).
  const anchor = mkNd([1], [2147483647], "int32");
  assert.strictEqual(anchor.dot(anchor), 4611686014132420609, "2147483647^2 = 4611686014132420609 is rounded to the nearest double (…608), not wrapped");
  for (let t = 0; t < 80; t++) {
    const n = rng.nextInt(1, 12);
    const a = gen(n);
    const b = gen(n);
    const na = mkNd([n], a, "int32");
    const nb = mkNd([n], b, "int32");
    assert.strictEqual(na.dot(nb), bigDot(a, b), `dot n=${n}`);
    assert.strictEqual(na.norm(), Math.sqrt(bigDot(a, a)), `norm n=${n}`);
    assertSameValues([bigDot(a, b) / (Math.sqrt(bigDot(a, a)) * Math.sqrt(bigDot(b, b)))], [na.cosineSimilarity(nb)], false, `cosineSimilarity n=${n}`);
    // matmul: [2,n] @ [n,2] (2-D core), every output element against the BigInt oracle.
    const A = gen(2 * n);
    const B = gen(n * 2);
    const r = mkNd([2, n], A, "int32").matmul(mkNd([n, 2], B, "int32"));
    const expect: number[] = [];
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        expect.push(bigDot(Array.from({ length: n }, (_, q) => A[i * n + q] ?? 0), Array.from({ length: n }, (_, q) => B[q * 2 + j] ?? 0)));
      }
    }
    assertSameValues(expect, r.data, false, `matmul [2,${n}]@[${n},2]`);
  }
});

test("dt3 cosineSimilarity float32: the float32 nsq underflows to 0 and overflows to Infinity where float64 would not (pinned against the independent reference AND the float64 contrast)", () => {
  // nsq underflow: (1e-30)^2 = 1e-60 is below the float32 range -> 0; num = 0 too -> 0/0 = NaN. float64 -> 1.
  const tiny = Float32Array.from([1e-30, 1e-30]);
  const tinyF32 = mkNd([2], tiny, "float32");
  assert.ok(Number.isNaN(tinyF32.cosineSimilarity(tinyF32)), "float32: underflow -> NaN");
  assertSameValues([refCosF32(tiny, tiny)], [tinyF32.cosineSimilarity(tinyF32)], true, "tiny vs reference");
  assert.ok(Math.abs(mkNd([2], tiny, "float64").cosineSimilarity(mkNd([2], tiny, "float64")) - 1) < 1e-12, "float64 contrast: ~1");
  // nsq overflow: a = [1e30] -> nsq(a) = 1e60 -> Infinity (float32); b = [1]: num = 1e30 (finite), den = Inf * 1 = Inf -> 0. float64 -> 1.
  const a = mkNd([1], [1e30], "float32");
  const b = mkNd([1], [1], "float32");
  assert.strictEqual(a.cosineSimilarity(b), 0, "float32: overflow of nsq(a) -> den Infinity -> 0");
  assertSameValues([refCosF32(Float32Array.of(1e30), Float32Array.of(1))], [a.cosineSimilarity(b)], true, "overflow vs reference");
  assert.ok(Math.abs(mkNd([1], [1e30], "float64").cosineSimilarity(mkNd([1], [1], "float64")) - 1) < 1e-12, "float64 contrast: ~1");
});

test("dt3 norm(): bool is a PERMANENT runtime-only rejection (BOOL_ARITHMETIC_MESSAGE) — non-empty AND size-0 (A3(a), COVENANT M2 v10); the other dtypes compute", () => {
  assert.throws(() => mkNd([3], [1, 0, 1], "bool").norm(), boolMsgRegex, "norm() on bool, non-empty");
  assert.throws(() => mkNd([0], [], "bool").norm(), boolMsgRegex, "norm() on bool, size-0");
  assert.throws(() => mkNd([2, 2], [1, 0, 1, 0], "bool").norm(), boolMsgRegex, "norm() on bool, rank 2");
  assert.strictEqual(mkNd([2], [3, 4], "float64").norm(), 5);
  assert.strictEqual(mkNd([2], [3, 4], "float32").norm(), 5);
  assert.strictEqual(mkNd([2], [3, 4], "int32").norm(), 5);
  assert.strictEqual(mkNd([0], [], "float32").norm(), 0);
  assert.strictEqual(mkNd([0], [], "int32").norm(), 0);
});

test("dt3 rank 0 and size-0 per dtype: sum/mean, matmul with k = 0, empty dot/norm/cosineSimilarity", () => {
  for (const dtype of DTYPES) {
    const ctx = `[${dtype}]`;
    const expectedDtype = REDUCE_EXPECTED[dtype];
    // Rank 0: sum()/mean() of a scalar are the scalar itself (n = 1); an axis is out of range.
    const scalar = mkNd([], [dtype === "bool" ? 1 : 7], dtype);
    assert.deepStrictEqual([...scalar.sum().shape], [], `${ctx} rank-0 sum shape`);
    assert.deepStrictEqual([...scalar.sum().data], [dtype === "bool" ? 1 : 7], `${ctx} rank-0 sum`);
    assert.deepStrictEqual([...scalar.mean().data], [dtype === "bool" ? 1 : 7], `${ctx} rank-0 mean`);
    assert.strictEqual(scalar.sum().dtype, expectedDtype);
    assert.throws(() => scalar.sum(0), /reduce: axis 0 is out of range for shape \[\] \(rank 0\)/, `${ctx} rank-0 sum(0)`);
    // size-0: sum -> 0, mean -> NaN (NumPy-conformant), in every dtype; a size-0 axis likewise.
    const empty = mkNd([0], [], dtype);
    assert.deepStrictEqual([...empty.sum().data], [0], `${ctx} empty sum`);
    assert.ok(Number.isNaN(empty.mean().data[0]), `${ctx} empty mean is NaN`);
    const zeroRows = mkNd([0, 3], [], dtype);
    assert.deepStrictEqual([...zeroRows.sum(0).data], [0, 0, 0], `${ctx} size-0 axis sum`);
    assert.ok([...zeroRows.mean(0).data].every((v) => Number.isNaN(v)), `${ctx} size-0 axis mean`);
    assert.deepStrictEqual([...zeroRows.sum(1).shape], [0], `${ctx} size-0 other axis: shape`);
  }
  for (const da of NUMERIC_DTYPES) {
    for (const db of NUMERIC_DTYPES) {
      const ctx = `[${da}]x[${db}]`;
      const k0 = mkNd([2, 0], [], da).matmul(mkNd([0, 3], [], db));
      assert.deepStrictEqual([...k0.shape], [2, 3], `${ctx} matmul k=0 shape`);
      assert.deepStrictEqual([...k0.data], [0, 0, 0, 0, 0, 0], `${ctx} matmul k=0 is zeros`);
      assert.strictEqual(k0.dtype, PROMOTE_DIV_EXPECTED[da][db]);
      assert.strictEqual(mkNd([0], [], da).dot(mkNd([0], [], db)), 0, `${ctx} empty dot`);
      assert.ok(Number.isNaN(mkNd([0], [], da).cosineSimilarity(mkNd([0], [], db))), `${ctx} empty cosineSimilarity is NaN`);
    }
  }
});

test("dt3 error messages: the float32 compute paths throw the SAME messages as the float64 paths and as the pre-dt3 runtime functions (D5), per error", () => {
  const msgOf = (fn: () => unknown): string => {
    let m = "";
    assert.throws(fn, (e: unknown) => {
      m = (e as Error).message;
      return true;
    });
    return m;
  };
  // sum/mean axis out of range (positive, negative, rank 0).
  for (const [shape, axis] of [[[2, 3], 2], [[2, 3], -3], [[], 0], [[0], 1]] as [number[], number][]) {
    const oracleMsg = msgOf(() => sumRuntime(shape, new Float64Array(prod(shape)), axis));
    for (const dtype of DTYPES) {
      const nd = mkNd(shape, new Array<number>(prod(shape)).fill(0), dtype);
      assert.strictEqual(msgOf(() => nd.sum(axis)), oracleMsg, `sum(${axis}) [${dtype}] shape=[${shape}]`);
      assert.strictEqual(msgOf(() => nd.mean(axis)), oracleMsg, `mean(${axis}) [${dtype}] shape=[${shape}]`);
    }
  }
  // matmul: scalar first/second, inner mismatch, batch-broadcast mismatch — float32 vs float64 vs the old function.
  const mm: [number[], number[]][] = [[[], [2, 2]], [[2, 2], []], [[2, 3], [4, 2]], [[2, 2, 3], [3, 3, 2]], [[3], [4, 2]], [[2, 3], [4]]];
  for (const [sa, sb] of mm) {
    const oracleMsg = msgOf(() => matmulRuntime(sa, new Float64Array(prod(sa)), sb, new Float64Array(prod(sb))));
    for (const [da, db] of [["float64", "float64"], ["float32", "float32"], ["int32", "float64"], ["float32", "float64"]] as [DType, DType][]) {
      const a = mkNd(sa, new Array<number>(prod(sa)).fill(0), da);
      const b = mkNd(sb, new Array<number>(prod(sb)).fill(0), db);
      assert.strictEqual(msgOf(() => a.matmul(b)), oracleMsg, `matmul ${JSON.stringify(sa)} @ ${JSON.stringify(sb)} [${da}]x[${db}]`);
    }
  }
  // dot/cosineSimilarity: operand rank and length mismatches.
  const vp: [number[], number[]][] = [[[2, 2], [4]], [[4], [2, 2]], [[3], [4]]];
  for (const [sa, sb] of vp) {
    for (const op of ["dot", "cosineSimilarity"] as const) {
      const f64a = mkNd(sa, new Array<number>(prod(sa)).fill(0), "float64");
      const f64b = mkNd(sb, new Array<number>(prod(sb)).fill(0), "float64");
      const oracleMsg = msgOf(() => f64a[op](f64b));
      assert.ok(new RegExp(`^${op}: `).test(oracleMsg), `${op} message prefix: ${oracleMsg}`);
      for (const dtype of ["float32", "int32"] as const) {
        const a = mkNd(sa, new Array<number>(prod(sa)).fill(0), dtype);
        const b = mkNd(sb, new Array<number>(prod(sb)).fill(0), dtype);
        assert.strictEqual(msgOf(() => a[op](b)), oracleMsg, `${op} ${JSON.stringify(sa)} / ${JSON.stringify(sb)} [${dtype}]`);
      }
    }
  }
});

test("dt3 check order (D4): runtime matches the editor — a SHAPE error outranks the bool rejection; with valid shapes bool is rejected", () => {
  const boolMat = mkNd([2, 3], [1, 0, 1, 0, 1, 0], "bool");
  const f64Wrong = mkNd([4, 2], [1, 2, 3, 4, 5, 6, 7, 8], "float64");
  const boolVec3 = mkNd([3], [1, 0, 1], "bool");
  const f64Vec4 = mkNd([4], [1, 2, 3, 4], "float64");
  const f64Mat = mkNd([2, 2], [1, 2, 3, 4], "float64");
  // bool receiver + shape mismatch -> the SHAPE message.
  assert.throws(() => boolMat.matmul(f64Wrong), /^Error: matmul: inner dimensions 3 and 4 do not match$/);
  assert.throws(() => mkNd([2, 3], [1, 2, 3, 4, 5, 6], "float64").matmul(mkNd([4, 2], [1, 0, 1, 0, 1, 0, 1, 0], "bool")), /^Error: matmul: inner dimensions 3 and 4 do not match$/);
  // bool receiver + a batch-broadcast mismatch ([2] vs [3] batch dims; inner dims agree) -> the broadcast message.
  assert.throws(
    () => mkNd([2, 2, 3], new Array<number>(12).fill(1), "bool").matmul(mkNd([3, 3, 2], new Array<number>(18).fill(1), "float64")),
    /^Error: broadcast: shapes \[2\] and \[3\] are not broadcast-compatible at axis -1/,
  );
  assert.throws(() => boolVec3.dot(f64Vec4), /^Error: dot: vector lengths 3 and 4 do not match$/);
  assert.throws(() => f64Vec4.dot(boolVec3), /^Error: dot: vector lengths 4 and 3 do not match$/);
  assert.throws(() => boolVec3.cosineSimilarity(f64Vec4), /^Error: cosineSimilarity: vector lengths 3 and 4 do not match$/);
  assert.throws(() => boolMat.dot(boolMat), /^Error: dot: expected a 1-D vector as the first operand/, "rank error outranks bool, dot");
  assert.throws(() => boolVec3.dot(f64Mat), /^Error: dot: expected a 1-D vector as the second operand/, "rank error outranks bool, dot (second operand)");
  assert.throws(() => boolMat.cosineSimilarity(boolMat), /^Error: cosineSimilarity: expected a 1-D vector as the first operand/);
  assert.throws(() => mkNd([], [1], "bool").matmul(f64Mat), /^Error: matmul: scalar operand \(rank 0\) is not allowed as the first argument/);
  // Valid shapes: bool is rejected with BOOL_ARITHMETIC_MESSAGE.
  assert.throws(() => boolMat.matmul(mkNd([3, 2], [1, 2, 3, 4, 5, 6], "float64")), boolMsgRegex);
  assert.throws(() => mkNd([3], [1, 2, 3], "float64").dot(boolVec3), boolMsgRegex);
  assert.throws(() => mkNd([3], [1, 2, 3], "float64").cosineSimilarity(boolVec3), boolMsgRegex);
});

test("dt3 views (transposed / sliced / composed): classified by exact shape + strides right after construction, then every reduction and product matches the oracle on the hand-built expected contents", () => {
  const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
  const boolValues = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0];
  for (const dtype of DTYPES) {
    const vals = dtype === "bool" ? boolValues : values;
    const base = NDArray.fromArray([3, 4], vals, { dtype });
    const stored = Array.from(base.data); // the converted contents (float32 rounding / bool 0-1)
    // Transposed view [4,3]: expected[j*3+i] = base[i*4+j].
    const t = base.transpose();
    assert.deepStrictEqual([...t.shape], [4, 3], `[${dtype}] transpose shape`);
    assert.deepStrictEqual([...t.strides], [3, 1], `[${dtype}] transpose strides (natural row-major, fresh buffer)`);
    const tExpected = Array.from({ length: 12 }, (_, f) => stored[(f % 3) * 4 + Math.floor(f / 3)] ?? 0);
    // Sliced view rows 1..2 (start 1, stop 3) -> [2,4]: contents stored[4..11].
    const s = base.slice(...wideSpecs({ start: 1, stop: 3 }, null));
    assert.deepStrictEqual([...s.shape], [2, 4], `[${dtype}] slice shape`);
    assert.deepStrictEqual([...s.strides], [4, 1], `[${dtype}] slice strides`);
    const sExpected = stored.slice(4, 12);
    // Composed: slice then transpose -> [4,2]: expected[j*2+i] = sExpected[i*4+j].
    const c = s.transpose();
    assert.deepStrictEqual([...c.shape], [4, 2], `[${dtype}] composed shape`);
    assert.deepStrictEqual([...c.strides], [2, 1], `[${dtype}] composed strides`);
    const cExpected = Array.from({ length: 8 }, (_, f) => sExpected[(f % 2) * 4 + Math.floor(f / 2)] ?? 0);
    for (const [name, view, expected, shape] of [
      ["transposed", t, tExpected, [4, 3]],
      ["sliced", s, sExpected, [2, 4]],
      ["composed", c, cExpected, [4, 2]],
    ] as const) {
      const v = loose(view);
      const ctx = `[${dtype}] ${name}`;
      assert.deepStrictEqual([...v.data], expected, `${ctx}: contents`);
      for (const axis of [undefined, 0, 1, -1]) {
        const got = axis === undefined ? v.sum() : v.sum(axis);
        const gotMean = axis === undefined ? v.mean() : v.mean(axis);
        if (dtype === "float32") {
          const f32 = Float32Array.from(expected);
          assertSameValues(refSumF32(shape, f32, axis).data, got.data, true, `${ctx} sum axis=${axis}`);
          assertSameValues(refMeanF32(shape, f32, axis).data, gotMean.data, true, `${ctx} mean axis=${axis}`);
        } else {
          assertSameValues(sumRuntime(shape, Float64Array.from(expected), axis).data, got.data, false, `${ctx} sum axis=${axis}`);
          assertSameValues(meanRuntime(shape, Float64Array.from(expected), axis).data, gotMean.data, false, `${ctx} mean axis=${axis}`);
        }
      }
    }
    // Products over views: transposed @ original ([4,3] @ [3,4]) and sliced @ composed ([2,4] @ [4,2]).
    if (dtype !== "bool") {
      const p1 = loose(t).matmul(loose(base));
      const p2 = loose(s).matmul(loose(c));
      const e1 = dtype === "float32" ? refMatmulF32([4, 3], Float32Array.from(tExpected), [3, 4], Float32Array.from(stored)) : matmulRuntime([4, 3], Float64Array.from(tExpected), [3, 4], Float64Array.from(stored));
      const e2 = dtype === "float32" ? refMatmulF32([2, 4], Float32Array.from(sExpected), [4, 2], Float32Array.from(cExpected)) : matmulRuntime([2, 4], Float64Array.from(sExpected), [4, 2], Float64Array.from(cExpected));
      assertSameValues(e1.data, p1.data, dtype === "float32", `[${dtype}] transposed @ original`);
      assertSameValues(e2.data, p2.data, dtype === "float32", `[${dtype}] sliced @ composed`);
    }
  }
});

test("dt3 results are fresh buffers: inputs are never mutated or aliased by sum/mean/matmul (float64 inputs are NOT copied on the f64 path, yet never written)", () => {
  for (const dtype of DTYPES) {
    const nd = mkNd([2, 2], [1, 2, 3, 4].map((v) => (dtype === "bool" ? v % 2 : v)), dtype);
    const before = Array.from(nd.data);
    const results = [nd.sum(), nd.mean(), nd.sum(0), nd.mean(1)];
    for (const r of results) assert.notStrictEqual(r.data, nd.data, `[${dtype}] result must not alias the input buffer`);
    if (dtype !== "bool") results.push(nd.matmul(nd));
    for (const r of results) assert.notStrictEqual(r.data.buffer, nd.data.buffer, `[${dtype}] result buffer is fresh`);
    assert.deepStrictEqual(Array.from(nd.data), before, `[${dtype}] input untouched`);
  }
});

// --- dt3 diagnostic CONTENTS (M3, Arbeitsregel 2): real tsc on a throwaway
// fixture OUTSIDE the repo. The bool rejection and the shape errors of
// matmul/dot/cosineSimilarity surface at the ARGUMENT with word-identical text
// to the runtime messages (never hand-typed: the runtime side comes from real
// thrown errors / the imported constant), and when BOTH a shape error and a bool
// operand are present the SHAPE message wins at compile time exactly as it does
// at runtime (D4). `norm()` on bool compiles clean (A3(a)).
test("dt3 diagnostic CONTENTS (M3): bool rejection and shape errors name word-identical messages at the argument; the shape message outranks bool; norm() on bool compiles clean", () => {
  const runtimeMessage = (fn: () => unknown): string => {
    let m = "";
    assert.throws(fn, (e: unknown) => {
      m = (e as Error).message;
      return true;
    });
    return m;
  };
  const shapeMatmul = runtimeMessage(() => mkNd([2, 3], new Array<number>(6).fill(0), "float32").matmul(mkNd([4, 5], new Array<number>(20).fill(0), "float32")));
  const shapeDot = runtimeMessage(() => mkNd([3], [0, 0, 0], "float32").dot(mkNd([4], [0, 0, 0, 0], "float32")));
  const shapeCos = runtimeMessage(() => mkNd([3], [0, 0, 0], "float32").cosineSimilarity(mkNd([4], [0, 0, 0, 0], "float32")));

  const dir = mkdtempSync(join(tmpdir(), "numtype-dt3-diag-pin-"));
  try {
    const ndarrayPath = fileURLToPath(new URL("../src/ndarray.ts", import.meta.url).href);
    const ambientPath = fileURLToPath(new URL("../src/ambient.d.ts", import.meta.url).href);
    const repoRoot = fileURLToPath(new URL("../..", import.meta.url).href);
    writeFileSync(
      join(dir, "probe.ts"),
      [
        `import { NDArray } from ${JSON.stringify(ndarrayPath)};`, // 1
        `const f32Mat = NDArray.zeros([2, 3], "float32");`, // 2
        `const boolMat34 = NDArray.zeros([3, 4], "bool");`, // 3
        `const boolMat45 = NDArray.zeros([4, 5], "bool");`, // 4
        `const boolVec3 = NDArray.zeros([3], "bool");`, // 5
        `const f32Vec3 = NDArray.zeros([3], "float32");`, // 6
        `const f32Vec4 = NDArray.zeros([4], "float32");`, // 7
        `const boolVec4 = NDArray.zeros([4], "bool");`, // 8
        `f32Mat.matmul(boolMat34);`, // 9  bool ARGUMENT
        `boolVec3.dot(f32Vec3);`, // 10 bool RECEIVER
        `f32Vec3.cosineSimilarity(boolVec3);`, // 11 bool ARGUMENT
        `f32Mat.matmul(boolMat45);`, // 12 shape error outranks the bool argument
        `boolVec3.dot(f32Vec4);`, // 13 length mismatch outranks the bool receiver
        `f32Vec3.cosineSimilarity(boolVec4);`, // 14 length mismatch outranks the bool argument
        `const normOfBool = boolVec3.norm();`, // 15 compiles clean (A3(a))
        `void normOfBool;`, // 16
        ``,
      ].join("\n"),
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
    assert.strictEqual(probeErrors.length, 6, `expected exactly SIX fixture errors (lines 9-14; line 15 norm() must compile):\n${out}`);

    // tsc prints the message as the content of a quoted string-literal type: `JSON.stringify(...).slice(1, -1)`
    // reproduces its escaping, so neither side is hand-typed.
    const escaped = (msg: string) => JSON.stringify(msg).slice(1, -1);
    const quoted = (msg: string) => `"${escaped(msg)}"`;
    const atLine = (n: number): string => probeErrors.find((l) => l.includes(`probe.ts(${n},`)) ?? "";
    const cases: [number, string, string][] = [
      [9, BOOL_ARITHMETIC_MESSAGE, "matmul: bool ARGUMENT"],
      [10, BOOL_ARITHMETIC_MESSAGE, "dot: bool RECEIVER"],
      [11, BOOL_ARITHMETIC_MESSAGE, "cosineSimilarity: bool ARGUMENT"],
      [12, shapeMatmul, "matmul: shape error (3 vs 4) with a bool argument"],
      [13, shapeDot, "dot: length mismatch with a bool receiver"],
      [14, shapeCos, "cosineSimilarity: length mismatch with a bool argument"],
    ];
    for (const [line, msg, what] of cases) {
      assert.ok(atLine(line).includes(quoted(msg)), `${what} (probe line ${line}) must surface "${msg}" verbatim:\n${out}`);
    }
    for (const line of [12, 13, 14]) {
      assert.ok(!atLine(line).includes(escaped(BOOL_ARITHMETIC_MESSAGE)), `the shape error must OUTRANK the bool message at probe line ${line}:\n${out}`);
    }
    assert.strictEqual(atLine(15), "", `norm() on bool must compile clean:\n${out}`);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
