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
import { assertDataBitIdentical, assertShapeEqual } from "./assert-helpers.ts";
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

  const sliced = base.slice(1); // NDArray<[3]>, row 1: [16, 25, 36]
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
  const row = m.slice(1); // shape [3]: [4, 5, 6]
  assert.strictEqual(row.item(0), 4, "sliced row item(0)");
  assert.strictEqual(row.item(1), 5, "sliced row item(1)");
  assert.strictEqual(row.item(2), 6, "sliced row item(2)");
  assert.strictEqual(row.item(-1), 6, "sliced row item(-1)");
});
