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
 */
import assert from "node:assert";
import { test } from "node:test";
import { NDArray } from "../src/ndarray.ts";
import { elementwiseBinary, meanRuntime, sumRuntime } from "../src/runtime.ts";
import { assertDataBitIdentical, assertShapeEqual } from "./assert-helpers.ts";
import { genDataSpecial, makeRng, nextF64Special, SPECIAL_VALUES, type Rng } from "./prng.ts";

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
