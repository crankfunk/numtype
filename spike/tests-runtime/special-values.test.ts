/**
 * Differential tests (Kern 10, docs/kern-10-special-values-spec.md): IEEE-754
 * special-value coverage for the project's "bit-identical" claim. Every
 * prior differential file (add/sum/elementwise/vector/matmul/transpose)
 * draws its operands from `genData`, which by construction never emits
 * NaN/Infinity/-0/subnormals (see `prng.ts`'s `nextF64` doc comment) — so
 * until this file, the claim has only been exercised for normal finite
 * values. This file injects `SPECIAL_VALUES` (via `genDataSpecial`) into
 * every op that has a resident kernel, plus curated deterministic fixtures
 * for the specific failure modes the spec names: SIMD-matmul subnormal
 * flushing, +/-0 accumulation, Inf/NaN propagation, and NaN-payload
 * byte-exactness through pure data movement (transpose).
 *
 * A coherent NEW file rather than an addition to an existing differential
 * file (spec D2: lower blast radius, append-only-friendly, precedent
 * keepdims.test.ts). Listed in package.json's `test:resident` (needs
 * `WNDArray`) — `test-scripts-guard.test.ts` enforces that listing.
 */
import assert from "node:assert";
import { test } from "node:test";
import { NDArray } from "../src/ndarray.ts";
import { dotRuntime, elementwiseBinary, matmulRuntime, normSqRuntime, sumRuntime, transposeRuntime } from "../src/runtime.ts";
import { wasmAdd, wasmMatmul, wasmSum, wasmTranspose } from "../src/wasm/backend.ts";
import { initCore } from "../src/wasm/loader.ts";
import { WNDArray, type AnyWNDArray } from "../src/wasm/resident.ts";
import { assertDataBitIdentical, assertShapeEqual, bitsOf } from "./assert-helpers.ts";
import { genBroadcastShapes, genDataSpecial, makeRng, SPECIAL_VALUES, type Rng } from "./prng.ts";

const core = await initCore();

function genShape(rng: Rng, minRank: number, maxRank: number): number[] {
  const rank = rng.nextInt(minRank, maxRank);
  return Array.from({ length: rank }, () => rng.nextInt(1, 8));
}

/** Scalar bit-identity assertion, `Object.is`-based (same standard as
 * `assertDataBitIdentical`'s per-element check, just for a lone `number`).
 * `vector.test.ts` has an unexported local helper of the same name and
 * contract; replicated here rather than imported/hoisted, matching this
 * file's own "independent blast radius" rationale (spec D2) rather than
 * coupling two differential files together for one three-line helper. */
function assertScalarBitIdentical(expected: number, actual: number, context: string): void {
  assert.ok(
    Object.is(expected, actual),
    `${context}: expected ${expected} (0x${bitsOf(expected).toString(16)}), got ${actual} (0x${bitsOf(actual).toString(16)})`,
  );
}

// A value strictly between 0 (exclusive) and the smallest normal f64
// (2.2250738585072014e-308, exclusive) is a genuine positive subnormal, and
// symmetrically for negative.
const MIN_NORMAL = 2.2250738585072014e-308;
function isSubnormalNonZero(v: number): boolean {
  return v !== 0 && Math.abs(v) < MIN_NORMAL;
}

// =============================================================================
// D1 non-vacuity (mandatory): `genDataSpecial` must actually produce EVERY
// `SPECIAL_VALUES` class over enough draws — a generator bug here would
// silently vacuum every test below it (KB lesson: generator coverage is the
// suite's own burden of proof).
// =============================================================================

test("genDataSpecial: non-vacuity — every SPECIAL_VALUES class appears at least once (specialProb=1, forced-special draws)", () => {
  const rng = makeRng(0x53504543494e5f31n); // "SPECIN_1"-ish
  const data = genDataSpecial(rng, [400], 1.0);
  const values = Array.from(data);
  for (const special of SPECIAL_VALUES) {
    const found = values.some((v) => Object.is(v, special));
    assert.ok(
      found,
      `genDataSpecial(specialProb=1) over 400 draws must produce ${special} (0x${bitsOf(special).toString(16)}) at least once`,
    );
  }
  // Explicit, named restatement of the spec's own emphasis: -0 confirmed
  // Object.is-distinguished from +0, and at least one subnormal confirmed.
  assert.ok(
    values.some((v) => Object.is(v, -0)),
    "genDataSpecial must produce -0 (Object.is-distinguished from +0)",
  );
  assert.ok(
    values.some((v) => Object.is(v, 0) && !Object.is(v, -0)),
    "genDataSpecial must also produce +0, distinctly from -0",
  );
  assert.ok(
    values.some((v) => isSubnormalNonZero(v)),
    "genDataSpecial must produce at least one genuine (nonzero) subnormal",
  );
});

test("genDataSpecial: at the default specialProb=0.35, special draws still appear at roughly the configured rate (sanity band, non-vacuity for the rate every op test below actually uses)", () => {
  const rng = makeRng(0x53504543494e5f30n); // "SPECIN_0"-ish
  const data = genDataSpecial(rng, [2000]); // default specialProb = 0.35
  const specialCount = Array.from(data).filter((v) => SPECIAL_VALUES.some((s) => Object.is(v, s))).length;
  // Expect ~700/2000; a loose band (not a precision binomial test) rules out
  // a generator that silently never special-izes (~0) or always does (~2000).
  assert.ok(
    specialCount > 400 && specialCount < 1000,
    `expected a substantial-but-partial special fraction near 700/2000 at specialProb=0.35, got ${specialCount}/2000`,
  );
});

// =============================================================================
// add: three-way (naive elementwiseBinary(+) / v1 wasmAdd / resident
// WNDArray.add), special-value-injected broadcast operand pairs.
// =============================================================================
{
  const rng = makeRng(0x4144445f53504543n); // "ADD_SPEC"
  const CASE_COUNT = 60;
  for (let c = 0; c < CASE_COUNT; c++) {
    const { aShape, bShape } = genBroadcastShapes(rng);
    const aData = genDataSpecial(rng, aShape);
    const bData = genDataSpecial(rng, bShape);

    test(`add special case ${c}: a=[${aShape.join(",")}] b=[${bShape.join(",")}]`, () => {
      const ref = elementwiseBinary(aShape, aData, bShape, bData, (x, y) => x + y);
      const v1 = wasmAdd(core, aShape, aData, bShape, bData);
      const ctx = `add special case ${c} a=[${aShape.join(",")}] b=[${bShape.join(",")}]`;
      assertShapeEqual(ref.shape, v1.shape, `${ctx} [v1]`);
      assertDataBitIdentical(ref.data, v1.data, `${ctx} [v1]`);

      const a = WNDArray.fromArray(core, aShape, aData);
      const b = WNDArray.fromArray(core, bShape, bData);
      try {
        const got = a.add(b);
        try {
          assertShapeEqual(ref.shape, got.shape, `${ctx} [resident]`);
          assertDataBitIdentical(ref.data, got.toArray(), `${ctx} [resident]`);
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

// =============================================================================
// sub/mul/div: two-way (naive elementwiseBinary / resident WNDArray.{sub,
// mul,div} — no v1 kernel exists for these three, per the project's own
// documented split), special-value-injected broadcast operand pairs.
// =============================================================================
type ElementwiseOp = "sub" | "mul" | "div";
const ELEMENTWISE_OPS: readonly ElementwiseOp[] = ["sub", "mul", "div"];

function refOp(op: ElementwiseOp): (x: number, y: number) => number {
  if (op === "sub") return (x, y) => x - y;
  if (op === "mul") return (x, y) => x * y;
  return (x, y) => x / y;
}

function callResidentElementwise(op: ElementwiseOp, a: AnyWNDArray, b: AnyWNDArray): AnyWNDArray {
  if (op === "sub") return a.sub(b);
  if (op === "mul") return a.mul(b);
  return a.div(b);
}

const ELEMENTWISE_SEED: Record<ElementwiseOp, bigint> = {
  sub: 0x5355425f53504543n, // "SUB_SPEC"
  mul: 0x4d554c5f53504543n, // "MUL_SPEC"
  div: 0x4449565f53504543n, // "DIV_SPEC"
};

for (const op of ELEMENTWISE_OPS) {
  const rng = makeRng(ELEMENTWISE_SEED[op]);
  const CASE_COUNT = 60;
  for (let c = 0; c < CASE_COUNT; c++) {
    const { aShape, bShape } = genBroadcastShapes(rng);
    const aData = genDataSpecial(rng, aShape);
    const bData = genDataSpecial(rng, bShape);

    test(`${op} special case ${c}: a=[${aShape.join(",")}] b=[${bShape.join(",")}]`, () => {
      const ref = elementwiseBinary(aShape, aData, bShape, bData, refOp(op));
      const a = WNDArray.fromArray(core, aShape, aData);
      const b = WNDArray.fromArray(core, bShape, bData);
      try {
        const got = callResidentElementwise(op, a, b);
        try {
          const ctx = `${op} special case ${c} a=[${aShape.join(",")}] b=[${bShape.join(",")}]`;
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

// =============================================================================
// sum: three-way (naive sumRuntime / v1 wasmSum / resident WNDArray.sum),
// both full (axis===undefined) and per-axis, special-value-injected data.
// =============================================================================
{
  const rng = makeRng(0x53554d5f414c5350n); // "SUM_ALSP"
  const CASE_COUNT = 60;
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genDataSpecial(rng, shape);

    test(`sum_all special case ${c}: shape=[${shape.join(",")}]`, () => {
      const ref = sumRuntime(shape, data, undefined);
      const v1 = wasmSum(core, shape, data, undefined);
      const ctx = `sum_all special case ${c} shape=[${shape.join(",")}]`;
      assertShapeEqual(ref.shape, v1.shape, `${ctx} [v1]`);
      assertDataBitIdentical(ref.data, v1.data, `${ctx} [v1]`);

      const w = WNDArray.fromArray(core, shape, data);
      try {
        const got = w.sum();
        try {
          assertShapeEqual(ref.shape, got.shape, `${ctx} [resident]`);
          assertDataBitIdentical(ref.data, got.toArray(), `${ctx} [resident]`);
        } finally {
          got.dispose();
        }
      } finally {
        w.dispose();
      }
    });
  }
}

{
  const rng = makeRng(0x53554d5f41585350n); // "SUM_AXSP"
  const CASE_COUNT = 60;
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 1, 4);
    const data = genDataSpecial(rng, shape);
    const rank = shape.length;
    const positiveAxis = rng.nextInt(0, rank - 1);
    const axis = rng.nextBool() ? positiveAxis - rank : positiveAxis;

    test(`sum_axis special case ${c}: shape=[${shape.join(",")}] axis=${axis}`, () => {
      const ref = sumRuntime(shape, data, axis);
      const v1 = wasmSum(core, shape, data, axis);
      const ctx = `sum_axis special case ${c} shape=[${shape.join(",")}] axis=${axis}`;
      assertShapeEqual(ref.shape, v1.shape, `${ctx} [v1]`);
      assertDataBitIdentical(ref.data, v1.data, `${ctx} [v1]`);

      const w = WNDArray.fromArray(core, shape, data);
      try {
        const got = w.sum(axis);
        try {
          assertShapeEqual(ref.shape, got.shape, `${ctx} [resident]`);
          assertDataBitIdentical(ref.data, got.toArray(), `${ctx} [resident]`);
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
// matmul: three-way (naive matmulRuntime / v1 wasmMatmul / resident
// WNDArray.matmul), special-value-injected square operands at n in
// {2,3,8,16,32}.
//
// Path finding (verified by reading spike/src/wasm/backend.ts and
// spike/src/wasm/resident.ts, not assumed): `wasmMatmul` ALWAYS calls the
// scalar `nt_matmul` entry point; `WNDArray.matmul` ALWAYS calls
// `nt_matmul_blocked` (the packed+SIMD128 kernel) — there is no runtime size
// threshold in EITHER surface (unlike the Kern-06 threaded/non-threaded
// auto-routing elsewhere in this project). The "path" is determined by which
// SURFACE you call, not by matrix size. Reading
// crates/core/src/kernels/matmul_blocked.rs's `micro_tile` confirms every
// n>=2 exercises the `f64x2` `accumulate_pair` micro-step for at least one
// adjacent output-column pair (MC=NC=KC=32, so n in {2,3,8,16,32} all fit in
// a single tile); n=3 additionally exercises the scalar `accumulate_single`
// tail (odd leftover column) in the SAME call — this is where subnormal
// preservation under WASM SIMD128 (no flush-to-zero) is most at risk, hence
// the dedicated curated fixture below.
// =============================================================================
{
  const sizes: readonly number[] = [2, 3, 8, 16, 32];
  const rng = makeRng(0x4d41544d554c5350n); // "MATMULSP"
  const CASES_PER_SIZE = 8;
  for (const n of sizes) {
    for (let c = 0; c < CASES_PER_SIZE; c++) {
      const shape: number[] = [n, n];
      const aData = genDataSpecial(rng, shape);
      const bData = genDataSpecial(rng, shape);

      test(`matmul special case n=${n} #${c}`, () => {
        const ref = matmulRuntime(shape, aData, shape, bData);
        const v1 = wasmMatmul(core, shape, aData, shape, bData);
        const ctx = `matmul special n=${n} #${c}`;
        assertShapeEqual(ref.shape, v1.shape, `${ctx} [v1]`);
        assertDataBitIdentical(ref.data, v1.data, `${ctx} [v1]`);

        const a = WNDArray.fromArray(core, shape, aData);
        const b = WNDArray.fromArray(core, shape, bData);
        try {
          const got = a.matmul(b);
          try {
            assertShapeEqual(ref.shape, got.shape, `${ctx} [resident]`);
            assertDataBitIdentical(ref.data, got.toArray(), `${ctx} [resident]`);
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
}

// -----------------------------------------------------------------------
// Curated fixture: subnormal preservation through the SIMD-blocked matmul
// kernel — the spec's single most valuable new test. WASM SIMD128 has NO
// flush-to-zero (unlike some native SIMD modes); if the blocked kernel (or a
// toolchain stage) flushed subnormals to zero, this would diverge from the
// naive JS reference, which never flushes (V8 respects subnormals in plain
// arithmetic).
//
// Shape 3x2 @ 2x3 (m=3, k=2, n=3): n=3 means every output row exercises BOTH
// the f64x2 accumulate_pair micro-step (columns 0,1) AND the scalar
// accumulate_single tail (column 2) in the SAME call (matmul_blocked.rs's
// micro_tile: nc=3 -> one pair, one single). Every operand entry is 0 or a
// small-integer multiple of Number.MIN_VALUE (the smallest positive
// subnormal); every product/sum below stays deep in subnormal range, nowhere
// near the ~2.2e-308 boundary into normal range, so IEEE-754 f64 arithmetic
// over these exact values is itself exact (no rounding/overflow ambiguity to
// reason about) — hand-verified: out = [[MV,0,2MV],[0,4MV,12MV],[MV,4MV,14MV]].
// -----------------------------------------------------------------------
test("matmul fixture: subnormal preservation through the SIMD-blocked kernel (naive/v1/resident three-way + explicit non-flush check)", () => {
  const MV = Number.MIN_VALUE;
  const aShape = [3, 2];
  const bShape = [2, 3];
  const aData = new Float64Array([MV, 0, 0, MV * 4, MV, MV * 4]);
  const bData = new Float64Array([1, 0, 2, 0, 1, 3]);

  const ref = matmulRuntime(aShape, aData, bShape, bData);
  assert.ok(
    Array.from(ref.data).some(isSubnormalNonZero),
    `fixture setup: expected >=1 subnormal reference output, got [${Array.from(ref.data).join(",")}]`,
  );

  const v1 = wasmMatmul(core, aShape, aData, bShape, bData);
  assertShapeEqual(ref.shape, v1.shape, "subnormal matmul [v1]");
  assertDataBitIdentical(ref.data, v1.data, "subnormal matmul [v1]");

  const a = WNDArray.fromArray(core, aShape, aData);
  const b = WNDArray.fromArray(core, bShape, bData);
  try {
    const got = a.matmul(b);
    try {
      assertShapeEqual(ref.shape, got.shape, "subnormal matmul [resident/SIMD]");
      const gotData = got.toArray();
      assertDataBitIdentical(ref.data, gotData, "subnormal matmul [resident/SIMD]");
      // Explicit, named non-flush check: every subnormal reference element
      // must survive as the SAME nonzero subnormal on the SIMD surface, not
      // flushed to +0/-0.
      for (let i = 0; i < ref.data.length; i++) {
        const r = ref.data[i] ?? 0;
        if (isSubnormalNonZero(r)) {
          const g = gotData[i] ?? 0;
          assert.ok(!Object.is(g, 0) && !Object.is(g, -0), `resident/SIMD element ${i} flushed subnormal ${r} to zero (got ${g})`);
        }
      }
    } finally {
      got.dispose();
    }
  } finally {
    a.dispose();
    b.dispose();
  }
});

// -----------------------------------------------------------------------
// Curated fixtures: +/-0 accumulation in sum, all three surfaces.
//
// SPEC DEVIATION (found empirically, not assumed — see kern-10 executor
// report): the spec text (D2) claims `sum([-0,-0,...]) === -0`. That is
// FALSE for this codebase's actual `sumRuntime`: the accumulator seed is
// `let total = 0` (+0, runtime.ts line ~214, pre-existing/frozen, untouched
// by this phase). IEEE-754 round-to-nearest (JS's only rounding mode)
// defines `+0 + -0 == +0` (verified: `Object.is(0 + -0, -0) === false` in
// Node) — so the VERY FIRST accumulation step poisons the chain to +0
// permanently; no array of any length containing only -0/+0 can ever sum to
// -0 through this seed-0 accumulator. The differential claim this fixture
// CAN and DOES prove is the honest one: all three surfaces agree the result
// is +0 (not that any of them independently produces -0, which is
// unreachable via this reduction as implemented).
// -----------------------------------------------------------------------
test("sum fixture: sum([-0,-0,-0]) === +0 (Object.is) on naive/v1/resident — NOT -0 (accumulator seed is +0, see comment above)", () => {
  const shape = [3];
  const data = new Float64Array([-0, -0, -0]);
  const ref = sumRuntime(shape, data, undefined);
  assert.ok(
    Object.is(ref.data[0], 0) && !Object.is(ref.data[0], -0),
    `reference sum of all -0 must be +0 given the seed-0 accumulator (+0 + -0 == +0 in round-to-nearest), got ${ref.data[0]}`,
  );

  const v1 = wasmSum(core, shape, data, undefined);
  assertDataBitIdentical(ref.data, v1.data, "sum([-0,-0,-0]) [v1]");

  const w = WNDArray.fromArray(core, shape, data);
  try {
    const got = w.sum();
    try {
      assertDataBitIdentical(ref.data, got.toArray(), "sum([-0,-0,-0]) [resident]");
    } finally {
      got.dispose();
    }
  } finally {
    w.dispose();
  }
});

test("sum fixture: sum([+0,-0]) === +0 (Object.is) on naive/v1/resident", () => {
  const shape = [2];
  const data = new Float64Array([0, -0]);
  const ref = sumRuntime(shape, data, undefined);
  assert.ok(Object.is(ref.data[0], 0) && !Object.is(ref.data[0], -0), `reference sum([+0,-0]) must be +0, got ${ref.data[0]}`);

  const v1 = wasmSum(core, shape, data, undefined);
  assertDataBitIdentical(ref.data, v1.data, "sum([+0,-0]) [v1]");

  const w = WNDArray.fromArray(core, shape, data);
  try {
    const got = w.sum();
    try {
      assertDataBitIdentical(ref.data, got.toArray(), "sum([+0,-0]) [resident]");
    } finally {
      got.dispose();
    }
  } finally {
    w.dispose();
  }
});

// -----------------------------------------------------------------------
// Curated fixtures: Inf/NaN propagation through accumulating/arithmetic ops.
// -----------------------------------------------------------------------
test("add fixture: [Inf,-Inf] + [0,0] -> [Inf,-Inf] unchanged, on naive/v1/resident", () => {
  const shape = [2];
  const aData = new Float64Array([Infinity, -Infinity]);
  const bData = new Float64Array([0, 0]);
  const ref = elementwiseBinary(shape, aData, shape, bData, (x, y) => x + y);
  assert.strictEqual(ref.data[0], Infinity);
  assert.strictEqual(ref.data[1], -Infinity);

  const v1 = wasmAdd(core, shape, aData, shape, bData);
  assertDataBitIdentical(ref.data, v1.data, "[Inf,-Inf]+[0,0] [v1]");

  const a = WNDArray.fromArray(core, shape, aData);
  const b = WNDArray.fromArray(core, shape, bData);
  try {
    const got = a.add(b);
    try {
      assertDataBitIdentical(ref.data, got.toArray(), "[Inf,-Inf]+[0,0] [resident]");
    } finally {
      got.dispose();
    }
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("add fixture: [Inf] + [-Inf] -> [NaN], on naive/v1/resident", () => {
  const shape = [1];
  const aData = new Float64Array([Infinity]);
  const bData = new Float64Array([-Infinity]);
  const ref = elementwiseBinary(shape, aData, shape, bData, (x, y) => x + y);
  assert.ok(Number.isNaN(ref.data[0]));

  const v1 = wasmAdd(core, shape, aData, shape, bData);
  assertDataBitIdentical(ref.data, v1.data, "[Inf]+[-Inf] [v1]");

  const a = WNDArray.fromArray(core, shape, aData);
  const b = WNDArray.fromArray(core, shape, bData);
  try {
    const got = a.add(b);
    try {
      assertDataBitIdentical(ref.data, got.toArray(), "[Inf]+[-Inf] [resident]");
    } finally {
      got.dispose();
    }
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("mul fixture: MAX_VALUE * MAX_VALUE -> Infinity (overflow), on naive/resident (no v1 kernel for mul)", () => {
  const shape = [1];
  const aData = new Float64Array([Number.MAX_VALUE]);
  const bData = new Float64Array([Number.MAX_VALUE]);
  const ref = elementwiseBinary(shape, aData, shape, bData, (x, y) => x * y);
  assert.strictEqual(ref.data[0], Infinity);

  const a = WNDArray.fromArray(core, shape, aData);
  const b = WNDArray.fromArray(core, shape, bData);
  try {
    const got = a.mul(b);
    try {
      assertDataBitIdentical(ref.data, got.toArray(), "MAX*MAX [resident]");
    } finally {
      got.dispose();
    }
  } finally {
    a.dispose();
    b.dispose();
  }
});

// =============================================================================
// dot: two-way (naive dotRuntime / resident WNDArray.dot — no v1 kernel),
// scalar comparator, special-value-injected 1-D operands.
//
// COVERAGE SCOPE (adversarial-verifier finding, Kern 10): these special-value
// passes prove bit-identical PROPAGATION of NaN/±Inf/±0 through the reduction,
// NOT accumulation-ORDER sensitivity. At ~35 % special injection, Inf/NaN
// dominance masks the rounding-sensitive normal values (once both +Inf and
// -Inf are summed the accumulator is NaN regardless of order), so a reversed
// accumulation flips only ~1/619 cases here. Order-sensitivity for dot/norm is
// covered by vector.test.ts (all-normal data, larger n) — mutation-measured
// there at 134/230. Do NOT read these passes as an order-bug guard.
// =============================================================================
{
  const rng = makeRng(0x444f545f53504543n); // "DOT_SPEC"
  const CASE_COUNT = 60;
  for (let c = 0; c < CASE_COUNT; c++) {
    const n = rng.nextInt(0, 24);
    const aData = genDataSpecial(rng, [n]);
    const bData = genDataSpecial(rng, [n]);

    test(`dot special case ${c}: n=${n}`, () => {
      const ref = dotRuntime([n], aData, [n], bData);
      const a = WNDArray.fromArray(core, [n], aData);
      const b = WNDArray.fromArray(core, [n], bData);
      try {
        const got = a.dot(b);
        assertScalarBitIdentical(ref, got, `dot special case ${c} n=${n}`);
      } finally {
        a.dispose();
        b.dispose();
      }
    });
  }
}

// =============================================================================
// norm: two-way (naive Math.sqrt(normSqRuntime) via NDArray.norm() / resident
// WNDArray.norm), scalar comparator, special-value-injected data, ranks 0..4.
// =============================================================================
{
  const rng = makeRng(0x4e524d5f53504543n); // "NRM_SPEC"
  const CASE_COUNT = 60;
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genDataSpecial(rng, shape);

    test(`norm special case ${c}: shape=[${shape.join(",")}]`, () => {
      const ref = Math.sqrt(normSqRuntime(data));
      const naive = NDArray.fromArray(shape, data).norm();
      assertScalarBitIdentical(ref, naive, `norm special case ${c} naive`);

      const w = WNDArray.fromArray(core, shape, data);
      try {
        assertScalarBitIdentical(ref, w.norm(), `norm special case ${c} resident`);
      } finally {
        w.dispose();
      }
    });
  }
}

// =============================================================================
// cosineSimilarity: two-way, scalar comparator, special-value-injected 1-D
// operands. Not a row of the spec's op-coverage table, but explicitly named
// in its "Skalar-Ops (dot, norm, cosine)" note — in scope.
// =============================================================================
{
  const rng = makeRng(0x434f535f53504543n); // "COS_SPEC"
  const CASE_COUNT = 30;
  for (let c = 0; c < CASE_COUNT; c++) {
    const n = rng.nextInt(1, 12);
    const aData = genDataSpecial(rng, [n]);
    const bData = genDataSpecial(rng, [n]);

    test(`cosineSimilarity special case ${c}: n=${n}`, () => {
      const num = dotRuntime([n], aData, [n], bData);
      const den = Math.sqrt(normSqRuntime(aData)) * Math.sqrt(normSqRuntime(bData));
      const ref = num / den;

      const naive = NDArray.fromArray([n], aData).cosineSimilarity(NDArray.fromArray([n], bData));
      assertScalarBitIdentical(ref, naive, `cosineSimilarity special case ${c} naive`);

      const a = WNDArray.fromArray(core, [n], aData);
      const b = WNDArray.fromArray(core, [n], bData);
      try {
        assertScalarBitIdentical(ref, a.cosineSimilarity(b), `cosineSimilarity special case ${c} resident`);
      } finally {
        a.dispose();
        b.dispose();
      }
    });
  }
}

// =============================================================================
// transpose: three-way (naive transposeRuntime / v1 wasmTranspose / resident
// WNDArray.transpose+toArray), special-value-injected data, ranks 0..4. Pure
// data movement — this random pass proves NaN/Inf/-0/subnormals survive a
// real WASM memory round-trip unscathed under `Object.is` (value-class)
// equality; the STRICTLY stronger byte-exact NaN-payload claim is the
// separate curated fixture below (D3).
// =============================================================================
{
  const rng = makeRng(0x5452415f53504543n); // "TRA_SPEC"
  const CASE_COUNT = 60;
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genDataSpecial(rng, shape);

    test(`transpose special case ${c}: shape=[${shape.join(",")}]`, () => {
      const ref = transposeRuntime(shape, data);
      const v1 = wasmTranspose(core, shape, data);
      const ctx = `transpose special case ${c} shape=[${shape.join(",")}]`;
      assertShapeEqual(ref.shape, v1.shape, `${ctx} [v1]`);
      assertDataBitIdentical(ref.data, v1.data, `${ctx} [v1]`);

      const w = WNDArray.fromArray(core, shape, data);
      try {
        const view = w.transpose();
        try {
          assertShapeEqual(ref.shape, view.shape, `${ctx} [resident]`);
          assertDataBitIdentical(ref.data, view.toArray(), `${ctx} [resident]`);
        } finally {
          view.dispose();
        }
      } finally {
        w.dispose();
      }
    });
  }
}

// -----------------------------------------------------------------------
// D3 movement-payload-sharpening fixture: a NaN with a NON-canonical payload
// (0x7ff8_0000_dead_beef) must survive pure data movement (transpose)
// BYTE-EXACT, not merely "still NaN" — checked via `bitsOf`, not `Object.is`
// (which treats every NaN as equal regardless of payload). Curated, not part
// of the random pass, per spec D3.
// -----------------------------------------------------------------------
function nonCanonicalNaN(): number {
  const bits = new BigUint64Array([0x7ff8_0000_dead_beefn]);
  return new Float64Array(bits.buffer)[0] ?? Number.NaN;
}

test("transpose fixture: a non-canonical NaN payload survives naive/v1/resident data movement byte-exact (bitsOf, not Object.is)", () => {
  const shape = [2, 2];
  const nan = nonCanonicalNaN();
  assert.ok(Number.isNaN(nan), "sanity: constructed value must actually be NaN");
  const nanBits = bitsOf(nan);
  assert.strictEqual(nanBits, 0x7ff8_0000_dead_beefn, "sanity: constructed NaN must carry the intended non-canonical payload bits");

  const data = new Float64Array([nan, 1, 2, 3]);
  const ref = transposeRuntime(shape, data);
  // The NaN starts at flat index 0 = logical [0,0]; for a 2x2, [0,0]^T =
  // [0,0], so it lands at output flat index 0 too. Confirm the reference
  // itself preserves the exact bits (pure JS array copy, no arithmetic —
  // transposeRuntime's inner loop is a plain `out[flat] = data[inOffset]`)
  // before testing the WASM surfaces against it.
  assert.strictEqual(bitsOf(ref.data[0] ?? 0), nanBits, "naive reference must preserve the exact NaN payload");

  const v1 = wasmTranspose(core, shape, data);
  assert.strictEqual(bitsOf(v1.data[0] ?? 0), nanBits, "v1 (nt_transpose) must preserve the exact NaN payload");

  const w = WNDArray.fromArray(core, shape, data);
  try {
    const view = w.transpose(); // O(1) metadata view — no data movement yet
    try {
      const got = view.toArray(); // real WASM memory movement (nt_materialize gather, since a transposed view is never "contiguous")
      assert.strictEqual(bitsOf(got[0] ?? 0), nanBits, "resident transpose+toArray must preserve the exact NaN payload");
    } finally {
      view.dispose();
    }
  } finally {
    w.dispose();
  }
});
