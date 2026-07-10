/**
 * Kern 04 differential tests: the blocked + packed + SIMD128 matmul kernel
 * (`nt_matmul_blocked`, which `WNDArray.matmul()` now calls
 * unconditionally) must stay bit-identical to the naive TS reference — the
 * same non-negotiable differential contract as every other phase, just
 * with dimensions large enough to actually cross the kernel's internal
 * M/N/K tile boundaries (MC = NC = KC = 32; see
 * crates/core/src/kernels/matmul_blocked.rs). The existing resident/strided
 * suites (dims capped at 8 — see prng.ts / strided.test.ts) already
 * re-exercise the blocked kernel, unchanged, for every one of their cases
 * now that resident.ts routes through it; THIS file is what specifically
 * proves the blocking/packing logic itself: exact tile-boundary values,
 * one-past, remainders, and multiples up to ~3x the largest tile dimension,
 * on both contiguous and transposed-view operands, plus batch and
 * broadcast-batch.
 *
 * Same view-construction trick as strided.test.ts (transposition is an
 * involution, pinned there — not re-derived here).
 *
 * Wired into `pnpm test:resident` (explicit file list in package.json —
 * the known footgun; see CLAUDE.md).
 */
import { test } from "node:test";
import { matmulRuntime, transposeRuntime } from "../src/runtime.ts";
import { initCore } from "../src/wasm/loader.ts";
import { WNDArray, type AnyWNDArray } from "../src/wasm/resident.ts";
import { assertDataBitIdentical, assertShapeEqual } from "./assert-helpers.ts";
import { genBroadcastShapes, genData, makeRng, type Rng } from "./prng.ts";

const core = await initCore();

// Kernel tile sizes (crates/core/src/kernels/matmul_blocked.rs) — every
// dimension chosen below is relative to these so cases actually straddle
// tile boundaries, not just "large".
const MC = 32;
const NC = 32;
const KC = 32;

interface Operand {
  readonly arr: AnyWNDArray;
  /** Every handle that must be disposed when the case is done (for a view:
   * the base AND the view). */
  readonly owners: readonly AnyWNDArray[];
}

function makeContiguous(shape: readonly number[], refData: Float64Array): Operand {
  const arr = WNDArray.fromArray(core, shape, refData);
  return { arr, owners: [arr] };
}

/** A transpose VIEW whose logical shape/content is `shape`/`refData` (see
 * strided.test.ts's file header for the involution trick this relies on). */
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

function runCase(name: string, aShape: number[], bShape: number[], aView: boolean, bView: boolean, rng: Rng): void {
  test(name, () => {
    const aData = genData(rng, aShape);
    const bData = genData(rng, bShape);
    const ref = matmulRuntime(aShape, aData, bShape, bData);
    const a = makeOperand(aView, aShape, aData);
    const b = makeOperand(bView, bShape, bData);
    try {
      const got = a.arr.matmul(b.arr);
      try {
        assertShapeEqual(ref.shape, got.shape, name);
        assertDataBitIdentical(ref.data, got.toArray(), name);
      } finally {
        got.dispose();
      }
    } finally {
      disposeAll(a, b);
    }
  });
}

// --- randomized cases: m/n up to 3xMC/NC, k up to 2xKC, view/contig mix ----
{
  const rng = makeRng(0x424c4b445f524e44n); // "BLKD_RND"
  const CASE_COUNT = 130;
  for (let c = 0; c < CASE_COUNT; c++) {
    const m = rng.nextInt(1, 3 * MC);
    const k = rng.nextInt(1, 2 * KC);
    const n = rng.nextInt(1, 3 * NC);
    let aView = rng.nextBool();
    let bView = rng.nextBool();
    if (!aView && !bView) {
      if (rng.nextBool()) aView = true;
      else bView = true;
    }
    runCase(
      `blocked matmul random case ${c}: [${m},${k}]${aView ? "ᵛ" : ""} @ [${k},${n}]${bView ? "ᵛ" : ""}`,
      [m, k],
      [k, n],
      aView,
      bView,
      rng,
    );
  }
}

// --- explicit tile-boundary cases: exactly at / one-below / one-above / ---
// multiples of MC, NC, KC, on both operand positions and both view states.
{
  const rng = makeRng(0x424c4b445f424e44n); // "BLKD_BND"
  const mnBoundary = [MC - 1, MC, MC + 1, 2 * MC - 1, 2 * MC, 2 * MC + 1, 3 * MC];
  const kBoundary = [KC - 1, KC, KC + 1, 2 * KC, 3 * KC];
  // Cross m x k x n boundary values pairwise (not full cross-product with
  // n too, to keep runtime bounded) — m and n each independently walk the
  // MC/NC boundary list, k independently walks the KC boundary list, zipped
  // rather than fully crossed.
  for (let i = 0; i < mnBoundary.length; i++) {
    const m = mnBoundary[i] ?? MC;
    const n = mnBoundary[mnBoundary.length - 1 - i] ?? NC;
    const k = kBoundary[i % kBoundary.length] ?? KC;
    const aView = i % 2 === 0;
    const bView = i % 3 === 0;
    runCase(`blocked matmul tile-boundary case ${i}: [${m},${k}]${aView ? "ᵛ" : ""} @ [${k},${n}]${bView ? "ᵛ" : ""}`, [m, k], [k, n], aView, bView, rng);
  }
  // A few dedicated large-K cases (K up to 3xKC) at fixed, tile-boundary-ish
  // m/n, to specifically stress the K-tiling (kb) loop's ascending-order
  // accumulation across many k-blocks.
  const bigK = [KC, KC + 1, 2 * KC - 1, 2 * KC, 3 * KC];
  for (let i = 0; i < bigK.length; i++) {
    const k = bigK[i] ?? KC;
    const m = MC + 1;
    const n = NC - 1;
    const aView = i % 2 === 1;
    const bView = i % 2 === 0;
    runCase(`blocked matmul big-K case ${i}: [${m},${k}]${aView ? "ᵛ" : ""} @ [${k},${n}]${bView ? "ᵛ" : ""}`, [m, k], [k, n], aView, bView, rng);
  }
}

// --- batch and broadcast-batch, with m/k/n large enough to cross tiles ----
{
  const rng = makeRng(0x424c4b445f424154n); // "BLKD_BAT"
  const CASE_COUNT = 20;
  for (let c = 0; c < CASE_COUNT; c++) {
    const m = rng.nextInt(MC - 1, 2 * MC + 1);
    const k = rng.nextInt(KC - 1, KC + 65);
    const n = rng.nextInt(NC - 1, 2 * NC + 1);
    const { aShape: batchA, bShape: batchB } = genBroadcastShapes(rng, 2);
    const aShape = [...batchA, m, k];
    const bShape = [...batchB, k, n];
    let aView = rng.nextBool();
    let bView = rng.nextBool();
    if (!aView && !bView) {
      if (rng.nextBool()) aView = true;
      else bView = true;
    }
    runCase(
      `blocked matmul batch case ${c}: a=[${aShape.join(",")}]${aView ? "ᵛ" : ""} b=[${bShape.join(",")}]${bView ? "ᵛ" : ""}`,
      aShape,
      bShape,
      aView,
      bView,
      rng,
    );
  }
}

// --- odd-dimension remainders: dims that leave a size-1 remainder against
// every tile size at once (MC, NC, and KC), forcing the scalar tail path in
// both the n-pair loop and (indirectly, via kb blocking) the k dimension.
{
  const rng = makeRng(0x424c4b445f4f4444n); // "BLKD_ODD"
  const odd = [
    { m: MC + 1, k: KC + 1, n: NC + 1 },
    { m: 2 * MC + 1, k: 2 * KC + 1, n: 2 * NC + 1 },
    { m: 1, k: KC + 1, n: 1 },
    { m: MC - 1, k: 1, n: NC - 1 },
  ];
  for (let i = 0; i < odd.length; i++) {
    const { m, k, n } = odd[i]!;
    const aView = i % 2 === 0;
    const bView = i % 2 === 1;
    runCase(`blocked matmul odd-remainder case ${i}: [${m},${k}]${aView ? "ᵛ" : ""} @ [${k},${n}]${bView ? "ᵛ" : ""}`, [m, k], [k, n], aView, bView, rng);
  }
}
