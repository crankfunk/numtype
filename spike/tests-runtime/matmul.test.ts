/**
 * Differential test: full NumPy `matmul` (1-D promotion, batch-dim
 * broadcasting, 2-D core). WASM (`wasmMatmul`) must be bit-identical to
 * the naive TS reference (`matmulRuntime`).
 */
import { test } from "node:test";
import { matmulRuntime } from "../src/runtime.ts";
import { wasmMatmul } from "../src/wasm/backend.ts";
import { initCore } from "../src/wasm/loader.ts";
import { assertDataBitIdentical, assertShapeEqual } from "./assert-helpers.ts";
import { genBroadcastShapes, genData, makeRng, type Rng } from "./prng.ts";

const core = await initCore();
const CASE_COUNT = 150;
const SEED = 0x4d41_544d_554c_2121n; // "MATMUL!!"-ish

function genSmallBatch(rng: Rng): number[] {
  const rank = rng.nextInt(0, 2);
  return Array.from({ length: rank }, () => rng.nextInt(1, 8));
}

/** Generate an (aShape, bShape) pair valid for `matmul`, including 1-D
 * promotion on either/both sides and batch-dim broadcasting (with
 * broadcast-1 batch-dim placements) for the 2-D-or-more case. */
function genMatmulShapes(rng: Rng): { aShape: number[]; bShape: number[] } {
  const k = rng.nextInt(1, 8);
  const m = rng.nextInt(1, 8);
  const n = rng.nextInt(1, 8);

  const aIs1D = rng.nextInt(0, 4) === 0; // ~20%
  const bIs1D = rng.nextInt(0, 4) === 0; // ~20%

  if (aIs1D && bIs1D) {
    return { aShape: [k], bShape: [k] };
  }
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

const rng = makeRng(SEED);
for (let c = 0; c < CASE_COUNT; c++) {
  const { aShape, bShape } = genMatmulShapes(rng);
  const aData = genData(rng, aShape);
  const bData = genData(rng, bShape);

  test(`matmul case ${c}: a=[${aShape.join(",")}] b=[${bShape.join(",")}]`, () => {
    const ref = matmulRuntime(aShape, aData, bShape, bData);
    const got = wasmMatmul(core, aShape, aData, bShape, bData);
    const ctx = `matmul case ${c} a=[${aShape.join(",")}] b=[${bShape.join(",")}]`;
    assertShapeEqual(ref.shape, got.shape, ctx);
    assertDataBitIdentical(ref.data, got.data, ctx);
  });
}
