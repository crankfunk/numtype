/**
 * Differential test: sum-reduce, both full (`axis === undefined`) and
 * per-axis (including negative axes). WASM (`wasmSum`) must be
 * bit-identical to the naive TS reference (`sumRuntime`).
 */
import { test } from "node:test";
import { sumRuntime } from "../src/runtime.ts";
import { wasmSum } from "../src/wasm/backend.ts";
import { initCore } from "../src/wasm/loader.ts";
import { assertDataBitIdentical, assertShapeEqual } from "./assert-helpers.ts";
import { genData, makeRng, type Rng } from "./prng.ts";

const core = await initCore();
const CASE_COUNT = 150;

function genShape(rng: Rng, minRank: number, maxRank: number): number[] {
  const rank = rng.nextInt(minRank, maxRank);
  return Array.from({ length: rank }, () => rng.nextInt(1, 8));
}

// --- sum_all: any rank 0..4 (shape doesn't affect sum_all's own logic,
// only `data.length` does — still exercised across ranks since NDArray
// always carries a real shape). ---------------------------------------
{
  const rng = makeRng(0x53554d5f414c4c00n); // "SUM_ALL\0"
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 0, 4);
    const data = genData(rng, shape);

    test(`sum_all case ${c}: shape=[${shape.join(",")}]`, () => {
      const ref = sumRuntime(shape, data, undefined);
      const got = wasmSum(core, shape, data, undefined);
      const ctx = `sum_all case ${c} shape=[${shape.join(",")}]`;
      assertShapeEqual(ref.shape, got.shape, ctx);
      assertDataBitIdentical(ref.data, got.data, ctx);
    });
  }
}

// --- sum_axis: rank 1..4 (need at least one axis to reduce), axis chosen
// both in positive and negative form. ------------------------------------
{
  const rng = makeRng(0x53554d5f41584953n); // "SUM_AXIS"
  for (let c = 0; c < CASE_COUNT; c++) {
    const shape = genShape(rng, 1, 4);
    const data = genData(rng, shape);
    const rank = shape.length;
    const positiveAxis = rng.nextInt(0, rank - 1);
    const useNegative = rng.nextBool();
    const axis = useNegative ? positiveAxis - rank : positiveAxis;

    test(`sum_axis case ${c}: shape=[${shape.join(",")}] axis=${axis}`, () => {
      const ref = sumRuntime(shape, data, axis);
      const got = wasmSum(core, shape, data, axis);
      const ctx = `sum_axis case ${c} shape=[${shape.join(",")}] axis=${axis}`;
      assertShapeEqual(ref.shape, got.shape, ctx);
      assertDataBitIdentical(ref.data, got.data, ctx);
    });
  }
}
