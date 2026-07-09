/**
 * Differential test: reverse-every-axis transpose. WASM (`wasmTranspose`)
 * must be bit-identical to the naive TS reference (`transposeRuntime`).
 */
import { test } from "node:test";
import { transposeRuntime } from "../src/runtime.ts";
import { wasmTranspose } from "../src/wasm/backend.ts";
import { initCore } from "../src/wasm/loader.ts";
import { assertDataBitIdentical, assertShapeEqual } from "./assert-helpers.ts";
import { genData, makeRng, type Rng } from "./prng.ts";

const core = await initCore();
const CASE_COUNT = 150;
const SEED = 0x5452_414e_5350_4f53n; // "TRANSPOS"-ish

function genShape(rng: Rng, minRank: number, maxRank: number): number[] {
  const rank = rng.nextInt(minRank, maxRank);
  return Array.from({ length: rank }, () => rng.nextInt(1, 8));
}

const rng = makeRng(SEED);
for (let c = 0; c < CASE_COUNT; c++) {
  const shape = genShape(rng, 0, 4);
  const data = genData(rng, shape);

  test(`transpose case ${c}: shape=[${shape.join(",")}]`, () => {
    const ref = transposeRuntime(shape, data);
    const got = wasmTranspose(core, shape, data);
    const ctx = `transpose case ${c} shape=[${shape.join(",")}]`;
    assertShapeEqual(ref.shape, got.shape, ctx);
    assertDataBitIdentical(ref.data, got.data, ctx);
  });
}
