/**
 * Differential test: broadcasting elementwise add. WASM (`wasmAdd`) must be
 * bit-identical to the naive TS reference (`elementwiseBinary`) across
 * ranks 0..4, dims 1..8, with broadcast-1 placements on either operand.
 */
import { test } from "node:test";
import { elementwiseBinary } from "../src/runtime.ts";
import { wasmAdd } from "../src/wasm/backend.ts";
import { initCore } from "../src/wasm/loader.ts";
import { assertDataBitIdentical, assertShapeEqual } from "./assert-helpers.ts";
import { genBroadcastShapes, genData, makeRng } from "./prng.ts";

const core = await initCore();
const CASE_COUNT = 150;
const SEED = 0x4144_445f_5345_4544n; // "ADD_SEED" ascii-ish, arbitrary fixed seed

const rng = makeRng(SEED);
for (let c = 0; c < CASE_COUNT; c++) {
  const { aShape, bShape } = genBroadcastShapes(rng);
  const aData = genData(rng, aShape);
  const bData = genData(rng, bShape);

  test(`add case ${c}: a=[${aShape.join(",")}] b=[${bShape.join(",")}]`, () => {
    const ref = elementwiseBinary(aShape, aData, bShape, bData, (x, y) => x + y);
    const got = wasmAdd(core, aShape, aData, bShape, bData);
    const ctx = `add case ${c} a=[${aShape.join(",")}] b=[${bShape.join(",")}]`;
    assertShapeEqual(ref.shape, got.shape, ctx);
    assertDataBitIdentical(ref.data, got.data, ctx);
  });
}
