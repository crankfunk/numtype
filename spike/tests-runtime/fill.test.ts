/**
 * Differential test: fill. Not one of the spec's four "ops" requiring
 * >=100 cases (it has no `runtime.ts` counterpart — `NDArray.zeros`/`.ones`
 * just use `new Float64Array(n)`/`.fill(1)` directly) but still worth a
 * light differential check since it's an ABI export with its own kernel.
 */
import { test } from "node:test";
import { wasmFill } from "../src/wasm/backend.ts";
import { initCore } from "../src/wasm/loader.ts";
import { assertDataBitIdentical } from "./assert-helpers.ts";
import { makeRng } from "./prng.ts";

const core = await initCore();
const CASE_COUNT = 30;
const SEED = 0x46494c4c5f534544n; // "FILL_SED"-ish

const rng = makeRng(SEED);
for (let c = 0; c < CASE_COUNT; c++) {
  const len = rng.nextInt(0, 64);
  const value = rng.nextF64();

  test(`fill case ${c}: len=${len} value=${value}`, () => {
    const ref = new Float64Array(len).fill(value);
    const got = wasmFill(core, len, value);
    assertDataBitIdentical(ref, got, `fill case ${c} len=${len} value=${value}`);
  });
}
