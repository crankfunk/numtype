/**
 * fromArray Float64Array overload (Kern 02 follow-up): both `NDArray` and
 * `WNDArray` accept a `Float64Array` directly at the copy-IN boundary —
 * removing the ~100x `Array.from` conversion tax the chain bench measured
 * (docs/kern-02-ergebnisse.md). Covers: (a) the typed-array path is
 * bit-identical to the number[] path, (b) the input is COPIED, never
 * aliased — mutating the source afterward must not affect the constructed
 * array (both classes).
 */
import assert from "node:assert";
import { test } from "node:test";
import { NDArray } from "../src/ndarray.ts";
import { initCore } from "../src/wasm/loader.ts";
import { WNDArray } from "../src/wasm/resident.ts";
import { assertDataBitIdentical } from "./assert-helpers.ts";
import { genData, makeRng } from "./prng.ts";

const rng = makeRng(0x4652_4f4d_4152_52n); // "FROMARR"

test("NDArray.fromArray: Float64Array input is bit-identical to the number[] path", () => {
  const source = genData(rng, [3, 4]);
  const viaTyped = NDArray.fromArray([3, 4], source);
  const viaList = NDArray.fromArray([3, 4], Array.from(source));
  assertDataBitIdentical(viaList.data, viaTyped.data, "NDArray typed vs list path");
});

test("NDArray.fromArray copies a Float64Array input (never aliases)", () => {
  const source = genData(rng, [2, 2]);
  const before0 = source[0] ?? 0;
  const arr = NDArray.fromArray([2, 2], source);
  source[0] = before0 + 1;
  assert.strictEqual(arr.data[0], before0, "mutating the source must not affect the array");
  assert.notStrictEqual(arr.data, source, "backing buffer must be a fresh copy");
});

test("WNDArray.fromArray: Float64Array input is bit-identical to the number[] path", async () => {
  const core = await initCore();
  const source = genData(rng, [3, 4]);
  const viaTyped = WNDArray.fromArray(core, [3, 4], source);
  const viaList = WNDArray.fromArray(core, [3, 4], Array.from(source));
  try {
    assertDataBitIdentical(viaList.toArray(), viaTyped.toArray(), "WNDArray typed vs list path");
  } finally {
    viaTyped.dispose();
    viaList.dispose();
  }
});

test("WNDArray.fromArray copies a Float64Array input (never aliases)", async () => {
  const core = await initCore();
  const source = genData(rng, [2, 2]);
  const before0 = source[0] ?? 0;
  const arr = WNDArray.fromArray(core, [2, 2], source);
  try {
    source[0] = before0 + 1;
    assert.strictEqual(arr.toArray()[0], before0, "mutating the source must not affect the resident array");
  } finally {
    arr.dispose();
  }
});
