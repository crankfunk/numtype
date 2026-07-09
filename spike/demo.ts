/**
 * NumType Spike 01 demo — run with `pnpm demo` (or `node spike/demo.ts`
 * directly; Node 24 executes .ts natively, no build step).
 *
 * Builds arrays, broadcasts an elementwise add, runs a 2-D matmul, sums and
 * transposes — all with results small/simple enough to hand-verify.
 *
 * Kern 01 addition: every showcase op also runs through the WASM backend
 * (same input shape/data, read straight off the already-built `NDArray`s)
 * and the two results are asserted equal inline — bit-for-bit, via
 * `Object.is` — so this script fails loudly on any TS/WASM divergence
 * instead of silently printing two different answers.
 */
import { type AnyNDArray, NDArray } from "./src/ndarray.ts";
import { wasmAdd, wasmMatmul, wasmSum, wasmTranspose } from "./src/wasm/backend.ts";
import { initCore } from "./src/wasm/loader.ts";

function printArray(label: string, arr: AnyNDArray): void {
  console.log(`${label} shape=[${arr.shape.join(",")}]`);
  console.log(JSON.stringify(arr.toNestedArray()));
  console.log();
}

/** Compare a WASM-backend result against the already-printed TS `NDArray`
 * result: same shape, bit-identical data (`Object.is`, distinguishes -0/+0
 * and handles NaN correctly — matches the differential suite's standard).
 * Throws loudly (this script's whole point per spec) on any divergence. */
function assertBackendsAgree(
  label: string,
  ref: { shape: readonly number[]; data: Float64Array },
  got: { shape: readonly number[]; data: Float64Array },
): void {
  const refShape: readonly number[] = ref.shape;
  const shapesEqual = refShape.length === got.shape.length && refShape.every((d, i) => d === got.shape[i]);
  if (!shapesEqual) {
    throw new Error(`${label}: TS/WASM shape divergence: [${refShape.join(",")}] vs [${got.shape.join(",")}]`);
  }
  if (ref.data.length !== got.data.length) {
    throw new Error(`${label}: TS/WASM data-length divergence: ${ref.data.length} vs ${got.data.length}`);
  }
  for (let i = 0; i < ref.data.length; i++) {
    if (!Object.is(ref.data[i], got.data[i])) {
      throw new Error(`${label}: TS/WASM bit divergence at index ${i}: ${ref.data[i]} vs ${got.data[i]}`);
    }
  }
  console.log(`  [wasm] ${label} shape=[${got.shape.join(",")}] — matches TS bit-for-bit`);
  console.log(JSON.stringify(NDArray.fromArray(got.shape, Array.from(got.data)).toNestedArray()));
  console.log();
}

console.log("=== NumType Spike 01 demo ===\n");

console.log("Loading WASM core...");
const core = await initCore();
console.log("WASM core loaded.\n");

// --- Broadcasting add: [2,3] + [3] -> [2,3] --------------------------------
// A = [[1,2,3],[4,5,6]], B = [10,20,30]
// Expected: [[11,22,33],[14,25,36]]
const a = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
const b = NDArray.fromArray([3], [10, 20, 30]);
const sum = a.add(b);
printArray("A            ", a);
printArray("B            ", b);
printArray("A + B (bcast)", sum);
assertBackendsAgree("A + B (bcast)", sum, wasmAdd(core, a.shape, a.data, b.shape, b.data));

// --- 2-D matmul: [2,3] x [3,2] -> [2,2] ------------------------------------
// M1 = [[1,2,3],[4,5,6]], M2 = [[7,8],[9,10],[11,12]]
// Expected (hand-computable):
//   row0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [7+18+33, 8+20+36]     = [58, 64]
//   row1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [28+45+66, 32+50+72]   = [139, 154]
const m1 = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
const m2 = NDArray.fromArray([3, 2], [7, 8, 9, 10, 11, 12]);
const product = m1.matmul(m2);
printArray("M1           ", m1);
printArray("M2           ", m2);
printArray("M1 @ M2      ", product);
assertBackendsAgree("M1 @ M2", product, wasmMatmul(core, m1.shape, m1.data, m2.shape, m2.data));

// --- Sum reduction along an axis: [2,3,4].sum(1) -> [2,4] -------------------
const cube = NDArray.zeros([2, 3, 4]);
for (let i = 0; i < cube.data.length; i++) cube.data[i] = i + 1;
const reduced = cube.sum(1);
printArray("cube         ", cube);
printArray("cube.sum(1)  ", reduced);
assertBackendsAgree("cube.sum(1)", reduced, wasmSum(core, cube.shape, cube.data, 1));

// --- Transpose: [2,3] -> [3,2] ----------------------------------------------
const transposed = a.transpose();
printArray("A.transpose()", transposed);
assertBackendsAgree("A.transpose()", transposed, wasmTranspose(core, a.shape, a.data));

console.log("=== demo complete: TS and WASM backends agree on every showcase op ===");
