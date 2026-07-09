/**
 * NumType Spike 01 demo — run with `pnpm demo` (or `node spike/demo.ts`
 * directly; Node 24 executes .ts natively, no build step).
 *
 * Builds arrays, broadcasts an elementwise add, runs a 2-D matmul, sums and
 * transposes — all with results small/simple enough to hand-verify.
 */
import { type AnyNDArray, NDArray } from "./src/ndarray.ts";

function printArray(label: string, arr: AnyNDArray): void {
  console.log(`${label} shape=[${arr.shape.join(",")}]`);
  console.log(JSON.stringify(arr.toNestedArray()));
  console.log();
}

console.log("=== NumType Spike 01 demo ===\n");

// --- Broadcasting add: [2,3] + [3] -> [2,3] --------------------------------
// A = [[1,2,3],[4,5,6]], B = [10,20,30]
// Expected: [[11,22,33],[14,25,36]]
const a = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);
const b = NDArray.fromArray([3], [10, 20, 30]);
const sum = a.add(b);
printArray("A            ", a);
printArray("B            ", b);
printArray("A + B (bcast)", sum);

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

// --- Sum reduction along an axis: [2,3,4].sum(1) -> [2,4] -------------------
const cube = NDArray.zeros([2, 3, 4]);
for (let i = 0; i < cube.data.length; i++) cube.data[i] = i + 1;
const reduced = cube.sum(1);
printArray("cube         ", cube);
printArray("cube.sum(1)  ", reduced);

// --- Transpose: [2,3] -> [3,2] ----------------------------------------------
const transposed = a.transpose();
printArray("A.transpose()", transposed);

console.log("=== demo complete ===");
