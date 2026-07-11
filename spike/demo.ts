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
 *
 * Kern 02 addition: the same showcase ops run a THIRD time through the
 * resident (zero-copy) backend — `WNDArray.fromArray` in, the same chain
 * of ops pointer-to-pointer, asserted bit-identical to the already-printed
 * TS results, with every resident handle explicitly `dispose()`d at the
 * end of that section (demonstrating the full lifecycle, not relying on
 * the GC backstop).
 */
import { NDArray, type NDArrayView } from "./src/ndarray.ts";
import { wasmAdd, wasmMatmul, wasmSum, wasmTranspose } from "./src/wasm/backend.ts";
import { initCore } from "./src/wasm/loader.ts";
import { type AnyWNDArray, WNDArray } from "./src/wasm/resident.ts";
import type { Shape } from "./src/dim.ts";

function printArray(label: string, arr: NDArrayView<Shape>): void {
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
  console.log(JSON.stringify(NDArray.fromArray(got.shape, got.data).toNestedArray()));
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

// --- Kern 02: same showcase ops via the resident (zero-copy) backend ------
console.log("=== Resident (WASM zero-copy) backend ===\n");

/** Compare a `WNDArray` result against an already-printed TS `NDArray`
 * result: same shape, bit-identical data. Throws loudly on any divergence,
 * same standard as `assertBackendsAgree` above. */
function assertResidentAgrees(
  label: string,
  ref: { shape: readonly number[]; data: Float64Array },
  got: AnyWNDArray,
): void {
  const refShape: readonly number[] = ref.shape;
  const gotShape: readonly number[] = got.shape;
  const shapesEqual = refShape.length === gotShape.length && refShape.every((d, i) => d === gotShape[i]);
  if (!shapesEqual) {
    throw new Error(`${label}: TS/resident shape divergence: [${refShape.join(",")}] vs [${gotShape.join(",")}]`);
  }
  const gotData = got.toArray();
  if (ref.data.length !== gotData.length) {
    throw new Error(`${label}: TS/resident data-length divergence: ${ref.data.length} vs ${gotData.length}`);
  }
  for (let i = 0; i < ref.data.length; i++) {
    if (!Object.is(ref.data[i], gotData[i])) {
      throw new Error(`${label}: TS/resident bit divergence at index ${i}: ${ref.data[i]} vs ${gotData[i]}`);
    }
  }
  console.log(`  [resident] ${label} shape=[${gotShape.join(",")}] — matches TS bit-for-bit`);
  console.log(JSON.stringify(got.toNestedArray()));
  console.log();
}

// Resident twins of A, B, M1, M2, cube — built once via the explicit
// copy-IN boundary (`fromArray`), then chained resident, pointer-to-pointer.
const rA = WNDArray.fromArray(core, a.shape, a.data);
const rB = WNDArray.fromArray(core, b.shape, b.data);
const rSum = rA.add(rB);
assertResidentAgrees("A + B (bcast)", sum, rSum);

const rM1 = WNDArray.fromArray(core, m1.shape, m1.data);
const rM2 = WNDArray.fromArray(core, m2.shape, m2.data);
const rProduct = rM1.matmul(rM2);
assertResidentAgrees("M1 @ M2", product, rProduct);

const rCube = WNDArray.fromArray(core, cube.shape, cube.data);
const rReduced = rCube.sum(1);
assertResidentAgrees("cube.sum(1)", reduced, rReduced);

const rTransposed = rA.transpose();
assertResidentAgrees("A.transpose()", transposed, rTransposed);

// --- Kern 03: transpose is an O(1) VIEW — feed it straight into matmul ----
// `rM2.transpose()` allocates nothing and copies nothing (shared buffer,
// reversed strides); the strided matmul kernel reads it in place. The
// reference chains the same two ops on the naive runtime.
const viewRef = m2.transpose().matmul(m2);
const rM2View = rM2.transpose(); // O(1), no kernel call
const rViewProduct = rM2View.matmul(rM2);
assertResidentAgrees("M2ᵀ (view) @ M2", viewRef, rViewProduct);

// --- Kern 05: slice is ALSO an O(1) VIEW — feed it straight into sum() ----
// `rA.slice(1)` picks row index 1 of A ([[1,2,3],[4,5,6]]) -> [4,5,6],
// folding the integer spec into the offset (allocates nothing, copies
// nothing — same mechanism as transpose()); the strided sum kernel then
// reads that row in place. The reference chains the same two ops on the
// naive runtime.
const sliceSumRef = a.slice(1).sum();
const rASlice = rA.slice(1); // O(1), no kernel call
const rSliceSum = rASlice.sum();
assertResidentAgrees("A.slice(1).sum()", sliceSumRef, rSliceSum);

// Explicit dispose at the end of this section — every resident handle
// created above, released deterministically (not left to the GC backstop).
// Note the order freedom refcounting buys: rM2 was already safe to dispose
// before rM2View (shared buffer lives until the LAST handle releases).
rA.dispose();
rB.dispose();
rSum.dispose();
rM1.dispose();
rM2.dispose();
rProduct.dispose();
rCube.dispose();
rReduced.dispose();
rTransposed.dispose();
rM2View.dispose();
rViewProduct.dispose();
rASlice.dispose();
rSliceSum.dispose();

// --- Kern 07: elementwise sub/mul/div + vector ops (dot/norm/cosineSimilarity) ---
// Small embedding-flavored vectors (the RAG use case the spec targets):
// two 4-dim "embeddings" and a per-dimension scaling/offset vector. The v1
// backend deliberately does NOT appear here — it stays the frozen
// four-op (add/matmul/sum/transpose) performance baseline (spec: "the v1
// backend deliberately does NOT appear for the new ops").
console.log("\n=== Kern 07: elementwise sub/mul/div + dot/norm/cosineSimilarity ===\n");

const embA = NDArray.fromArray([4], [0.2, 0.4, 0.1, 0.8]);
const embB = NDArray.fromArray([4], [0.5, 0.1, 0.3, 0.2]);
const scale = NDArray.fromArray([4], [2, 2, 2, 2]);

const diff = embA.sub(embB);
printArray("embA - embB  ", diff);
const scaled = embA.mul(scale);
printArray("embA * scale ", scaled);
const halved = embA.div(scale);
printArray("embA / scale ", halved);

const dot = embA.dot(embB);
const normA = embA.norm();
const normB = embB.norm();
const cosine = embA.cosineSimilarity(embB);
console.log(`embA . embB (dot)        = ${dot}`);
console.log(`||embA|| (norm)          = ${normA}`);
console.log(`||embB|| (norm)          = ${normB}`);
console.log(`cosineSimilarity(A,B)    = ${cosine}\n`);

const rEmbA = WNDArray.fromArray(core, embA.shape, embA.data);
const rEmbB = WNDArray.fromArray(core, embB.shape, embB.data);
const rScale = WNDArray.fromArray(core, scale.shape, scale.data);

const rDiff = rEmbA.sub(rEmbB);
assertResidentAgrees("embA - embB", diff, rDiff);
const rScaled = rEmbA.mul(rScale);
assertResidentAgrees("embA * scale", scaled, rScaled);
const rHalved = rEmbA.div(rScale);
assertResidentAgrees("embA / scale", halved, rHalved);

const rDot = rEmbA.dot(rEmbB);
const rNormA = rEmbA.norm();
const rNormB = rEmbB.norm();
const rCosine = rEmbA.cosineSimilarity(rEmbB);
if (!Object.is(dot, rDot)) throw new Error(`dot: TS/resident bit divergence: ${dot} vs ${rDot}`);
if (!Object.is(normA, rNormA)) throw new Error(`norm(embA): TS/resident bit divergence: ${normA} vs ${rNormA}`);
if (!Object.is(normB, rNormB)) throw new Error(`norm(embB): TS/resident bit divergence: ${normB} vs ${rNormB}`);
if (!Object.is(cosine, rCosine)) throw new Error(`cosineSimilarity: TS/resident bit divergence: ${cosine} vs ${rCosine}`);
console.log(`  [wasm resident] dot=${rDot} norm(A)=${rNormA} norm(B)=${rNormB} cosineSimilarity=${rCosine} — matches TS bit-for-bit\n`);

rEmbA.dispose();
rEmbB.dispose();
rScale.dispose();
rDiff.dispose();
rScaled.dispose();
rHalved.dispose();

// --- Kern 08: reshape/flatten (docs/kern-08-reshape-flatten-spec.md) -------
// Same elements, new shape — naive always copies; resident views if
// contiguous (O(1), shares the buffer), else materializes. `flatten()`'s
// return type is the Spike-04 payoff: a STATICALLY COMPUTED literal rank-1
// shape (see the `bigFlattened` hover below: `NDArray<[1048576]>`, not
// `NDArray<[number]>`, for a [1024,1024] input — verified by the assertion
// right after it, no `as`/manual proof needed).
console.log("\n=== Kern 08: reshape/flatten ===\n");

const cube234 = NDArray.zeros([2, 3, 4]);
for (let i = 0; i < cube234.data.length; i++) cube234.data[i] = i + 1;
const reshaped432 = cube234.reshape([4, 3, 2]);
printArray("cube234              ", cube234);
printArray("cube234.reshape([4,3,2])", reshaped432);

const flatCube = cube234.flatten();
printArray("cube234.flatten()    ", flatCube);
console.log(`  flatten() hover shape: NDArray<[${flatCube.shape.join(",")}]> (statically computed, not just runtime-correct)\n`);

// Digit-multiplication stress: [1024,1024] -> [1048576], the phase's own
// headline example — hover shows NDArray<[1048576]>, a computed LITERAL.
const bigArr = NDArray.zeros([1024, 1024]);
const bigFlattened = bigArr.flatten();
const _bigFlattenedHoverCheck: NDArray<[1048576]> = bigFlattened; // compile-time proof the hover is exactly [1048576]
console.log(`bigArr.flatten() shape = [${bigFlattened.shape.join(",")}] (hover: NDArray<[1048576]>)\n`);

const rCube234 = WNDArray.fromArray(core, cube234.shape, cube234.data);
const rReshaped = rCube234.reshape([4, 3, 2]);
assertResidentAgrees("cube234.reshape([4,3,2])", reshaped432, rReshaped);
const rFlatCube = rCube234.flatten();
assertResidentAgrees("cube234.flatten()", flatCube, rFlatCube);

// View-routing proof: a contiguous resident handle's reshape/flatten shares
// the SAME buffer pointer (O(1), no kernel call) — printed, not just tested.
console.log(`  [resident] reshape ptr match (view, not copy): ${rReshaped.describe().ptr === rCube234.describe().ptr}`);
console.log(`  [resident] flatten ptr match (view, not copy): ${rFlatCube.describe().ptr === rCube234.describe().ptr}\n`);

rCube234.dispose();
rReshaped.dispose();
rFlatCube.dispose();

console.log("=== demo complete: TS, WASM v1, and WASM resident all agree on every showcase op ===");
