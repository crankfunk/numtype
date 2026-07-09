/**
 * WASM-backed twin of `spike/src/runtime.ts`'s exported ops: same
 * signatures/semantics, but the actual number-crunching happens in the
 * Rust kernels via the ABI in `crates/core/src/abi.rs`. Each op here:
 * copies operands into WASM scratch memory, runs the kernel, copies the
 * result out, frees the scratch. Copy-in/copy-out is deliberate for v1
 * (see spec, "out of scope": zero-copy resident buffers are v2).
 *
 * Shape computation (broadcast/matmul-promotion) is still done TS-side,
 * reusing `runtime.ts`'s own (already-correct, already-tested) helpers —
 * this is the "TS computes output shapes" half of the spec's contract; the
 * Rust kernels independently re-derive the same shapes and re-validate
 * (the "Rust re-validates" half). A divergence between the two would
 * surface either as a non-zero status (caught below and turned into a
 * thrown Error) or, for compatible-but-wrong-shape bugs, as a length
 * mismatch against the buffer TS sized — both are failures a differential
 * test will catch.
 *
 * **Memory rule (hard):** never cache a `Float64Array`/`Uint32Array` view
 * across any call that may allocate (`memory.grow` detaches views — proven
 * empirically, see docs/kern-01-ergebnisse.md). Every helper below
 * re-derives its view fresh from `core.memory.buffer` immediately before
 * use, after all allocations for a given call have already happened.
 */

import { product, runtimeBroadcastShape } from "../runtime.ts";
import type { CoreExports } from "./loader.ts";

interface ScratchBuf {
  readonly ptr: number;
  readonly bytes: number;
}

function allocBytes(core: CoreExports, bytes: number): ScratchBuf {
  const ptr = core.nt_alloc(bytes);
  if (ptr === 0 && bytes !== 0) {
    throw new Error(`wasm backend: nt_alloc(${bytes}) failed (out of memory)`);
  }
  return { ptr, bytes };
}

function freeBuf(core: CoreExports, buf: ScratchBuf): void {
  core.nt_free(buf.ptr, buf.bytes);
}

/** Allocate + write a shape as a little-endian u32 array. Returns the
 * scratch buffer (rank * 4 bytes). */
function writeShape(core: CoreExports, shape: readonly number[]): ScratchBuf {
  const buf = allocBytes(core, shape.length * 4);
  // Re-derive the view fresh, after allocation, right before writing.
  const view = new Uint32Array(core.memory.buffer, buf.ptr, shape.length);
  view.set(shape);
  return buf;
}

/** Allocate + write an f64 data array. Returns the scratch buffer
 * (len * 8 bytes). */
function writeData(core: CoreExports, data: Float64Array): ScratchBuf {
  const buf = allocBytes(core, data.length * 8);
  const view = new Float64Array(core.memory.buffer, buf.ptr, data.length);
  view.set(data);
  return buf;
}

/** Read `len` f64s starting at `ptr` out as an independent copy (never a
 * live view — the source buffer is about to be freed). */
function readDataCopy(core: CoreExports, ptr: number, len: number): Float64Array {
  const view = new Float64Array(core.memory.buffer, ptr, len);
  return Float64Array.from(view);
}

/** Broadcasting elementwise add — WASM-backed twin of
 * `elementwiseBinary(..., (x, y) => x + y)`. */
export function wasmAdd(
  core: CoreExports,
  aShape: readonly number[],
  aData: Float64Array,
  bShape: readonly number[],
  bData: Float64Array,
): { shape: number[]; data: Float64Array } {
  const outShape = runtimeBroadcastShape(aShape, bShape); // throws with named shapes on incompatibility
  const outLen = product(outShape);

  const aShapeBuf = writeShape(core, aShape);
  const bShapeBuf = writeShape(core, bShape);
  const aDataBuf = writeData(core, aData);
  const bDataBuf = writeData(core, bData);
  const outDataBuf = allocBytes(core, outLen * 8);

  try {
    const status = core.nt_add(
      aShapeBuf.ptr,
      aShape.length,
      aDataBuf.ptr,
      aData.length,
      bShapeBuf.ptr,
      bShape.length,
      bDataBuf.ptr,
      bData.length,
      outDataBuf.ptr,
      outLen,
    );
    if (status !== 0) {
      throw new Error(
        `wasm backend nt_add: status ${status} for shapes [${aShape.join(",")}] and [${bShape.join(",")}]`,
      );
    }
    return { shape: outShape, data: readDataCopy(core, outDataBuf.ptr, outLen) };
  } finally {
    freeBuf(core, aShapeBuf);
    freeBuf(core, bShapeBuf);
    freeBuf(core, aDataBuf);
    freeBuf(core, bDataBuf);
    freeBuf(core, outDataBuf);
  }
}

/** Full NumPy `matmul` — WASM-backed twin of `matmulRuntime`. 1-D promotion
 * and the final squeeze happen here (TS-side, mirroring `matmulRuntime`
 * exactly); the 2-D-core-plus-batch-broadcast part is delegated to
 * `nt_matmul`. */
export function wasmMatmul(
  core: CoreExports,
  aShapeIn: readonly number[],
  aData: Float64Array,
  bShapeIn: readonly number[],
  bData: Float64Array,
): { shape: number[]; data: Float64Array } {
  if (aShapeIn.length === 0) {
    throw new Error(`matmul: scalar operand (rank 0) is not allowed as the first argument (got shape [])`);
  }
  if (bShapeIn.length === 0) {
    throw new Error(`matmul: scalar operand (rank 0) is not allowed as the second argument (got shape [])`);
  }

  const aPromoted = aShapeIn.length === 1;
  const bPromoted = bShapeIn.length === 1;
  const aShape = aPromoted ? [1, aShapeIn[0] ?? 1] : [...aShapeIn];
  const bShape = bPromoted ? [bShapeIn[0] ?? 1, 1] : [...bShapeIn];

  const m = aShape[aShape.length - 2] ?? 1;
  const k1 = aShape[aShape.length - 1] ?? 1;
  const k2 = bShape[bShape.length - 2] ?? 1;
  const n = bShape[bShape.length - 1] ?? 1;
  if (k1 !== k2) {
    throw new Error(`matmul: inner dimensions ${k1} and ${k2} do not match`);
  }

  const batchA = aShape.slice(0, -2);
  const batchB = bShape.slice(0, -2);
  const batchOut = runtimeBroadcastShape(batchA, batchB);
  const batchRank = batchOut.length;
  const outFullShape = [...batchOut, m, n];
  const outLen = product(outFullShape);

  const aShapeBuf = writeShape(core, aShape);
  const bShapeBuf = writeShape(core, bShape);
  const aDataBuf = writeData(core, aData);
  const bDataBuf = writeData(core, bData);
  const outDataBuf = allocBytes(core, outLen * 8);

  let outData: Float64Array;
  try {
    const status = core.nt_matmul(
      aShapeBuf.ptr,
      aShape.length,
      aDataBuf.ptr,
      aData.length,
      bShapeBuf.ptr,
      bShape.length,
      bDataBuf.ptr,
      bData.length,
      outDataBuf.ptr,
      outLen,
    );
    if (status !== 0) {
      throw new Error(
        `wasm backend nt_matmul: status ${status} for shapes [${aShapeIn.join(",")}] and [${bShapeIn.join(",")}]`,
      );
    }
    outData = readDataCopy(core, outDataBuf.ptr, outLen);
  } finally {
    freeBuf(core, aShapeBuf);
    freeBuf(core, bShapeBuf);
    freeBuf(core, aDataBuf);
    freeBuf(core, bDataBuf);
    freeBuf(core, outDataBuf);
  }

  // Squeezing a size-1 axis never changes the flat (row-major) data layout,
  // only the shape metadata — mirrors matmulRuntime exactly.
  let finalShape = outFullShape;
  if (aPromoted) {
    finalShape = [...finalShape.slice(0, batchRank), ...finalShape.slice(batchRank + 1)];
  }
  if (bPromoted) {
    finalShape = finalShape.slice(0, -1);
  }

  return { shape: finalShape, data: outData };
}

/** Sum-reduce — WASM-backed twin of `sumRuntime`. `axis === undefined`
 * sums every element (via `nt_sum_all`); otherwise reduces along one axis
 * (via `nt_sum_axis`), with the same axis normalization/range check as
 * `sumRuntime` (so the thrown message matches the reference exactly). */
export function wasmSum(
  core: CoreExports,
  shape: readonly number[],
  data: Float64Array,
  axis: number | undefined,
): { shape: number[]; data: Float64Array } {
  if (axis === undefined) {
    const aDataBuf = writeData(core, data);
    const outDataBuf = allocBytes(core, 8);
    try {
      const status = core.nt_sum_all(aDataBuf.ptr, data.length, outDataBuf.ptr);
      if (status !== 0) {
        throw new Error(`wasm backend nt_sum_all: unexpected status ${status}`);
      }
      return { shape: [], data: readDataCopy(core, outDataBuf.ptr, 1) };
    } finally {
      freeBuf(core, aDataBuf);
      freeBuf(core, outDataBuf);
    }
  }

  const rank = shape.length;
  const normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw new Error(`reduce: axis ${axis} is out of range for shape [${shape.join(",")}] (rank ${rank})`);
  }

  const outShape = [...shape.slice(0, normAxis), ...shape.slice(normAxis + 1)];
  const outLen = product(outShape);

  const shapeBuf = writeShape(core, shape);
  const dataBuf = writeData(core, data);
  const outDataBuf = allocBytes(core, outLen * 8);

  try {
    const status = core.nt_sum_axis(shapeBuf.ptr, shape.length, dataBuf.ptr, data.length, axis, outDataBuf.ptr, outLen);
    if (status !== 0) {
      throw new Error(`wasm backend nt_sum_axis: status ${status} for shape [${shape.join(",")}] axis ${axis}`);
    }
    return { shape: outShape, data: readDataCopy(core, outDataBuf.ptr, outLen) };
  } finally {
    freeBuf(core, shapeBuf);
    freeBuf(core, dataBuf);
    freeBuf(core, outDataBuf);
  }
}

/** Reverse every axis — WASM-backed twin of `transposeRuntime`. */
export function wasmTranspose(
  core: CoreExports,
  shape: readonly number[],
  data: Float64Array,
): { shape: number[]; data: Float64Array } {
  const outShape = [...shape].reverse();
  const outLen = product(outShape);

  const shapeBuf = writeShape(core, shape);
  const dataBuf = writeData(core, data);
  const outDataBuf = allocBytes(core, outLen * 8);

  try {
    const status = core.nt_transpose(shapeBuf.ptr, shape.length, dataBuf.ptr, data.length, outDataBuf.ptr, outLen);
    if (status !== 0) {
      throw new Error(`wasm backend nt_transpose: status ${status} for shape [${shape.join(",")}]`);
    }
    return { shape: outShape, data: readDataCopy(core, outDataBuf.ptr, outLen) };
  } finally {
    freeBuf(core, shapeBuf);
    freeBuf(core, dataBuf);
    freeBuf(core, outDataBuf);
  }
}

/** Fill `len` elements with `value` — WASM-backed twin of
 * `new Float64Array(len).fill(value)`. */
export function wasmFill(core: CoreExports, len: number, value: number): Float64Array {
  const outDataBuf = allocBytes(core, len * 8);
  try {
    const status = core.nt_fill(outDataBuf.ptr, len, value);
    if (status !== 0) {
      throw new Error(`wasm backend nt_fill: unexpected status ${status}`);
    }
    return readDataCopy(core, outDataBuf.ptr, len);
  } finally {
    freeBuf(core, outDataBuf);
  }
}
