/**
 * Minimal naive runtime: plain loops over `Float64Array`, row-major
 * (C-contiguous) strides, correct NumPy broadcasting semantics. Correctness
 * only — no performance work (see spec, "out of scope").
 *
 * This is also where the *gradual-typing* escape hatch cashes out: a
 * dynamic (`number`, not a literal) dim never produces a compile error, so
 * these functions defensively re-validate actual shapes at runtime and
 * throw a descriptive `Error` on a real mismatch the type checker couldn't
 * see statically.
 */

export function product(shape: readonly number[]): number {
  return shape.reduce((acc, d) => acc * d, 1);
}

/** Row-major (C-contiguous) strides for a shape. */
export function computeStrides(shape: readonly number[]): number[] {
  const strides = new Array<number>(shape.length).fill(0);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = acc;
    acc *= shape[i] ?? 1;
  }
  return strides;
}

/** Decode a flat row-major index into per-axis indices. */
export function unravel(flat: number, shape: readonly number[], strides: readonly number[]): number[] {
  return shape.map((dim, i) => Math.floor(flat / (strides[i] ?? 1)) % dim);
}

/**
 * NumPy broadcast of two *runtime* shapes (mirrors the type-level
 * `Broadcast<A, B>`). Throws a descriptive error naming the incompatible
 * shapes/dims on failure — this is the runtime backstop for dims the type
 * system left dynamic (`number`).
 */
export function runtimeBroadcastShape(a: readonly number[], b: readonly number[]): number[] {
  const rank = Math.max(a.length, b.length);
  const result = new Array<number>(rank);
  for (let i = 0; i < rank; i++) {
    const ai = a[a.length - 1 - i] ?? 1;
    const bi = b[b.length - 1 - i] ?? 1;
    if (ai === bi || ai === 1 || bi === 1) {
      result[rank - 1 - i] = ai === 1 ? bi : ai;
    } else {
      throw new Error(
        `broadcast: shapes [${a.join(",")}] and [${b.join(",")}] are not broadcast-compatible ` +
          `at axis ${-1 - i} (dims ${ai} and ${bi})`,
      );
    }
  }
  return result;
}

/** Pad a shape/strides pair with leading (dim=1, stride=0) axes up to `rank`. */
function alignToRank(
  shape: readonly number[],
  strides: readonly number[],
  rank: number,
): { shape: number[]; strides: number[] } {
  const pad = rank - shape.length;
  return {
    shape: [...new Array<number>(pad).fill(1), ...shape],
    strides: [...new Array<number>(pad).fill(0), ...strides],
  };
}

/** Effective strides for indexing under broadcasting: any axis of size 1 in
 * the (rank-aligned) source contributes stride 0, regardless of its
 * naturally-computed stride. */
function effectiveStrides(alignedShape: readonly number[], alignedStrides: readonly number[]): number[] {
  return alignedShape.map((d, i) => (d === 1 ? 0 : (alignedStrides[i] ?? 0)));
}

/** Generic broadcasting elementwise binary op over two Float64Arrays. */
export function elementwiseBinary(
  aShape: readonly number[],
  aData: Float64Array,
  bShape: readonly number[],
  bData: Float64Array,
  op: (x: number, y: number) => number,
): { shape: number[]; data: Float64Array } {
  const outShape = runtimeBroadcastShape(aShape, bShape);
  const rank = outShape.length;

  const aAligned = alignToRank(aShape, computeStrides(aShape), rank);
  const bAligned = alignToRank(bShape, computeStrides(bShape), rank);
  const aEff = effectiveStrides(aAligned.shape, aAligned.strides);
  const bEff = effectiveStrides(bAligned.shape, bAligned.strides);

  const outStrides = computeStrides(outShape);
  const size = product(outShape);
  const out = new Float64Array(size);

  for (let flat = 0; flat < size; flat++) {
    const idx = unravel(flat, outShape, outStrides);
    let aOff = 0;
    let bOff = 0;
    for (let i = 0; i < rank; i++) {
      const ix = idx[i] ?? 0;
      aOff += ix * (aEff[i] ?? 0);
      bOff += ix * (bEff[i] ?? 0);
    }
    out[flat] = op(aData[aOff] ?? 0, bData[bOff] ?? 0);
  }
  return { shape: outShape, data: out };
}

/**
 * Full NumPy `matmul` at runtime: 1-D promotion (with squeeze), batch-dim
 * broadcasting, naive O(batch * m * n * k) triple loop for the 2-D core.
 */
export function matmulRuntime(
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

  const aPromoted = aShapeIn.length === 1; // prepend a 1 (squeezed from the result afterwards)
  const bPromoted = bShapeIn.length === 1; // append a 1 (squeezed from the result afterwards)

  const aShape = aPromoted ? [1, aShapeIn[0] ?? 1] : aShapeIn;
  const bShape = bPromoted ? [bShapeIn[0] ?? 1, 1] : bShapeIn;

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

  const aStridesFull = computeStrides(aShape);
  const bStridesFull = computeStrides(bShape);

  const aBatchAligned = alignToRank(batchA, aStridesFull.slice(0, -2), batchRank);
  const bBatchAligned = alignToRank(batchB, bStridesFull.slice(0, -2), batchRank);
  const aBatchEff = effectiveStrides(aBatchAligned.shape, aBatchAligned.strides);
  const bBatchEff = effectiveStrides(bBatchAligned.shape, bBatchAligned.strides);

  const aRowStride = aStridesFull[aStridesFull.length - 2] ?? 0;
  const aColStride = aStridesFull[aStridesFull.length - 1] ?? 0;
  const bRowStride = bStridesFull[bStridesFull.length - 2] ?? 0;
  const bColStride = bStridesFull[bStridesFull.length - 1] ?? 0;

  const outFullShape = [...batchOut, m, n];
  const outStridesFull = computeStrides(outFullShape);
  const outBatchStrides = outStridesFull.slice(0, batchRank);
  const batchStridesPlain = computeStrides(batchOut);

  const batchSize = product(batchOut);
  const out = new Float64Array(batchSize * m * n);

  for (let bIdx = 0; bIdx < batchSize; bIdx++) {
    const multi = unravel(bIdx, batchOut, batchStridesPlain);
    let aBatchOff = 0;
    let bBatchOff = 0;
    let outBatchOff = 0;
    for (let i = 0; i < batchRank; i++) {
      const ix = multi[i] ?? 0;
      aBatchOff += ix * (aBatchEff[i] ?? 0);
      bBatchOff += ix * (bBatchEff[i] ?? 0);
      outBatchOff += ix * (outBatchStrides[i] ?? 0);
    }
    for (let mi = 0; mi < m; mi++) {
      for (let ni = 0; ni < n; ni++) {
        let sum = 0;
        for (let ki = 0; ki < k1; ki++) {
          const aVal = aData[aBatchOff + mi * aRowStride + ki * aColStride] ?? 0;
          const bVal = bData[bBatchOff + ki * bRowStride + ni * bColStride] ?? 0;
          sum += aVal * bVal;
        }
        out[outBatchOff + mi * n + ni] = sum;
      }
    }
  }

  // Squeezing a size-1 axis never changes the flat (row-major) data layout,
  // only the shape metadata — no data reshuffling needed.
  let finalShape = outFullShape;
  if (aPromoted) {
    finalShape = [...finalShape.slice(0, batchRank), ...finalShape.slice(batchRank + 1)];
  }
  if (bPromoted) {
    finalShape = finalShape.slice(0, -1);
  }

  return { shape: finalShape, data: out };
}

/** Sum-reduce along one axis (negative axes count from the end), or over
 * every element if `axis` is `undefined`. */
export function sumRuntime(
  shape: readonly number[],
  data: Float64Array,
  axis: number | undefined,
): { shape: number[]; data: Float64Array } {
  if (axis === undefined) {
    let total = 0;
    for (let i = 0; i < data.length; i++) total += data[i] ?? 0;
    return { shape: [], data: Float64Array.from([total]) };
  }

  const rank = shape.length;
  const normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw new Error(`reduce: axis ${axis} is out of range for shape [${shape.join(",")}] (rank ${rank})`);
  }

  const outShape = [...shape.slice(0, normAxis), ...shape.slice(normAxis + 1)];
  const strides = computeStrides(shape);
  const outStrides = computeStrides(outShape);
  const outSize = product(outShape);
  const out = new Float64Array(outSize);
  const axisDim = shape[normAxis] ?? 1;
  const axisStride = strides[normAxis] ?? 0;

  for (let outFlat = 0; outFlat < outSize; outFlat++) {
    const idx = unravel(outFlat, outShape, outStrides);
    let baseOffset = 0;
    let outAxis = 0;
    for (let inAxis = 0; inAxis < rank; inAxis++) {
      if (inAxis === normAxis) continue;
      baseOffset += (idx[outAxis] ?? 0) * (strides[inAxis] ?? 0);
      outAxis++;
    }
    let total = 0;
    for (let a = 0; a < axisDim; a++) {
      total += data[baseOffset + a * axisStride] ?? 0;
    }
    out[outFlat] = total;
  }
  return { shape: outShape, data: out };
}

/** Reverse every axis (NumPy's `.T` generalized to N-D). */
export function transposeRuntime(shape: readonly number[], data: Float64Array): { shape: number[]; data: Float64Array } {
  const rank = shape.length;
  const outShape = [...shape].reverse();
  const inStrides = computeStrides(shape);
  const outStrides = computeStrides(outShape);
  const size = product(shape);
  const out = new Float64Array(size);

  for (let flat = 0; flat < size; flat++) {
    const outIdx = unravel(flat, outShape, outStrides);
    let inOffset = 0;
    for (let i = 0; i < rank; i++) {
      const originalAxis = rank - 1 - i;
      inOffset += (outIdx[i] ?? 0) * (inStrides[originalAxis] ?? 0);
    }
    out[flat] = data[inOffset] ?? 0;
  }
  return { shape: outShape, data: out };
}
