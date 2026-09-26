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

/**
 * One axis's slice specification at the runtime boundary — structurally
 * identical to `slice.ts`'s type-level `SliceSpecInput`, declared
 * independently here so this module stays the standalone, type-file-free
 * value layer it already is (mirrors the existing `Shape`/`readonly
 * number[]` split: dim.ts declares `Shape`, this file never imports it).
 *
 * NumPy semantics (see docs/kern-05-slicing-spec.md for the full fixture
 * table this normalizer is pinned against):
 *  - `number` — index the axis; negative counts from the end; out-of-bounds
 *    (after normalization) THROWS (indices never clamp).
 *  - `null` — take the axis in full.
 *  - `{ start?, stop?, step? }` — range slice; `step` defaults to 1 and must
 *    be `>= 1` (throw otherwise — negative steps need signed strides, out of
 *    scope since Kern 03); `start`/`stop` default to `0`/`d`, negative values
 *    count from the end, and the result CLAMPS to `[0, d]` (never throws).
 */
export type SliceSpec = number | null | { readonly start?: number; readonly stop?: number; readonly step?: number };

/** One axis's spec, post-normalization: either a dropped `index` (with its
 * already-bounds-checked absolute position) or a surviving `range` (with its
 * already-clamped absolute `start`, resulting element `dim`, and `step`). */
export type NormalizedAxisSpec =
  | { readonly kind: "index"; readonly i: number }
  | { readonly kind: "range"; readonly start: number; readonly dim: number; readonly step: number };

/** Normalize one axis's raw spec against its dim `d` (see `SliceSpec`'s doc
 * comment for the exact semantics; `axis` is only for error messages).
 * Indices/step are assumed integral (NumPy itself rejects fractional slice
 * indices with a `TypeError`) — a non-integer value would otherwise produce
 * a non-integer element offset/stride that silently corrupts every strided
 * read downstream, so this is checked explicitly rather than left latent. */
function normalizeAxisSpec(d: number, spec: SliceSpec, axis: number): NormalizedAxisSpec {
  if (spec === null) {
    return { kind: "range", start: 0, dim: d, step: 1 };
  }
  if (typeof spec === "number") {
    if (!Number.isInteger(spec)) {
      throw new Error(`slice: index ${spec} for axis ${axis} is not an integer`);
    }
    const i = spec < 0 ? spec + d : spec;
    if (i < 0 || i >= d) {
      throw new Error(`slice: index ${spec} is out of bounds for axis ${axis} with dim ${d}`);
    }
    return { kind: "index", i };
  }

  const step = spec.step ?? 1;
  if (!Number.isInteger(step) || step < 1) {
    throw new Error(`slice: step ${step} for axis ${axis} is invalid (must be an integer >= 1; negative steps are out of scope)`);
  }
  let start = spec.start ?? 0;
  if (!Number.isInteger(start)) {
    throw new Error(`slice: start ${start} for axis ${axis} is not an integer`);
  }
  if (start < 0) start += d;
  start = Math.min(Math.max(start, 0), d);

  let stop = spec.stop ?? d;
  if (!Number.isInteger(stop)) {
    throw new Error(`slice: stop ${stop} for axis ${axis} is not an integer`);
  }
  if (stop < 0) stop += d;
  stop = Math.min(Math.max(stop, 0), d);

  const dim = Math.max(0, Math.ceil((stop - start) / step));
  return { kind: "range", start, dim, step };
}

/**
 * Normalize a per-axis spec list against `shape`. One spec per LEADING axis
 * (trailing axes are implicitly "taken in full" — callers that need that
 * distinction check `specs.length` themselves; this function only normalizes
 * the axes it was given specs for). Throws if there are more specs than
 * axes (mirrors the type layer's `SliceSpecsGuard` compile-time rejection —
 * this is the runtime backstop for gradual/dynamic-rank callers the type
 * layer couldn't check statically).
 *
 * Shared, byte-for-byte, by BOTH `sliceRuntime` (naive, copy-based) and
 * `WNDArray.slice` (resident, O(1) view) — a deliberate, documented
 * differential blind spot: the two backends share spec *parsing* but
 * diverge in *data movement*, which is where the differential test suite's
 * value actually lies. This function's own semantics are pinned directly by
 * the fixture-table unit tests instead.
 */
export function normalizeSliceSpecs(shape: readonly number[], specs: readonly SliceSpec[]): NormalizedAxisSpec[] {
  if (specs.length > shape.length) {
    throw new Error(`slice: ${specs.length} specs given for rank ${shape.length} shape [${shape.join(",")}]`);
  }
  return specs.map((spec, axis) => normalizeAxisSpec(shape[axis] ?? 0, spec, axis));
}

/**
 * Naive reference slice: copy-based gather. Trailing axes beyond
 * `specs.length` are taken in full. Walks the SAME per-axis
 * offset/stride algebra `WNDArray.slice`'s O(1) view construction does
 * (shared normalizer, diverging data movement — see `normalizeSliceSpecs`'s
 * doc comment) but immediately gathers into a fresh contiguous buffer
 * instead of returning view metadata, since `NDArray` never aliases.
 */
export function sliceRuntime(
  shape: readonly number[],
  data: Float64Array,
  specs: readonly NormalizedAxisSpec[],
): { shape: number[]; data: Float64Array } {
  const originalStrides = computeStrides(shape);
  const outShape: number[] = [];
  const viewStrides: number[] = [];
  let offset = 0;

  for (let axis = 0; axis < shape.length; axis++) {
    const stride = originalStrides[axis] ?? 0;
    const spec = specs[axis];
    if (spec === undefined) {
      // Trailing axis, beyond the given specs: taken in full.
      outShape.push(shape[axis] ?? 0);
      viewStrides.push(stride);
      continue;
    }
    if (spec.kind === "index") {
      offset += spec.i * stride;
    } else {
      offset += spec.start * stride;
      outShape.push(spec.dim);
      viewStrides.push(stride * spec.step);
    }
  }

  const size = product(outShape);
  const out = new Float64Array(size);
  const outStrides = computeStrides(outShape);
  for (let flat = 0; flat < size; flat++) {
    const idx = unravel(flat, outShape, outStrides);
    let srcOffset = offset;
    for (let i = 0; i < viewStrides.length; i++) {
      srcOffset += (idx[i] ?? 0) * (viewStrides[i] ?? 0);
    }
    out[flat] = data[srcOffset] ?? 0;
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

// ---------------------------------------------------------------------------
// Kern 07 (docs/kern-07-elementwise-vector-spec.md): dot/norm vector-op
// support. Appended strictly after all pre-existing content in this file
// (freeze discipline — `runtime.ts` is the pinned reference for the frozen
// v1 differential suite; every line above this point is byte-for-byte
// unchanged). `sub`/`mul`/`div` themselves need NO new runtime.ts code —
// they reuse the existing `elementwiseBinary` with a pinned closure
// (`(x, y) => x - y` etc.), exactly like `add` already does.
// ---------------------------------------------------------------------------

/**
 * Shared vector-pair validator for `dot`/`cosineSimilarity`: both operands
 * must be rank-1 with equal length. SHARED between the naive (`NDArray`)
 * and resident (`WNDArray`) surfaces — spec parsing/validation is shared,
 * data paths diverge (same rationale, and the same documented differential
 * blind spot, as `normalizeSliceSpecs`, Kern 05). Checked in the
 * message-table order (first operand rank, second operand rank, length
 * mismatch); mirrors the compile-time `DotCheck` guard's message wording
 * verbatim (spike/src/vector.ts) — pinned directly by unit tests of these
 * exact strings, not just by the differential suite.
 */
export function assertVectorPair(op: string, aShape: readonly number[], bShape: readonly number[]): void {
  if (aShape.length !== 1) {
    throw new Error(`${op}: expected a 1-D vector as the first operand (got shape [${aShape.join(",")}])`);
  }
  if (bShape.length !== 1) {
    throw new Error(`${op}: expected a 1-D vector as the second operand (got shape [${bShape.join(",")}])`);
  }
  const aLen = aShape[0] ?? 0;
  const bLen = bShape[0] ?? 0;
  if (aLen !== bLen) {
    throw new Error(`${op}: vector lengths ${aLen} and ${bLen} do not match`);
  }
}

/**
 * 1-D inner product: `sum(a[i] * b[i])`, single accumulator, strictly
 * ascending index, seed `0` — mirrors `dot_strided`'s bit-identity contract
 * exactly (no FMA, no reordering; this makes `dotRuntime(a, a)` bit-
 * identical to `sumRuntime` of the elementwise-squared array, which the
 * differential suite pins). Callers validate the operand pair via
 * `assertVectorPair` FIRST (same contract as `sliceRuntime` taking
 * already-normalized specs) — this function assumes `aShape`/`bShape` are
 * already known-valid (rank 1, equal length) and does not re-check.
 */
export function dotRuntime(aShape: readonly number[], aData: Float64Array, bShape: readonly number[], bData: Float64Array): number {
  const n = aShape[0] ?? 0;
  let acc = 0;
  for (let i = 0; i < n; i++) {
    acc += (aData[i] ?? 0) * (bData[i] ?? 0);
  }
  return acc;
}

/**
 * Sum of squares over every element of `data`, in flat order — single
 * accumulator, strictly ascending index, seed `0`, `acc += v * v`. For the
 * naive backend `data` is always already the shape's logical row-major
 * flattening (every `NDArray` op materializes a fresh contiguous buffer —
 * see ndarray.ts/module docs), so flat order IS logical order here; mirrors
 * `norm_sq_strided`'s bit-identity contract. `Math.sqrt` of this result is
 * `norm()` (TS-side, per the spec's pinned composition). Any rank/size,
 * including size-0 (-> `0`).
 */
export function normSqRuntime(data: Float64Array): number {
  let acc = 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i] ?? 0;
    acc += v * v;
  }
  return acc;
}

// ---------------------------------------------------------------------------
// Kern 08 (docs/kern-08-reshape-flatten-spec.md): reshape/flatten runtime
// validator. Appended strictly after all pre-existing content in this file
// (freeze discipline — every line above this point is byte-for-byte
// unchanged). `reshape` itself needs NO further runtime.ts code beyond this
// validator: data movement is a straight copy (naive) or a view/materialize
// choice (resident, via the already-existing `nt_materialize` entry point) —
// logical row-major order is invariant under a reshape, so there is no
// per-element gather/scatter algebra to add here (unlike `sliceRuntime`).
// ---------------------------------------------------------------------------

/**
 * Shared reshape/flatten validator: both `NDArray.reshape` and
 * `WNDArray.reshape` call this BEFORE any allocation (same "validate first"
 * discipline as `assertVectorPair`/`normalizeSliceSpecs`). Checked in the
 * message-table order (dim validity first, per-axis left to right, then
 * product) — mirrors the compile-time `ReshapeCheck` guard's message wording
 * verbatim (spike/src/reshape.ts). `flatten` never calls this (always valid
 * by construction: `[product(shape)]` trivially has a matching product).
 */
export function assertReshapeArgs(oldShape: readonly number[], newShape: readonly number[]): void {
  for (const d of newShape) {
    if (!Number.isInteger(d) || d < 0) {
      throw new Error(`reshape: invalid dimension ${d} in shape [${newShape.join(",")}] (dims must be non-negative integers)`);
    }
  }
  const oldSize = product(oldShape);
  const newSize = product(newShape);
  if (oldSize !== newSize) {
    throw new Error(`reshape: cannot reshape array of size ${oldSize} into shape [${newShape.join(",")}]`);
  }
}

/**
 * Output shape for a `keepdims=True` sum-reduction (Kern 09, NumPy semantics):
 * the axis-reduced result of `sumRuntime`, but with the reduced axis kept as
 * size-1 instead of removed. The reduction DATA is byte-identical either way —
 * an axis of length 1 contributes nothing to the row-major order and leaves
 * `product`/`data.length` unchanged — so callers run the ordinary reduce and
 * swap in this shape (`NDArray.sum` and `WNDArray.sum` share this one helper,
 * making the two surfaces shape-identical by construction). `axis === undefined`
 * reduces every axis ⇒ all-ones of the input rank. `axis` is assumed already
 * range-validated by the caller's reduce path (same axis normalization as
 * `sumRuntime`); this is metadata only and throws nothing itself.
 */
export function keepDimsShape(shape: readonly number[], axis: number | undefined): number[] {
  if (axis === undefined) return shape.map(() => 1);
  const rank = shape.length;
  const normAxis = axis < 0 ? rank + axis : axis;
  return shape.map((d, i) => (i === normAxis ? 1 : d));
}

// ---------------------------------------------------------------------------
// Op-Scheibe W1 (docs/op-w1-argmax-topk-spec.md): argmax/topk runtime
// reference. Appended strictly after all pre-existing content in this file
// (freeze discipline — every line above this point is byte-for-byte
// unchanged). Pure reference functions with NO kernel counterpart (D1/M1 —
// this slice adds no WASM surface at all, unlike every op above).
// ---------------------------------------------------------------------------

/** Does `el` beat the current running maximum `max` (D4's pinned total
 * order, shared by `argmaxRuntime` and `topkRuntime`'s comparator below)?
 * NaN counts as MAXIMAL (NumPy `argmax` behavior) — an element beats the
 * max iff it's a NaN the max isn't, or it's numerically greater; a tie
 * (including `0`/`-0`, compared via plain `>`, never `Object.is`, and
 * including NaN-vs-NaN) leaves the max/first-seen index in place, since
 * callers only ever call this as a strict "does the NEW element win"
 * challenger check (never `>=`). */
function beatsMax(el: number, max: number): boolean {
  return (Number.isNaN(el) && !Number.isNaN(max)) || el > max;
}

/**
 * Index of the maximum element (Op-Scheibe W1, D4): niladic (`axis ===
 * undefined`) flattens row-major over every element, mirroring
 * `sumRuntime`'s own "reduce everything" branch exactly (same `{shape: [],
 * data: Float64Array.from([...])}` return shape — `NDArray.argmax()`
 * unwraps `.data[0]` to produce the public `number` result, D2). An empty
 * receiver throws (`argmax` has no answer for zero elements); the message
 * stem is pinned, compile-time-unreachable (no static claim is possible for
 * a niladic call — no argument to hang a `Guard` on, D2).
 *
 * `axis` normalization/validation is EXACTLY `sumRuntime`'s own (same
 * negative-axis wraparound, same out-of-range throw, WORD-FOR-WORD the same
 * message stem as `runtime.ts:222` above — the compile-time `ReduceAxis`
 * guard this function's caller reuses unmodified produces that exact string
 * as ITS OWN message, D4). Unlike `sumRuntime`, an axis whose OWN dim is `0`
 * also throws (the empty-receiver stem) — summing zero elements is `0`, a
 * well-defined answer; taking the argmax of zero candidates is not.
 */
export function argmaxRuntime(
  shape: readonly number[],
  data: Float64Array,
  axis: number | undefined,
): { shape: number[]; data: Float64Array } {
  if (axis === undefined) {
    if (data.length === 0) {
      throw new Error(`argmax: attempt to get argmax of an empty array`);
    }
    let maxIdx = 0;
    let maxVal = data[0] ?? 0;
    for (let i = 1; i < data.length; i++) {
      const v = data[i] ?? 0;
      if (beatsMax(v, maxVal)) {
        maxVal = v;
        maxIdx = i;
      }
    }
    return { shape: [], data: Float64Array.from([maxIdx]) };
  }

  const rank = shape.length;
  const normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw new Error(`reduce: axis ${axis} is out of range for shape [${shape.join(",")}] (rank ${rank})`);
  }
  const axisDim = shape[normAxis] ?? 1;
  if (axisDim === 0) {
    throw new Error(`argmax: attempt to get argmax of an empty array`);
  }

  const outShape = [...shape.slice(0, normAxis), ...shape.slice(normAxis + 1)];
  const strides = computeStrides(shape);
  const outStrides = computeStrides(outShape);
  const outSize = product(outShape);
  const out = new Float64Array(outSize);
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
    let maxIdx = 0;
    let maxVal = data[baseOffset] ?? 0;
    for (let a = 1; a < axisDim; a++) {
      const v = data[baseOffset + a * axisStride] ?? 0;
      if (beatsMax(v, maxVal)) {
        maxVal = v;
        maxIdx = a;
      }
    }
    out[outFlat] = maxIdx;
  }
  return { shape: outShape, data: out };
}

/** `topk`'s selection-order comparator (D4): NaN entries sort first; among
 * two non-NaN entries, the larger VALUE sorts first. A tie (either two NaNs,
 * or two equal non-NaN values, `0`/`-0` included via plain `>`/`<`, never
 * `Object.is`) is left to the caller's trailing `|| (i - j)` index
 * tiebreak — this function alone only orders by value, returning a negative
 * number when `a` should sort before `b` (the `Array.prototype.sort`
 * comparator contract). */
function topkCompareValues(a: number, b: number): number {
  const aNaN = Number.isNaN(a);
  const bNaN = Number.isNaN(b);
  if (aNaN && bNaN) return 0;
  if (aNaN) return -1;
  if (bNaN) return 1;
  if (a > b) return -1;
  if (a < b) return 1;
  return 0;
}

/**
 * Top-`k` values + indices of a rank-1 `data` (Op-Scheibe W1, D4). Validated
 * in the SAME order, with the SAME message stems, as the compile-time
 * `TopkCheck` (vector.ts): rank first, then `k`'s own validity (non-integer
 * or negative), then `k` against the vector's length. `k = 0` and `k =
 * length` both succeed (an empty result / the whole vector, sorted).
 *
 * Selection: a size-`k` bounded max-heap of "badness" (the root holds the
 * WORST of the `k` currently held elements under the total order below),
 * O(n log k) — NOT a full O(n log n) sort of all `n` indices. See
 * docs/op-topk-selection-spec.md v6 (Phase 2, outcome "reiner Heap", D8): the
 * heap was measured against the former full sort over a 92-cell grid
 * (docs/op-topk-selection-ergebnisse.md) and won unconditionally — e.g.
 * n=1e6, k=1 from 280 ms to 3.8 ms (factor 74) — with no dual violation
 * anywhere in the grid, so the whole vector's `k/n` range is safe and the
 * replacement is branch-free (t* = 1.0). This body replaces the former full
 * sort IN PLACE (a deviation from runtime.ts's append-only convention,
 * pre-approved in the spec: see the "Vorab-Genehmigung der In-Place-Abweichung"
 * paragraph before D8/D9). `topkCompareValues` is unchanged and reused here.
 *
 * Selection ORDER is unchanged from the former full sort and provably
 * bit-identical to it: every index is ordered by `topkCompareValues` (NaN
 * first, then descending value), ties broken by ascending index — and because
 * that comparator combined with `|| (idxA - idxB)` is a STRICT TOTAL ORDER on
 * the distinct indices `0..n-1`, there is exactly one correct top-`k` index
 * set in exactly one order, so heap and full sort MUST agree (spec section
 * "Die tragende Beobachtung"). `values[i] = data[indices[i]]` is a plain
 * `Float64Array`-element read/copy (never routed through any arithmetic), so a
 * NaN's exact bit payload survives untouched. The former full-sort form lives
 * on verbatim as the test oracle `topkOracleFullSort`
 * (spike/tests-runtime/argmax-topk.test.ts), diffed bit-for-bit against this
 * function over NaN payloads, `+0`/`-0`, ties and edge cases (D11).
 */
export function topkRuntime(
  shape: readonly number[],
  data: Float64Array,
  k: number,
): { values: Float64Array; indices: Float64Array } {
  if (shape.length !== 1) {
    throw new Error(`topk: expected a 1-D vector (got shape [${shape.join(",")}])`);
  }
  if (!Number.isInteger(k) || k < 0) {
    throw new Error(`topk: k must be a non-negative integer (got ${k})`);
  }
  const n = shape[0] ?? 0;
  if (k > n) {
    throw new Error(`topk: k=${k} exceeds the vector length ${n}`);
  }
  if (k === 0) {
    return { values: new Float64Array(0), indices: new Float64Array(0) };
  }

  // Size-`k` max-heap of "badness": the root holds the WORST of the `k`
  // currently held elements under the shared total order, so a new candidate
  // that ranks BEFORE the root evicts it and resifts. Two parallel typed
  // arrays (values + source indices) rather than an object heap — the
  // observable semantics (bit-identity) and complexity (O(n log k)) are the
  // binding parts, not the representation (spec D2).
  const heapVal = new Float64Array(k);
  const heapIdx = new Float64Array(k);
  let size = 0;

  const cmp = (aVal: number, aIdx: number, bVal: number, bIdx: number): number =>
    topkCompareValues(aVal, bVal) || aIdx - bIdx;

  const siftUp = (start: number): void => {
    let i = start;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      const pv = heapVal[parent] ?? 0;
      const pi = heapIdx[parent] ?? 0;
      const cv = heapVal[i] ?? 0;
      const ci = heapIdx[i] ?? 0;
      if (cmp(cv, ci, pv, pi) > 0) {
        heapVal[i] = pv;
        heapIdx[i] = pi;
        heapVal[parent] = cv;
        heapIdx[parent] = ci;
        i = parent;
      } else break;
    }
  };

  const siftDown = (start: number): void => {
    let i = start;
    for (;;) {
      const l = 2 * i + 1;
      const r = 2 * i + 2;
      let worst = i;
      if (l < size && cmp(heapVal[l] ?? 0, heapIdx[l] ?? 0, heapVal[worst] ?? 0, heapIdx[worst] ?? 0) > 0) worst = l;
      if (r < size && cmp(heapVal[r] ?? 0, heapIdx[r] ?? 0, heapVal[worst] ?? 0, heapIdx[worst] ?? 0) > 0) worst = r;
      if (worst === i) break;
      const iv = heapVal[i] ?? 0;
      const ii = heapIdx[i] ?? 0;
      heapVal[i] = heapVal[worst] ?? 0;
      heapIdx[i] = heapIdx[worst] ?? 0;
      heapVal[worst] = iv;
      heapIdx[worst] = ii;
      i = worst;
    }
  };

  for (let i = 0; i < n; i++) {
    const v = data[i] ?? 0;
    if (size < k) {
      heapVal[size] = v;
      heapIdx[size] = i;
      siftUp(size);
      size++;
    } else {
      const rv = heapVal[0] ?? 0;
      const ri = heapIdx[0] ?? 0;
      if (cmp(v, i, rv, ri) < 0) {
        heapVal[0] = v;
        heapIdx[0] = i;
        siftDown(0);
      }
    }
  }

  // Final O(k log k) sort of the held indices — SAME comparator expression as
  // the former full sort, `values[i]` re-read from the ORIGINAL `data` array
  // (never a cached heap value), so the data flow into `values`/`indices` is
  // identical to the former code (spec D2 point 4).
  const order = Array.from({ length: size }, (_, i) => heapIdx[i] ?? 0);
  order.sort((ia, ib) => topkCompareValues(data[ia] ?? 0, data[ib] ?? 0) || ia - ib);

  const values = new Float64Array(size);
  const indices = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    const srcIdx = order[i] ?? 0;
    indices[i] = srcIdx;
    values[i] = data[srcIdx] ?? 0;
  }
  return { values, indices };
}

// ---------------------------------------------------------------------------
// Op-Scheibe W2 (docs/op-w2-scalar-mean-spec.md): scalar-overload + `mean`
// runtime reference. Appended strictly after all pre-existing content in
// this file (freeze discipline — every line above this point is byte-for-byte
// unchanged). Pure reference functions with NO kernel counterpart (D1/M1 —
// like W1's argmax/topk above, this slice adds no WASM surface at all).
// ---------------------------------------------------------------------------

/**
 * Elementwise `data[i] op s` for a single scalar `s`, in strictly ascending
 * index order, into a fresh `Float64Array` (D3, docs/op-w2-scalar-mean-spec.md):
 * the runtime backing for `NDArray.add/sub/mul/div`'s new scalar overload.
 * Shape is untouched by this function — the caller (`NDArray`'s scalar branch)
 * passes `this.shape` straight through, unlike the binary `elementwiseBinary`
 * path (D2's shape-preserving, NumPy-scalar semantics: no [1]-broadcast temp,
 * no shape change even at rank 0). Pure IEEE 754 throughout, same as the
 * existing binary closures (`add`/`sub`/`mul`/`div` on `NDArray`) — no
 * zero/NaN/Infinity guards, no throws; `x/0 -> +/-Infinity`, `0/0 -> NaN`,
 * signed zeros/infinities propagate per the standard. A `switch` dispatcher
 * (rather than four separate exported closures) — a deliberate, spec-freed
 * style choice (Baustein-0 Nit 4) — keeps the four ops' loops each a single
 * straight-line pass, no per-element branch inside the hot loop.
 */
export function scalarElementwiseRuntime(op: "add" | "sub" | "mul" | "div", data: Float64Array, s: number): Float64Array {
  const out = new Float64Array(data.length);
  switch (op) {
    case "add":
      for (let i = 0; i < data.length; i++) out[i] = (data[i] ?? 0) + s;
      break;
    case "sub":
      for (let i = 0; i < data.length; i++) out[i] = (data[i] ?? 0) - s;
      break;
    case "mul":
      for (let i = 0; i < data.length; i++) out[i] = (data[i] ?? 0) * s;
      break;
    case "div":
      for (let i = 0; i < data.length; i++) out[i] = (data[i] ?? 0) / s;
      break;
  }
  return out;
}

/**
 * Mean-reduce along `axis` (negative axes count from the end), or over every
 * element if `axis` is `undefined` (D4/D5, docs/op-w2-scalar-mean-spec.md):
 * `sumRuntime`'s own reduction, then EXACTLY ONE division per output element
 * by `n` — the pinned determinism decision (`sum/n` per element, NEVER
 * `sum * (1/n)`, which rounds differently in f64). `n` is `shape[normAxis]`
 * for the axis form, or the total input element count (`product(shape)`) for
 * the full-reduction form — mirroring `sumRuntime`'s own axis-vs-undefined
 * split exactly.
 *
 * Axis validation/normalization and the out-of-range throw are entirely
 * `sumRuntime`'s own (this function adds no separate check) — so the thrown
 * message is, by construction, WORD-FOR-WORD `sumRuntime`'s `reduce: axis …`
 * stem (same guarantee D4 gave `argmaxRuntime` above, here for free via
 * delegation rather than a hand-duplicated check). Consequently `n`'s own
 * axis normalization below only ever runs once `sumRuntime` has already
 * proven the axis in range.
 *
 * size-0 (an empty receiver, or a size-0 axis): the reduced sum is `0` and
 * `n` is `0`, so every output element is `0/0 -> NaN` — NumPy-conformant
 * ("mean of empty is NaN"), never a throw (unlike `argmaxRuntime`, which
 * DOES throw on an empty/size-0-axis receiver — `mean` follows `sum`'s
 * always-well-defined-on-size-0 precedent instead, D5).
 */
export function meanRuntime(
  shape: readonly number[],
  data: Float64Array,
  axis: number | undefined,
): { shape: number[]; data: Float64Array } {
  const summed = sumRuntime(shape, data, axis);
  let n: number;
  if (axis === undefined) {
    n = product(shape);
  } else {
    const rank = shape.length;
    const normAxis = axis < 0 ? rank + axis : axis;
    n = shape[normAxis] ?? 1;
  }
  const out = new Float64Array(summed.data.length);
  for (let i = 0; i < out.length; i++) {
    out[i] = (summed.data[i] ?? 0) / n;
  }
  return { shape: summed.shape, data: out };
}

// ---------------------------------------------------------------------------
// Op-Scheibe W3 (docs/op-w3-sqrt-spec.md): appended strictly after all
// pre-existing content — nothing above this comment is touched.
// ---------------------------------------------------------------------------

/**
 * Elementwise `Math.sqrt(data[i])`, in strictly ascending index order, into a
 * fresh `Float64Array` (D2, docs/op-w3-sqrt-spec.md): the runtime backing for
 * `NDArray.sqrt()`. Shape is untouched by this function — `sqrt` is
 * shape-preserving at every rank, so the caller (`NDArray.sqrt`) passes
 * `this.shape` straight through unchanged, exactly like the scalar-op
 * closures above.
 *
 * IEEE-754 exactness (the Baustein-0-verified basis for `sqrt`'s exemption
 * from the transcendental non-goal, docs/op-w3-sqrt-spec.md "Berührte
 * Covenant-Invarianten"): ECMA-262 `sec-math.sqrt` defines `Math.sqrt`'s
 * result via the exact correctly-rounded real square root (the same
 * correctly-rounded contract `+`/`-`/`*`/`/` carry) — UNLIKE every
 * transcendental `Math.*` method (`exp`/`log`/`sin`/...), which the spec
 * explicitly marks "implementation-approximated". `Math.sqrt` is therefore
 * bit-deterministic across conforming engines, same as the four arithmetic
 * ops this file already treats as exact.
 *
 * Pinned IEEE edges (tests, D2): `sqrt(-0) === -0` (a genuine IEEE edge —
 * `Object.is`-distinguished from `+0`), `sqrt(-x) -> NaN` for finite `x > 0`,
 * `sqrt(NaN) -> NaN`, `sqrt(Infinity) -> Infinity`, subnormal inputs pass
 * through exactly (bit-compared against a direct `Math.sqrt` reference — the
 * op's OWN definition IS `Math.sqrt`, so the test proves faithful
 * pass-through, not independent mathematics). size-0 input -> an empty
 * output array, no special-casing needed (the loop body never runs).
 */
export function sqrtRuntime(data: Float64Array): Float64Array {
  const out = new Float64Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = Math.sqrt(data[i] ?? 0);
  return out;
}

// ---------------------------------------------------------------------------
// Op-Scheibe W4 (docs/op-w4-stack-spec.md, D3): `stackRuntime`, the runtime
// backing for `NDArray.stack(rows)`. Appended strictly after all pre-
// existing content in this file (freeze discipline — nothing above this
// comment is touched).
// ---------------------------------------------------------------------------

/**
 * Runtime validation + data movement for `NDArray.stack(rows)` (D3): the
 * SAME per-row, LEFT-TO-RIGHT order the compile-time fold uses (`StackFold`,
 * vector.ts) — rank checked before length for each row in turn, erroring at
 * the FIRST offending row rather than collecting every problem, so which
 * error fires for a given malformed input matches the type-level story
 * exactly (not required for soundness — the type layer only ever needs to
 * be "never wrong, only incomplete" — but keeping the two READING as the
 * same algorithm is worth the discipline). Three gepinnte stems, mirrored
 * verbatim by `StackEmptyMessage`/`StackRankMessage`/
 * `StackLengthMismatchMessage` (vector.ts):
 *  - `stack: expected at least one row` — an empty `rows` array (checked
 *    first, same precedence as the type-level `Shapes["length"] extends 0`
 *    gate, F3);
 *  - `stack: expected 1-D rows (got shape [...] at index i)` — the first
 *    row whose OWN shape isn't rank 1;
 *  - `stack: row length mismatch (expected D, got X at index i)` — the
 *    first row whose length disagrees with the FIRST row's length (`D`,
 *    fixed once seen — same "wide-Sentinel" merge policy the type-level
 *    `StackDimMerge` documents, F6, though at runtime every dim is always
 *    a concrete number, so there is no separate "wide" case to represent —
 *    only the ordinary equality check).
 *
 * D=0 rows are VALID (`[[],[]]` -> shape `[2, 0]`, D2/D3) — the length
 * `0 === 0` comparison falls out of the ordinary equality check above with
 * no special-casing needed.
 *
 * Data movement (D3): row-major `Float64Array#set` at each row's offset —
 * the EXACT algorithm `examples/rag-demo/embedding.ts`'s `embedMatrix`
 * hand-rolled before this op existed (`flat.set(embedText(text, dims), row
 * * dims)`), the very friction (F5, docs/dogfooding-rag-ergebnisse.md) that
 * motivated this op-slice. `Float64Array#set` is a raw bit-for-bit typed-
 * array copy (never routes through `Number`/arithmetic), so a NaN's exact
 * payload bits survive unchanged (D3: "NaN-Payloads byte-erhalten
 * (Movement-Op)") — the same guarantee `transposeRuntime`'s own plain-copy
 * inner loop already carries (special-values.test.ts's transpose fixture).
 */
export function stackRuntime(
  rows: ReadonlyArray<{ shape: readonly number[]; data: Float64Array }>,
): { shape: number[]; data: Float64Array } {
  if (rows.length === 0) {
    throw new Error("stack: expected at least one row");
  }
  let d: number | undefined;
  for (let i = 0; i < rows.length; i++) {
    const shape = rows[i]?.shape ?? [];
    if (shape.length !== 1) {
      throw new Error(`stack: expected 1-D rows (got shape [${shape.join(",")}] at index ${i})`);
    }
    const di = shape[0] ?? 0;
    if (d === undefined) {
      d = di;
    } else if (di !== d) {
      throw new Error(`stack: row length mismatch (expected ${d}, got ${di} at index ${i})`);
    }
  }
  const n = rows.length;
  const dim = d ?? 0;
  const out = new Float64Array(n * dim);
  for (let i = 0; i < n; i++) {
    out.set(rows[i]?.data ?? new Float64Array(0), i * dim);
  }
  return { shape: [n, dim], data: out };
}

/**
 * Op-Scheibe W5 (docs/op-w5-item-spec.md, D3): `item(...indices)`'s runtime
 * backstop — the direct scalar read `NDArray.item` exists for (`x.item(i,
 * j, ...)`, NumPy's own full-indexing accessor). Three checks, in order,
 * mirroring `normalizeAxisSpec`'s own per-axis checks above but generalized
 * to ALL axes at once (full indexing, never partial, D1):
 *  - Arity: `indices.length` must equal `shape.length` exactly — a
 *    RUNTIME-ONLY stem (F3, vector.ts's `ItemGuard` doc comment: missing/
 *    excess arguments are already a native TS2554 at compile time for any
 *    call whose rank the type layer can see statically, so this throw is
 *    reached only by gradual/dynamic-rank callers, or a rank-0 receiver
 *    called via a wide `number[]` spread).
 *  - Per axis: NumPy negative-index normalization (`i < 0 -> i + d`, the
 *    SAME rule `normalizeAxisSpec` already applies for `slice`), then a
 *    bounds check — word-for-word the same STEM `ItemMark`'s
 *    `ItemNotIntegerMessage`/`ItemOutOfBoundsMessage` (vector.ts) build at
 *    the type layer (M3: "Stems wortgleich zur Runtime"), minus the
 *    axis-count/shape suffix those type-level messages append purely for
 *    editor context — the same stem-vs-editor-context split
 *    `normalizeAxisSpec`'s own slice throws already carry relative to
 *    `slice.ts`'s `IndexOutOfBoundsMessage`/`StepInvalidMessage`. A size-0
 *    dim needs no special case: EVERY normalized index for `d = 0` fails
 *    `0 <= i < 0`, so it is caught unconditionally by the ordinary bounds
 *    check (pinned by test, per D3's own note).
 *  - Offset: the normalized per-axis indices dotted with `computeStrides`
 *    — a direct strided read, never a kernel (M1 v5: kernel-less; FOLLOWUPS
 *    tracks parity as a deferred item). NaN/±0 pass through byte-exact — a
 *    plain `Float64Array` indexed read never routes through arithmetic that
 *    could touch the value's bits.
 */
export function itemRuntime(shape: readonly number[], data: Float64Array, indices: readonly number[]): number {
  if (indices.length !== shape.length) {
    throw new Error(`item: expected ${shape.length} indices (got ${indices.length})`);
  }
  const strides = computeStrides(shape);
  let offset = 0;
  for (let axis = 0; axis < shape.length; axis++) {
    const raw = indices[axis] ?? 0;
    if (!Number.isInteger(raw)) {
      throw new Error(`item: index ${raw} for axis ${axis} is not an integer`);
    }
    const d = shape[axis] ?? 0;
    const i = raw < 0 ? raw + d : raw;
    if (i < 0 || i >= d) {
      throw new Error(`item: index ${raw} is out of bounds for axis ${axis} with dim ${d}`);
    }
    offset += i * (strides[axis] ?? 0);
  }
  return data[offset] ?? NaN;
}

/**
 * WASM parity S3 (docs/wasm-parity-item-stack-spec.md, D4): the
 * validation-and-offset half of `itemRuntime`'s own logic above, split out
 * so `WNDArray.item` (resident.ts) can reuse the identical arity/integer/
 * bounds checks and stem wording WITHOUT going through `itemRuntime`
 * itself. `itemRuntime` always derives its own strides via
 * `computeStrides(shape)` and reads from offset 0 — correct for its
 * contiguous `Float64Array` callers, but wrong for a resident VIEW (strided,
 * arbitrary `offset`, Kern 03). This helper takes `strides`/`base` directly
 * from the caller instead of deriving them, so the SAME three checks work
 * for both a contiguous handle (natural strides, offset 0 — passes through
 * identically to `itemRuntime`) and an arbitrary strided view.
 *
 * Deliberately a DUPLICATE of `itemRuntime`'s three checks (arity -> per-axis
 * integer -> per-axis bounds, same stems, same order) rather than a
 * refactor that routes `itemRuntime` through this helper — `itemRuntime`
 * stays this scheibe's Orakel, byte-unchanged (D4: a proof that rewrites
 * its own reference in the same commit proves less; the `runtime.ts`
 * append-only convention). Cross-surface message equality between
 * `itemRuntime` and this helper is a TEST obligation (T4), not a
 * structural guarantee — the price of duplication, paid off mechanically.
 */
export function itemOffsetStrided(
  shape: readonly number[],
  strides: readonly number[],
  base: number,
  indices: readonly number[],
): number {
  if (indices.length !== shape.length) {
    throw new Error(`item: expected ${shape.length} indices (got ${indices.length})`);
  }
  let offset = base;
  for (let axis = 0; axis < shape.length; axis++) {
    const raw = indices[axis] ?? 0;
    if (!Number.isInteger(raw)) {
      throw new Error(`item: index ${raw} for axis ${axis} is not an integer`);
    }
    const d = shape[axis] ?? 0;
    const i = raw < 0 ? raw + d : raw;
    if (i < 0 || i >= d) {
      throw new Error(`item: index ${raw} is out of bounds for axis ${axis} with dim ${d}`);
    }
    offset += i * (strides[axis] ?? 0);
  }
  return offset;
}

/**
 * WASM parity S3 (docs/wasm-parity-item-stack-spec.md, D4): the validation
 * half of `stackRuntime`'s own logic above, split out so `WNDArray.stack`
 * (resident.ts) can reuse the identical empty/rank/length-mismatch checks
 * and stem wording without also inheriting `stackRuntime`'s own
 * `Float64Array`-copying data movement — `WNDArray.stack` replaces that
 * movement with N `nt_materialize` calls into one WASM output buffer (D5),
 * so only the VALIDATION half is shared logic here. Returns the row count
 * and the shared row length so the caller can size its output buffer
 * BEFORE touching any WASM memory — mirrors `stackRuntime`'s own
 * "validate everything, then allocate" order, and this codebase's own
 * "shape computation before any allocation" convention (see e.g.
 * `WNDArray.add`).
 *
 * Deliberately a DUPLICATE of `stackRuntime`'s three checks (empty -> per-row
 * rank -> per-row length-vs-first, same stems, same order) rather than a
 * refactor of `stackRuntime` itself — same D4 rationale (and the same T4
 * cross-surface message-equality test obligation) as `itemOffsetStrided`
 * above.
 */
export function stackValidateShapes(shapes: readonly (readonly number[])[]): { n: number; d: number } {
  if (shapes.length === 0) {
    throw new Error("stack: expected at least one row");
  }
  let d: number | undefined;
  for (let i = 0; i < shapes.length; i++) {
    const shape = shapes[i] ?? [];
    if (shape.length !== 1) {
      throw new Error(`stack: expected 1-D rows (got shape [${shape.join(",")}] at index ${i})`);
    }
    const di = shape[0] ?? 0;
    if (d === undefined) {
      d = di;
    } else if (di !== d) {
      throw new Error(`stack: row length mismatch (expected ${d}, got ${di} at index ${i})`);
    }
  }
  return { n: shapes.length, d: d ?? 0 };
}

/**
 * D3 (docs/release-0.3.0-spec.md): shared pretty-printer for `toString()`
 * and the Node inspect hook on both `NDArray` and `WNDArray` — one nested-
 * value walk over whatever `toNestedArray()` already produced, so the two
 * backends render identically (`NDArray<[2, 3]> [[1, 2, 3], [4, 5, 6]]`).
 * Not itself a shape/stride walk (each class already has its own
 * `toNestedArray()` reading its own backing store) — purely "how to print
 * a value": numbers via plain `String()` (`NaN` -> `"NaN"`, matching this
 * codebase's existing stem-formatting elsewhere), arrays joined with ", ".
 *
 * Large arrays are summarized the way NumPy's default print options do it:
 * above 1000 elements in total, every axis longer than 6 shows only its
 * first and last 3 entries around a literal `...`
 * (`NDArray<[10000]> [0, 1, 2, ..., 9997, 9998, 9999]`). Display only —
 * `toJSON()`/`toNestedArray()` stay complete.
 */
const DISPLAY_SUMMARY_THRESHOLD = 1000;
const DISPLAY_EDGE_ITEMS = 3;
export function formatNDArrayDisplay(className: string, shape: readonly number[], nested: unknown): string {
  const summarize = product(shape) > DISPLAY_SUMMARY_THRESHOLD;
  const formatValue = (value: unknown): string => {
    if (!Array.isArray(value)) return String(value);
    const parts =
      summarize && value.length > 2 * DISPLAY_EDGE_ITEMS
        ? [...value.slice(0, DISPLAY_EDGE_ITEMS).map(formatValue), "...", ...value.slice(-DISPLAY_EDGE_ITEMS).map(formatValue)]
        : value.map(formatValue);
    return `[${parts.join(", ")}]`;
  };
  return `${className}<[${shape.join(", ")}]> ${formatValue(nested)}`;
}

// ---------------------------------------------------------------------------
// dt1 (docs/dtype-dt1-spec.md): dtype core on `NDArray` — storage, creation,
// conversion, and the dtype-neutral movement ops for float64/float32/int32/
// bool. Appended strictly after all pre-existing content in this file
// (freeze discipline — nothing above this comment is touched; every
// pre-existing exported function, including `transposeRuntime`/
// `sliceRuntime` above, keeps its exact byte-for-byte behavior and stays the
// float64-only oracle the new functions below are tested against). Reuses
// the design from docs/dtype-design-spec.md v2.1 (D1-D10) and the prototype
// on branch `proto/dtype` (059d852) where its scope matches dt1's K1/K2;
// the prototype LOCKED transpose/slice/reshape/flatten to float64-only,
// which dt1's K3 must instead implement for every dtype — the twins below
// are new, with no prototype precedent (dt1 spec, "Vorprüfung").
// ---------------------------------------------------------------------------

/** K1 (D1): the four dtypes this rollout supports (float16/int64/uint8/
 * complex are explicit non-goals, docs/dtype-design-spec.md "Nicht-Ziele").
 * `NDArray<S, D extends DType = "float64">`'s default keeps every existing
 * 1-type-argument use (`NDArray<[2, 3]>`) valid and meaning float64,
 * unchanged. */
export type DType = "float64" | "float32" | "int32" | "bool";

/** K1 (D2): `NDArray.data`'s storage per dtype — all four are ECMAScript-
 * standard typed arrays (M5/Z1 unaffected: no new dependency); bool stores
 * 0/1 in a `Uint8Array`. */
export type DataOf<D extends DType> = D extends "float32" ? Float32Array : D extends "int32" ? Int32Array : D extends "bool" ? Uint8Array : Float64Array;

/** The shape-erased union of every concrete `DataOf<D>` — this file's own
 * value-layer need for a dtype-generic buffer (mirrors `SliceSpec`'s role: a
 * plain value-layer type, independent of the type-level `DataOf<D>` above,
 * that this module's runtime functions pass around before the caller's own
 * generic `D` narrows it back down). */
export type DataOfRuntime = Float64Array | Float32Array | Int32Array | Uint8Array;

/** F1 fix (dt1 post-review): every `switch (dtype)` below over the closed
 * `DType` union was missing a `default` arm, so an invalid dtype string
 * smuggled in at runtime (`"invalid" as DType`, e.g. past an `as any`/JSON
 * boundary) fell through the switch with no return value — `data` silently
 * became `undefined` instead of throwing (M2 violation: a confidently-wrong
 * silent result, not an honest failure). This helper is both the shared
 * throw site (one message shape for all four call sites) AND the
 * exhaustiveness proof: passing anything but `never` here is now a TS2345
 * compile error at the call site, so a future fifth `DType` member that
 * forgets to update one of these switches fails the BUILD, not just at
 * runtime. */
function assertNeverDType(dtype: never, context: string): never {
  throw new Error(`${context}: invalid dtype ${JSON.stringify(dtype)} (expected one of "float64" | "float32" | "int32" | "bool")`);
}

/** K2 (D3): allocate a fresh all-zeros buffer for `dtype` (the runtime
 * backing for `NDArray.zeros`). */
export function zerosData(dtype: DType, size: number): DataOfRuntime {
  switch (dtype) {
    case "float64":
      return new Float64Array(size);
    case "float32":
      return new Float32Array(size);
    case "int32":
      return new Int32Array(size);
    case "bool":
      return new Uint8Array(size);
    default:
      return assertNeverDType(dtype, "zerosData");
  }
}

/** K2 (D3): allocate a fresh all-ones buffer for `dtype` (the runtime
 * backing for `NDArray.ones`) — `1` is exactly representable in every one of
 * the four storage kinds (including bool's 0/1 convention). */
export function onesData(dtype: DType, size: number): DataOfRuntime {
  switch (dtype) {
    case "float64":
      return new Float64Array(size).fill(1);
    case "float32":
      return new Float32Array(size).fill(1);
    case "int32":
      return new Int32Array(size).fill(1);
    case "bool":
      return new Uint8Array(size).fill(1);
    default:
      return assertNeverDType(dtype, "onesData");
  }
}

/**
 * K2 (D3): convert + validate a plain numeric source (`number[]` or any
 * existing typed array, read element-by-element via `ArrayLike<number>`)
 * into `dtype`'s storage, for the explicit-`dtype` path of `fromArray`.
 * Runtime backstop for exactly the cases the type layer leaves gradual (a
 * wide `number[]`, or a literal array whose per-element values the checker
 * can't/doesn't prove) — mirrors `astypeConvert`'s per-dtype rules below,
 * except int32/bool here THROW on a bad value rather than truncating/
 * coercing (D3: "Werte werden geprüft und konvertiert ... sonst Throw" — a
 * *constructor* input is validated strictly; `astype` is the *converting*
 * op, D3's own separate rule set below).
 *  - float64/float32: no validation beyond the numeric conversion itself
 *    (float32 via `Math.fround`, same correctly-rounded contract D8 relies
 *    on elsewhere).
 *  - int32: every value must already be an integer within [-2^31, 2^31-1],
 *    else throw (this is `fromArray`, not `astype` — no truncation here).
 *  - bool: every value must be exactly `0` or `1`, else throw.
 */
export function convertToDType(dtype: DType, values: ArrayLike<number>): DataOfRuntime {
  const n = values.length;
  switch (dtype) {
    case "float64": {
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) out[i] = values[i] ?? 0;
      return out;
    }
    case "float32": {
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = Math.fround(values[i] ?? 0);
      return out;
    }
    case "int32": {
      const out = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        const v = values[i] ?? 0;
        if (!Number.isInteger(v) || v < -2147483648 || v > 2147483647) {
          throw new Error(`fromArray: value ${v} at index ${i} is not a valid int32 (must be an integer in [-2147483648, 2147483647])`);
        }
        out[i] = v;
      }
      return out;
    }
    case "bool": {
      const out = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        const v = values[i] ?? 0;
        if (v !== 0 && v !== 1) {
          throw new Error(`fromArray: value ${v} at index ${i} is not a valid bool (must be 0 or 1)`);
        }
        out[i] = v;
      }
      return out;
    }
    default:
      return assertNeverDType(dtype, "convertToDType");
  }
}

/**
 * K2 (D3): `astype` conversion rules, keyed ONLY by the TARGET dtype (the
 * source value is always already a plain JS `number` once read out of any
 * typed array, so the conversion rule never needs to branch on the source's
 * own dtype — D2's "bool -> numerisch 0/1" is exactly the float64/float32/
 * int32 branches applied to values that happen to already be 0/1).
 *  - -> float64: passthrough numeric copy.
 *  - -> float32: `Math.fround` (D8's correctly-rounded contract).
 *  - -> int32: truncate TOWARD ZERO (`Math.trunc`, NumPy's own truncation
 *    direction) but THROW for NaN/±Infinity/out-of-[-2^31,2^31-1] (D3:
 *    "strenger als NumPy, das dort undefiniert ist") — deliberately NOT the
 *    same rule as `convertToDType`'s int32 branch (which rejects ANY
 *    non-integer; this one only rejects the un-truncatable/out-of-range
 *    cases and otherwise truncates).
 *  - -> bool: `x !== 0` (NaN -> `true`, same as NumPy — `NaN !== 0` is
 *    `true` under plain IEEE comparison already, no special case needed).
 */
export function astypeConvert(target: DType, data: DataOfRuntime): DataOfRuntime {
  const n = data.length;
  switch (target) {
    case "float64": {
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) out[i] = data[i] ?? 0;
      return out;
    }
    case "float32": {
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = Math.fround(data[i] ?? 0);
      return out;
    }
    case "int32": {
      const out = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        const v = data[i] ?? 0;
        if (!Number.isFinite(v)) {
          throw new Error(`astype("int32"): value ${v} at index ${i} is not finite (NaN/Infinity cannot convert to int32)`);
        }
        const t = Math.trunc(v);
        if (t < -2147483648 || t > 2147483647) {
          throw new Error(`astype("int32"): value ${v} at index ${i} (truncated to ${t}) is out of int32 range [-2147483648, 2147483647]`);
        }
        out[i] = t;
      }
      return out;
    }
    case "bool": {
      const out = new Uint8Array(n);
      for (let i = 0; i < n; i++) out[i] = (data[i] ?? 0) !== 0 ? 1 : 0;
      return out;
    }
    default:
      return assertNeverDType(target, "astypeConvert");
  }
}

/** K3 Vorprüfung (dt1 spec "Reihenfolge der Umsetzung" step 1): allocate a
 * fresh typed array of the SAME concrete class as `data`, with `size`
 * elements (zero-initialized) — `transposeDtyped`/`sliceDtyped` below need
 * their OUTPUT allocated in the same typed-array class as their INPUT,
 * unlike `transposeRuntime`/`sliceRuntime` above (which allocate
 * unconditionally `Float64Array` and stay byte-identical, untouched). */
export function sameKindArray(data: DataOfRuntime, size: number): DataOfRuntime {
  if (data instanceof Float64Array) return new Float64Array(size);
  if (data instanceof Float32Array) return new Float32Array(size);
  if (data instanceof Int32Array) return new Int32Array(size);
  if (data instanceof Uint8Array) return new Uint8Array(size);
  // F1 fix (dt1 post-review): the pre-fix `return new Float64Array(size)`
  // fallback here silently mislabeled ANY unrecognized input as float64 data
  // (M2 violation) instead of surfacing the impossible state. Every real
  // `DataOfRuntime` value is one of the four `instanceof` checks above; this
  // is reachable only via an `as any`/foreign-typed-array bypass of the type
  // layer, and the honest response is to throw, not to guess a dtype.
  throw new Error(`sameKindArray: unrecognized typed array (not Float64Array | Float32Array | Int32Array | Uint8Array)`);
}

/**
 * K3 (D5, "D unverändert"): dtype-generic twin of `transposeRuntime` above —
 * the IDENTICAL algorithm (same per-element offset walk), allocating its
 * output via `sameKindArray` instead of unconditionally `Float64Array`.
 * `transposeRuntime` itself is untouched (freeze discipline) and remains
 * the float64-only oracle this function is tested against (dt1 spec
 * "Vorprüfung": tested for all four dtypes against the float64 result,
 * converted, plus an assertion on the output's own constructor/dtype).
 */
export function transposeDtyped(shape: readonly number[], data: DataOfRuntime): { shape: number[]; data: DataOfRuntime } {
  const rank = shape.length;
  const outShape = [...shape].reverse();
  const inStrides = computeStrides(shape);
  const outStrides = computeStrides(outShape);
  const size = product(shape);
  const out = sameKindArray(data, size);

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

/**
 * K3 (D5, "D unverändert"): dtype-generic twin of `sliceRuntime` above —
 * the IDENTICAL per-axis offset/stride algebra and gather loop, allocating
 * its output via `sameKindArray` instead of unconditionally `Float64Array`.
 * `sliceRuntime` itself is untouched (freeze discipline) and remains the
 * float64-only oracle this function is tested against (same "Vorprüfung"
 * discipline as `transposeDtyped` above).
 */
export function sliceDtyped(
  shape: readonly number[],
  data: DataOfRuntime,
  specs: readonly NormalizedAxisSpec[],
): { shape: number[]; data: DataOfRuntime } {
  const originalStrides = computeStrides(shape);
  const outShape: number[] = [];
  const viewStrides: number[] = [];
  let offset = 0;

  for (let axis = 0; axis < shape.length; axis++) {
    const stride = originalStrides[axis] ?? 0;
    const spec = specs[axis];
    if (spec === undefined) {
      // Trailing axis, beyond the given specs: taken in full.
      outShape.push(shape[axis] ?? 0);
      viewStrides.push(stride);
      continue;
    }
    if (spec.kind === "index") {
      offset += spec.i * stride;
    } else {
      offset += spec.start * stride;
      outShape.push(spec.dim);
      viewStrides.push(stride * spec.step);
    }
  }

  const size = product(outShape);
  const out = sameKindArray(data, size);
  const outStrides = computeStrides(outShape);
  for (let flat = 0; flat < size; flat++) {
    const idx = unravel(flat, outShape, outStrides);
    let srcOffset = offset;
    for (let i = 0; i < viewStrides.length; i++) {
      srcOffset += (idx[i] ?? 0) * (viewStrides[i] ?? 0);
    }
    out[flat] = data[srcOffset] ?? 0;
  }
  return { shape: outShape, data: out };
}

/**
 * K4 (O2(a)): word-identical (M3 message parity) runtime half of the lock
 * for every computing op dt1 does not yet implement for `D != "float64"`
 * (add/sub/mul/div/matmul/dot/cosineSimilarity/sum/mean/argmax/topk/stack/
 * sqrt/norm — dt2-dt5 unlock these progressively, docs/dtype-dt1-spec.md
 * K4). Every locked op calls this FIRST, defense-in-depth for the
 * argument-less (niladic) forms the compile-time `DTypeLock` (ndarray.ts)
 * cannot reach at all (no argument position to hang a conditional type on —
 * the exact reason a `this`-parameter alternative was tempting and O2
 * rejected it for opaque TS2684 diagnostics) — so a bypass (e.g. via `any`)
 * or a niladic call on a non-float64 receiver still throws instead of
 * silently computing a wrong answer (M2's runtime backstop holds even where
 * the compile-time claim can't reach).
 */
export function lockedOpMessage(op: string, dtype: DType): string {
  return `${op}: dtype '${dtype}' is not implemented for non-float64 arrays yet (use astype("float64") first)`;
}

export function assertFloat64Locked(op: string, dtype: DType): void {
  if (dtype !== "float64") {
    throw new Error(lockedOpMessage(op, dtype));
  }
}

// ---------------------------------------------------------------------------
// dt2 (docs/dtype-dt2-spec.md), Commit A — P1 (Promote) + P2 (array⊕array for
// add/sub/mul/div). Prototype precedent: proto/dtype 90e7aae (stage 3,
// Promote + add's array overload). The bool-arithmetic message below is the
// SAME string the compile-time `Promote<A,B>`/`PromoteDiv<A,B>` (ndarray.ts)
// embed in their `ShapeError` branch via `typeof` (M3, single source, never
// re-typed by hand) — a `const` string declaration's inferred TYPE is
// already the literal (verified empirically against real tsc 7.0.2, no
// `as const` needed for a bare top-level string constant).
// ---------------------------------------------------------------------------

/** D4/D7 (dt2 P1/P4): bool has no arithmetic at all — every `Promote`/
 * `PromoteDiv` rejection involving bool, and the permanent scalar rejection
 * (P4, replacing dt1's transitional `lockedOpMessage` for this case), uses
 * this EXACT stem, compile-time and runtime alike (M3). */
export const BOOL_ARITHMETIC_MESSAGE = "dtype 'bool' has no arithmetic — use astype() to convert first";

/** D4: the three dtypes `Promote`/`PromoteDiv`'s tables cover directly —
 * `bool` is handled separately (it always rejects, D4/D7), never a
 * `PROMOTE_NUMERIC` key. */
export type NumericDType = Exclude<DType, "bool">;

/** D4 (dt2 P1): the single source of truth for NUMERIC dtype promotion
 * (`add`/`sub`/`mul`'s array⊕array rule — `bool` excluded, handled
 * separately by `promoteDType` below). The compile-time `Promote<A,B>`
 * (ndarray.ts) reads this SAME object via `typeof` for its non-bool branch,
 * so type and runtime tables cannot drift (spec P1: "Typebene und
 * identische Laufzeit-Tabelle aus EINER Quelle"). */
export const PROMOTE_NUMERIC = {
  float64: { float64: "float64", float32: "float64", int32: "float64" },
  float32: { float64: "float64", float32: "float32", int32: "float64" },
  int32: { float64: "float64", float32: "float64", int32: "int32" },
} as const;

/**
 * D4 (dt2 P1): runtime dtype promotion for `add`/`sub`/`mul`. Throws
 * `BOOL_ARITHMETIC_MESSAGE` (word-for-word matching the compile-time
 * `Promote<A,B>` rejection, M3) when EITHER operand is `bool` — bool has no
 * arithmetic, only `astype` converts it (D7). Otherwise looks up
 * `PROMOTE_NUMERIC` directly — the single source the compile-time type
 * reads too.
 */
export function promoteDType(a: DType, b: DType): NumericDType {
  if (a === "bool" || b === "bool") {
    throw new Error(BOOL_ARITHMETIC_MESSAGE);
  }
  return PROMOTE_NUMERIC[a as NumericDType][b as NumericDType];
}

/**
 * D5 (dt2 P2): runtime dtype promotion for `div` — ALWAYS floating-point,
 * never int32 (unlike `promoteDType` above, which lets int32⊕int32 stay
 * int32): float32⊕float32 → float32, every other combination (including
 * int32⊕int32) → float64. Throws `BOOL_ARITHMETIC_MESSAGE` when either
 * operand is bool, same as `promoteDType`.
 */
export function promoteDTypeDiv(a: DType, b: DType): "float32" | "float64" {
  if (a === "bool" || b === "bool") {
    throw new Error(BOOL_ARITHMETIC_MESSAGE);
  }
  return a === "float32" && b === "float32" ? "float32" : "float64";
}

/**
 * D4/D5/D8 (dt2 P2): dtype-aware, broadcasting elementwise `add`/`sub`/`mul`
 * — the SAME broadcast/index algorithm `elementwiseBinary` above already
 * implements (duplicated here rather than generalizing that frozen
 * function's own signature, frozen-baseline discipline: `elementwiseBinary`
 * stays byte-unchanged, still the exact float64 oracle the existing
 * differential suite pins, and now doubles as dt2's own float64⊕float64
 * test oracle, v1.1 Baustein 0), generalized over `DataOfRuntime` and the
 * promoted result dtype (`promoteDType`).
 *
 * Two's-complement wrap for an int32 result: `(a+b)|0`/`(a-b)|0` for
 * add/sub, but `Math.imul(a,b)` EXCLUSIVELY for mul (dt2 spec B1/v1.1):
 * `(a*b)|0` first computes the product in float64, which silently loses
 * precision once it exceeds 2^53 — e.g. `2147483647 * 2147483647 | 0`
 * evaluates to `0`, not the correct wrapped `1` — while `Math.imul`
 * computes the wrapped 32-bit product directly, never through a float64
 * intermediate. A float32 result (only reachable when BOTH operands are
 * float32, the sole `PROMOTE_NUMERIC` cell yielding it) is computed in f64
 * and `Math.fround`ed per element (D8's correctly-rounded contract for
 * `+ − ×`, doubly-rounded-is-harmless per Figueroa 1995). Throws
 * `BOOL_ARITHMETIC_MESSAGE` (via `promoteDType`) if either operand is bool.
 */
export function elementwiseBinaryTyped(
  op: "add" | "sub" | "mul",
  aShape: readonly number[],
  aData: DataOfRuntime,
  aDtype: DType,
  bShape: readonly number[],
  bData: DataOfRuntime,
  bDtype: DType,
): { shape: number[]; data: DataOfRuntime; resultDtype: NumericDType } {
  const resultDtype = promoteDType(aDtype, bDtype);
  const outShape = runtimeBroadcastShape(aShape, bShape);
  const rank = outShape.length;

  const aAligned = alignToRank(aShape, computeStrides(aShape), rank);
  const bAligned = alignToRank(bShape, computeStrides(bShape), rank);
  const aEff = effectiveStrides(aAligned.shape, aAligned.strides);
  const bEff = effectiveStrides(bAligned.shape, bAligned.strides);

  const outStrides = computeStrides(outShape);
  const size = product(outShape);
  const out: DataOfRuntime = resultDtype === "int32" ? new Int32Array(size) : resultDtype === "float32" ? new Float32Array(size) : new Float64Array(size);

  for (let flat = 0; flat < size; flat++) {
    const idx = unravel(flat, outShape, outStrides);
    let aOff = 0;
    let bOff = 0;
    for (let i = 0; i < rank; i++) {
      const ix = idx[i] ?? 0;
      aOff += ix * (aEff[i] ?? 0);
      bOff += ix * (bEff[i] ?? 0);
    }
    const x = aData[aOff] ?? 0;
    const y = bData[bOff] ?? 0;
    if (resultDtype === "int32") {
      (out as Int32Array)[flat] = op === "mul" ? Math.imul(x, y) : op === "add" ? (x + y) | 0 : (x - y) | 0;
    } else if (resultDtype === "float32") {
      const raw = op === "add" ? x + y : op === "sub" ? x - y : x * y;
      (out as Float32Array)[flat] = Math.fround(raw);
    } else {
      (out as Float64Array)[flat] = op === "add" ? x + y : op === "sub" ? x - y : x * y;
    }
  }
  return { shape: outShape, data: out, resultDtype };
}

/**
 * D5/D8 (dt2 P2): dtype-aware, broadcasting elementwise `div` — same
 * broadcast/index algorithm as `elementwiseBinaryTyped` above, but its own
 * function (not a third `op` branch there): `div`'s result dtype follows
 * `promoteDTypeDiv`, a DIFFERENT table (always floating-point, int32⊕int32
 * → float64 rather than staying int32), so there is no int32 output branch
 * to wrap here at all — only a possible float32 `Math.fround` per element
 * (D8), reachable exactly when both operands are float32. Pure IEEE 754
 * division throughout (`x/0 → ±Infinity`, `0/0 → NaN`), same as the
 * existing float64-only `elementwiseBinary` path. Throws
 * `BOOL_ARITHMETIC_MESSAGE` (via `promoteDTypeDiv`) if either operand is
 * bool.
 */
export function elementwiseDivTyped(
  aShape: readonly number[],
  aData: DataOfRuntime,
  aDtype: DType,
  bShape: readonly number[],
  bData: DataOfRuntime,
  bDtype: DType,
): { shape: number[]; data: DataOfRuntime; resultDtype: "float32" | "float64" } {
  const resultDtype = promoteDTypeDiv(aDtype, bDtype);
  const outShape = runtimeBroadcastShape(aShape, bShape);
  const rank = outShape.length;

  const aAligned = alignToRank(aShape, computeStrides(aShape), rank);
  const bAligned = alignToRank(bShape, computeStrides(bShape), rank);
  const aEff = effectiveStrides(aAligned.shape, aAligned.strides);
  const bEff = effectiveStrides(bAligned.shape, bAligned.strides);

  const outStrides = computeStrides(outShape);
  const size = product(outShape);
  const out: DataOfRuntime = resultDtype === "float32" ? new Float32Array(size) : new Float64Array(size);

  for (let flat = 0; flat < size; flat++) {
    const idx = unravel(flat, outShape, outStrides);
    let aOff = 0;
    let bOff = 0;
    for (let i = 0; i < rank; i++) {
      const ix = idx[i] ?? 0;
      aOff += ix * (aEff[i] ?? 0);
      bOff += ix * (bEff[i] ?? 0);
    }
    const raw = (aData[aOff] ?? 0) / (bData[bOff] ?? 0);
    if (resultDtype === "float32") (out as Float32Array)[flat] = Math.fround(raw);
    else (out as Float64Array)[flat] = raw;
  }
  return { shape: outShape, data: out, resultDtype };
}
