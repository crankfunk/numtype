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
 * Selection order: every index is sorted by `topkCompareValues` (NaN first,
 * then descending value), ties broken by ascending index — the first `k`
 * entries of that order become the result. `values[i] = data[indices[i]]`
 * is a plain `Float64Array`-element read/copy (never routed through any
 * arithmetic), so a NaN's exact bit payload survives untouched.
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

  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((i, j) => topkCompareValues(data[i] ?? 0, data[j] ?? 0) || i - j);

  const values = new Float64Array(k);
  const indices = new Float64Array(k);
  for (let i = 0; i < k; i++) {
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
