/**
 * Zero-copy residency layer (Kern 02) + strided views (Kern 03) — the
 * resident twin of `NDArray<S>`. See docs/kern-02-residency-spec.md and
 * docs/kern-03-strided-spec.md for the full contracts; this header covers
 * the invariants that matter for correctness/safety, not the why.
 *
 * A `WNDArray<S>` does NOT hold its data in a JS-side `Float64Array`: the
 * data lives in the WASM core's own linear memory, tracked by a shared
 * `ResidentBuffer` record (`ptr`/`bytes`/`refs`). Since Kern 03 a handle is
 * a **view** onto that buffer: `(shape, strides, offset)` metadata over a
 * possibly shared allocation. `transpose()` is an O(1) metadata operation —
 * reversed shape + reversed strides, same buffer, no kernel call. Because
 * `WNDArray` exposes no mutation, views are semantically indistinguishable
 * from copies; sharing is observable only through memory/lifecycle.
 *
 * Ops call the Kern-03 *strided* kernels pointer-to-pointer (contiguous
 * handles simply pass their natural strides and offset 0 — one code path
 * for everything; the non-strided exports remain in use by the v1 backend
 * only). Operand DATA is never copied for an op — only the tiny per-call
 * shape/stride metadata (u32 arrays) is marshalled fresh each call. Copies
 * happen at the explicit boundaries: `fromArray` (in), `toArray`/
 * `toNestedArray` (out), and the explicit `contiguous()`.
 *
 * **Memory rule (hard, same as v1):** never store a typed-array view across
 * a call boundary — `memory.grow` detaches every existing view. Every read
 * derives a fresh view from `core.memory.buffer` immediately before use.
 * What IS safely stored long-lived is the raw numeric `ptr`/`len` — plain
 * numbers, unaffected by `memory.grow`.
 *
 * **Output aliasing:** every op allocates a *fresh* WASM buffer for its
 * result — it never writes into an operand's buffer. Kernels assume
 * non-overlapping `out` vs. operands; this invariant keeps that assumption
 * sound. (Views deliberately alias each other's *input* buffer — that is
 * their point — but an op's output never aliases anything.)
 *
 * **Lifecycle (refcounted since Kern 03):**
 *  - The shared `ResidentBuffer` starts at `refs = 1`; each view (`transpose`)
 *    increments. `dispose()` marks *this handle* disposed, unregisters it
 *    from the `FinalizationRegistry`, and decrements — the WASM allocation
 *    is freed exactly when `refs` hits 0. Disposing a base while views live
 *    keeps the buffer alive and the views fully usable (and vice versa).
 *  - A second `dispose()` on the same handle is a safe no-op (checked via
 *    the per-handle `disposed_` flag, never via `ptr` — a legitimately
 *    empty array has `ptr === 0` from the very start).
 *  - The `FinalizationRegistry` is the GC backstop: a handle dropped
 *    without `dispose()` still has its reference released once collected.
 *    The held value is the shared `ResidentBuffer` record — plain data plus
 *    the long-lived core handle, NEVER a reference to the `WNDArray` itself
 *    (that would keep it permanently reachable and defeat collection).
 *    Any interleaving of dispose/GC across the handles of one buffer
 *    releases each reference exactly once, so no leak and no double-free.
 *  - Every op/read on a disposed handle throws immediately, naming the
 *    operation, before touching any WASM memory.
 *  - **Error paths leak nothing:** per-call scratch (shape/stride arrays)
 *    is tracked in a list freed in `finally` — this also covers an
 *    out-of-memory throw *between* allocations, closing the Kern-02-era
 *    gap where scratch allocated before a failing output `nt_alloc` leaked
 *    (the v1 backend still has that gap; see FOLLOWUPS.md). A non-zero
 *    kernel status frees the already-allocated output buffer before
 *    throwing; input handles are never touched by a failing op.
 */

import type { Broadcast } from "../broadcast.ts";
import { type Dim, type Mutable, type Shape } from "../dim.ts";
import type { Guard, OkShape } from "../ndarray.ts";
import type { MatMul } from "../matmul.ts";
import type { ReduceAxis, Transpose } from "../reduce.ts";
import { computeStrides, product, runtimeBroadcastShape } from "../runtime.ts";
import type { CoreExports } from "./loader.ts";

interface ScratchBuf {
  readonly ptr: number;
  readonly bytes: number;
}

function allocBytes(core: CoreExports, bytes: number): ScratchBuf {
  const ptr = core.nt_alloc(bytes);
  if (ptr === 0 && bytes !== 0) {
    throw new Error(`resident: nt_alloc(${bytes}) failed (out of memory)`);
  }
  return { ptr, bytes };
}

function freeBuf(core: CoreExports, buf: ScratchBuf): void {
  core.nt_free(buf.ptr, buf.bytes);
}

/** Allocate + write a little-endian u32 array (shape or strides) — the same
 * tiny, per-call, copy-based marshalling v1 uses (spec: "not worth
 * residency"). */
function writeU32Array(core: CoreExports, values: readonly number[]): ScratchBuf {
  const buf = allocBytes(core, values.length * 4);
  const view = new Uint32Array(core.memory.buffer, buf.ptr, values.length);
  view.set(values);
  return buf;
}

// --- shared buffer record + GC backstop + test-observability ---------------

/** One WASM allocation, shared by a base handle and any views onto it.
 * Plain data plus the core handle — deliberately free of any reference to
 * the `WNDArray` handles themselves (see module doc comment). */
interface ResidentBuffer {
  readonly core: CoreExports;
  readonly ptr: number;
  readonly bytes: number;
  /** Full allocation length in ELEMENTS — the `data_len` every strided
   * kernel call passes for bounds validation. */
  readonly lenElems: number;
  refs: number;
}

let residentFreeCount = 0;

/** Test-only observability hook: how many times a *resident data buffer*
 * (as opposed to per-op ephemeral scratch) has actually been freed —
 * i.e. how often a buffer's refcount reached 0, whether via `dispose()` or
 * the `FinalizationRegistry` backstop. NOT a dispose counter: releasing a
 * view of a still-referenced buffer does not increment it. Exists so the
 * leak-plateau/GC/refcount tests have a deterministic signal instead of
 * only inferring frees from `core.memory.buffer.byteLength` (which can
 * never shrink, only plateau). */
export function getResidentFreeCount(): number {
  return residentFreeCount;
}

function retainBuffer(buf: ResidentBuffer): void {
  buf.refs++;
}

function releaseBuffer(buf: ResidentBuffer): void {
  buf.refs--;
  if (buf.refs === 0) {
    buf.core.nt_free(buf.ptr, buf.bytes);
    residentFreeCount++;
  }
}

/** Held value is the shared `ResidentBuffer` record — never a reference to
 * the `WNDArray` itself (see module doc comment for why that matters). */
const registry = new FinalizationRegistry<ResidentBuffer>((buf) => {
  releaseBuffer(buf);
});

/**
 * Erased top type for heterogeneous containers, non-generic helpers, and
 * loop variables reassigned across ops with different (dynamic-rank)
 * shapes — same rationale and same fix as `ndarray.ts`'s `AnyNDArray`: the
 * argument-side error guards make this class's measured variance
 * invariant, so e.g. a `let cur: WNDArray<number[]>` cannot be reassigned
 * the `WNDArray<readonly number[]>` a chained `.add()`/`.matmul()` call
 * returns under dynamic-rank shapes. `any` as the type argument bypasses
 * the variance comparison entirely, same idiom as `NDArray<any>`.
 */
export type AnyWNDArray = WNDArray<any>;

/**
 * Resident twin of `NDArray<S>`. See module doc comment for the full
 * lifecycle contract. Surface mirrors `NDArray`: `zeros`/`ones`/`fromArray`,
 * `add`/`matmul`/`sum`/`transpose`, `toArray`/`toNestedArray`, plus the
 * residency-specific `dispose()`/`disposed` and the Kern-03 `contiguous()`.
 */
export class WNDArray<S extends Shape> {
  readonly shape: S;
  /** Element strides of this handle's view onto its buffer — row-major
   * (`computeStrides(shape)`) for freshly created arrays, permuted for
   * transpose views. Runtime observability only; strides have no type-level
   * representation. */
  readonly strides: readonly number[];
  /** Base element offset into the buffer. Always 0 in Kern 03 (only
   * transpose views exist); the ABI supports nonzero offsets so slicing
   * later needs no ABI revision. */
  private readonly offset: number;
  private readonly buf: ResidentBuffer;
  private disposed_: boolean;

  /** Reference accounting happens at the call sites: a fresh buffer record
   * is created with `refs = 1` (this constructor takes that reference); a
   * view retains BEFORE constructing. The constructor itself never touches
   * `refs`. */
  private constructor(buf: ResidentBuffer, shape: S, strides: readonly number[], offset: number) {
    this.buf = buf;
    this.shape = shape;
    this.strides = strides;
    this.offset = offset;
    this.disposed_ = false;
    registry.register(this, buf, this);
  }

  private get core(): CoreExports {
    return this.buf.core;
  }

  /** Whether `dispose()` has already run on this handle (once true, every
   * op/read throws instead of touching WASM memory). Views onto the same
   * buffer each carry their own flag. */
  get disposed(): boolean {
    return this.disposed_;
  }

  private assertLive(op: string): void {
    if (this.disposed_) {
      throw new Error(`WNDArray.${op}: array has been disposed`);
    }
  }

  // Generic (rather than `WNDArray<Shape>`) so a concrete `WNDArray<B>`
  // argument always type-checks here regardless of variance: per
  // docs/spike-01-ergebnisse.md's verified finding, the argument-side error
  // guards make this class's measured variance invariant, so
  // `WNDArray<[2, 3]>` is NOT assignable to a fixed `WNDArray<Shape>`
  // parameter type — the same reason `ndarray.ts` needs `AnyWNDArray =
  // WNDArray<any>` instead of `WNDArray<Shape>` for its own erased handle.
  private assertSameCore<B extends Shape>(other: WNDArray<B>, op: string): void {
    if (this.buf.core !== other.buf.core) {
      throw new Error(`WNDArray.${op}: operands belong to different WASM core instances`);
    }
  }

  /** Whether this handle's metadata is plain row-major over the whole
   * buffer (natural strides, offset 0) — the `toArray` fast-path test. A
   * double-transposed view qualifies again by construction. */
  private isContiguous(): boolean {
    if (this.offset !== 0) return false;
    const natural = computeStrides(this.shape);
    return natural.length === this.strides.length && natural.every((s, i) => s === this.strides[i]);
  }

  /** Release this handle's reference on the shared buffer, mark the handle
   * disposed, and unregister it from the GC backstop (`this` doubles as the
   * unregister token). The WASM allocation is freed exactly when the last
   * handle (base or view) releases. A second call is a safe no-op: guarded
   * by the per-handle `disposed_` flag, never by inspecting `ptr`. */
  dispose(): void {
    if (this.disposed_) return;
    this.disposed_ = true;
    registry.unregister(this);
    releaseBuffer(this.buf);
  }

  /** Wrap a freshly allocated WASM buffer (ownership transfers: the new
   * handle holds the record's initial `refs = 1`) with natural row-major
   * metadata. */
  private static fresh<S extends Shape>(core: CoreExports, shape: S, ptr: number, lenElems: number): WNDArray<S> {
    const buf: ResidentBuffer = { core, ptr, bytes: lenElems * 8, lenElems, refs: 1 };
    return new WNDArray<S>(buf, shape, computeStrides(shape), 0);
  }

  /** An all-zeros resident array of the given shape. WASM's global
   * allocator does not zero memory on allocation (`std::alloc::alloc`, not
   * `calloc`) — the zero-fill is an explicit `nt_fill(ptr, len, 0)` kernel
   * call, entirely on the WASM side (no data ever crosses the boundary). */
  static zeros<const S extends Shape>(core: CoreExports, shape: S): WNDArray<Mutable<S>> {
    const len = product(shape);
    const buf = allocBytes(core, len * 8);
    const status = core.nt_fill(buf.ptr, len, 0);
    if (status !== 0) {
      freeBuf(core, buf);
      throw new Error(`WNDArray.zeros: nt_fill unexpected status ${status}`);
    }
    return WNDArray.fresh<Mutable<S>>(core, [...shape] as Mutable<S>, buf.ptr, len);
  }

  /** An all-ones resident array of the given shape (see `zeros` for why
   * this goes through `nt_fill` rather than a zero-then-copy). */
  static ones<const S extends Shape>(core: CoreExports, shape: S): WNDArray<Mutable<S>> {
    const len = product(shape);
    const buf = allocBytes(core, len * 8);
    const status = core.nt_fill(buf.ptr, len, 1);
    if (status !== 0) {
      freeBuf(core, buf);
      throw new Error(`WNDArray.ones: nt_fill unexpected status ${status}`);
    }
    return WNDArray.fresh<Mutable<S>>(core, [...shape] as Mutable<S>, buf.ptr, len);
  }

  /** Build a resident array from flat row-major values — the explicit
   * copy-IN boundary (spec). Accepts a plain list or a `Float64Array`
   * (mirrors `NDArray.fromArray`); either way the copy into WASM memory is
   * a single `view.set()` — for a `Float64Array` that is memcpy-fast,
   * removing the ~100x `Array.from` conversion tax the Kern-02 chain bench
   * measured. Throws at runtime if `values.length` doesn't match the
   * shape's element count (same check as `NDArray.fromArray`). */
  static fromArray<const S extends Shape>(
    core: CoreExports,
    shape: S,
    values: readonly number[] | Float64Array,
  ): WNDArray<Mutable<S>> {
    const len = product(shape);
    if (values.length !== len) {
      throw new Error(`fromArray: expected ${len} values for shape [${shape.join(",")}], got ${values.length}`);
    }
    const buf = allocBytes(core, len * 8);
    const view = new Float64Array(core.memory.buffer, buf.ptr, len);
    view.set(values);
    return WNDArray.fresh<Mutable<S>>(core, [...shape] as Mutable<S>, buf.ptr, len);
  }

  /** Broadcasting elementwise add — resident twin of `NDArray.add`/
   * `wasmAdd`, routed through the strided kernel (contiguous handles pass
   * natural strides): operand data stays resident; only the four shape/
   * stride arrays are marshalled per call, and a fresh output buffer is
   * allocated for the result (never aliasing either operand). */
  add<B extends Shape>(other: Guard<Broadcast<S, B>, WNDArray<B>>): WNDArray<OkShape<Broadcast<S, B>>> {
    this.assertLive("add");
    const o = other as unknown as WNDArray<B>;
    o.assertLive("add");
    this.assertSameCore(o, "add");

    // Shape computation (and any incompatibility throw) happens BEFORE any
    // allocation — a rejected call never leaks scratch because none was
    // ever allocated for it.
    const outShape = runtimeBroadcastShape(this.shape, o.shape);
    const outLen = product(outShape);

    const scratch: ScratchBuf[] = [];
    try {
      const aShapeBuf = writeU32Array(this.core, this.shape);
      scratch.push(aShapeBuf);
      const aStridesBuf = writeU32Array(this.core, this.strides);
      scratch.push(aStridesBuf);
      const bShapeBuf = writeU32Array(this.core, o.shape);
      scratch.push(bShapeBuf);
      const bStridesBuf = writeU32Array(this.core, o.strides);
      scratch.push(bStridesBuf);
      const outDataBuf = allocBytes(this.core, outLen * 8);

      const status = this.core.nt_add_strided(
        aShapeBuf.ptr,
        this.shape.length,
        aStridesBuf.ptr,
        this.offset,
        this.buf.ptr,
        this.buf.lenElems,
        bShapeBuf.ptr,
        o.shape.length,
        bStridesBuf.ptr,
        o.offset,
        o.buf.ptr,
        o.buf.lenElems,
        outDataBuf.ptr,
        outLen,
      );
      if (status !== 0) {
        freeBuf(this.core, outDataBuf); // fresh output buffer never escapes on failure
        throw new Error(
          `wasm resident nt_add_strided: status ${status} for shapes [${this.shape.join(",")}] and [${o.shape.join(",")}]`,
        );
      }
      return WNDArray.fresh<OkShape<Broadcast<S, B>>>(
        this.core,
        outShape as OkShape<Broadcast<S, B>>,
        outDataBuf.ptr,
        outLen,
      );
    } finally {
      // Ephemeral per-call scratch: always freed — success, kernel failure,
      // or an OOM throw between the allocations above.
      for (const buf of scratch) freeBuf(this.core, buf);
    }
  }

  /** Full NumPy `matmul` — resident twin of `NDArray.matmul`/`wasmMatmul`,
   * routed through the strided kernel. 1-D promotion/squeeze is TS-side
   * shape bookkeeping only (mirrors `matmulRuntime`/`wasmMatmul` exactly);
   * it never touches operand data — the same resident buffer is passed
   * straight to the kernel under promoted shape metadata. The axis added
   * by promotion has dim 1 and carries stride 0 (never multiplied by a
   * nonzero index, so any value would do; 0 keeps the bounds check tight). */
  matmul<B extends Shape>(other: Guard<MatMul<S, B>, WNDArray<B>>): WNDArray<OkShape<MatMul<S, B>>> {
    this.assertLive("matmul");
    const o = other as unknown as WNDArray<B>;
    o.assertLive("matmul");
    this.assertSameCore(o, "matmul");

    const aShapeIn = this.shape;
    const bShapeIn = o.shape;
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
    const aStrides = aPromoted ? [0, this.strides[0] ?? 0] : [...this.strides];
    const bStrides = bPromoted ? [o.strides[0] ?? 0, 0] : [...o.strides];

    const m = aShape[aShape.length - 2] ?? 1;
    const k1 = aShape[aShape.length - 1] ?? 1;
    const k2 = bShape[bShape.length - 2] ?? 1;
    const n = bShape[bShape.length - 1] ?? 1;
    if (k1 !== k2) {
      throw new Error(`matmul: inner dimensions ${k1} and ${k2} do not match`);
    }

    const batchA = aShape.slice(0, -2);
    const batchB = bShape.slice(0, -2);
    const batchOut = runtimeBroadcastShape(batchA, batchB); // throws before any allocation
    const batchRank = batchOut.length;
    const outFullShape = [...batchOut, m, n];
    const outLen = product(outFullShape);

    const scratch: ScratchBuf[] = [];
    try {
      const aShapeBuf = writeU32Array(this.core, aShape);
      scratch.push(aShapeBuf);
      const aStridesBuf = writeU32Array(this.core, aStrides);
      scratch.push(aStridesBuf);
      const bShapeBuf = writeU32Array(this.core, bShape);
      scratch.push(bShapeBuf);
      const bStridesBuf = writeU32Array(this.core, bStrides);
      scratch.push(bStridesBuf);
      const outDataBuf = allocBytes(this.core, outLen * 8);

      const status = this.core.nt_matmul_blocked(
        aShapeBuf.ptr,
        aShape.length,
        aStridesBuf.ptr,
        this.offset,
        this.buf.ptr,
        this.buf.lenElems,
        bShapeBuf.ptr,
        bShape.length,
        bStridesBuf.ptr,
        o.offset,
        o.buf.ptr,
        o.buf.lenElems,
        outDataBuf.ptr,
        outLen,
      );
      if (status !== 0) {
        freeBuf(this.core, outDataBuf);
        throw new Error(
          `wasm resident nt_matmul_blocked: status ${status} for shapes [${aShapeIn.join(",")}] and [${bShapeIn.join(",")}]`,
        );
      }

      // Squeezing a size-1 axis never changes the flat (row-major) data
      // layout, only the shape metadata — mirrors matmulRuntime exactly.
      let finalShape = outFullShape;
      if (aPromoted) {
        finalShape = [...finalShape.slice(0, batchRank), ...finalShape.slice(batchRank + 1)];
      }
      if (bPromoted) {
        finalShape = finalShape.slice(0, -1);
      }
      return WNDArray.fresh<OkShape<MatMul<S, B>>>(
        this.core,
        finalShape as OkShape<MatMul<S, B>>,
        outDataBuf.ptr,
        outLen,
      );
    } finally {
      for (const buf of scratch) freeBuf(this.core, buf);
    }
  }

  /** Sum-reduce along `axis` (negative counts from the end); omit `axis` to
   * sum every element down to a rank-0 array. Resident twin of
   * `NDArray.sum`/`wasmSum`, routed through the strided kernels. The full
   * sum accumulates in this view's LOGICAL row-major order (kernel
   * contract) — bit-identical to summing the materialized equivalent. */
  sum<const Axis extends number | undefined = undefined>(
    axis?: Guard<ReduceAxis<S, Axis>, Axis>,
  ): WNDArray<OkShape<ReduceAxis<S, Axis>>> {
    this.assertLive("sum");
    const axisNum = axis as unknown as Axis | undefined;

    if (axisNum === undefined) {
      const scratch: ScratchBuf[] = [];
      try {
        const shapeBuf = writeU32Array(this.core, this.shape);
        scratch.push(shapeBuf);
        const stridesBuf = writeU32Array(this.core, this.strides);
        scratch.push(stridesBuf);
        const outDataBuf = allocBytes(this.core, 8);

        const status = this.core.nt_sum_all_strided(
          shapeBuf.ptr,
          this.shape.length,
          stridesBuf.ptr,
          this.offset,
          this.buf.ptr,
          this.buf.lenElems,
          outDataBuf.ptr,
        );
        if (status !== 0) {
          freeBuf(this.core, outDataBuf);
          throw new Error(`wasm resident nt_sum_all_strided: status ${status} for shape [${this.shape.join(",")}]`);
        }
        return WNDArray.fresh<OkShape<ReduceAxis<S, Axis>>>(
          this.core,
          [] as OkShape<ReduceAxis<S, Axis>>,
          outDataBuf.ptr,
          1,
        );
      } finally {
        for (const buf of scratch) freeBuf(this.core, buf);
      }
    }

    const rank = this.shape.length;
    const normAxis = axisNum < 0 ? rank + axisNum : axisNum;
    if (normAxis < 0 || normAxis >= rank) {
      throw new Error(`reduce: axis ${axisNum} is out of range for shape [${this.shape.join(",")}] (rank ${rank})`);
    }

    const outShape = [...this.shape.slice(0, normAxis), ...this.shape.slice(normAxis + 1)];
    const outLen = product(outShape);

    const scratch: ScratchBuf[] = [];
    try {
      const shapeBuf = writeU32Array(this.core, this.shape);
      scratch.push(shapeBuf);
      const stridesBuf = writeU32Array(this.core, this.strides);
      scratch.push(stridesBuf);
      const outDataBuf = allocBytes(this.core, outLen * 8);

      const status = this.core.nt_sum_axis_strided(
        shapeBuf.ptr,
        rank,
        stridesBuf.ptr,
        this.offset,
        this.buf.ptr,
        this.buf.lenElems,
        axisNum,
        outDataBuf.ptr,
        outLen,
      );
      if (status !== 0) {
        freeBuf(this.core, outDataBuf);
        throw new Error(
          `wasm resident nt_sum_axis_strided: status ${status} for shape [${this.shape.join(",")}] axis ${axisNum}`,
        );
      }
      return WNDArray.fresh<OkShape<ReduceAxis<S, Axis>>>(
        this.core,
        outShape as OkShape<ReduceAxis<S, Axis>>,
        outDataBuf.ptr,
        outLen,
      );
    } finally {
      for (const buf of scratch) freeBuf(this.core, buf);
    }
  }

  /** Reverse every axis (NumPy's `.T` generalized to N-D). Since Kern 03
   * this is an **O(1) view**: reversed shape + reversed strides over the
   * SAME buffer — no kernel call, no allocation, no data movement. The
   * buffer is freed only when the last handle onto it is released; a
   * double transpose yields a contiguous-strided view again. */
  transpose(): WNDArray<Transpose<S>> {
    this.assertLive("transpose");
    retainBuffer(this.buf);
    return new WNDArray<Transpose<S>>(
      this.buf,
      [...this.shape].reverse() as Transpose<S>,
      [...this.strides].reverse(),
      this.offset,
    );
  }

  /** Materialize this view into a fresh, independently owned, contiguous
   * row-major array of the same shape (Kern 03). Deliberately ALWAYS
   * copies, even when the receiver is already contiguous — predictable
   * ownership (the result is never a view) beats a micro-optimization.
   * This is the explicit escape hatch when strided reads would be paid
   * repeatedly (e.g. a transposed operand feeding many ops). */
  contiguous(): WNDArray<S> {
    this.assertLive("contiguous");
    const len = product(this.shape);

    const scratch: ScratchBuf[] = [];
    try {
      const shapeBuf = writeU32Array(this.core, this.shape);
      scratch.push(shapeBuf);
      const stridesBuf = writeU32Array(this.core, this.strides);
      scratch.push(stridesBuf);
      const outDataBuf = allocBytes(this.core, len * 8);

      const status = this.core.nt_materialize(
        shapeBuf.ptr,
        this.shape.length,
        stridesBuf.ptr,
        this.offset,
        this.buf.ptr,
        this.buf.lenElems,
        outDataBuf.ptr,
        len,
      );
      if (status !== 0) {
        freeBuf(this.core, outDataBuf);
        throw new Error(`wasm resident nt_materialize: status ${status} for shape [${this.shape.join(",")}]`);
      }
      // `[...t] as unknown as S`: the spread of a readonly tuple S types as
      // S[number][], which TS won't narrow back to S directly — the values
      // ARE the same dims, so the round trip through unknown is sound here.
      return WNDArray.fresh<S>(this.core, [...this.shape] as unknown as S, outDataBuf.ptr, len);
    } finally {
      for (const buf of scratch) freeBuf(this.core, buf);
    }
  }

  /** Read back as an independent `Float64Array` copy in LOGICAL row-major
   * order — the explicit copy-OUT boundary (spec). Never a live view: the
   * returned array is detached from WASM memory by construction, so it
   * stays valid even after `dispose()` or a later `memory.grow`. A
   * contiguous handle copies directly; a strided view gathers through
   * `nt_materialize` into ephemeral scratch first. */
  toArray(): Float64Array {
    this.assertLive("toArray");
    const len = product(this.shape);

    if (this.isContiguous()) {
      const view = new Float64Array(this.core.memory.buffer, this.buf.ptr, len);
      return Float64Array.from(view);
    }

    const scratch: ScratchBuf[] = [];
    try {
      const shapeBuf = writeU32Array(this.core, this.shape);
      scratch.push(shapeBuf);
      const stridesBuf = writeU32Array(this.core, this.strides);
      scratch.push(stridesBuf);
      const outDataBuf = allocBytes(this.core, len * 8);
      scratch.push(outDataBuf); // ephemeral here — copied out below, always freed

      const status = this.core.nt_materialize(
        shapeBuf.ptr,
        this.shape.length,
        stridesBuf.ptr,
        this.offset,
        this.buf.ptr,
        this.buf.lenElems,
        outDataBuf.ptr,
        len,
      );
      if (status !== 0) {
        throw new Error(`wasm resident nt_materialize (toArray): status ${status} for shape [${this.shape.join(",")}]`);
      }
      // Fresh view derived AFTER the last allocation above (memory rule).
      const view = new Float64Array(this.core.memory.buffer, outDataBuf.ptr, len);
      return Float64Array.from(view);
    } finally {
      for (const buf of scratch) freeBuf(this.core, buf);
    }
  }

  /** Read back as a plain nested JS array (any rank), for printing/tests —
   * same shape/stride walk as `NDArray.toNestedArray` (over the logical
   * row-major copy `toArray` returns, so views need no special casing). */
  toNestedArray(): unknown {
    this.assertLive("toNestedArray");
    const data = this.toArray();
    const strides = computeStrides(this.shape);
    const build = (axis: number, offset: number): unknown => {
      if (axis === this.shape.length) return data[offset] ?? 0;
      const dim: Dim = this.shape[axis] ?? 0;
      const stride = strides[axis] ?? 0;
      const out: unknown[] = [];
      for (let i = 0; i < dim; i++) out.push(build(axis + 1, offset + i * stride));
      return out;
    };
    return build(0, 0);
  }
}
