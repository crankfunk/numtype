/**
 * Zero-copy residency layer (Kern 02) — the resident twin of `NDArray<S>`.
 * See docs/kern-02-residency-spec.md for the full contract; this header
 * covers the invariants that matter for correctness/safety, not the why.
 *
 * A `WNDArray<S>` does NOT hold its data in a JS-side `Float64Array`: the
 * data lives as a `(ptr, len)` pair into the WASM core's own linear memory
 * (`core.memory.buffer`), allocated once at construction and freed exactly
 * once (via `dispose()` or, as a backstop, GC). Ops call the *same* Rust
 * kernels as the v1 backend (`spike/src/wasm/backend.ts`) pointer-to-pointer:
 * operand DATA is never copied into or out of WASM for an op — only the
 * tiny per-call shape metadata (u32 arrays) is marshalled fresh each call,
 * exactly as the spec allows ("not worth residency"). Copies only happen at
 * the two explicit boundaries: `fromArray` (in) and `toArray`/
 * `toNestedArray` (out).
 *
 * **Memory rule (hard, same as v1):** never store a typed-array view across
 * a call boundary — `memory.grow` detaches every existing view. Every read
 * (`toArray`, shape marshalling) derives a fresh view from
 * `core.memory.buffer` immediately before use. What IS safely stored
 * long-lived is the raw numeric `ptr`/`len` — plain numbers, unaffected by
 * `memory.grow` (an offset into linear memory stays valid; only the JS
 * typed-array *view object* wrapping the old, now-replaced `ArrayBuffer*
 * detaches).
 *
 * **Output aliasing:** every op allocates a *fresh* WASM buffer for its
 * result and wraps it in a new `WNDArray` — it never writes into an input's
 * buffer and never returns a `WNDArray` that shares a buffer with an
 * operand. Kernels assume non-overlapping `out` vs. operands; this
 * invariant is what makes that assumption sound from the TS side too.
 *
 * **Lifecycle:**
 *  - `dispose()` frees the WASM allocation, marks the handle disposed, and
 *    unregisters from the `FinalizationRegistry` (using `this` as the
 *    unregister token — a `FinalizationRegistry` only ever holds *weak*
 *    references to both the registration target and the unregister token,
 *    so this does not itself keep the instance alive or reachable). A
 *    second `dispose()` is a safe no-op (checked via the `disposed_` flag,
 *    not by inspecting `ptr` — a legitimately empty array, e.g. shape
 *    `[0, 3]`, has `ptr === 0` from the very start, which is NOT "already
 *    disposed").
 *  - The module-level `FinalizationRegistry` is the GC backstop: if a
 *    `WNDArray` is dropped without an explicit `dispose()`, its allocation
 *    is still freed once the GC collects it. The registry's held value
 *    carries only `{ core, ptr, bytes }` — plain data plus a reference to
 *    the long-lived WASM core handle, NEVER a reference to the `WNDArray`
 *    instance itself (that would keep it permanently reachable and defeat
 *    collection entirely).
 *  - Every op/`toArray`/`toNestedArray` call on a disposed handle throws
 *    immediately, naming the operation, before touching any WASM memory.
 *  - **Error paths leak nothing:** a non-zero kernel status frees the
 *    already-allocated *output* buffer before throwing (its ephemeral shape
 *    scratch buffers are always freed too, success or failure); input
 *    handles are never touched by a failing op and remain fully valid.
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

/** Allocate + write a shape as a little-endian u32 array — the same tiny,
 * per-call, copy-based marshalling v1 uses (spec: "not worth residency"). */
function writeShape(core: CoreExports, shape: readonly number[]): ScratchBuf {
  const buf = allocBytes(core, shape.length * 4);
  const view = new Uint32Array(core.memory.buffer, buf.ptr, shape.length);
  view.set(shape);
  return buf;
}

// --- GC backstop + test-observability -------------------------------------

interface FreeHandle {
  readonly core: CoreExports;
  readonly ptr: number;
  readonly bytes: number;
}

let residentFreeCount = 0;

/** Test-only observability hook: how many times a *resident array's own
 * data buffer* (as opposed to per-op ephemeral shape/output scratch) has
 * actually been freed, whether via `dispose()` or the `FinalizationRegistry`
 * backstop. Not part of the product API — exists so the leak-plateau and
 * GC-backstop tests have a deterministic signal instead of only inferring
 * frees from `core.memory.buffer.byteLength` (which can never shrink, only
 * plateau, since WASM memory only grows). */
export function getResidentFreeCount(): number {
  return residentFreeCount;
}

function releaseDataBuf(core: CoreExports, ptr: number, bytes: number): void {
  core.nt_free(ptr, bytes);
  residentFreeCount++;
}

/** Held value carries `{ core, ptr, bytes }` only — never a reference to the
 * `WNDArray` itself (see module doc comment for why that matters). */
const registry = new FinalizationRegistry<FreeHandle>((held) => {
  releaseDataBuf(held.core, held.ptr, held.bytes);
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
 * residency-specific `dispose()`/`disposed`.
 */
export class WNDArray<S extends Shape> {
  readonly shape: S;
  private readonly core: CoreExports;
  private ptr: number;
  private readonly bytes: number;
  private readonly len: number;
  private disposed_: boolean;

  private constructor(core: CoreExports, shape: S, ptr: number, len: number) {
    this.core = core;
    this.shape = shape;
    this.ptr = ptr;
    this.len = len;
    this.bytes = len * 8;
    this.disposed_ = false;
    registry.register(this, { core, ptr, bytes: this.bytes }, this);
  }

  /** Whether `dispose()` has already run on this handle (once true, every
   * op/read throws instead of touching WASM memory). */
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
  // parameter type — the same reason `ndarray.ts` needs `AnyNDArray =
  // NDArray<any>` instead of `NDArray<Shape>` for its own erased handle.
  private assertSameCore<B extends Shape>(other: WNDArray<B>, op: string): void {
    if (this.core !== other.core) {
      throw new Error(`WNDArray.${op}: operands belong to different WASM core instances`);
    }
  }

  /** Free the WASM allocation, mark this handle disposed, and unregister
   * from the GC backstop (`this` doubles as the unregister token it was
   * registered with — see the constructor). A second call is a safe no-op:
   * guarded by the `disposed_` flag, never by inspecting `ptr` (a
   * legitimately empty array has `ptr === 0` from construction, not from
   * having been disposed). */
  dispose(): void {
    if (this.disposed_) return;
    this.disposed_ = true;
    registry.unregister(this);
    releaseDataBuf(this.core, this.ptr, this.bytes);
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
    return new WNDArray<Mutable<S>>(core, [...shape] as Mutable<S>, buf.ptr, len);
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
    return new WNDArray<Mutable<S>>(core, [...shape] as Mutable<S>, buf.ptr, len);
  }

  /** Build a resident array from a flat row-major values list — the
   * explicit copy-IN boundary (spec). Throws at runtime if `values.length`
   * doesn't match the shape's element count (same check as
   * `NDArray.fromArray`). */
  static fromArray<const S extends Shape>(core: CoreExports, shape: S, values: readonly number[]): WNDArray<Mutable<S>> {
    const len = product(shape);
    if (values.length !== len) {
      throw new Error(`fromArray: expected ${len} values for shape [${shape.join(",")}], got ${values.length}`);
    }
    const buf = allocBytes(core, len * 8);
    const view = new Float64Array(core.memory.buffer, buf.ptr, len);
    view.set(values);
    return new WNDArray<Mutable<S>>(core, [...shape] as Mutable<S>, buf.ptr, len);
  }

  /** Broadcasting elementwise add — resident twin of `NDArray.add`/
   * `wasmAdd`: operand data stays resident; only the two shape arrays are
   * marshalled per call, and a fresh output buffer is allocated for the
   * result (never aliasing either operand). */
  add<B extends Shape>(other: Guard<Broadcast<S, B>, WNDArray<B>>): WNDArray<OkShape<Broadcast<S, B>>> {
    this.assertLive("add");
    const o = other as unknown as WNDArray<B>;
    o.assertLive("add");
    this.assertSameCore(o, "add");

    // Shape computation (and any incompatibility throw) happens BEFORE any
    // allocation — mirrors v1's error-path ordering: a rejected call never
    // leaks a scratch buffer because none was ever allocated for it.
    const outShape = runtimeBroadcastShape(this.shape, o.shape);
    const outLen = product(outShape);

    const aShapeBuf = writeShape(this.core, this.shape);
    const bShapeBuf = writeShape(this.core, o.shape);
    const outDataBuf = allocBytes(this.core, outLen * 8);
    try {
      const status = this.core.nt_add(
        aShapeBuf.ptr,
        this.shape.length,
        this.ptr,
        this.len,
        bShapeBuf.ptr,
        o.shape.length,
        o.ptr,
        o.len,
        outDataBuf.ptr,
        outLen,
      );
      if (status !== 0) {
        freeBuf(this.core, outDataBuf); // fresh output buffer never escapes on failure
        throw new Error(
          `wasm resident nt_add: status ${status} for shapes [${this.shape.join(",")}] and [${o.shape.join(",")}]`,
        );
      }
      return new WNDArray<OkShape<Broadcast<S, B>>>(this.core, outShape as OkShape<Broadcast<S, B>>, outDataBuf.ptr, outLen);
    } finally {
      // Ephemeral per-call shape scratch: always freed, success or failure.
      freeBuf(this.core, aShapeBuf);
      freeBuf(this.core, bShapeBuf);
    }
  }

  /** Full NumPy `matmul` — resident twin of `NDArray.matmul`/`wasmMatmul`.
   * 1-D promotion/squeeze is TS-side shape bookkeeping only (mirrors
   * `matmulRuntime`/`wasmMatmul` exactly); it never touches operand data —
   * the same resident `ptr`/`len` are passed straight to the kernel under
   * the promoted shape metadata. */
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

    const aShapeBuf = writeShape(this.core, aShape);
    const bShapeBuf = writeShape(this.core, bShape);
    const outDataBuf = allocBytes(this.core, outLen * 8);

    let result: WNDArray<OkShape<MatMul<S, B>>>;
    try {
      const status = this.core.nt_matmul(
        aShapeBuf.ptr,
        aShape.length,
        this.ptr,
        this.len,
        bShapeBuf.ptr,
        bShape.length,
        o.ptr,
        o.len,
        outDataBuf.ptr,
        outLen,
      );
      if (status !== 0) {
        freeBuf(this.core, outDataBuf);
        throw new Error(
          `wasm resident nt_matmul: status ${status} for shapes [${aShapeIn.join(",")}] and [${bShapeIn.join(",")}]`,
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
      result = new WNDArray<OkShape<MatMul<S, B>>>(this.core, finalShape as OkShape<MatMul<S, B>>, outDataBuf.ptr, outLen);
    } finally {
      freeBuf(this.core, aShapeBuf);
      freeBuf(this.core, bShapeBuf);
    }
    return result;
  }

  /** Sum-reduce along `axis` (negative counts from the end); omit `axis` to
   * sum every element down to a rank-0 array. Resident twin of
   * `NDArray.sum`/`wasmSum`. */
  sum<const Axis extends number | undefined = undefined>(
    axis?: Guard<ReduceAxis<S, Axis>, Axis>,
  ): WNDArray<OkShape<ReduceAxis<S, Axis>>> {
    this.assertLive("sum");
    const axisNum = axis as unknown as Axis | undefined;

    if (axisNum === undefined) {
      const outDataBuf = allocBytes(this.core, 8);
      const status = this.core.nt_sum_all(this.ptr, this.len, outDataBuf.ptr);
      if (status !== 0) {
        freeBuf(this.core, outDataBuf);
        throw new Error(`wasm resident nt_sum_all: unexpected status ${status}`);
      }
      return new WNDArray<OkShape<ReduceAxis<S, Axis>>>(this.core, [] as OkShape<ReduceAxis<S, Axis>>, outDataBuf.ptr, 1);
    }

    const rank = this.shape.length;
    const normAxis = axisNum < 0 ? rank + axisNum : axisNum;
    if (normAxis < 0 || normAxis >= rank) {
      throw new Error(`reduce: axis ${axisNum} is out of range for shape [${this.shape.join(",")}] (rank ${rank})`);
    }

    const outShape = [...this.shape.slice(0, normAxis), ...this.shape.slice(normAxis + 1)];
    const outLen = product(outShape);

    const shapeBuf = writeShape(this.core, this.shape);
    const outDataBuf = allocBytes(this.core, outLen * 8);
    try {
      const status = this.core.nt_sum_axis(shapeBuf.ptr, rank, this.ptr, this.len, axisNum, outDataBuf.ptr, outLen);
      if (status !== 0) {
        freeBuf(this.core, outDataBuf);
        throw new Error(`wasm resident nt_sum_axis: status ${status} for shape [${this.shape.join(",")}] axis ${axisNum}`);
      }
      return new WNDArray<OkShape<ReduceAxis<S, Axis>>>(this.core, outShape as OkShape<ReduceAxis<S, Axis>>, outDataBuf.ptr, outLen);
    } finally {
      freeBuf(this.core, shapeBuf);
    }
  }

  /** Reverse every axis (NumPy's `.T` generalized to N-D). Resident twin of
   * `NDArray.transpose`/`wasmTranspose`. */
  transpose(): WNDArray<Transpose<S>> {
    this.assertLive("transpose");
    const outShape = [...this.shape].reverse();
    const outLen = product(outShape);

    const shapeBuf = writeShape(this.core, this.shape);
    const outDataBuf = allocBytes(this.core, outLen * 8);
    try {
      const status = this.core.nt_transpose(shapeBuf.ptr, this.shape.length, this.ptr, this.len, outDataBuf.ptr, outLen);
      if (status !== 0) {
        freeBuf(this.core, outDataBuf);
        throw new Error(`wasm resident nt_transpose: status ${status} for shape [${this.shape.join(",")}]`);
      }
      return new WNDArray<Transpose<S>>(this.core, outShape as Transpose<S>, outDataBuf.ptr, outLen);
    } finally {
      freeBuf(this.core, shapeBuf);
    }
  }

  /** Read back as an independent `Float64Array` copy — the explicit
   * copy-OUT boundary (spec). Never a live view: the returned array is
   * detached from WASM memory by construction (`Float64Array.from`), so it
   * stays valid even after `dispose()` or a later `memory.grow`. */
  toArray(): Float64Array {
    this.assertLive("toArray");
    const view = new Float64Array(this.core.memory.buffer, this.ptr, this.len);
    return Float64Array.from(view);
  }

  /** Read back as a plain nested JS array (any rank), for printing/tests —
   * same shape/stride walk as `NDArray.toNestedArray`. */
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
