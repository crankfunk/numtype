/**
 * Public function surface: `NDArray<S>` + constructors + the four ops
 * (add, matmul, sum, transpose). See docs/spike-01-ergebnisse.md for the
 * error-surfacing pattern write-up and the alternative considered.
 *
 * Error-surfacing pattern (chosen): each op's "other-shape" parameter is
 * typed as a conditional — when the computed result type is a
 * `ShapeError<Message>`, the parameter type becomes an object requiring a
 * `__shapeError: Message` property. The actual argument (a plain
 * `NDArray<B>`) obviously lacks that property, so `tsc` reports a "missing
 * property" error *at that argument*, and the property's type is the
 * literal message string — so the message (naming the offending shapes)
 * appears verbatim in the error. When shapes ARE compatible, the
 * conditional resolves to plain `NDArray<B>` and the call is unconstrained.
 * See docs/spike-01-ergebnisse.md for the alternative (return-type-only
 * error surfacing) and why it was rejected: it type-checks fine at the
 * mistake's call site and only errors later, wherever the result is first
 * consumed — which can be far from the actual mistake, especially in
 * chained calls.
 */

import type { Broadcast } from "./broadcast.ts";
import { type Dim, type Mutable, type Shape, type ShapeError } from "./dim.ts";
import type { MatMul } from "./matmul.ts";
import type { ReduceAxis, Transpose } from "./reduce.ts";
import { computeStrides, elementwiseBinary, matmulRuntime, product, sumRuntime, transposeRuntime } from "./runtime.ts";

/** Narrow a possibly-erroring computed shape down to a real `Shape`,
 * excluding the `ShapeError` branch. Only ever evaluated at call sites
 * where the compatible branch already applies (the incompatible branch is
 * rejected earlier, at the argument itself, by `Guard`).
 *
 * Exported (type-only, Kern 02): `spike/src/wasm/resident.ts`'s `WNDArray`
 * reuses this exact type machinery so its ops surface shape errors at the
 * argument identically to `NDArray` — see docs/kern-02-residency-spec.md. */
export type OkShape<S> = S extends ShapeError<string> ? never : S extends Shape ? S : never;

/** The argument-side guard: forces a "missing property" error, naming the
 * shape mismatch, at the *argument* when `Result` is a `ShapeError`.
 *
 * Exported (type-only, Kern 02): see `OkShape` above. */
export type Guard<Result, Actual> = Result extends ShapeError<infer Message> ? { readonly __shapeError: Message } : Actual;

/**
 * Erased top type for heterogeneous containers and non-generic helpers.
 *
 * `NDArray<Shape>` does NOT work as an implicit supertype: sound
 * dynamic-rank degradation puts `S` in method-parameter positions (the
 * argument-side error guards), which makes the class's measured variance
 * invariant — `NDArray<[2, 3]>` is then not assignable to
 * `NDArray<readonly number[]>`. (Pre-fix this assignment only type-checked
 * because the unsound degradations happened to satisfy the variance probe;
 * verified empirically against Guard-, intersection- and single-root
 * formulations — see docs/spike-01-ergebnisse.md, post-verification
 * addendum.) `any` as the type argument bypasses the variance comparison
 * entirely, TS's own idiom for a variance-erased handle.
 */
export type AnyNDArray = NDArray<any>;

export class NDArray<S extends Shape> {
  readonly shape: S;
  readonly data: Float64Array;

  private constructor(shape: S, data: Float64Array) {
    this.shape = shape;
    this.data = data;
  }

  /** An all-zeros array of the given shape. `const S` means callers never
   * need `as const` — `NDArray.zeros([2, 3])` infers `S` from the literal
   * `[2, 3]`. `Mutable<S>` strips the `readonly` a `const` type param would
   * otherwise attach, so the hover matches every other op's plain-tuple
   * display: `NDArray<[2, 3]>`, not `NDArray<readonly [2, 3]>`. */
  static zeros<const S extends Shape>(shape: S): NDArray<Mutable<S>> {
    return new NDArray<Mutable<S>>([...shape] as Mutable<S>, new Float64Array(product(shape)));
  }

  /** An all-ones array of the given shape. */
  static ones<const S extends Shape>(shape: S): NDArray<Mutable<S>> {
    return new NDArray<Mutable<S>>([...shape] as Mutable<S>, new Float64Array(product(shape)).fill(1));
  }

  /** Build an array from flat row-major values. Accepts a plain list or a
   * `Float64Array` — the typed-array path copies via the copy constructor
   * (memcpy-fast); forcing typed-array callers through `number[]` would
   * cost ~100x at the boundary (docs/kern-02-ergebnisse.md, chain-bench
   * finding). The input is always copied, never aliased. Throws at runtime
   * if `values.length` doesn't match the shape's element count. */
  static fromArray<const S extends Shape>(shape: S, values: readonly number[] | Float64Array): NDArray<Mutable<S>> {
    const size = product(shape);
    if (values.length !== size) {
      throw new Error(`fromArray: expected ${size} values for shape [${shape.join(",")}], got ${values.length}`);
    }
    const data = values instanceof Float64Array ? new Float64Array(values) : Float64Array.from(values);
    return new NDArray<Mutable<S>>([...shape] as Mutable<S>, data);
  }

  /** Broadcasting elementwise add. */
  add<B extends Shape>(other: Guard<Broadcast<S, B>, NDArray<B>>): NDArray<OkShape<Broadcast<S, B>>> {
    const o = other as unknown as NDArray<B>;
    const { shape, data } = elementwiseBinary(this.shape, this.data, o.shape, o.data, (x, y) => x + y);
    return new NDArray<OkShape<Broadcast<S, B>>>(shape as OkShape<Broadcast<S, B>>, data);
  }

  /** Full NumPy `matmul`: 2-D product, 1-D promotion, batch broadcasting. */
  matmul<B extends Shape>(other: Guard<MatMul<S, B>, NDArray<B>>): NDArray<OkShape<MatMul<S, B>>> {
    const o = other as unknown as NDArray<B>;
    const { shape, data } = matmulRuntime(this.shape, this.data, o.shape, o.data);
    return new NDArray<OkShape<MatMul<S, B>>>(shape as OkShape<MatMul<S, B>>, data);
  }

  /** Sum-reduce along `axis` (negative counts from the end); omit `axis` to
   * sum every element down to a rank-0 array. */
  sum<const Axis extends number | undefined = undefined>(
    axis?: Guard<ReduceAxis<S, Axis>, Axis>,
  ): NDArray<OkShape<ReduceAxis<S, Axis>>> {
    const axisNum = axis as unknown as Axis | undefined;
    const { shape, data } = sumRuntime(this.shape, this.data, axisNum);
    return new NDArray<OkShape<ReduceAxis<S, Axis>>>(shape as OkShape<ReduceAxis<S, Axis>>, data);
  }

  /** Reverse every axis (NumPy's `.T` generalized to N-D). */
  transpose(): NDArray<Transpose<S>> {
    const { shape, data } = transposeRuntime(this.shape, this.data);
    return new NDArray<Transpose<S>>(shape as Transpose<S>, data);
  }

  /** Row-major strides for the current shape (introspection helper). */
  strides(): number[] {
    return computeStrides(this.shape);
  }

  /** Read back as a plain nested JS array (any rank), for printing/tests. */
  toNestedArray(): unknown {
    const strides = computeStrides(this.shape);
    const build = (axis: number, offset: number): unknown => {
      if (axis === this.shape.length) return this.data[offset] ?? 0;
      const dim: Dim = this.shape[axis] ?? 0;
      const stride = strides[axis] ?? 0;
      const out: unknown[] = [];
      for (let i = 0; i < dim; i++) out.push(build(axis + 1, offset + i * stride));
      return out;
    };
    return build(0, 0);
  }
}
