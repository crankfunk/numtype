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
import {
  assertVectorPair,
  computeStrides,
  dotRuntime,
  elementwiseBinary,
  matmulRuntime,
  normalizeSliceSpecs,
  normSqRuntime,
  product,
  sliceRuntime,
  type SliceSpec,
  sumRuntime,
  transposeRuntime,
} from "./runtime.ts";
import type { SliceShape, SliceSpecInput, SliceSpecsGuard } from "./slice.ts";
import type { DotCheck } from "./vector.ts";

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
 * A minimal, checker-ENFORCED covariant read view (Spike 05,
 * docs/spike-05-variance-design-spec.md). `out S` makes the compiler itself
 * prove, at every compile, that widening a concrete `NDArrayView<[2, 3]>` to
 * `NDArrayView<Shape>` (or to `NDArrayView<readonly number[]>`) is sound —
 * the annotation is a declaration-site regression pin for every
 * instantiation at once. WITH ONE KNOWN LOOPHOLE the maintainer must carry
 * (verified empirically, Spike-05 fresh-context pass): TypeScript checks
 * METHOD-SHORTHAND parameters bivariantly (the same long-standing exemption
 * `strictFunctionTypes` grants methods), so a future member written
 * shorthand-style with `S` in argument position — `resizeTo(s: S): void` —
 * would COMPILE despite genuinely breaking covariance. Only a
 * property-typed function member (`resizeTo: (s: S) => void`) triggers
 * TS2636. House rule, therefore: this view must NEVER gain a member that
 * consumes `S`, and any future function-typed member is to be declared
 * property-style, where the annotation actually enforces.
 *
 * Kept to EXACTLY these three members — the omissions are load-bearing,
 * not oversights (probe evidence in the spec):
 *  - No op methods (add/matmul/sum/transpose/slice/…): sound dynamic-rank
 *    degradation needs `S` in ARGUMENT position (the `Guard<Result, Actual>`
 *    pattern above, on `NDArray` itself) to surface shape errors at the
 *    call site — consuming `S` genuinely breaks covariance (whether or not
 *    the checker catches the particular syntax, per the loophole above), so
 *    guard-bearing members and a covariant view are mutually exclusive
 *    (the two-of-three rule, docs/spike-01-ergebnisse.md, applied to a
 *    read-only view). Op consumers stay on generic `NDArray<S>` — a plain
 *    generic function never needs variance, it re-derives `S` per call.
 *  - No shape-COMPUTING members either (e.g. a hypothetical
 *    `transpose(): NDArrayView<Transpose<S>>`), even though such a member
 *    takes no `S`-typed argument: measured this session (scratchpad
 *    `variance-probes{,-2}.ts`), that member declaration itself fails with
 *    TS2636. The `out` check is ABSTRACT — it must hold for any two related
 *    S/super-S, not merely the shapes this codebase happens to build — and
 *    `Transpose` is not *provably* monotone under that abstract check, only
 *    *factually* monotone for concrete instantiations. Proof of the
 *    distinction: the identical member WITHOUT the `out` annotation type-
 *    checks fine (the structural widening check runs on concrete
 *    instantiations only, where `Transpose` happens to behave) — but that
 *    is covariance-BY-ACCIDENT, exactly the failure mode Spike 01 rejected
 *    for `NDArray` itself: unenforced, silently breakable by a future
 *    non-monotone shape op, and only catchable point-wise by tests. The
 *    `out` annotation trades that away for a compile-time proof, which is
 *    only available to a view with no computing members.
 *  - No `data` field: a `Float64Array` is naive-backend-specific; keeping
 *    the view to shape/strides/nested-array keeps it satisfiable by
 *    resident/strided backends (`WNDArray`-style) too, for a later phase
 *    (FOLLOWUPS.md) — out of scope here.
 *
 * Two honest caveats on "read view":
 *  - `readonly shape: S` blocks reassigning the PROPERTY, not mutating the
 *    tuple's elements — `view.shape[0] = 99` type-checks. This is a
 *    PRE-EXISTING latent hole on `NDArray` itself (same member type), not
 *    introduced or widened by this view; a deep-readonly shape is a
 *    deliberate open decision (FOLLOWUPS.md — it interacts with the
 *    clean-hover house rule).
 *  - This is ordinary STRUCTURAL typing: any object with these three
 *    members satisfies the view, real `NDArray` or not — unlike
 *    `AnyNDArray` below, which stays de-facto nominal because `NDArray`'s
 *    private constructor blocks structural impostors. For a read-only
 *    surface that looseness is by design, but the capability difference
 *    between the two top types is real and worth knowing.
 */
export interface NDArrayView<out S extends Shape> {
  readonly shape: S;
  strides(): number[];
  toNestedArray(): unknown;
}

/**
 * Erased top type for heterogeneous containers and non-generic helpers that
 * also need to CALL ops (add/matmul/sum/transpose/slice/…) on a
 * shape-erased handle — a deliberately UNSAFE escape hatch, unsafe in BOTH
 * directions: `any` as the type argument bypasses the variance comparison
 * entirely (TS's own idiom for a variance-erased handle), so nothing stops
 * an `AnyNDArray` from flowing back into a precisely-shaped
 * `NDArray<[2, 3]>` binding either.
 *
 * `NDArray<Shape>` does NOT work as an implicit supertype: sound
 * dynamic-rank degradation puts `S` in method-parameter positions (the
 * argument-side error guards), which makes the class's measured variance
 * invariant — `NDArray<[2, 3]>` is then not assignable to
 * `NDArray<readonly number[]>`. (Pre-fix this assignment only type-checked
 * because the unsound degradations happened to satisfy the variance probe;
 * verified empirically against Guard-, intersection- and single-root
 * formulations — see docs/spike-01-ergebnisse.md, post-verification
 * addendum.)
 *
 * Reach for `AnyNDArray` ONLY when you need to CALL ops on a
 * heterogeneously-shaped array. For read-only access (shape/strides/
 * toNestedArray) prefer `NDArrayView<Shape>` (above) instead — it is the
 * safe, checker-enforced top type: covariant by a proven annotation, not by
 * erasure, so it cannot silently unerase in the write direction the way
 * `any` can.
 */
export type AnyNDArray = NDArray<any>;

export class NDArray<S extends Shape> implements NDArrayView<S> {
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

  /** Broadcasting elementwise subtract (Kern 07). Structural mirror of
   * `add` — same `Broadcast`/`Guard`/`OkShape` pattern, pinned closure
   * `(x, y) => x - y`. */
  sub<B extends Shape>(other: Guard<Broadcast<S, B>, NDArray<B>>): NDArray<OkShape<Broadcast<S, B>>> {
    const o = other as unknown as NDArray<B>;
    const { shape, data } = elementwiseBinary(this.shape, this.data, o.shape, o.data, (x, y) => x - y);
    return new NDArray<OkShape<Broadcast<S, B>>>(shape as OkShape<Broadcast<S, B>>, data);
  }

  /** Broadcasting elementwise multiply (Kern 07). Structural mirror of
   * `add` — pinned closure `(x, y) => x * y`. */
  mul<B extends Shape>(other: Guard<Broadcast<S, B>, NDArray<B>>): NDArray<OkShape<Broadcast<S, B>>> {
    const o = other as unknown as NDArray<B>;
    const { shape, data } = elementwiseBinary(this.shape, this.data, o.shape, o.data, (x, y) => x * y);
    return new NDArray<OkShape<Broadcast<S, B>>>(shape as OkShape<Broadcast<S, B>>, data);
  }

  /** Broadcasting elementwise divide (Kern 07). Structural mirror of `add`
   * — pinned closure `(x, y) => x / y`. Pure IEEE 754: no zero checks, no
   * throws (`x/0 -> +/-Infinity`, `0/0 -> NaN`, signed zeros/infinities
   * propagate per the standard — a documented divergence from NumPy, which
   * additionally warns; see spec). */
  div<B extends Shape>(other: Guard<Broadcast<S, B>, NDArray<B>>): NDArray<OkShape<Broadcast<S, B>>> {
    const o = other as unknown as NDArray<B>;
    const { shape, data } = elementwiseBinary(this.shape, this.data, o.shape, o.data, (x, y) => x / y);
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

  /** 1-D inner product (Kern 07): `a.dot(b)` for two rank-1 arrays of equal
   * length. Returns a plain `number` — deliberately leaves the `NDArray`
   * world (see spec's "dot(other)" design note: `sum()` is a reduction
   * that stays chainable, `dot`/`norm`/`cosineSimilarity` are scalar
   * consumer ops that terminate a chain). Rank != 1 (either operand) or a
   * length mismatch is a compile error at the argument (`DotCheck`) and a
   * runtime throw (`assertVectorPair`) for the gradual/dynamic cases the
   * type layer couldn't check statically. */
  dot<B extends Shape>(other: Guard<DotCheck<S, B, "dot">, NDArray<B>>): number {
    const o = other as unknown as NDArray<B>;
    assertVectorPair("dot", this.shape, o.shape);
    return dotRuntime(this.shape, this.data, o.shape, o.data);
  }

  /** L2/Frobenius norm over ALL elements (Kern 07), any rank (mirrors
   * `np.linalg.norm`'s default: flatten, then L2) — no guard, since a
   * niladic method has no argument to hang one on and every rank is valid
   * by this op's own semantics. `Math.sqrt` is IEEE-correctly-rounded, so
   * this is bit-identical to `WNDArray.norm` iff the underlying sum of
   * squares is (which the differential suite asserts). */
  norm(): number {
    return Math.sqrt(normSqRuntime(this.data));
  }

  /** Cosine similarity (Kern 07): same rank-1/equal-length operand contract
   * as `dot` (own error-message prefix). The pinned expression (spec,
   * identical on both surfaces): `num = dot(a,b)`, `den =
   * sqrt(normSq(a)) * sqrt(normSq(b))`, `return num / den`. Pure IEEE, no
   * epsilon guards: a zero vector on either side makes `den` (or both
   * `num` and `den`) `0`, yielding `NaN`; an adversarial magnitude split
   * can underflow `den` to `0` with `num != 0`, yielding `+/-Infinity`. */
  cosineSimilarity<B extends Shape>(other: Guard<DotCheck<S, B, "cosineSimilarity">, NDArray<B>>): number {
    const o = other as unknown as NDArray<B>;
    assertVectorPair("cosineSimilarity", this.shape, o.shape);
    const num = dotRuntime(this.shape, this.data, o.shape, o.data);
    const den = Math.sqrt(normSqRuntime(this.data)) * Math.sqrt(normSqRuntime(o.data));
    return num / den;
  }

  /** Reverse every axis (NumPy's `.T` generalized to N-D). */
  transpose(): NDArray<Transpose<S>> {
    const { shape, data } = transposeRuntime(this.shape, this.data);
    return new NDArray<Transpose<S>>(shape as Transpose<S>, data);
  }

  /** Basic (NumPy-style) slicing: one spec per leading axis, trailing axes
   * taken in full — see docs/kern-05-slicing-spec.md for the full semantics
   * table. `const Specs` means callers never write `as const` (same
   * rationale as `zeros`/`ones`/`fromArray`'s `const S`), which also lets
   * literal `start`/`stop`/`step` values reach the type layer. Always a
   * fresh COPY (naive reference; `WNDArray.slice` is the O(1) view twin —
   * both share `normalizeSliceSpecs`, see its doc comment for why that's a
   * deliberate, documented differential blind spot). Too many specs is a
   * compile error at the offending argument (`SliceSpecsGuard`, see
   * slice.ts) and a runtime throw (`normalizeSliceSpecs`) for gradual/
   * dynamic-rank callers the type layer couldn't check statically. */
  slice<const Specs extends readonly SliceSpecInput[]>(
    ...specs: SliceSpecsGuard<S, Specs>
  ): NDArray<OkShape<SliceShape<S, Specs>>> {
    const rawSpecs = specs as unknown as readonly SliceSpec[];
    const norm = normalizeSliceSpecs(this.shape, rawSpecs);
    const { shape, data } = sliceRuntime(this.shape, this.data, norm);
    return new NDArray<OkShape<SliceShape<S, Specs>>>(shape as OkShape<SliceShape<S, Specs>>, data);
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
