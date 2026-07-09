/**
 * Core dimension/shape primitives + the gradual-typing escape hatch.
 *
 * Design constraints (see docs/spike-01-type-layer-spec.md):
 *  - No arithmetic on dim *values* — only equality and 1-detection.
 *  - All recursive types are tail-recursive / accumulator-based.
 *  - A plain `number` dim is "dynamic": never a compile error, always
 *    propagates and degrades to a runtime check.
 */

/** A single dimension: a literal (e.g. `3`) when statically known, or the
 * wide `number` type when dynamic (unknown until runtime). */
export type Dim = number;

/** A shape is a readonly tuple of dims. Rank = tuple length (small ints —
 * safe to use tuple-length arithmetic on; never used for dim *values*). */
export type Shape = readonly Dim[];

/**
 * Branded error channel. Carries a human-readable message that names the
 * offending shapes/dims. Never widen this to `never`/`unknown` — the
 * message must survive to the point where a consumer sees the type error.
 */
declare const ShapeErrorBrand: unique symbol;
export type ShapeError<Message extends string> = {
  readonly [ShapeErrorBrand]: Message;
};

/** True iff a ShapeError of any message. Useful for lenient assertions in
 * tests that don't want to pin down exact wording. */
export type IsShapeError<T> = T extends ShapeError<string> ? true : false;

/** Is this Dim the wide, dynamic `number` type (as opposed to a specific
 * numeric literal)? The standard "is this the wide type" probe: only the
 * wide `number` type has `number extends T` hold. */
export type IsDynamicDim<T extends Dim> = number extends T ? true : false;

/**
 * Is this Shape's RANK statically unknown — a non-tuple array type like
 * `number[]`, or a variadic tuple like `[2, ...number[]]`? Fixed tuples have
 * a literal `length`; only dynamic-rank shapes have `length: number`. Ops
 * must degrade such shapes to `Dim[]` (gradual: accept, check at runtime) —
 * recursing into them via tuple destructuring silently treats them as
 * empty/exhausted and produces confidently-wrong results.
 */
export type IsDynamicRank<S extends Shape> = number extends S["length"] ? true : false;

/**
 * Per-axis broadcast compatibility (NumPy rule): equal, or one of them is 1.
 * A dynamic dim on either side is accepted unconditionally (gradual — we
 * can't know statically, so we don't error, and the result degrades to
 * dynamic too, since we no longer know the resolved dim statically).
 */
export type CompatDim<A extends Dim, B extends Dim> = IsDynamicDim<A> extends true
  ? Dim
  : IsDynamicDim<B> extends true
    ? Dim
    : A extends B
      ? A
      : B extends A
        ? B
        : A extends 1
          ? B
          : B extends 1
            ? A
            : ShapeError<`dims ${A} and ${B} are not broadcast-compatible (neither equal nor 1)`>;

/**
 * Strict dim equality for contraction axes (matmul inner dims): NO
 * broadcast "1" special-casing — a dynamic dim on either side matches
 * unconditionally (checked at runtime instead).
 */
export type DimEq<A extends Dim, B extends Dim> = IsDynamicDim<A> extends true
  ? true
  : IsDynamicDim<B> extends true
    ? true
    : A extends B
      ? (B extends A ? true : false)
      : false;

/** Tail-recursive (accumulator) reverse of a tuple. Used by Transpose and by
 * Broadcast's right-to-left alignment. */
export type Reverse<T extends readonly unknown[], Acc extends readonly unknown[] = []> = T extends readonly [
  infer Head,
  ...infer Rest,
]
  ? Reverse<Rest, [Head, ...Acc]>
  : Acc;

/**
 * Strip the `readonly` modifier a `const` type parameter attaches to an
 * inferred tuple (spreading a readonly tuple into a fresh tuple literal
 * yields a plain, mutable one with the same element types). Used so
 * `zeros`/`ones`/`fromArray` — which must use `const` type params so
 * callers never write `as const` — display exactly as clean a tuple as
 * every other op (`Broadcast`/`MatMul`/`ReduceAxis` already resolve to
 * plain tuples on their own): `NDArray<[2, 4]>`, never
 * `NDArray<readonly [2, 4]>`.
 */
export type Mutable<T extends readonly unknown[]> = [...T];

/** Tail-recursive join of a shape's dims into a comma-separated string, for
 * error messages / display. Rank-bounded (small), fine even non-tail. */
type JoinDims<S extends readonly Dim[], Acc extends string = ""> = S extends readonly [
  infer Head extends Dim,
  ...infer Rest extends readonly Dim[],
]
  ? JoinDims<Rest, Acc extends "" ? `${Head}` : `${Acc},${Head}`>
  : Acc;

/** Render a shape as `[2,3,4]` for messages/hovers. A dynamic-rank shape
 * cannot enumerate its dims — rendering it via JoinDims would misleadingly
 * print `[]` (or only the fixed prefix of a variadic tuple). */
export type ShowShape<S extends Shape> = IsDynamicRank<S> extends true ? "[unknown rank]" : `[${JoinDims<S>}]`;
