/**
 * Basic (NumPy-style) slicing, at the type level. See
 * docs/kern-05-slicing-spec.md for the full binding spec (semantics table,
 * scope, acceptance criteria); this header covers only the type-layer design.
 *
 * Design constraints (spec, non-negotiable):
 *  - NEVER tuple-length arithmetic over dim *values* (dims can be 1024+) —
 *    only over RANK (small ints, tuple length). All recursion here walks one
 *    axis (one tuple element) per step, tail-recursive/accumulator, exactly
 *    like broadcast.ts/matmul.ts/reduce.ts.
 *  - Wide `S` (`number[]`, unknown rank) degrades wholly, checked FIRST,
 *    before any tuple recursion (KB `ts-wide-types-vor-tupel-rekursion-abfangen`).
 *  - An integer spec drops its axis STATICALLY even when the index value is
 *    a plain (non-literal) `number` — the rank effect never depends on the
 *    concrete value, only on which of the three spec *shapes* (integer /
 *    `null` / range object) was used.
 *
 * Per-axis spec forms (mirrors the runtime `SliceSpec` in runtime.ts — kept
 * as a separately-declared, structurally-identical type here rather than
 * imported, the same standalone-module split dim.ts/runtime.ts already use
 * for `Shape` vs plain `readonly number[]`):
 *  - `number` — index the axis and remove it (rank -1).
 *  - `null` — take the axis in full; its literal dim survives untouched.
 *  - `{ start?, stop?, step? }` — range slice; the axis survives. CORE rule:
 *    its dim degrades to `number` (the exact resulting count depends on dim
 *    VALUES, which tuple-recursion here never computes arithmetically).
 *    STRETCH (slice-literal.ts, gated — see that file for the precise
 *    supported subset): a literal non-negative `start`/`stop` with `step`
 *    omitted-or-`1` instead yields a computed LITERAL dim, via digit-string
 *    (not tuple-length) arithmetic; anything outside that subset still
 *    degrades to `number`, identically to the core rule.
 *
 * Two independent computations share the (S, Specs) pair:
 *  - `SliceShape<S, Specs>` — the resulting shape (this file's main export).
 *  - `SliceSpecsGuard<S, Specs>` — the REST-PARAMETER type used at the
 *    `slice()` call site to surface "too many specs" as a compile error at
 *    the offending argument. This can't reuse the `Guard<Result, Actual>`
 *    helper from ndarray.ts as-is: `Guard` collapses its whole `Actual` type
 *    to a single non-array `{ __shapeError }` object on failure, which is
 *    illegal for a *rest* parameter (TS2370 "a rest parameter must be of an
 *    array type" — confirmed empirically; a conditional type that could
 *    resolve to a non-array branch is rejected at the method's OWN
 *    declaration, breaking every call, not just the bad ones). Instead,
 *    `SliceSpecsGuard` stays a tuple the same length as `Specs` at every
 *    step (satisfies TS2370 unconditionally) and, only once the rank is
 *    exhausted, retypes each EXCESS position as the `{ __shapeError }`
 *    object (via the homomorphic `ErrorTuple` mapped type — same idiom as
 *    reduce.ts's `AllOnes`). The result: a "missing property" error lands
 *    on the first excess argument, naming the rank and spec count, and every
 *    within-rank call stays untouched by this mechanism.
 */

import { type Dim, type IsDynamicRank, type Shape, type ShowShape } from "./dim.ts";
import type { LiteralRangeDim } from "./slice-literal.ts";

/** One axis's slice specification, at the type level. See file header. */
export type SliceSpecInput = number | null | { readonly start?: number; readonly stop?: number; readonly step?: number };

/** Is this tuple's LENGTH statically unknown (a wide array like `T[]`, as
 * opposed to a fixed-length tuple)? Same "wide type" probe as
 * `IsDynamicRank` in dim.ts, generalized to any tuple (here: the `Specs`
 * rest-parameter tuple, not a `Shape`). A caller spreading a plain
 * `SliceSpecInput[]` (no `as const`, no literal tuple) hits this — honest
 * gradual degrade, not a guess. */
type IsDynamicLength<T extends readonly unknown[]> = number extends T["length"] ? true : false;

// ---------------------------------------------------------------------------
// SliceShape: the resulting shape.
// ---------------------------------------------------------------------------

/** Tail-recursive accumulator walk over `S` and `Specs` in lockstep. Once
 * `Specs` is exhausted, whatever remains of `S` (the trailing axes) is
 * appended verbatim — "trailing axes are taken in full" from the spec. If
 * `S` exhausts first (too many specs), recursion just stops and returns
 * `Acc` — the argument-side `SliceSpecsGuard` already rejects that call at
 * the argument, so this branch is never reached by code that compiles.
 *
 * Range-object dims go through `LiteralRangeDim<Head, SHead>`
 * (slice-literal.ts, the gated stretch): a literal, non-negative
 * `start`/`stop` with `step` omitted-or-`1` yields a computed literal dim;
 * everything else honestly degrades to `Dim` (`number`) — the exact same
 * fallback a core-only (no-stretch) implementation would use. Dropping the
 * stretch is therefore a one-line revert of this single call site (back to
 * a bare `Dim`), never a structural change to this walk. */
type SliceShapeAcc<
  S extends readonly Dim[],
  Specs extends readonly SliceSpecInput[],
  Acc extends readonly Dim[] = [],
> = Specs extends readonly [infer Head extends SliceSpecInput, ...infer SpecsTail extends readonly SliceSpecInput[]]
  ? S extends readonly [infer SHead extends Dim, ...infer STail extends readonly Dim[]]
    ? Head extends null
      ? SliceShapeAcc<STail, SpecsTail, [...Acc, SHead]> // null: keep the literal dim
      : Head extends number
        ? SliceShapeAcc<STail, SpecsTail, Acc> // integer: drop the axis (static, regardless of the value)
        : SliceShapeAcc<STail, SpecsTail, [...Acc, LiteralRangeDim<Head, SHead>]> // range object: literal dim (stretch) or `number` (core fallback)
    : Acc // S exhausted, Specs still has entries: too-many-specs; SliceSpecsGuard already flags the argument
  : [...Acc, ...S]; // Specs exhausted: append S's remaining (trailing) axes literally, in full

/**
 * Resulting shape of `arr.slice(...specs)`. Wide `S` (unknown rank) and wide
 * `Specs` (unknown spec count) both degrade wholly to `readonly Dim[]` —
 * checked first, before any tuple recursion (spec: "wide-type guard FIRST").
 */
export type SliceShape<S extends Shape, Specs extends readonly SliceSpecInput[]> = IsDynamicRank<S> extends true
  ? readonly Dim[]
  : IsDynamicLength<Specs> extends true
    ? readonly Dim[]
    : SliceShapeAcc<S, Specs>;

// ---------------------------------------------------------------------------
// SliceSpecsGuard: the rest-parameter arity guard (see file header).
// ---------------------------------------------------------------------------

/** Homomorphic mapped type: same length/tuple-ness as `T`, every element
 * retyped to the branded error-carrying object. Same idiom as reduce.ts's
 * `AllOnes` — a mapped type over a tuple via `keyof T` preserves the
 * tuple's array-ness, which is exactly what keeps `SliceSpecsGuard` legal in
 * a rest-parameter position even on the error path. */
type ErrorTuple<T extends readonly unknown[], Msg extends string> = { [K in keyof T]: { readonly __shapeError: Msg } };

/** The too-many-specs message: names both counts and the shape, mirroring
 * every other op's error-message convention (`broadcast.ts`/`matmul.ts`). */
type TooManySpecsMessage<S extends Shape, Specs extends readonly SliceSpecInput[]> =
  `slice: ${Specs["length"]} specs given for rank ${S["length"]} shape ${ShowShape<S>}`;

/** Tail-recursive accumulator: walks `S` and `Specs` in lockstep, passing
 * each in-range spec's OWN type through unchanged. Once `S` is exhausted
 * with `Specs` entries still remaining, every remaining position is retyped
 * to the shared error object (`ErrorTuple`) — every excess argument is
 * flagged, not just the first. */
type ValidateSpecsAcc<
  S extends readonly Dim[],
  Specs extends readonly SliceSpecInput[],
  Msg extends string,
  Passed extends readonly unknown[] = [],
> = S extends readonly [Dim, ...infer STail extends readonly Dim[]]
  ? Specs extends readonly [infer Head, ...infer SpecsTail extends readonly SliceSpecInput[]]
    ? ValidateSpecsAcc<STail, SpecsTail, Msg, [...Passed, Head]>
    : [...Passed, ...Specs] // Specs shorter than/equal to the remaining rank: nothing to flag (Specs is empty here)
  : Specs extends readonly []
    ? [...Passed] // exact match: rank and spec count agree exactly
    : [...Passed, ...ErrorTuple<Specs, Msg>]; // rank exhausted, Specs has excess: flag every excess position

/**
 * The `slice()` method's rest-parameter type. Dynamic rank / dynamic spec
 * count both pass `Specs` through untouched (can't validate arity
 * statically — gradual, runtime-checked instead, same honesty rule as
 * `SliceShape`). Otherwise: too many specs retypes the excess arguments so
 * `tsc` reports a "missing property" error naming the rank and spec count,
 * landing on the first excess argument; a valid call's `Specs` passes
 * through completely unchanged (every element keeps its own inferred type).
 */
export type SliceSpecsGuard<S extends Shape, Specs extends readonly SliceSpecInput[]> = IsDynamicRank<S> extends true
  ? Specs
  : IsDynamicLength<Specs> extends true
    ? Specs
    : ValidateSpecsAcc<S, Specs, TooManySpecsMessage<S, Specs>>;
