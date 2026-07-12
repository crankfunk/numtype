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
 *    supported subset, extended by Spike 06 for negative literal
 *    `start`/`stop` and literal `step >= 1`, docs/spike-06-range-literals-spec.md):
 *    a literal `start`/`stop` (negative or non-negative) with
 *    `step` omitted, literal `1`, or a literal plain-digit integer `>= 2`
 *    instead yields a computed LITERAL dim, via digit-string (not
 *    tuple-length) arithmetic; anything outside that subset still degrades
 *    to `number`, identically to the core rule. A literal `step` PROVABLY
 *    invalid at runtime (`0`, negative, or non-integer) is instead a
 *    COMPILE ERROR at that argument (`SliceSpecsGuard` below, Spike 06).
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

import { type Dim, type RankUnknowable, type Shape, type ShowShape } from "./dim.ts";
import type { ExtractStep, LiteralIndexBounds, LiteralRangeDim, LiteralStepInvalid } from "./slice-literal.ts";

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
 * Resulting shape of `arr.slice(...specs)`. Wide `S` (unknown rank), a
 * MIXED-rank shape union (D-V1.3, `RankUnknowable`), and wide `Specs`
 * (unknown spec count) all degrade wholly to `readonly Dim[]` — checked
 * first, before any tuple recursion (spec: "wide-type guard FIRST").
 */
export type SliceShape<S extends Shape, Specs extends readonly SliceSpecInput[]> = RankUnknowable<S> extends true
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

/** Spike 03 (docs/spike-03-index-bounds-spec.md): the out-of-bounds message
 * for a provably-invalid literal integer index — same wording stem as the
 * runtime's own throw in `normalizeAxisSpec` (runtime.ts), extended with the
 * full shape for editor context. */
type IndexOutOfBoundsMessage<I extends number, Axis extends number, D extends Dim, S extends Shape> =
  `slice: index ${I} is out of bounds for axis ${Axis} with dim ${D} (shape ${ShowShape<S>})`;

/** Spike 06 (docs/spike-06-range-literals-spec.md): the step-invalid message
 * for a provably-invalid literal range-spec `step` — same wording stem as
 * the runtime's own throw in `normalizeAxisSpec` (runtime.ts:300), extended
 * with the full shape for editor context (same convention as
 * `IndexOutOfBoundsMessage`). `StepV` is the step's own literal value
 * (`ExtractStep<Head>`, unconstrained at the extraction site but always a
 * `number` by the time `LiteralStepInvalid<Head>` says `"invalid"`, since
 * that verdict requires a `step` property to exist at all). */
type StepInvalidMessage<StepV extends number, Axis extends number, S extends Shape> =
  `slice: step ${StepV} for axis ${Axis} is invalid (must be an integer >= 1; negative steps are out of scope) (shape ${ShowShape<S>})`;

/** Tail-recursive accumulator: walks `S` and `Specs` in lockstep, passing
 * each in-range spec's OWN type through unchanged. Three error mechanisms,
 * all landing at the offending ARGUMENT:
 *  - Once `S` is exhausted with `Specs` entries still remaining, every
 *    remaining position is retyped to the shared error object
 *    (`ErrorTuple`) — every excess argument is flagged, not just the first.
 *  - Spike 03: an integer spec whose literal value is PROVABLY out of
 *    bounds for its axis's literal dim (`LiteralIndexBounds` = `"out"` —
 *    the runtime would be guaranteed to throw) is retyped, individually, to
 *    the branded error object naming index, axis, dim, and shape. `"in"`
 *    and `"unknown"` (wide/dynamic/union/non-integer-literal) pass through
 *    unchanged — the static check is never wrong, only incomplete; the
 *    runtime backstop stays authoritative for everything unproven.
 *  - Spike 06: a range-spec's literal `step` that is PROVABLY invalid
 *    (`LiteralStepInvalid` = `"invalid"` — plain-digit `0`, negative, or
 *    dot-form non-integer) is retyped the same way, naming the step, axis,
 *    and shape. `null` specs skip this check entirely (never step-guard-
 *    relevant); `"unknown"` (valid/wide/union/exponent-form steps) passes
 *    through unchanged, same never-wrong-only-incomplete discipline.
 * `FullS` threads the ORIGINAL shape (unconsumed) purely for messages;
 * `Passed["length"]` doubles as the current axis index (one entry is
 * accumulated per consumed spec — rank-bounded, the allowed kind of
 * tuple-length arithmetic). */
type ValidateSpecsAcc<
  S extends readonly Dim[],
  Specs extends readonly SliceSpecInput[],
  Msg extends string,
  FullS extends Shape,
  Passed extends readonly unknown[] = [],
> = S extends readonly [infer SHead extends Dim, ...infer STail extends readonly Dim[]]
  ? Specs extends readonly [infer Head, ...infer SpecsTail extends readonly SliceSpecInput[]]
    ? [Head] extends [number]
      ? LiteralIndexBounds<Head, SHead> extends "out"
        ? ValidateSpecsAcc<
            STail,
            SpecsTail,
            Msg,
            FullS,
            [...Passed, { readonly __shapeError: IndexOutOfBoundsMessage<Head, Passed["length"], SHead, FullS> }]
          >
        : ValidateSpecsAcc<STail, SpecsTail, Msg, FullS, [...Passed, Head]>
      : [Head] extends [null]
        ? ValidateSpecsAcc<STail, SpecsTail, Msg, FullS, [...Passed, Head]>
        : [LiteralStepInvalid<Head>] extends ["invalid"]
          ? ExtractStep<Head> extends infer StepV extends number
            ? ValidateSpecsAcc<
                STail,
                SpecsTail,
                Msg,
                FullS,
                [...Passed, { readonly __shapeError: StepInvalidMessage<StepV, Passed["length"], FullS> }]
              >
            : ValidateSpecsAcc<STail, SpecsTail, Msg, FullS, [...Passed, Head]> // defensive: step wasn't actually a number (shouldn't happen)
          : ValidateSpecsAcc<STail, SpecsTail, Msg, FullS, [...Passed, Head]>
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
 * A MIXED-rank shape union (D-V1.3, `RankUnknowable`) passes `Specs` through
 * unchanged too — without this, a mixed-rank receiver could silently accept
 * an arity that is only valid for SOME of its rank union's members (each
 * member distributes its own arity check independently; a literal call
 * matching one member's shape sails through even when another member would
 * reject it) — gradual, runtime-checked instead, same honesty rule as
 * `SliceShape`.
 */
export type SliceSpecsGuard<S extends Shape, Specs extends readonly SliceSpecInput[]> = RankUnknowable<S> extends true
  ? Specs
  : IsDynamicLength<Specs> extends true
    ? Specs
    : ValidateSpecsAcc<S, Specs, TooManySpecsMessage<S, Specs>, S>;
