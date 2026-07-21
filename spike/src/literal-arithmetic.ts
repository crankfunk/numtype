/**
 * (Renamed from `slice-literal.ts` in Item 11 / S1, 2026-07-17 — the sole
 * historical reference to the old name kept for traceability. Since Spike 04
 * this file hosts ALL digit-string arithmetic — subtraction, addition,
 * schoolbook multiplication (`LiteralShapeProduct`), long division (`DivCeil`)
 * — plus the literal classifiers, not just slice-dim machinery, so the old
 * slice-specific name was misleading. `git log --follow` preserves full
 * history across the rename.)
 *
 * Kern 05 STRETCH GOAL (extended by Spike 06 — negative literal start/stop
 * and literal steps >= 1 — docs/spike-06-range-literals-spec.md): statically
 * computed literal dims for range slices, via from-scratch DIGIT-STRING
 * arithmetic — NOT tuple-length arithmetic over dim values (that would be
 * O(n) tuple construction per dim, blowing the ~1000 tail-recursion ceiling
 * for any dim past a few hundred; see docs/kern-05-slicing-spec.md and
 * CLAUDE.md's TS-limits section). Cost axis here is DIGIT COUNT: subtracting
 * two 7-digit numbers costs ~7 recursion steps, not ~1,000,000 — the same
 * "rank/count is fine, value is not" distinction the rest of this codebase
 * already draws, just pushed one level deeper (digits of a value, rather
 * than the value itself).
 *
 * Isolated in its OWN file, deliberately: this is the gated, "allowed to
 * fail and be dropped" part of the phase (spec: go/no-go on
 * Instantiations <= 3x baseline, wall time <= 2x baseline, clean hovers,
 * errors still at the argument). `slice.ts` imports three symbols from here:
 * `LiteralRangeDim` (Kern 05 stretch, extended by Spike 06 — dropping it
 * means reverting that one call site back to the core's honest `Dim`
 * degrade), `LiteralIndexBounds` (Spike 03 — dropping it means reverting
 * `ValidateSpecsAcc`'s bounds branch to an unconditional pass-through), and
 * `LiteralStepInvalid` (Spike 06 — dropping it means reverting
 * `ValidateSpecsAcc`'s step-guard branch the same way). No other file
 * depends on anything below.
 *
 * SUPPORTED LITERAL SUBSET (precise, by design — see file body for why):
 *  - `start`/`stop`, if present, must be literal integers — non-negative
 *    (unchanged from Kern 05, `min(v, d)`) OR negative (Spike 06:
 *    `v + d`, clamped to `[0, d]`, exactly mirroring the runtime's own
 *    normalization). Wide (non-literal) `number`, non-integer (`1.5`), and
 *    exponent-form (`1e21`) literals are NOT supported and degrade to
 *    `Dim`, honestly and gradually — identical to the core rule.
 *  - `step` must be omitted, the literal `1` (existing fast path, kept
 *    byte-identical), OR a literal plain-digit integer `>= 2` (Spike 06:
 *    `ceil((stop - start) / step)` via schoolbook long division). A step
 *    that is PROVABLY invalid at runtime — plain-digit `0`, negative
 *    (`-${digits}`), or dot-form non-integer (`1.5`) — is instead REJECTED
 *    at compile time (`LiteralStepInvalid`, wired into `slice.ts`'s
 *    `ValidateSpecsAcc`, the Spike-03 idiom) rather than silently degraded;
 *    a wide/union/exponent-form step still degrades the computed dim to
 *    `Dim` with NO guard claim (never-wrong-only-incomplete — flagging
 *    `1e21` as invalid would lie, since it IS a valid runtime step).
 *  - The axis's own dim must itself be a literal (an already-dynamic `Dim`
 *    can never yield a computed literal here either way).
 *  - A union literal (or `never`) for `start`/`stop`/`step` degrades the
 *    WHOLE computation to `Dim`, uniformly — the Spike-04 "union-free by
 *    construction" boundary-filter rule, reused here (a deliberate Spike 06
 *    behavior sharpening: the Kern-05-era code let such unions distribute
 *    into the digit pipeline unaudited; see
 *    docs/spike-06-range-literals-spec.md's "Union rule alignment").
 *
 * ALGORITHM (mirrors `normalizeAxisSpec` in runtime.ts exactly, at the type
 * level): `start' = clamp(start, d)` — non-negative: `min(start, d)`;
 * negative: `d - |start|`, or `0` when `|start| > d` (Spike 06's scoping
 * insight: this needs only a COMPARISON plus the existing UNSIGNED subtract,
 * never a signed add — correcting this file's original Kern-05 note below).
 * `stop'` resolves the same way. `dim = start' < stop' ? (step == 1 ?
 * stop' - start' : ceil((stop' - start') / step)) : 0`.
 */

import { type Dim, type IsDynamicDim, type IsDynamicRank, type Shape } from "./dim.ts";

type Digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9";

/** Digit -> its value 0-9, used ONLY to build small (<=19-element) fixed
 * tuples below — this is the bounded, "rank-level" kind of tuple-length use
 * the project already allows elsewhere (e.g. reduce.ts's `Decrement`), never
 * applied to a dim VALUE. */
type DigitValue<D extends Digit> = D extends "0"
  ? 0
  : D extends "1"
    ? 1
    : D extends "2"
      ? 2
      : D extends "3"
        ? 3
        : D extends "4"
          ? 4
          : D extends "5"
            ? 5
            : D extends "6"
              ? 6
              : D extends "7"
                ? 7
                : D extends "8"
                  ? 8
                  : 9;

/** `N` (always 0..19 here) -> a tuple of that length. */
type Tup<N extends number, Acc extends readonly unknown[] = []> = Acc["length"] extends N ? Acc : Tup<N, [...Acc, unknown]>;

/** A tuple's length (always 0..9 here) -> its digit char. */
type TupDigit<T extends readonly unknown[]> = T["length"] extends 0
  ? "0"
  : T["length"] extends 1
    ? "1"
    : T["length"] extends 2
      ? "2"
      : T["length"] extends 3
        ? "3"
        : T["length"] extends 4
          ? "4"
          : T["length"] extends 5
            ? "5"
            : T["length"] extends 6
              ? "6"
              : T["length"] extends 7
                ? "7"
                : T["length"] extends 8
                  ? "8"
                  : T["length"] extends 9
                    ? "9"
                    : never;

/** Subtract two digits with an incoming borrow (0 or 1): `[resultDigit,
 * borrowOut]`. Built via bounded tuple-length subtraction (`[...Sub,
 * ...infer Rest]` peels off exactly `Sub`'s length), never a 200-entry
 * lookup table and never proportional to a dim value. */
type SubDigit<A extends Digit, B extends Digit, BorrowIn extends 0 | 1> = Tup<DigitValue<B>> extends infer BT extends readonly unknown[]
  ? (BorrowIn extends 1 ? [...BT, unknown] : BT) extends infer Sub extends readonly unknown[]
    ? Tup<DigitValue<A>> extends [...Sub, ...infer NoBorrowRest]
      ? [TupDigit<NoBorrowRest>, 0]
      : [...Tup<DigitValue<A>>, ...Tup<10>] extends [...Sub, ...infer BorrowRest]
        ? [TupDigit<BorrowRest>, 1]
        : never
    : never
  : never;

/** Tail-recursive (accumulator) string reversal. */
type ReverseStr<S extends string, Acc extends string = ""> = S extends `${infer Head}${infer Rest}` ? ReverseStr<Rest, `${Head}${Acc}`> : Acc;

/** `"00" -> "0"`, `"007" -> "7"`, `"070" -> "70"` (leading zeros only). */
type StripLeadingZeros<S extends string> = S extends `0${infer Rest}` ? (Rest extends "" ? "0" : StripLeadingZeros<Rest>) : S;

/** Is `S` composed ONLY of decimal digits (non-empty)? Rejects `-`, `.`,
 * `e`-notation, and (since a non-literal `number`'s template-literal form
 * can't be walked character-by-character this way) any wide `number`. */
type IsPlainDigits<S extends string> = S extends ""
  ? false
  : S extends `${infer Head}${infer Rest}`
    ? Head extends Digit
      ? Rest extends ""
        ? true
        : IsPlainDigits<Rest>
      : false
    : false;

/** Multi-digit subtraction, digit-by-digit from the LEAST significant end
 * (via `ReverseStr`), threading a borrow. Once `BRev` is exhausted but
 * `ARev` isn't, subtracts `"0"` for the remaining (higher) digits of `A` —
 * this is what lets `A` and `B` have different digit counts without
 * separate zero-padding. */
type SubRev<ARev extends string, BRev extends string, BorrowIn extends 0 | 1, Acc extends string = ""> = ARev extends `${infer AHead extends Digit}${infer ARest}`
  ? BRev extends `${infer BHead extends Digit}${infer BRest}`
    ? SubDigit<AHead, BHead, BorrowIn> extends [infer RDigit extends Digit, infer BorrowOut extends 0 | 1]
      ? SubRev<ARest, BRest, BorrowOut, `${Acc}${RDigit}`>
      : never
    : SubDigit<AHead, "0", BorrowIn> extends [infer RDigit extends Digit, infer BorrowOut extends 0 | 1]
      ? SubRev<ARest, "", BorrowOut, `${Acc}${RDigit}`>
      : never
  : Acc;

/** `A - B` for non-negative integer digit strings. PRECONDITION `A >= B` —
 * never checked here; every call site below establishes it via `Compare`
 * first (mirrors the "callers establish preconditions" idiom already used
 * for e.g. `OkShape` in ndarray.ts). */
type MultiSub<A extends string, B extends string> = StripLeadingZeros<ReverseStr<SubRev<ReverseStr<A>, ReverseStr<B>, 0>>>;

/** Length comparison first (peels one char off each per step; a shorter
 * digit string is always numerically smaller since neither side ever has
 * leading zeros — both come from `${literalNumber}`). */
type LenCompare<A extends string, B extends string> = A extends `${infer _AHead}${infer ARest}`
  ? B extends `${infer _BHead}${infer BRest}`
    ? LenCompare<ARest, BRest>
    : "gt"
  : B extends `${infer _BHead}${infer BRest}`
    ? "lt"
    : "eq";

/** All `"AB"` pairs where digit A > digit B — a fixed, bounded 45-entry
 * lookup (10 digits -> 45 unordered pairs), not proportional to any dim
 * value. */
type DigitGtPairs =
  | "10"
  | "20"
  | "21"
  | "30"
  | "31"
  | "32"
  | "40"
  | "41"
  | "42"
  | "43"
  | "50"
  | "51"
  | "52"
  | "53"
  | "54"
  | "60"
  | "61"
  | "62"
  | "63"
  | "64"
  | "65"
  | "70"
  | "71"
  | "72"
  | "73"
  | "74"
  | "75"
  | "76"
  | "80"
  | "81"
  | "82"
  | "83"
  | "84"
  | "85"
  | "86"
  | "87"
  | "90"
  | "91"
  | "92"
  | "93"
  | "94"
  | "95"
  | "96"
  | "97"
  | "98";

type DigitCompare<A extends Digit, B extends Digit> = A extends B ? "eq" : `${A}${B}` extends DigitGtPairs ? "gt" : "lt";

type LexCompareSameLen<A extends string, B extends string> = A extends `${infer AHead extends Digit}${infer ARest}`
  ? B extends `${infer BHead extends Digit}${infer BRest}`
    ? DigitCompare<AHead, BHead> extends "eq"
      ? LexCompareSameLen<ARest, BRest>
      : DigitCompare<AHead, BHead>
    : never
  : "eq";

/** Compare two non-negative integer digit strings: length first, then
 * lexicographic (only meaningful/invoked once lengths are equal). */
export type Compare<A extends string, B extends string> = LenCompare<A, B> extends infer LC
  ? LC extends "eq"
    ? LexCompareSameLen<A, B>
    : LC
  : never;

type Min<A extends string, B extends string> = Compare<A, B> extends "gt" ? B : A;

/** A literal, non-negative integer's digit string, or the sentinel
 * `"unsupported"` (never a valid `${T}` for any actual number, so this is
 * an unambiguous failure signal — deliberately NOT `never`, which would
 * satisfy every subsequent `extends` check and silently "succeed" wrong;
 * this is the standard `never`-always-matches gotcha, confirmed while
 * prototyping this file, and the reason for the explicit sentinel). */
export type NonNegDigits<T> = [T] extends [number]
  ? IsDynamicDim<T> extends true
    ? "unsupported"
    : IsPlainDigits<`${T}`> extends true
      ? `${T}`
      : "unsupported" // negative / non-integer literal — out of the supported subset
  : "unsupported"; // not a number at all (shouldn't happen once callers pre-check, defensive)

/**
 * Resolve one literal `start`/`stop` boundary against dim digit string `DS`,
 * or the sentinel `"unsupported"` — shared by `ResolveStart`/`ResolveStop`
 * (see their thin wrappers below). Boundary-filters `[T] extends [never]`
 * and `IsUnion<T>` FIRST, before any digit-string form is attempted (Spike
 * 04's rule, reused: a union/never input degrades the WHOLE computation
 * uniformly, rather than letting it distribute through the arithmetic
 * below unaudited — Spike 06's deliberate behavior sharpening of the
 * Kern-05-era code, pinned by a test). `IsDynamicDim<T>` is checked before
 * any `` `${T}` `` template destructuring (same defensive ordering
 * `NonNegDigits` already uses): a wide, non-literal `number`'s template
 * form can't be walked character-by-character the way `IsPlainDigits`/the
 * `` `-${infer Abs}` `` pattern need to.
 *
 * Spike 06's negative branch (`` `${T}` extends `-${infer Abs}` ``): the
 * runtime normalizes a negative `start`/`stop` via `v += d`, then clamps to
 * `[0, d]`. For a negative literal `v = -|v|`, `d + v = d - |v|` — needing
 * only a COMPARISON (`|v| > d` clamps to `0`, i.e. `v` was past the front
 * even after normalizing) plus the existing UNSIGNED `MultiSub` (`d - |v|`
 * when `|v| <= d`) — never a signed add. The non-negative branch reuses
 * `NonNegDigits` unchanged, adding only the `Min` clamp against `DS` (the
 * existing Kern-05 behavior, now factored out to this shared helper).
 */
type ResolveBoundaryDigits<T, DS extends string, WhenUndefined extends string> = [T] extends [never]
  ? "unsupported"
  : [T] extends [undefined]
    ? WhenUndefined
    : IsUnion<T> extends true
      ? "unsupported"
      : [T] extends [number]
        ? IsDynamicDim<T> extends true
          ? "unsupported"
          : `${T}` extends `-${infer Abs}`
            ? IsPlainDigits<Abs> extends true
              ? Compare<Abs, DS> extends "gt"
                ? "0" // |v| > d: past even v = -d, clamps to the front
                : MultiSub<DS, Abs> // d - |v| (== v + d, the runtime's own normalization)
              : "unsupported" // "-1.5" (dot-form), "-1e21" (exponent switchover — NOTE `${-1e5}` renders as plain "-100000" and IS computed; only magnitudes >= 1e21 render with an `e`)
            : NonNegDigits<T> extends infer TD extends string
              ? TD extends "unsupported"
                ? "unsupported"
                : Min<TD, DS> // existing Kern-05 non-negative clamp, unchanged
              : "unsupported"
        : "unsupported"; // not a number at all (defensive; callers pre-filter)

type ResolveStart<StartT, DS extends string> = ResolveBoundaryDigits<StartT, DS, "0">;
type ResolveStop<StopT, DS extends string> = ResolveBoundaryDigits<StopT, DS, DS>;

/**
 * Classify a literal `step` for the DIM-COMPUTATION path (the separate,
 * exported `LiteralStepInvalid<Spec>` below classifies it for the COMPILE-
 * ERROR guard instead — two different questions about the same value:
 * "can I compute a number from this" vs. "is this guaranteed to throw").
 * `"fast"` for omitted/literal `1` (the existing, byte-identical Kern-05
 * fast path — no division needed); a plain-digit string `>= "2"` for the
 * `DivCeil` path; `"unsupported"` for everything else — a literal `0` or
 * negative step (provably invalid: the GUARD flags these, computation just
 * honestly has nothing to compute), a non-integer/exponent-form/wide/union
 * step (no claim either way), or `never`. Same boundary-filter ordering as
 * `ResolveBoundaryDigits` (never, then the fast-path shortcut, then
 * `IsUnion`, then digit-string forms).
 */
type ResolveStepDigits<StepT> = [StepT] extends [never]
  ? "unsupported"
  : [StepT] extends [undefined | 1]
    ? "fast"
    : IsUnion<StepT> extends true
      ? "unsupported"
      : [StepT] extends [number]
        ? IsDynamicDim<StepT> extends true
          ? "unsupported"
          : `${StepT}` extends "0" | "1"
            ? "unsupported" // "1" is unreachable here (already fast-pathed above); "0": the guard's job
            : IsPlainDigits<`${StepT}`> extends true
              ? `${StepT}`
              : "unsupported" // negative / non-integer / exponent-form
        : "unsupported";

/** `start' < stop' ? (step == "fast" ? stop' - start' : ceil((stop' -
 * start') / step)) : 0` — the runtime's own `normalizeAxisSpec` formula
 * (both its step=1 fast path AND its general ceil-division formula), at the
 * type level. `StartS`/`StopS` are already fully resolved (clamped,
 * negative-normalized) by `ResolveStart`/`ResolveStop`; `StepR` is
 * `ResolveStepDigits`'s classification — `"fast"` or a plain-digit string
 * `>= "2"`, NEVER `"unsupported"` (the caller filters that out first). The
 * `"fast"` branch is exactly the original Kern-05 instantiation chain
 * (`Compare` then `MultiSub`, nothing else threaded through) — byte-
 * identical results, pinned by the untouched ST1-ST9 tests. */
type ComputeRangeDigits<StartS extends string, StopS extends string, StepR extends string> = Compare<StartS, StopS> extends "lt"
  ? MultiSub<StopS, StartS> extends infer Diff extends string
    ? StepR extends "fast"
      ? Diff
      : DivCeil<Diff, StepR>
    : never
  : "0";

// Extraction via a REQUIRED-property pattern (`{ readonly start: infer T }`,
// not `start?: infer T`): destructuring an OPTIONAL property against a
// concrete type that structurally lacks it infers `unknown` (there is no
// positional information to infer FROM) — NOT `undefined` — confirmed
// empirically while prototyping this file. A required-property pattern
// correctly fails the whole `extends` check when the property is absent,
// so the outer conditional's `: undefined` branch fires instead — the
// deliberately roundabout way to get `undefined` for "genuinely absent".
type ExtractStart<Spec> = Spec extends { readonly start: infer StartT } ? StartT : undefined;
type ExtractStop<Spec> = Spec extends { readonly stop: infer StopT } ? StopT : undefined;
/** Exported (Spike 06): `slice.ts`'s step-invalid guard message needs the
 * step's own literal VALUE to interpolate (`` `slice: step ${step} ...` ``,
 * mirroring the runtime's message verbatim) — the same extraction this
 * file already needed internally for `ResolveStepDigits`. */
export type ExtractStep<Spec> = Spec extends { readonly step: infer StepT } ? StepT : undefined;

/**
 * The stretch: a literal dim for a range spec, when `start`/`stop` are
 * literal integers (or omitted; non-negative OR — since Spike 06 —
 * negative) and `step` is omitted, the literal `1`, or — since Spike 06 —
 * a literal plain-digit integer `>= 2`. Anything outside that subset (see
 * file header) honestly degrades to `Dim` (`number`) — the exact same
 * fallback the core rule already uses, so dropping either the Kern-05
 * stretch or the Spike-06 extension is always a safe, behavior-preserving
 * revert.
 *
 * `Spec` is intentionally UNCONSTRAINED (not `extends RangeSpecLike`): the
 * call site in `slice.ts` passes its already-inferred `Head` — narrowed
 * in-line by two preceding `extends null` / `extends number` checks, which
 * TS does not propagate as a narrowed CONSTRAINT on that same named type
 * variable for a later generic argument position (confirmed empirically:
 * passing it to a `RangeSpecLike`-constrained parameter fails with
 * "`SliceSpecInput` is not assignable" even though `null`/`number` are
 * already excluded by that point). `ExtractStart`/`ExtractStop`/`ExtractStep`
 * below all resolve to `undefined` for a `Spec` that isn't shaped like a
 * range object anyway (including `null` or `number`), so relaxing the
 * constraint costs nothing: they degrade to `Dim` exactly as if `Spec` had
 * been the empty range object `{}`.
 */
export type LiteralRangeDim<Spec, D extends Dim> = IsDynamicDim<D> extends true
  ? Dim
  : ResolveStepDigits<ExtractStep<Spec>> extends infer StepR extends string
    ? StepR extends "unsupported"
      ? Dim
      : ResolveStart<ExtractStart<Spec>, `${D}`> extends infer StartS extends string
        ? StartS extends "unsupported"
          ? Dim
          : ResolveStop<ExtractStop<Spec>, `${D}`> extends infer StopS extends string
            ? StopS extends "unsupported"
              ? Dim
              : ComputeRangeDigits<StartS, StopS, StepR> extends infer DimDigits extends string
                ? DimDigits extends `${infer N extends number}`
                  ? N
                  : Dim
                : Dim
            : Dim
        : Dim
    : Dim;

// ---------------------------------------------------------------------------
// Spike 03 (docs/spike-03-index-bounds-spec.md): bounds classification for a
// literal integer INDEX spec against a literal dim. Appended below all
// pre-existing Kern-05 content; needs only `Compare` (a bounds check is a
// comparison — the signed ADDITION that kept negative start/stop out of the
// Kern-05 stretch is not required here, which is why negative literal
// INDICES are fully supported by this check even though negative literal
// start/stop still degrade in `LiteralRangeDim` above).
// ---------------------------------------------------------------------------

/**
 * Classify literal index `I` against dim `D`:
 *  - `"in"`  — provably in bounds (`0 <= i < d` after NumPy negative
 *    normalization `i < 0 -> i + d`, i.e. positive `i < d`, negative
 *    `abs(i) <= d`). The runtime (`normalizeAxisSpec` in runtime.ts) will
 *    not throw.
 *  - `"out"` — provably out of bounds; the runtime is GUARANTEED to throw
 *    `slice: index i is out of bounds ...`. The caller (slice.ts's
 *    `ValidateSpecsAcc`) turns exactly this verdict into a compile error at
 *    the offending argument.
 *  - `"unknown"` — no static claim (dynamic dim, wide `number` index, union
 *    literals, non-plain-digit literal forms like `1.5` or `1e21`). The
 *    runtime backstop stays authoritative — this type must be NEVER WRONG,
 *    only incomplete, so every ambiguous input lands here.
 *
 * Union safety (the reason for the `[C] extends [...]`-wrapped verdict
 * checks): `Compare` distributes over union digit strings (its parameters
 * appear naked in its own conditionals), so e.g. `I = 2 | 7` against `d = 5`
 * yields `C = "lt" | "gt"`. The tuple-wrapped SUBSET checks below accept a
 * (possibly-union) verdict only when EVERY member classifies the same way —
 * `[C] extends [("eq" | "gt")]` is true for `"eq"`, `"gt"`, and
 * `"eq" | "gt"` (all genuinely out of bounds) but false for anything
 * containing `"lt"`. A mixed verdict falls through to `"unknown"` instead
 * of picking a side.
 */
export type LiteralIndexBounds<I, D extends Dim> = IsDynamicDim<D> extends true
  ? "unknown"
  : [I] extends [number]
    ? IsDynamicDim<I> extends true
      ? "unknown"
      : `${I}` extends `-${infer Abs}`
        ? IsPlainDigits<Abs> extends true
          ? Compare<Abs, `${D}`> extends infer C
            ? [C] extends [("lt" | "eq")]
              ? "in" // abs(i) <= d (i = -d normalizes to index 0, still valid)
              : [C] extends ["gt"]
                ? "out" // abs(i) > d: past even i = -d
                : "unknown" // mixed union verdict (e.g. I = -1 | -9 on d = 5): no claim
            : never
          : "unknown" // "-1.5" and friends: not a plain-digit negative integer
        : IsPlainDigits<`${I}`> extends true
          ? Compare<`${I}`, `${D}`> extends infer C
            ? [C] extends ["lt"]
              ? "in" // i < d
              : [C] extends [("eq" | "gt")]
                ? "out" // i >= d (includes every literal index against d = 0)
                : "unknown" // mixed union verdict (e.g. I = 2 | 7 on d = 5): no claim
            : never
          : "unknown" // "1.5", "1e+21", mixed unions like "-1" | "2": no claim
    : "unknown"; // not a number at all (defensive; callers pre-filter)

// ---------------------------------------------------------------------------
// Spike 04 (docs/spike-04-shape-products-spec.md): the literal PRODUCT of a
// shape's dims — reshape/flatten's Phase-B enabler. Appended below all
// pre-existing content; needs `Compare`, `StripLeadingZeros`, `ReverseStr`,
// `IsPlainDigits`/`NonNegDigits`, `Digit`/`DigitValue`/`Tup`/`TupDigit` — no
// duplication of anything above. Unlike Spike 03 (a bounded finite verdict
// alphabet, safely union-tolerant via subset checks), a product is an
// UNBOUNDED value with no such alphabet, so this section's arithmetic is
// UNION-FREE BY CONSTRUCTION instead: any union (or `never`) dim is filtered
// out at the shape-walk boundary, before a single digit string is built, and
// no type below ever receives a naked union as `A`/`B`. Addition is needed
// here (Kern 05's stretch got away with subtraction+comparison only) because
// schoolbook multiplication is repeated shifted addition.
// ---------------------------------------------------------------------------

/** Add two digits with an incoming carry (0 or 1): `[resultDigit,
 * carryOut]` — the carry mirror of `SubDigit`. Built by concatenating three
 * bounded tuples (length <= 9+9+1 = 19: both digits plus an optional
 * carry-in unit) and peeling `Tup<10>` off once if that overflows — never a
 * 100-entry digit+digit lookup table and never proportional to a dim
 * value. */
type AddDigit<A extends Digit, B extends Digit, CarryIn extends 0 | 1> = [
  ...Tup<DigitValue<A>>,
  ...Tup<DigitValue<B>>,
  ...(CarryIn extends 1 ? Tup<1> : []),
] extends infer Sum extends readonly unknown[]
  ? Sum extends [...Tup<10>, ...infer Rest extends readonly unknown[]]
    ? [TupDigit<Rest>, 1]
    : [TupDigit<Sum>, 0]
  : never;

/** Multi-digit addition, digit-by-digit from the LEAST significant end (via
 * `ReverseStr`), threading a carry — `SubRev`'s addition mirror, but
 * SYMMETRIC in both operands: `SubRev` only ever needs to handle `BRev`
 * running out first, because its one caller (`MultiSub`) establishes
 * `A >= B` in LENGTH via `Compare` before calling it. `AddRev` has no such
 * precondition — the schoolbook accumulator below (`MulAccRev`) adds a
 * short shifted partial product into a long running total, and later a long
 * one into a short total, in either order, so both exhaustion orders must
 * be handled. A carry surviving past BOTH operands exhausting APPENDS one
 * more (highest) digit — addition's one failure mode subtraction never has
 * under its own precondition (e.g. `999 + 1 = 1000`: one digit longer than
 * either operand; `A - B` with `A >= B` can never grow past `A`'s length). */
type AddRev<ARev extends string, BRev extends string, CarryIn extends 0 | 1, Acc extends string = ""> = ARev extends `${infer AHead extends Digit}${infer ARest}`
  ? BRev extends `${infer BHead extends Digit}${infer BRest}`
    ? AddDigit<AHead, BHead, CarryIn> extends [infer RDigit extends Digit, infer CarryOut extends 0 | 1]
      ? AddRev<ARest, BRest, CarryOut, `${Acc}${RDigit}`>
      : never
    : AddDigit<AHead, "0", CarryIn> extends [infer RDigit extends Digit, infer CarryOut extends 0 | 1]
      ? AddRev<ARest, "", CarryOut, `${Acc}${RDigit}`>
      : never
  : BRev extends `${infer BHead extends Digit}${infer BRest}`
    ? AddDigit<"0", BHead, CarryIn> extends [infer RDigit extends Digit, infer CarryOut extends 0 | 1]
      ? AddRev<"", BRest, CarryOut, `${Acc}${RDigit}`>
      : never
    : CarryIn extends 1
      ? `${Acc}1` // both exhausted, carry survives: one more (highest) digit
      : Acc;

/** `Countdown.length` copies of `BTup`, concatenated: a tuple of length
 * `Countdown.length * BTup.length` — bounded to <= 9*9 = 81 at every call
 * site below (both operands are single digits 0-9). Recursion depth <= 9
 * (peels one `Countdown` element per step). This is the "repeated concat"
 * `MulAdd` needs to get a digit product, kept OUT of a 100-entry digit x
 * digit lookup table and never proportional to a dim VALUE (only ever to
 * bounded 0-9 digit values). */
type MulTup<Countdown extends readonly unknown[], BTup extends readonly unknown[], Acc extends readonly unknown[] = []> = Countdown extends [unknown, ...infer Rest]
  ? MulTup<Rest, BTup, [...Acc, ...BTup]>
  : Acc;

/** Split a bounded (<=89-element) tuple into `[remainderDigit,
 * quotientDigit]` by peeling `Tup<10>` off repeatedly — <= 8 peels, since
 * `floor(89/10) = 8` is the largest possible quotient any call site below
 * produces (see `MulAdd`'s comment for where 89 comes from). The quotient is
 * tracked as the accumulated peel-count's tuple LENGTH and converted via
 * `TupDigit` at the end — the same idiom `SubDigit`'s borrow branch already
 * uses at a single-peel scale, just iterated here. */
type DivMod10<T extends readonly unknown[], QAcc extends readonly unknown[] = []> = T extends [...Tup<10>, ...infer Rest extends readonly unknown[]]
  ? DivMod10<Rest, [...QAcc, unknown]>
  : [TupDigit<T>, TupDigit<QAcc>];

/** Digit x digit + a carry-in DIGIT (not just 0|1, unlike `AddDigit`'s
 * carry) -> `[resultDigit, carryOutDigit]`. Value bound: `9*9 + 8 = 89` —
 * the `+8` is the largest carry a PRIOR `MulAdd` step in the same
 * digit-times-digit chain (`MulDigitRev`) can hand off (`DivMod10`'s own
 * comment explains why 8, not 9, is the ceiling — the bound is
 * self-consistent: a chain of `MulAdd` calls can never produce a carry above
 * 8 into the NEXT call either). Built from the two bounded tuple helpers
 * above — never a 100-entry digit x digit table. */
type MulAdd<A extends Digit, B extends Digit, CarryIn extends Digit> = DivMod10<[...MulTup<Tup<DigitValue<A>>, Tup<DigitValue<B>>>, ...Tup<DigitValue<CarryIn>>]>;

/** `ARev` (a multi-digit number, reversed) times the single digit `B`,
 * threading the carry digit-position to digit-position via `MulAdd`. A
 * surviving final carry (nonzero once `ARev` is exhausted) becomes one more
 * (highest) digit — the schoolbook "carry the tens into a new column"
 * step. */
type MulDigitRev<ARev extends string, B extends Digit, CarryIn extends Digit = "0", Acc extends string = ""> = ARev extends `${infer AHead extends Digit}${infer ARest}`
  ? MulAdd<AHead, B, CarryIn> extends [infer RDigit extends Digit, infer CarryOut extends Digit]
    ? MulDigitRev<ARest, B, CarryOut, `${Acc}${RDigit}`>
    : never
  : CarryIn extends "0"
    ? Acc
    : `${Acc}${CarryIn}`;

/** Schoolbook long multiplication, tail-recursive over `B`'s digits
 * (`BRev`, peeled LEAST-significant first — the natural order to walk a
 * reversed string, and the order the shift amount grows in). `Shift` is a
 * string of zeros tracking the current column: moving the next partial
 * product one column further left (schoolbook: the tens' place, then
 * hundreds', ...) means appending one more zero in NORMAL (un-reversed)
 * terms — which is PREFIXING one more zero in REVERSED terms. Hence
 * `` `0${Shift}` `` grows as a prefix here, not a suffix — the "shift-left =
 * prefix zeros in reversed space" the binding spec calls for. `AccRev` (the
 * running total) is threaded in the SAME reversed form `AddRev` both
 * produces and consumes, so no per-step reverse/un-reverse round-trip is
 * needed — only `MulDigits` below converts back to normal form once, at the
 * very end. */
type MulAccRev<ARev extends string, BRev extends string, Shift extends string = "", AccRev extends string = "0"> = BRev extends `${infer BHead extends Digit}${infer BRest}`
  ? MulAccRev<ARev, BRest, `0${Shift}`, AddRev<AccRev, `${Shift}${MulDigitRev<ARev, BHead>}`, 0>>
  : AccRev;

/** `A * B` for non-negative integer digit strings: reverse both operands for
 * the least-significant-first walk `MulAccRev` needs, then reverse its
 * (also-reversed) result back. `StripLeadingZeros` is MANDATORY here, not
 * merely tidy: a `x 0` schoolbook path produces a string of all-zero digits
 * at full operand length (e.g. `"999" * "0"` walks out as `"000"`, not
 * `"0"`) — without stripping, a downstream
 * `` `${infer N extends number}` `` template-literal parse would simply
 * fail to match (leading zeros are not a valid `number` literal's string
 * form), silently degrading a perfectly decidable `0` product to `number`. */
type MulDigits<A extends string, B extends string> = StripLeadingZeros<ReverseStr<MulAccRev<ReverseStr<A>, ReverseStr<B>>>>;

/** Standard "is `T` a (2+-member) union" probe. `U` defaults to (and is
 * bound ONCE, undistributed, to the whole of) `T`; the naked `T extends
 * unknown` distributes over `T`'s members when `T` IS a union, giving each
 * member a chance to fail `[U] extends [T]` — true only when `T`'s single
 * member (not the whole original union) equals `U`, which is impossible
 * unless `T` had exactly one member to begin with. `[T] extends [never]` is
 * a separate, necessary special case FIRST: `never` is the empty union, so
 * `T extends unknown` for `T = never` distributes over ZERO members and
 * evaluates to `never` itself, never reaching a `true`/`false` verdict.
 * (`ProductAcc` below additionally guards `never` dims itself — NOT for this
 * helper's sake, which answers `false` for `never` via this branch, but
 * because `never` would otherwise sail on THROUGH `NonNegDigits`: `never`
 * satisfies every `extends` check, so `` IsPlainDigits<`${never}`> extends
 * true `` matches and hands back `` `${never}` `` = `never`, which then
 * propagates through the whole product as `never` instead of an honest
 * `number` degrade — the standard never-always-matches gotcha, empirically
 * confirmed by ablation while building this block.) */
export type IsUnion<T, U = T> = [T] extends [never] ? false : T extends unknown ? ([U] extends [T] ? false : true) : never;

/**
 * The product of a fixed-rank shape's dims, decided per-dim in this exact
 * order (binding spec's semantics table): `never` dim, then union dim, then
 * "outside the supported literal subset" (dynamic dim, negative,
 * non-integer, exponent-form — all already classified as `NonNegDigits`'s
 * `"unsupported"` sentinel) — each an unconditional, EARLY-EXIT `number`
 * degrade. No arithmetic is attempted on the remaining dims once any one dim
 * degrades: there is no partial literal to report for a shape with one
 * unknown axis. Every OTHER dim multiplies into a running digit-string
 * accumulator (`Acc`, starting at `"1"`, the empty product) via
 * `MulDigits` — union-free BY CONSTRUCTION, since a naked union dim can
 * never reach that branch.
 *
 * At exhaustion, the exact product is reportable as a literal only if it is
 * a SAFE integer (`<= Number.MAX_SAFE_INTEGER = 9_007_199_254_740_991`):
 * round-tripping a LARGER digit string through
 * `` `${infer N extends number}` `` would double-round through `number`'s
 * float64 representation, silently returning a WRONG literal — the worst
 * failure mode for this project's whole USP. This is a plain
 * `Number.isSafeInteger`-style boundary (strictly GREATER than the cap
 * degrades; `== the cap` itself is still exact and stays a literal), not a
 * representability special case. Note the cap check happens ONLY here, at
 * the very end: an intermediate product (e.g. two 16-digit dims multiplied
 * before a LATER `0` dim zeroes the whole shape out) may transiently exceed
 * the cap and is never truncated or degraded early — digit strings are
 * exact at any length, so only the FINAL value is judged against the cap.
 */
type ProductAcc<S extends Shape, Acc extends string = "1"> = S extends readonly [infer Head extends Dim, ...infer Rest extends Shape]
  ? [Head] extends [never]
    ? number
    : IsUnion<Head> extends true
      ? number
      : NonNegDigits<Head> extends infer HeadDigits extends string
        ? HeadDigits extends "unsupported"
          ? number
          : ProductAcc<Rest, MulDigits<Acc, HeadDigits>>
        : number // defensive: NonNegDigits<Dim> always yields a string
  : Compare<Acc, "9007199254740991"> extends "gt"
    ? number
    : Acc extends `${infer N extends number}`
      ? N
      : number; // defensive: Acc is always plain digits by construction

/**
 * The product of a shape's literal dims (reshape/flatten's Phase-B enabler,
 * docs/spike-04-shape-products-spec.md). Dynamic RANK is checked FIRST,
 * before any tuple recursion — the same ordering `SliceShape` elsewhere in
 * this codebase already uses: a rank-unknown shape (`number[]`, a variadic
 * tail) cannot be walked dim-by-dim at all, so it degrades WHOLLY, never
 * attempting to read a fixed prefix. A union of FIXED shapes (e.g.
 * `[2,3] | [4]`) is deliberately NOT filtered here — `S` is a naked type
 * parameter inside `ProductAcc`'s own recursive `S extends [Head,
 * ...Rest]` check, so it distributes there automatically once `ProductAcc`
 * is reached, running the whole pipeline independently per union member and
 * yielding the union of results (`6 | 4`); this is standard TS
 * conditional-type distribution, not special-cased code.
 */
export type LiteralShapeProduct<S extends Shape> = IsDynamicRank<S> extends true ? number : ProductAcc<S>;

// ---------------------------------------------------------------------------
// Spike 06 (docs/spike-06-range-literals-spec.md): negative literal
// start/stop (wired into `LiteralRangeDim`'s `ResolveBoundaryDigits` above —
// no new arithmetic needed there beyond `Compare` + the existing `MultiSub`)
// and literal steps >= 1 for range slices. Appended below all pre-existing
// content, needing only two genuinely NEW arithmetic primitives — `MultiAdd`
// (the `+1` in "ceil = floor + 1 iff remainder != 0") and `DivCeil`
// (schoolbook long division, for `ceil((stop-start)/step)`) — plus one new
// exported classifier, `LiteralStepInvalid`, for the compile-time guard
// `slice.ts` wires up (the Spike-03 idiom: a provably-invalid literal
// retypes its own argument to a branded error). No duplication of anything
// above: reuses `Compare`, `MultiSub`, `StripLeadingZeros`, `ReverseStr`,
// `IsPlainDigits`, `Digit`, and Spike-04's `AddRev`/`MulDigits`/`IsUnion`.
// ---------------------------------------------------------------------------

/** `A + B` for non-negative integer digit strings — the addition mirror of
 * `MultiSub`: both digit-by-digit from the LEAST significant end via
 * `ReverseStr`/`AddRev` (Spike 04's, no precondition on relative magnitude,
 * unlike `MultiSub`'s `A >= B`), converting back to normal
 * (most-significant-first) form at the end. `StripLeadingZeros` is kept for
 * consistency with every other public arithmetic export here, even though
 * this file's own call site (`DivCeil`'s `+1`) never actually produces a
 * leading zero (its LHS is already a stripped quotient digit string). */
type MultiAdd<A extends string, B extends string> = StripLeadingZeros<ReverseStr<AddRev<ReverseStr<A>, ReverseStr<B>, 0>>>;

/** Does `MulDigits(B, Q) <= Rem`? One bounded trial (a single `Compare`
 * plus one digit-by-single-digit `MulDigits` call) for `FindQuotientDigit`'s
 * descending scan. */
type TryQuotientDigit<B extends string, Rem extends string, Q extends Digit> = Compare<MulDigits<B, Q>, Rem> extends "gt" ? false : true;

/** The largest quotient digit `q` in `0..9` with `MulDigits(B, q) <= Rem` —
 * schoolbook long division's per-position step. A plain descending chain of
 * <= 10 bounded trials (`TryQuotientDigit`), not a search structure: `q = 0`
 * always terminates the chain (`MulDigits(B, "0") = "0" <= Rem` for any
 * non-negative `Rem`), so this never falls through to `never`. */
type FindQuotientDigit<B extends string, Rem extends string> = TryQuotientDigit<B, Rem, "9"> extends true
  ? "9"
  : TryQuotientDigit<B, Rem, "8"> extends true
    ? "8"
    : TryQuotientDigit<B, Rem, "7"> extends true
      ? "7"
      : TryQuotientDigit<B, Rem, "6"> extends true
        ? "6"
        : TryQuotientDigit<B, Rem, "5"> extends true
          ? "5"
          : TryQuotientDigit<B, Rem, "4"> extends true
            ? "4"
            : TryQuotientDigit<B, Rem, "3"> extends true
              ? "3"
              : TryQuotientDigit<B, Rem, "2"> extends true
                ? "2"
                : TryQuotientDigit<B, Rem, "1"> extends true
                  ? "1"
                  : "0";

/** Schoolbook long division's accumulator: walks `A`'s digits MOST-
 * significant-first (`ARest`, peeled directly off the front — `A` is
 * already in normal form, so unlike `SubRev`/`AddRev`/`MulAccRev` this walk
 * needs NO `ReverseStr` at all), threading the running remainder `Rem` and
 * the quotient-so-far `QAcc` (built by APPENDING each new quotient digit at
 * the end, which — since digits are produced in the same most-significant-
 * first order they're consumed — naturally yields the quotient in normal
 * form with no reversal step, unlike every accumulator elsewhere in this
 * file). At each position: bring down the next digit of `A` into the
 * remainder (`` `${Rem}${Head}` ``, leading-zero-stripped — `"0"` bringing
 * down into remainder `"0"` must collapse back to `"0"`, not `"00"`, or the
 * next `Compare` sees a spuriously "longer" remainder), find that
 * position's quotient digit, subtract its contribution, and recurse. Once
 * `A` is exhausted, returns `[quotient digits so far, final remainder]` —
 * `DivCeil` below strips the quotient's own leading zeros (an all-zero
 * numerator like `A = "0"` walks out as `"0"` already; a numerator smaller
 * than `B` walks out with leading zeros in `QAcc`, e.g. `"03"` for `3 / 7`). */
type DivCeilAcc<ARest extends string, B extends string, Rem extends string, QAcc extends string> = ARest extends `${infer Head extends Digit}${infer Tail}`
  ? StripLeadingZeros<`${Rem}${Head}`> extends infer NewRem extends string
    ? FindQuotientDigit<B, NewRem> extends infer Q extends Digit
      ? DivCeilAcc<Tail, B, MultiSub<NewRem, MulDigits<B, Q>>, `${QAcc}${Q}`>
      : never
    : never
  : readonly [QAcc, Rem];

/** `ceil(A / B)` for non-negative integer digit strings, via schoolbook long
 * division (`DivCeilAcc`) then `+1` iff the floor-division remainder is
 * nonzero. PRECONDITION `B >= "1"` (never checked here — every call site in
 * `ComputeRangeDigits` above passes a `StepR` already classified as a
 * plain-digit integer `>= "2"` by `ResolveStepDigits`, so this bound holds
 * with margin; mirrors the "callers establish preconditions" idiom `MultiSub`
 * already uses for `A >= B`). Cost: O(digits(A) x digits(B)) — <= 10 bounded
 * trials (`FindQuotientDigit`) per digit of `A`, each trial one
 * `MulDigits(B, q)` call costing O(digits(B)). */
type DivCeil<A extends string, B extends string> = DivCeilAcc<A, B, "", ""> extends readonly [infer Q extends string, infer R extends string]
  ? StripLeadingZeros<Q> extends infer QS extends string
    ? R extends "0"
      ? QS
      : MultiAdd<QS, "1">
    : never
  : never;

/** Does `S`'s literal template form contain the character `C` anywhere?
 * (`` `${string}${C}${string}` `` matches iff SOME split of `S` has `C` at
 * the split point — the standard template-literal "substring contains"
 * idiom, distinct from `IsPlainDigits`'s character-by-character walk.) */
type ContainsChar<S extends string, C extends string> = S extends `${string}${C}${string}` ? true : false;

/** Does `S` — a literal step's `` `${T}` `` form — PROVE a non-integer via
 * the "dot-form" pattern: contains `.` and does NOT contain `e`. An
 * integer literal's template form never renders with a decimal point, so
 * `.` alone would already be a safe signal — the `no e` half specifically
 * protects `1e21`-style forms: for `T` large enough that `${T}` renders in
 * JS's own scientific notation (`>= 1e21`, mirroring `Number.prototype
 * .toString`'s own switchover), the rendered form contains `e` but never a
 * literal `.` for an actual integer value, so this predicate already
 * answers `false` for those and never needs to inspect the exponent's own
 * digits — but the explicit `no e` guard future-proofs against any
 * mixed dot+exponent form (`1.5e10`) that a re-derivation might introduce,
 * keeping the "never wrong" guarantee independent of exactly how TS
 * canonicalizes a given numeric literal's string form. */
export type IsDotFormStep<S extends string> = ContainsChar<S, "."> extends true ? (ContainsChar<S, "e"> extends true ? false : true) : false;

/**
 * Classify a literal `step` for the COMPILE-TIME GUARD (`slice.ts`'s
 * `ValidateSpecsAcc`, the Spike-03 idiom): `"invalid"` ONLY for a step the
 * runtime is GUARANTEED to throw on via ONE of three provable forms —
 * plain-digit `0`, negative plain-digit (`` `-${digits}` ``), or dot-form
 * non-integer (`1.5`) — mirroring `normalizeAxisSpec`'s own check
 * (`!Number.isInteger(step) || step < 1`) exactly on this provable subset.
 * `"unknown"` for EVERYTHING else, including: omitted/valid steps (`1`, `2`,
 * ...  — never "invalid", obviously, but also not specially marked here,
 * since the guard only ever checks for the `"invalid"` verdict); a wide
 * `number`; exponent-form (`1e21` — `${1e21}` renders as `"1e+21"` in TS,
 * same as JS's own `Number.prototype.toString` switchover for magnitudes
 * `>= 1e21` — IS a valid integer step, so flagging it would LIE); and any
 * union step (boundary-filtered like every other union input in this file
 * — even a UNIFORMLY-invalid union like `0 | -1` still classifies as
 * `"unknown"` here, a deliberate scope simplification per the binding spec:
 * unlike `LiteralIndexBounds`'s subset-check tolerance for uniform union
 * verdicts, this classifier reuses the Spike-04 boundary-FILTER style
 * instead, since the semantics table's own last row lists "union" among the
 * unconditional "no guard claim" categories for `step`). Never wrong, only
 * incomplete: the runtime backstop stays authoritative for every case this
 * returns `"unknown"` for.
 */
export type LiteralStepInvalid<Spec> = ExtractStep<Spec> extends infer StepT
  ? [StepT] extends [never]
    ? "unknown"
    : [StepT] extends [undefined]
      ? "unknown" // omitted: defaults to step 1, always valid
      : IsUnion<StepT> extends true
        ? "unknown" // boundary-filtered: no uniform claim, even if every member is invalid
        : [StepT] extends [number]
          ? IsDynamicDim<StepT> extends true
            ? "unknown" // wide `number`: no claim
            : `${StepT}` extends `-${infer Abs}`
              ? IsPlainDigits<Abs> extends true
                ? "invalid" // negative plain-digit integer step
                : IsDotFormStep<`${StepT}`> extends true
                  ? "invalid" // "-1.5": dot-form is a proven non-integer regardless of sign (review fix — the spec's dot-form row is sign-agnostic)
                  : "unknown" // "-1e21": negative AND integral in fact, but its template form is "-1e+21" — unprovable (verifier-corrected example: `${-1e5}` renders as plain "-100000" and IS provable/"invalid")
              : `${StepT}` extends "0"
                ? "invalid" // the literal 0
                : IsDotFormStep<`${StepT}`> extends true
                  ? "invalid" // e.g. "1.5"
                  : "unknown" // valid steps (1, 2, 3, ...) and exponent-form ("1e+21"): no claim, never wrong
          : "unknown"
  : "unknown";

// ---------------------------------------------------------------------------
// Kern 08 (docs/kern-08-reshape-flatten-spec.md): the reshape stretch — lift
// a PROVABLY invalid literal dim of a `reshape()` new shape to a compile
// error at the argument (the Spike-03/06 idiom, reusing this file's private
// classification primitives — `IsPlainDigits`, `IsDotFormStep`, `IsUnion` —
// rather than duplicating them in reshape.ts, which is why this classifier
// lives here instead). Appended below all pre-existing content.
//
// Unlike a slice `step` (where a literal `0` is itself invalid), `0` is a
// VALID dim here (size-0 shapes are first-class in this codebase) — so the
// non-negative branch below is simpler than `LiteralStepInvalid`'s: any
// plain-digit string (including "0") is a valid dim, only a DOT-FORM
// literal (proven non-integer) is provably invalid on that side.
// ---------------------------------------------------------------------------

/** Classify one literal dim `D` of a `reshape()` new shape: `"invalid"` for
 * a PROVABLY invalid literal (negative plain-digit integer, or a sign-
 * agnostic dot-form non-integer), `"unknown"` for everything else (valid
 * dims INCLUDING `0`, a wide `number`, exponent-form, union, or `never`) —
 * never wrong, only incomplete, same discipline as `LiteralStepInvalid`. */
type LiteralDimInvalid<D extends Dim> = [D] extends [never]
  ? "unknown"
  : IsUnion<D> extends true
    ? "unknown" // boundary-filtered: no uniform claim, even if every member is invalid
    : [D] extends [number]
      ? IsDynamicDim<D> extends true
        ? "unknown" // wide `number`: no claim
        : `${D}` extends `-${infer Abs}`
          ? IsPlainDigits<Abs> extends true
            ? "invalid" // negative plain-digit integer dim
            : IsDotFormStep<`${D}`> extends true
              ? "invalid" // "-1.5": dot-form is a proven non-integer regardless of sign
              : "unknown" // "-1e21": unprovable from its template form
          : IsDotFormStep<`${D}`> extends true
            ? "invalid" // e.g. "1.5"
            : "unknown" // any plain-digit dim (incl. "0") or exponent-form: no claim, never wrong
      : "unknown";

/** Tail-recursive accumulator: walks `NS`'s dims left to right (mirrors the
 * runtime's own per-axis loop in `assertReshapeArgs`, runtime.ts) and
 * returns the FIRST provably-invalid dim's OWN literal value, or the
 * sentinel `"ok"` if none is found — deliberately NOT `never` (the standard
 * "never always matches" gotcha this file already guards against
 * elsewhere, e.g. `NonNegDigits`'s own `"unsupported"` sentinel), and
 * deliberately NOT the full message string either: this file's own
 * historical precedent (`LiteralIndexBounds`/`LiteralStepInvalid` return
 * bare verdicts, never pre-built messages) keeps it free of a `ShowShape`
 * import, which the append-only freeze discipline for this file forbids
 * (that would touch the pre-existing import line) — message construction
 * happens one layer up, in `reshape.ts`, exactly where `slice.ts` already
 * builds `IndexOutOfBoundsMessage`/`StepInvalidMessage` from this file's
 * bare verdicts. */
type ReshapeDimInvalidAcc<NS extends Shape> = NS extends readonly [infer Head extends Dim, ...infer Tail extends Shape]
  ? LiteralDimInvalid<Head> extends "invalid"
    ? Head
    : ReshapeDimInvalidAcc<Tail>
  : "ok";

/**
 * Lift a provably-invalid literal dim of a `reshape()` new shape `NS` to
 * its OWN literal value, or the sentinel `"ok"` if no dim is provably
 * invalid — `reshape.ts`'s `ReshapeCheckWithStretch` checks this BEFORE the
 * product check (mirroring `assertReshapeArgs`'s own check order: dim
 * validity first, per-axis left to right, then product) and builds the
 * final message itself via `ShowShape<NS>`. A union of whole shapes for
 * `NS` distributes naturally through `NS extends readonly [Head, ...Tail]`
 * above (same "not special-cased" precedent as `DotCheckStatic` in
 * vector.ts) — the systemic whole-shape-union guard bypass is a documented,
 * shared FOLLOWUPS item, not specific to this classifier.
 */
export type LiteralReshapeDimInvalid<NS extends Shape> = ReshapeDimInvalidAcc<NS>;
