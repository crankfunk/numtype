/**
 * Kern 05 STRETCH GOAL: statically computed literal dims for range slices,
 * via from-scratch DIGIT-STRING arithmetic — NOT tuple-length arithmetic
 * over dim values (that would be O(n) tuple construction per dim, blowing
 * the ~1000 tail-recursion ceiling for any dim past a few hundred; see
 * docs/kern-05-slicing-spec.md and CLAUDE.md's TS-limits section). Cost axis
 * here is DIGIT COUNT: subtracting two 7-digit numbers costs ~7 recursion
 * steps, not ~1,000,000 — the same "rank/count is fine, value is not"
 * distinction the rest of this codebase already draws, just pushed one
 * level deeper (digits of a value, rather than the value itself).
 *
 * Isolated in its OWN file, deliberately: this is the gated, "allowed to
 * fail and be dropped" part of the phase (spec: go/no-go on
 * Instantiations <= 3x baseline, wall time <= 2x baseline, clean hovers,
 * errors still at the argument). `slice.ts` imports two symbols from here:
 * `LiteralRangeDim` (Kern 05 stretch — dropping it means reverting that one
 * call site back to the core's honest `Dim` degrade) and
 * `LiteralIndexBounds` (Spike 03 — dropping it means reverting
 * `ValidateSpecsAcc`'s bounds branch to an unconditional pass-through). No
 * other file depends on anything below.
 *
 * SUPPORTED LITERAL SUBSET (precise, by design — see file body for why):
 *  - `step` must be omitted or the literal `1`. Any other step (including a
 *    literal `!= 1`, or a wide `number`) degrades to `Dim`.
 *  - `start`/`stop`, if present, must be NON-NEGATIVE literal integers.
 *    Negative literals are NOT supported by the stretch (they would need a
 *    signed add — `start + d` — on top of the unsigned subtract/compare
 *    already implemented here; scoped out to keep the arithmetic surface
 *    smaller and the correctness bar easier to clear). A negative literal
 *    start/stop still compiles — it just degrades to `Dim`, identical to
 *    the core rule; this is an honest, gradual narrowing, not a hole.
 *  - The axis's own dim must itself be a literal (an already-dynamic `Dim`
 *    can never yield a computed literal here either way).
 *  - Any wide (non-literal) `start`/`stop` degrades to `Dim`.
 *
 * ALGORITHM (mirrors `normalizeAxisSpec`'s step=1 case in runtime.ts
 * exactly, just at the type level): `start' = min(start, d)`,
 * `stop' = min(stop, d)`, `dim = start' < stop' ? stop' - start' : 0`.
 * (No `+ d` for negative normalization — see the negative-literal scope note
 * above; only non-negative literals reach this arithmetic at all.)
 */

import { type Dim, type IsDynamicDim } from "./dim.ts";

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
type Compare<A extends string, B extends string> = LenCompare<A, B> extends infer LC
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
type NonNegDigits<T> = [T] extends [number]
  ? IsDynamicDim<T> extends true
    ? "unsupported"
    : IsPlainDigits<`${T}`> extends true
      ? `${T}`
      : "unsupported" // negative / non-integer literal — out of the supported subset
  : "unsupported"; // not a number at all (shouldn't happen once callers pre-check, defensive)

type ResolveStart<StartT> = [StartT] extends [undefined] ? "0" : NonNegDigits<StartT>;
type ResolveStop<StopT, D extends Dim> = [StopT] extends [undefined] ? `${D}` : NonNegDigits<StopT>;

/** `start' = min(start, d)`, `stop' = min(stop, d)`,
 * `dim = start' < stop' ? stop' - start' : 0` — the runtime's own
 * `normalizeAxisSpec` step=1 formula, at the type level. */
type ComputeRangeDigits<StartS extends string, StopS extends string, DS extends string> = Min<StartS, DS> extends infer StartClamped extends string
  ? Min<StopS, DS> extends infer StopClamped extends string
    ? Compare<StartClamped, StopClamped> extends "lt"
      ? MultiSub<StopClamped, StartClamped>
      : "0"
    : never
  : never;

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
type ExtractStep<Spec> = Spec extends { readonly step: infer StepT } ? StepT : undefined;

/**
 * The stretch: a literal dim for a range spec, when `start`/`stop` are
 * literal non-negative integers (or omitted) and `step` is omitted or the
 * literal `1`. Anything outside that subset (see file header) honestly
 * degrades to `Dim` (`number`) — the exact same fallback the core rule
 * already uses, so dropping this call site entirely is always a safe,
 * behavior-preserving revert.
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
  : [ExtractStep<Spec>] extends [undefined | 1]
    ? ResolveStart<ExtractStart<Spec>> extends infer StartS extends string
      ? StartS extends "unsupported"
        ? Dim
        : ResolveStop<ExtractStop<Spec>, D> extends infer StopS extends string
          ? StopS extends "unsupported"
            ? Dim
            : ComputeRangeDigits<StartS, StopS, `${D}`> extends infer DimDigits extends string
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
