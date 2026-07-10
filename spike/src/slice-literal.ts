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
type IsUnion<T, U = T> = [T] extends [never] ? false : T extends unknown ? ([U] extends [T] ? false : true) : never;

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
