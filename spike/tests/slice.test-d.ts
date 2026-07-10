/**
 * Kern 05 type-level tests: `SliceShape`/`SliceSpecsGuard` (pure type
 * functions, docs/kern-05-slicing-spec.md's core rules), the STRETCH
 * (`LiteralRangeDim` via `slice.ts`'s wiring — literal computed dims for a
 * supported subset of range specs), plus `NDArray.slice` method-level checks
 * (positive threading, error-at-argument, hover cleanliness). Only `NDArray`
 * is exercised here, matching every other `*.test-d.ts` file in this suite
 * (`WNDArray` shares the exact same type machinery — `SliceShape`/`Guard`/
 * `OkShape` — imported unchanged, so pinning it once via `NDArray` covers
 * both; `WNDArray.slice`'s own correctness is covered by the runtime
 * differential suite instead).
 */
import { type AnyNDArray, NDArray } from "../src/ndarray.ts";
import type { LiteralIndexBounds } from "../src/slice-literal.ts";
import type { SliceShape, SliceSpecsGuard } from "../src/slice.ts";
import type { Equal, Expect } from "./test-utils.ts";

// --- Acceptance table: SliceShape (pure type function) --------------------

// Integer: drops the axis.
type T1 = Expect<Equal<SliceShape<[2, 3, 4], [1]>, [3, 4]>>;
// null: keeps the literal dim untouched.
type T2 = Expect<Equal<SliceShape<[2, 3, 4], [null]>, [2, 3, 4]>>;
// Range object OUTSIDE the stretch's supported subset (non-1 step) still
// degrades that one axis to `number` (core rule); trailing axes after it
// are still preserved literally. (A SUPPORTED range object instead computes
// a literal dim — see the "STRETCH" section below.)
type T3 = Expect<Equal<SliceShape<[2, 3, 4], [{ step: 2 }]>, [number, 3, 4]>>;
// Mixed: index + null + range (non-1 step, degrades) in one call.
type T4 = Expect<Equal<SliceShape<[2, 3, 4], [1, null, { step: 2 }]>, [3, number]>>;
// Zero specs: every axis is "trailing" -> full shape preserved.
type T5 = Expect<Equal<SliceShape<[2, 3, 4], []>, [2, 3, 4]>>;
// Fewer specs than rank: the remaining axes are taken in full, literally.
type T6 = Expect<Equal<SliceShape<[2, 3, 4], [1]>, [3, 4]>>;
// A negative literal index still drops the axis statically (the rank effect
// never depends on the concrete index value).
type T7 = Expect<Equal<SliceShape<[2, 3, 4], [-1]>, [3, 4]>>;
// A plain (non-literal) `number` index ALSO drops the axis statically.
type T8 = Expect<Equal<SliceShape<[2, 3, 4], [number]>, [3, 4]>>;
// Rank-0: zero specs on a rank-0 shape.
type T9 = Expect<Equal<SliceShape<[], []>, []>>;
// A dim that is ALREADY dynamic (`number`, known rank) stays `number` when
// kept via `null`, and the axis still drops cleanly via an integer spec.
type T11 = Expect<Equal<SliceShape<[2, number], [null, 1]>, [2]>>;
type T12 = Expect<Equal<SliceShape<[2, number], [1]>, [number]>>;

// --- STRETCH: literal computed dims for a supported subset of range specs -
// `start`/`stop` literal non-negative integers (or omitted), `step` omitted
// or literal `1` — see slice-literal.ts's file header for the precise
// supported subset and why it's scoped that way (negative literals and
// non-1 steps are explicitly NOT supported and degrade to `number`, same as
// the core rule — a strict superset of what the core alone could resolve).

// d=2: {start:1,stop:3} clamps stop to 2 -> dim = 2-1 = 1 (a LITERAL, not
// `number`); trailing axes 3,4 still preserved literally.
type ST1 = Expect<Equal<SliceShape<[2, 3, 4], [{ start: 1; stop: 3 }]>, [1, 3, 4]>>;
// Every axis sliced by a SUPPORTED range object -> every dim is computed.
type ST2 = Expect<Equal<SliceShape<[2, 3], [{ start: 0 }, { stop: 2 }]>, [2, 2]>>;
// {} (all defaults): dim = the full axis dim, unchanged but now LITERAL
// (start=0, stop=d).
type ST3 = Expect<Equal<SliceShape<[7], [{}]>, [7]>>;
// start >= stop (post-clamp) -> literal `0`, not `number`.
type ST4 = Expect<Equal<SliceShape<[10], [{ start: 5; stop: 2 }]>, [0]>>;
// step omitted vs literal `1` behave identically (both computable).
type ST5 = Expect<Equal<SliceShape<[5], [{ start: 1; stop: 4; step: 1 }]>, [3]>>;
// Larger dims (this project's own "1024+" examples) still resolve — digit
// count stays small (4 digits), never tuple-length-proportional to the
// value: docs/kern-05-slicing-spec.md / CLAUDE.md's TS-limits section.
type ST6 = Expect<Equal<SliceShape<[1024], [{ start: 100; stop: 1000 }]>, [900]>>;
type ST7 = Expect<Equal<SliceShape<[1024], [{}]>, [1024]>>;
// Borrow-chain hard cases (the arithmetic's riskiest paths — a silently
// wrong literal here would be the worst failure mode for the shape USP):
// 1000 - 1 = 999 (borrow propagates through three zeros) ...
type ST8 = Expect<Equal<SliceShape<[1000], [{ start: 1 }]>, [999]>>;
// ... and 100 - 99 = 1 (multi-digit borrow + leading-zero strip).
type ST9 = Expect<Equal<SliceShape<[100], [{ start: 99 }]>, [1]>>;

// --- STRETCH boundary: outside the supported subset degrades to `number` --

// Negative literal start: explicitly out of scope (would need signed add).
type SB1 = Expect<Equal<SliceShape<[5], [{ start: -1 }]>, [number]>>;
// A literal step other than 1: out of scope.
type SB2 = Expect<Equal<SliceShape<[5], [{ start: 1; stop: 4; step: 2 }]>, [number]>>;
// A non-literal (wide) start: can't compute a literal from it.
declare const wideNum: number;
type SB3 = Expect<Equal<SliceShape<[2, 3, 4], [{ start: typeof wideNum }]>, [number, 3, 4]>>;
// A dynamic (already-`number`) axis dim: nothing to compute from either.
type SB4 = Expect<Equal<SliceShape<[2, number], [null, { start: 1 }]>, [2, number]>>;

// --- Dynamic RANK / dynamic SPEC COUNT: degrade wholly, checked FIRST -----
// (per spec: "wide S degrades wholly, checked FIRST before any tuple
// recursion" — verified here for both sides of the (S, Specs) pair.)

type T13 = Expect<Equal<SliceShape<number[], [1, null]>, readonly number[]>>;
type T14 = Expect<Equal<SliceShape<[2, ...number[]], [1]>, readonly number[]>>;

// --- SliceSpecsGuard: the rest-parameter arity guard (message-shape check) -
// Exercised directly (pure type function) since the exact excess-position
// retyping and the exact message text ARE checkable at the type level here
// (unlike the Guard<Result,Actual> single-parameter idiom elsewhere, whose
// error only surfaces as call-site diagnostic text).

type BadSpecsMsg = `slice: 3 specs given for rank 2 shape [2,3]`;
// Exactly the EXCESS position (index 2) is retyped to the error object; the
// two in-range specs (1, null) keep their own original types unchanged.
type T15 = Expect<Equal<SliceSpecsGuard<[2, 3], [1, null, 2]>, [1, null, { readonly __shapeError: BadSpecsMsg }]>>;
// Exact match (spec count === rank): passes through completely unchanged.
type T16 = Expect<Equal<SliceSpecsGuard<[2, 3], [1, null]>, [1, null]>>;
// Fewer specs than rank: not an error (trailing axes) -> unchanged.
type T17 = Expect<Equal<SliceSpecsGuard<[2, 3], [1]>, [1]>>;
type T18 = Expect<Equal<SliceSpecsGuard<[2, 3], []>, []>>;
// Dynamic rank: passes through unchanged regardless of spec count (can't
// validate arity statically -> gradual, runtime-checked instead).
type T19 = Expect<Equal<SliceSpecsGuard<number[], [1, null, 2, 3, 4]>, [1, null, 2, 3, 4]>>;

// --- NDArray.slice(): method-level positive threading + hover cleanliness -

const arr = NDArray.zeros([2, 3, 4]);

const s1 = arr.slice(1);
type T20 = Expect<Equal<(typeof s1)["shape"], [3, 4]>>;

const s2 = arr.slice(null, 1);
type T21 = Expect<Equal<(typeof s2)["shape"], [2, 4]>>;

// Supported range spec -> the hover shows a clean, fully resolved literal
// tuple `NDArray<[1, 3, 4]>`, not `NDArray<[number, 3, 4]>`.
const s3 = arr.slice({ start: 1, stop: 3 });
type T22 = Expect<Equal<(typeof s3)["shape"], [1, 3, 4]>>;

const s4 = arr.slice();
type T23 = Expect<Equal<(typeof s4)["shape"], [2, 3, 4]>>;

// Non-1 step: outside the stretch's supported subset -> degrades to `number`.
const s5 = arr.slice(1, null, { step: 2 });
type T24 = Expect<Equal<(typeof s5)["shape"], [3, number]>>;

// Composition: slice then transpose, and transpose then slice, both thread
// through cleanly (the runtime differential suite exercises the DATA side
// of this; here we only pin that the TYPES compose).
const s6 = arr.slice(1).transpose();
type T25 = Expect<Equal<(typeof s6)["shape"], [4, 3]>>;

const s7 = arr.transpose().slice(1);
type T26 = Expect<Equal<(typeof s7)["shape"], [3, 2]>>;

// --- error-at-argument: too many specs ------------------------------------

// @ts-expect-error - 4 specs for rank 3 shape [2,3,4]: error must land on the excess (4th) argument
arr.slice(1, null, 2, 3);

// --- gradual typing: dynamic RANK stays callable and degrades honestly ----

declare const runtimeShape: number[];
const dynRank = NDArray.zeros(runtimeShape);
const dynSliced = dynRank.slice(1, null, 2, 3, 4, 5); // must NOT error (rank unknown -> arity unchecked)
type T27 = Expect<Equal<(typeof dynSliced)["shape"], readonly number[]>>;

// --- gradual typing: a `number` (dynamic) DIM never errors, keeps its kind -

declare const dynamicShape: readonly [2, number];
const dyn = NDArray.zeros(dynamicShape);
const dynDropped = dyn.slice(1); // integer spec: drops axis 0, keeps the still-dynamic axis 1
type T28 = Expect<Equal<(typeof dynDropped)["shape"], [number]>>;

// --- erased top type stays usable as a slice() receiver -------------------

const anyErased: AnyNDArray = NDArray.zeros([2, 3]);
void anyErased;

void s1;
void s2;
void s3;
void s4;
void s5;
void s6;
void s7;
void dynSliced;
void dynDropped;

// =============================================================================
// Spike 03 (docs/spike-03-index-bounds-spec.md): type-level bounds checks for
// literal integer indices against literal dims. Compile-time twin of the
// runtime-parity test in tests-runtime/slice.test.ts.
// =============================================================================

// --- LiteralIndexBounds: boundary exactness (pure type function) -----------

type B1 = Expect<Equal<LiteralIndexBounds<4, 5>, "in">>; // i = d-1: last valid
type B2 = Expect<Equal<LiteralIndexBounds<5, 5>, "out">>; // i = d: first invalid
type B3 = Expect<Equal<LiteralIndexBounds<-5, 5>, "in">>; // i = -d: normalizes to index 0
type B4 = Expect<Equal<LiteralIndexBounds<-6, 5>, "out">>; // i = -(d+1): past the front
type B5 = Expect<Equal<LiteralIndexBounds<0, 5>, "in">>;
// d = 0: no valid index exists at all.
type B6 = Expect<Equal<LiteralIndexBounds<0, 0>, "out">>;
type B7 = Expect<Equal<LiteralIndexBounds<-1, 0>, "out">>;
// Big dims: multi-digit Compare paths (length-compare AND same-length
// lexicographic), both sides of the boundary, both signs.
type B8 = Expect<Equal<LiteralIndexBounds<1023, 1024>, "in">>;
type B9 = Expect<Equal<LiteralIndexBounds<1024, 1024>, "out">>;
type B10 = Expect<Equal<LiteralIndexBounds<65535, 65536>, "in">>;
type B11 = Expect<Equal<LiteralIndexBounds<65536, 65536>, "out">>;
type B12 = Expect<Equal<LiteralIndexBounds<-65536, 65536>, "in">>;
type B13 = Expect<Equal<LiteralIndexBounds<-65537, 65536>, "out">>;
// No static claim (conservative "unknown"): wide index, dynamic dim,
// non-integer literal, exponent-form literal (an integer, but not provable
// from its template form — flagging it with the wrong reason would lie),
// and MIXED unions (members classify differently).
type B14 = Expect<Equal<LiteralIndexBounds<number, 5>, "unknown">>;
type B15 = Expect<Equal<LiteralIndexBounds<3, number>, "unknown">>;
type B16 = Expect<Equal<LiteralIndexBounds<1.5, 5>, "unknown">>;
type B17 = Expect<Equal<LiteralIndexBounds<1e21, 5>, "unknown">>;
type B18 = Expect<Equal<LiteralIndexBounds<2 | 7, 5>, "unknown">>;
type B19 = Expect<Equal<LiteralIndexBounds<-1 | 2, 5>, "unknown">>;
// A UNIFORM union (every member out of bounds) IS classified — the call is
// invalid whichever member the value turns out to be.
type B20 = Expect<Equal<LiteralIndexBounds<6 | 7, 5>, "out">>;

// --- SliceSpecsGuard: OOB positions retyped, message + axis exactness ------

// The error lands on exactly the offending position, naming index, axis,
// dim, and the full shape; in-range positions keep their own types.
type OobAxis1Msg = `slice: index 5 is out of bounds for axis 1 with dim 5 (shape [2,5])`;
type G1 = Expect<Equal<SliceSpecsGuard<[2, 5], [1, 5]>, [1, { readonly __shapeError: OobAxis1Msg }]>>;
// Boundary-valid calls pass through completely unchanged (both signs).
type G2 = Expect<Equal<SliceSpecsGuard<[2, 5], [1, 4]>, [1, 4]>>;
type G3 = Expect<Equal<SliceSpecsGuard<[2, 5], [-2, -5]>, [-2, -5]>>;
// null / range specs are never bounds-flagged (ranges CLAMP, per NumPy).
type G4 = Expect<Equal<SliceSpecsGuard<[2, 5], [null, { start: 99 }]>, [null, { start: 99 }]>>;
// Axis indexing is exact: the flagged axis is 2, not 0.
type OobAxis2Msg = `slice: index 5 is out of bounds for axis 2 with dim 5 (shape [5,5,5])`;
type G5 = Expect<Equal<SliceSpecsGuard<[5, 5, 5], [null, null, 5]>, [null, null, { readonly __shapeError: OobAxis2Msg }]>>;
// Both mechanisms in one call: an OOB literal at axis 0 AND an excess spec —
// each flagged with its own message.
type G6 = Expect<
  Equal<
    SliceSpecsGuard<[2], [5, null]>,
    [
      { readonly __shapeError: `slice: index 5 is out of bounds for axis 0 with dim 2 (shape [2])` },
      { readonly __shapeError: `slice: 2 specs given for rank 1 shape [2]` },
    ]
  >
>;

// --- NDArray.slice(): method-level error-at-argument + gradual pass-through -

// @ts-expect-error - index 2 is out of bounds for axis 0 with dim 2
arr.slice(2);
// @ts-expect-error - index -3 is past the front for axis 0 with dim 2
arr.slice(-3);
// Boundary-valid negative index compiles and threads the shape.
const okBoundary = arr.slice(-2);
type T29 = Expect<Equal<(typeof okBoundary)["shape"], [3, 4]>>;
// The error lands at the SECOND argument; the first (valid) is untouched.
// @ts-expect-error - index 3 is out of bounds for axis 1 with dim 3
arr.slice(1, 3);
// Wide `number` index: no static claim -> compiles (runtime backstop), and
// the axis still drops statically.
declare const wideIdx: number;
const wideCall = arr.slice(wideIdx);
type T30 = Expect<Equal<(typeof wideCall)["shape"], [3, 4]>>;
// Non-integer literal: no static claim -> compiles; the runtime's own
// "not an integer" error stays authoritative (pinned in tests-runtime).
const nonInt = arr.slice(1.5);
type T31 = Expect<Equal<(typeof nonInt)["shape"], [3, 4]>>;

void okBoundary;
void wideCall;
void nonInt;
