/**
 * Kern 08 (docs/kern-08-reshape-flatten-spec.md) type-level tests:
 * `ReshapeCheck<S, NS>` (spike/src/reshape.ts) and the CORE product-mismatch
 * guard, `LiteralReshapeDimInvalid<NS>` (spike/src/slice-literal.ts, the
 * stretch), plus `NDArray.reshape`/`NDArray.flatten` method-level DX
 * (positive threading, error-at-argument, hover cleanliness) — same house
 * idioms as `vector.test-d.ts` (bare `Guard` message pinning) and
 * `product.test-d.ts` (the enabler this phase consumes).
 */
import { type Guard, NDArray } from "../src/ndarray.ts";
import type { ReshapeCheck } from "../src/reshape.ts";
import type { LiteralReshapeDimInvalid } from "../src/slice-literal.ts";
import type { Equal, Expect } from "./test-utils.ts";

// ---------------------------------------------------------------------------
// LiteralReshapeDimInvalid: bare classifier (pure type function) — the
// stretch's own per-dim walk, returning either the first provably-invalid
// dim's OWN value or the sentinel "ok".
// ---------------------------------------------------------------------------

type LDI1 = Expect<Equal<LiteralReshapeDimInvalid<[-1, 4]>, -1>>;
// Left-to-right: the invalid dim at position 1 is still found and reported.
type LDI2 = Expect<Equal<LiteralReshapeDimInvalid<[4, -1]>, -1>>;
// Both invalid: the FIRST one (left to right) wins.
type LDI3 = Expect<Equal<LiteralReshapeDimInvalid<[-1, -2]>, -1>>;
// Every dim valid, including 0 (VALID here, unlike slice `step`).
type LDI4 = Expect<Equal<LiteralReshapeDimInvalid<[3, 4]>, "ok">>;
type LDI5 = Expect<Equal<LiteralReshapeDimInvalid<[0, 4]>, "ok">>;
type LDI6 = Expect<Equal<LiteralReshapeDimInvalid<[]>, "ok">>;
// Dot-form: a proven non-integer, sign-agnostic.
type LDI7 = Expect<Equal<LiteralReshapeDimInvalid<[1.5]>, 1.5>>;
type LDI8 = Expect<Equal<LiteralReshapeDimInvalid<[-1.5]>, -1.5>>;
// No claim: wide `number`, exponent-form (valid integer, unprovable
// template form), union, dynamic rank.
type LDI9 = Expect<Equal<LiteralReshapeDimInvalid<[number]>, "ok">>;
type LDI10 = Expect<Equal<LiteralReshapeDimInvalid<[1e21]>, "ok">>;
type LDI11 = Expect<Equal<LiteralReshapeDimInvalid<[2 | 3]>, "ok">>;
type LDI12 = Expect<Equal<LiteralReshapeDimInvalid<number[]>, "ok">>;
type LDI13 = Expect<Equal<LiteralReshapeDimInvalid<[2, ...number[]]>, "ok">>;

// ---------------------------------------------------------------------------
// ReshapeCheck / Guard message pinning — same idiom as vector.test-d.ts's
// bare `DotCheck`/`Guard` pinning, applied to the single-value
// `Guard<Result, Actual>` form (no rest-parameter positions to inspect,
// unlike `SliceSpecsGuard`).
// ---------------------------------------------------------------------------

// --- message 2: product mismatch -------------------------------------------

type ReshapeSizeMsg = "reshape: cannot reshape array of size 6 into shape [4,2]";
type RG1 = Expect<Equal<Guard<ReshapeCheck<[2, 3], [4, 2]>, [4, 2]>, { readonly __shapeError: ReshapeSizeMsg }>>;

// Same message stem interpolates the OLD size, whichever shape it came from.
type ReshapeSizeMsg2 = "reshape: cannot reshape array of size 24 into shape [5,5]";
type RG2 = Expect<Equal<Guard<ReshapeCheck<[4, 6], [5, 5]>, [5, 5]>, { readonly __shapeError: ReshapeSizeMsg2 }>>;

// --- message 1: invalid dim (stretch), checked BEFORE the product check ---

type ReshapeNegDimMsg = "reshape: invalid dimension -1 in shape [-1,4] (dims must be non-negative integers)";
type RG3 = Expect<Equal<Guard<ReshapeCheck<[12], [-1, 4]>, [-1, 4]>, { readonly __shapeError: ReshapeNegDimMsg }>>;

type ReshapeDotDimMsg = "reshape: invalid dimension 1.5 in shape [1.5,4] (dims must be non-negative integers)";
type RG4 = Expect<Equal<Guard<ReshapeCheck<[6], [1.5, 4]>, [1.5, 4]>, { readonly __shapeError: ReshapeDotDimMsg }>>;

// A shape violating BOTH (invalid dim AND a product that would mismatch
// anyway) reports message 1 — dim validity is checked first.
type ReshapeBothMsg = "reshape: invalid dimension -1 in shape [-1,100] (dims must be non-negative integers)";
type RG5 = Expect<Equal<Guard<ReshapeCheck<[6], [-1, 100]>, [-1, 100]>, { readonly __shapeError: ReshapeBothMsg }>>;

// --- ok cases: Guard passes the Actual type through unconstrained ---------

type RG6 = Expect<Equal<Guard<ReshapeCheck<[2, 3], [3, 2]>, [3, 2]>, [3, 2]>>;
type RG7 = Expect<Equal<Guard<ReshapeCheck<[], [1]>, [1]>, [1]>>; // rank-0 (size 1) <-> [1]
type RG8 = Expect<Equal<Guard<ReshapeCheck<[0, 3], [0]>, [0]>, [0]>>; // size-0 <-> size-0
type RG9 = Expect<Equal<Guard<ReshapeCheck<[2, 3], [2, 3]>, [2, 3]>, [2, 3]>>; // identity

// --- dynamic dim / dynamic rank on either side -> no claim (pass) ---------

type RG10 = Expect<Equal<Guard<ReshapeCheck<readonly [number, 3], [3, 2]>, [3, 2]>, [3, 2]>>;
type RG11 = Expect<Equal<Guard<ReshapeCheck<[2, 3], readonly [number]>, readonly [number]>, readonly [number]>>;
type RG12 = Expect<Equal<Guard<ReshapeCheck<number[], [3, 2]>, [3, 2]>, [3, 2]>>;
type RG13 = Expect<Equal<Guard<ReshapeCheck<[2, 3], number[]>, number[]>, number[]>>;

// --- union dim on either side -> no claim (pass; Spike-04 boundary-filter
// rule: a product verdict is an unbounded value, no subset check exists). --

type RG14 = Expect<Equal<Guard<ReshapeCheck<readonly [2 | 4, 3], [6]>, [6]>, [6]>>;
type RG15 = Expect<Equal<Guard<ReshapeCheck<[6], readonly [2 | 3, 2]>, readonly [2 | 3, 2]>, readonly [2 | 3, 2]>>;

// --- union of whole shapes (documented FOLLOWUPS boundary, shared with
// DotCheck/MatMul): the CURRENT (S) side may bypass the guard entirely. -----

type RG16 = Expect<Equal<Guard<ReshapeCheck<readonly [2, 3] | readonly [4], [6]>, [6]>, [6]>>;

// RG17 (over-cap product on either side, LiteralShapeProduct's own
// MAX_SAFE_INTEGER boundary) moved to
// spike/tests-stress/reshape-stress.test-d.ts (Infra 01 — digit-arithmetic
// stress split, docs/infra-01-stress-split.md).

// --- exponent-form dim: valid integer, unprovable template form — must
// pass (no claim, neither the stretch nor the core flags it). --------------

type RG18 = Expect<Equal<Guard<ReshapeCheck<[6], [1e21, 2]>, [1e21, 2]>, [1e21, 2]>>;

// ---------------------------------------------------------------------------
// NDArray.reshape()/flatten(): real class-level DX.
// ---------------------------------------------------------------------------

const arr23 = NDArray.zeros([2, 3]);

// --- ok: reshape hover ------------------------------------------------------

const reshaped = arr23.reshape([3, 2]);
type T1 = Expect<Equal<(typeof reshaped)["shape"], [3, 2]>>;

// identity reshape.
const identity = arr23.reshape([2, 3]);
type T2 = Expect<Equal<(typeof identity)["shape"], [2, 3]>>;

// rank change: [2,3] (size 6) -> [6] -> [1,6,1].
const to1d = arr23.reshape([6]);
type T3 = Expect<Equal<(typeof to1d)["shape"], [6]>>;
const toPadded = arr23.reshape([1, 6, 1]);
type T4 = Expect<Equal<(typeof toPadded)["shape"], [1, 6, 1]>>;

// --- ok: flatten hover, small computed literal -----------------------------

const flattened = arr23.flatten();
type T5 = Expect<Equal<(typeof flattened)["shape"], [6]>>;

// --- ok: flatten hover, big-dim digit-multiplication case (Spike-04 payoff,
// the phase's own headline example: [1024,1024] -> [1048576]). -------------

const bigArr = NDArray.zeros([1024, 1024]);
const bigFlattened = bigArr.flatten();
type T6 = Expect<Equal<(typeof bigFlattened)["shape"], [1048576]>>;

// T7/T8 (flatten at the MAX_SAFE_INTEGER cap boundary, method-level reuse of
// product.test-d.ts's P10/P11) moved to
// spike/tests-stress/reshape-stress.test-d.ts (Infra 01 — digit-arithmetic
// stress split, docs/infra-01-stress-split.md).

// --- ok: flatten of rank-0 -> [1], of size-0 -> [0] -------------------------

declare const rank0Arr: NDArray<[]>;
const rank0Flat = rank0Arr.flatten();
type T9 = Expect<Equal<(typeof rank0Flat)["shape"], [1]>>;

declare const size0Arr: NDArray<[0, 4]>;
const size0Flat = size0Arr.flatten();
type T10 = Expect<Equal<(typeof size0Flat)["shape"], [0]>>;

// --- error-at-argument: product mismatch ------------------------------------

// @ts-expect-error - array of size 6 cannot reshape into [4,2] (size 8)
arr23.reshape([4, 2]);

// --- error-at-argument: invalid literal dim (stretch), checked BEFORE the
// product check -------------------------------------------------------------

const arr12 = NDArray.zeros([12]);
// @ts-expect-error - -1 is an invalid dimension (the `-1`-inference deferral
// documented in FOLLOWUPS: it is just "a negative dim" here, not inferred)
arr12.reshape([-1, 4]);
// @ts-expect-error - 1.5 is a non-integer dimension
arr12.reshape([1.5, 12]);
// A shape violating both reports the dim-validity error (message 1 wins).
// @ts-expect-error - -1 is an invalid dimension (reported before the product check)
arr23.reshape([-1, 100]);

// --- gradual typing: dynamic dim / dynamic rank must NOT error (the runtime
// backstop, `assertReshapeArgs`, checks it instead). -------------------------

declare const dynRankShape: number[];
const dynRankArr = NDArray.zeros(dynRankShape);
dynRankArr.reshape([2, 3]); // must NOT error (dynamic rank on the receiver)
const dynResult = arr23.reshape(dynRankShape); // must NOT error (dynamic rank/length on the argument)
type T11 = Expect<Equal<(typeof dynResult)["shape"], number[]>>;

declare const dynDimShape: readonly [number, 3];
const dynDimArr = NDArray.zeros(dynDimShape);
dynDimArr.reshape([3, 2]); // must NOT error (dynamic dim on the receiver)

// A literal `-1` widened to `number` is no longer a provable-invalid literal
// -> compiles (the runtime backstop, `assertReshapeArgs`, still throws;
// pinned in tests-runtime/reshape.test.ts).
declare const wideNegDim: number;
arr12.reshape([wideNegDim, 4]); // must NOT error (no static claim on a wide dim)

// --- flatten(): niladic, no guard, callable on any rank/dynamic shape ------

const flatDyn = dynRankArr.flatten();
type T12 = Expect<Equal<(typeof flatDyn)["shape"], [number]>>;

void reshaped;
void identity;
void to1d;
void toPadded;
void flattened;
void bigFlattened;
void rank0Flat;
void size0Flat;
void dynResult;
void flatDyn;
