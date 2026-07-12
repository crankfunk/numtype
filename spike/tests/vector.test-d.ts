/**
 * Type-level tests for Kern 07's `DotCheck<S, B, Op>` (spike/src/vector.ts)
 * and the `dot`/`norm`/`cosineSimilarity`/`sub`/`mul`/`div` call-site DX on
 * the real `NDArray` class — see docs/kern-07-elementwise-vector-spec.md's
 * error-message table for the exact pinned strings.
 */
import { type Guard, NDArray } from "../src/ndarray.ts";
import type { DotCheck } from "../src/vector.ts";
import type { Equal, Expect } from "./test-utils.ts";

// ---------------------------------------------------------------------------
// Bare `DotCheck`/`Guard` pinning — exact message wording via the `Guard`
// property type (same idiom as `slice.test-d.ts`'s `SliceSpecsGuard`
// pinning, applied to the single-value `Guard<Result, Actual>` form).
// ---------------------------------------------------------------------------

// --- dot: message-table order (first rank, second rank, length) -----------

type DotFirstRankMsg = "dot: expected a 1-D vector as the first operand (got shape [2,3])";
type G1 = Expect<Equal<Guard<DotCheck<[2, 3], [3], "dot">, NDArray<[3]>>, { readonly __shapeError: DotFirstRankMsg }>>;

type DotSecondRankMsg = "dot: expected a 1-D vector as the second operand (got shape [2,3])";
type G2 = Expect<Equal<Guard<DotCheck<[3], [2, 3], "dot">, NDArray<[2, 3]>>, { readonly __shapeError: DotSecondRankMsg }>>;

type DotLenMsg = "dot: vector lengths 3 and 4 do not match";
type G3 = Expect<Equal<Guard<DotCheck<[3], [4], "dot">, NDArray<[4]>>, { readonly __shapeError: DotLenMsg }>>;

// --- cosineSimilarity: own message prefix, same structure -----------------

type CosFirstRankMsg = "cosineSimilarity: expected a 1-D vector as the first operand (got shape [2,3])";
type G4 = Expect<Equal<Guard<DotCheck<[2, 3], [3], "cosineSimilarity">, NDArray<[3]>>, { readonly __shapeError: CosFirstRankMsg }>>;

type CosSecondRankMsg = "cosineSimilarity: expected a 1-D vector as the second operand (got shape [2,3])";
type G5 = Expect<Equal<Guard<DotCheck<[3], [2, 3], "cosineSimilarity">, NDArray<[2, 3]>>, { readonly __shapeError: CosSecondRankMsg }>>;

type CosLenMsg = "cosineSimilarity: vector lengths 3 and 4 do not match";
type G6 = Expect<Equal<Guard<DotCheck<[3], [4], "cosineSimilarity">, NDArray<[4]>>, { readonly __shapeError: CosLenMsg }>>;

// --- ok case: Guard passes the Actual type through unconstrained ----------

type G7 = Expect<Equal<Guard<DotCheck<[3], [3], "dot">, NDArray<[3]>>, NDArray<[3]>>>;
type G8 = Expect<Equal<Guard<DotCheck<[0], [0], "dot">, NDArray<[0]>>, NDArray<[0]>>>; // size-0 vectors: valid

// --- dynamic dim / dynamic rank on either side -> no claim (pass) ----------

type G9 = Expect<Equal<Guard<DotCheck<readonly [number], [3], "dot">, NDArray<[3]>>, NDArray<[3]>>>;
type G10 = Expect<Equal<Guard<DotCheck<[3], readonly [number], "dot">, NDArray<readonly [number]>>, NDArray<readonly [number]>>>;
type G11 = Expect<Equal<Guard<DotCheck<number[], [3], "dot">, NDArray<[3]>>, NDArray<[3]>>>;
type G12 = Expect<Equal<Guard<DotCheck<[3], number[], "dot">, NDArray<number[]>>, NDArray<number[]>>>;

// --- union dim on either side -> no claim (pass; never a union verdict
// misread — the Spike-04/06 house rule this guard was built to honor from
// day one, unlike MatMul's pre-existing `DimEq` hazard). ---------------------

type G13 = Expect<Equal<Guard<DotCheck<readonly [2 | 3], [3], "dot">, NDArray<[3]>>, NDArray<[3]>>>;
type G14 = Expect<Equal<Guard<DotCheck<[3], readonly [3 | 4], "dot">, NDArray<readonly [3 | 4]>>, NDArray<readonly [3 | 4]>>>;

// ---------------------------------------------------------------------------
// Real `NDArray` class DX: the actual editor-visible behavior at a call site
// (mirrors ndarray.test-d.ts's own "call-site DX" checks for add/matmul).
// ---------------------------------------------------------------------------

const va = NDArray.zeros([3]);
const vb = NDArray.zeros([3]);

// --- ok case: return type is a plain `number` ------------------------------

const dotResult = va.dot(vb);
type T1 = Expect<Equal<typeof dotResult, number>>;

const cosResult = va.cosineSimilarity(vb);
type T2 = Expect<Equal<typeof cosResult, number>>;

// --- negative: rank-2 RECEIVER rejected — error lands at the ARGUMENT, that
// is where `Guard` puts it (same mechanism as every other op here; a
// receiver-side violation is still surfaced at the call's argument). -------

const badReceiver = NDArray.zeros([2, 3]);
// @ts-expect-error - receiver [2,3] is not rank-1: DotCheck names the first operand, error lands on `vb`
badReceiver.dot(vb);
// @ts-expect-error - same rule for cosineSimilarity
badReceiver.cosineSimilarity(vb);

// --- negative: rank-2 ARGUMENT rejected ------------------------------------

const badArg = NDArray.zeros([2, 3]);
// @ts-expect-error - [2,3] argument is not rank-1
va.dot(badArg);
// @ts-expect-error - same rule for cosineSimilarity
va.cosineSimilarity(badArg);

// --- negative: length mismatch rejected ------------------------------------

const wrongLen = NDArray.zeros([4]);
// @ts-expect-error - vector lengths 3 and 4 do not match
va.dot(wrongLen);
// @ts-expect-error - same rule for cosineSimilarity
va.cosineSimilarity(wrongLen);

// --- gradual typing: dynamic dim / dynamic rank on either side must NOT
// error (the runtime backstop, `assertVectorPair`, checks it instead). -----

declare const dynDimShape: readonly [number];
const dynDimArr = NDArray.zeros(dynDimShape);
dynDimArr.dot(vb); // must NOT error (dynamic dim on the receiver)
vb.dot(dynDimArr); // must NOT error (dynamic dim on the argument)

declare const dynRankShape: number[];
const dynRankArr = NDArray.zeros(dynRankShape);
dynRankArr.dot(vb); // must NOT error (dynamic rank on the receiver)
vb.dot(dynRankArr); // must NOT error (dynamic rank on the argument)
dynRankArr.cosineSimilarity(vb); // same for cosineSimilarity

// --- norm(): callable on ranks 0..2 and on a dynamic-rank shape, always
// returns a plain `number` — no guard exists (niladic, every rank valid). --

const normRank0 = NDArray.zeros([]).norm();
type T3 = Expect<Equal<typeof normRank0, number>>;

const normRank1 = NDArray.zeros([3]).norm();
type T4 = Expect<Equal<typeof normRank1, number>>;

const normRank2 = NDArray.zeros([2, 3]).norm();
type T5 = Expect<Equal<typeof normRank2, number>>;

const normDynRank = dynRankArr.norm();
type T6 = Expect<Equal<typeof normDynRank, number>>;

// ---------------------------------------------------------------------------
// sub/mul/div: broadcast-shape-error-at-argument pins (own to each method,
// though they share `Broadcast`) + result-shape hovers, incl. a broadcast
// case ([2,3] op [3] -> NDArray<[2,3]>) per the spec.
// ---------------------------------------------------------------------------

const s1 = NDArray.zeros([2, 3]);
const s2 = NDArray.zeros([3]);
const badBroadcast = NDArray.zeros([4]);

const subResult = s1.sub(s2);
type T7 = Expect<Equal<(typeof subResult)["shape"], [2, 3]>>;
// @ts-expect-error - [2,3] and [4] don't broadcast: error must land on `badBroadcast`
s1.sub(badBroadcast);

const mulResult = s1.mul(s2);
type T8 = Expect<Equal<(typeof mulResult)["shape"], [2, 3]>>;
// @ts-expect-error - [2,3] and [4] don't broadcast
s1.mul(badBroadcast);

const divResult = s1.div(s2);
type T9 = Expect<Equal<(typeof divResult)["shape"], [2, 3]>>;
// @ts-expect-error - [2,3] and [4] don't broadcast
s1.div(badBroadcast);

// ---------------------------------------------------------------------------
// Phase-D V1 (docs/phase-d-vorarbeiten-spec.md, Union-Guard-Fix): Facette (c)
// — a MIXED-rank shape union on EITHER `DotCheck` operand (receiver `S` or
// argument `B`) degrades to no-claim (`Pass`) via `RankUnknowable`, same
// treatment as a dynamic rank. Pre-fix this was already "gemischt accepted"
// (natural distribution mixed a `Pass` branch with a `ShapeError` branch,
// and the pre-existing distributive `Guard` already let a mixed union
// through) — not a confidently-wrong bug for `dot` specifically, but the fix
// still replaces the ad-hoc per-branch distribution with one clean gate.
// ---------------------------------------------------------------------------

type UC1 = Expect<Equal<DotCheck<[3] | [3, 4], [3], "dot">, true>>; // mixed-rank RECEIVER
type UC2 = Expect<Equal<DotCheck<readonly [number], [3] | [3, 4], "dot">, true>>; // mixed-rank ARGUMENT

declare const dotMixedRankRecv: NDArray<[3] | [3, 4]>;
const dotMixedResult = dotMixedRankRecv.dot(va); // must NOT error (mixed-rank receiver, no-claim)
type UC3 = Expect<Equal<typeof dotMixedResult, number>>;
void dotMixedResult;
