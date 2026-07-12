import type { Broadcast } from "../src/broadcast.ts";
import { type CompatDim, type DimEq, type IsShapeError } from "../src/dim.ts";
import { type Guard, NDArray } from "../src/ndarray.ts";
import type { Equal, Expect } from "./test-utils.ts";

// --- Acceptance table: Broadcast ---------------------------------------

type T1 = Expect<Equal<Broadcast<[2, 3], [3]>, [2, 3]>>;
type T2 = Expect<Equal<Broadcast<[8, 1, 6, 1], [7, 1, 5]>, [8, 7, 6, 5]>>;
type T3 = Expect<Equal<Broadcast<[256, 256, 3], [3]>, [256, 256, 3]>>;
type T4 = Expect<Equal<Broadcast<[], [2, 3]>, [2, 3]>>;
type T5 = Expect<Equal<Broadcast<[2, number], [2, 3]>, [2, number]>>;

// Negative cases: must be a ShapeError.
type T6 = Expect<IsShapeError<Broadcast<[2, 3], [4]>>>;
type T7 = Expect<IsShapeError<Broadcast<[5, 4], [2, 4]>>>;

// --- Extra symmetric / edge sanity (not in the table, cheap to cover) --

// Broadcast is symmetric for the compatible cases above.
type T8 = Expect<Equal<Broadcast<[3], [2, 3]>, [2, 3]>>;
type T9 = Expect<Equal<Broadcast<[2, 3], []>, [2, 3]>>;
// Both fully dynamic.
type T10 = Expect<Equal<Broadcast<[number, number], [number, number]>, [number, number]>>;
// A literal 1 vs a dynamic dim: dynamic wins (gradual — accept, degrade to `number`).
type T11 = Expect<Equal<Broadcast<[1], [number]>, [number]>>;

// Rank-16 broadcast (stress case lives in limits.test-d.ts too, but this is
// the smallest possible positive check of it here).
type Rank16A = [1, 2, 1, 4, 1, 6, 1, 8, 1, 10, 1, 12, 1, 14, 1, 16];
type Rank16B = [1, 1, 3, 1, 5, 1, 7, 1, 9, 1, 11, 1, 13, 1, 15, 1];
type Rank16Result = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
type T12 = Expect<Equal<Broadcast<Rank16A, Rank16B>, Rank16Result>>;

// --- Dynamic RANK (`number[]`, variadic tuples): degrade to `Dim[]` ------
// Recursing into a rank-unknown shape would silently treat it as exhausted
// and produce a confidently-wrong fixed tuple.

type T13 = Expect<Equal<Broadcast<number[], [2, 3]>, readonly number[]>>;
type T14 = Expect<Equal<Broadcast<[2, 3], number[]>, readonly number[]>>;
type T15 = Expect<Equal<Broadcast<number[], number[]>, readonly number[]>>;
type T16 = Expect<Equal<Broadcast<[2, ...number[]], [3]>, readonly number[]>>;

// =============================================================================
// Phase-D V1 (docs/phase-d-vorarbeiten-spec.md, Union-Guard-Fix): Facette (a)
// (union DIM -> no-claim/wide) and Facette (b) corrected (shape-union IN ONE
// type parameter of a single NDArray instance) — the reproducible form of
// the guard leak; a union of whole `NDArray<A>|NDArray<B>` INSTANCES is a
// separate, already-rejected control case, pinned in ndarray.test-d.ts.
// =============================================================================

// --- Facette (a): union DIM on either side -> no-claim, `true`/`Dim` -------
// (D-V1.2 house pattern: `VectorLenCheck` in vector.ts.) Pre-fix, `DimEq`
// distributed to `boolean` (rejecting even a member that DOES match) and
// `CompatDim` distributed to a raw `Dim | ShapeError<...>` union.

type UA1 = Expect<Equal<DimEq<2 | 7, 2>, true>>;
type UA2 = Expect<Equal<CompatDim<2 | 7, 2>, number>>;
type UA3 = Expect<Equal<DimEq<2, 2 | 7>, true>>; // symmetric: union on the OTHER side
type UA4 = Expect<Equal<CompatDim<2, 2 | 7>, number>>;

// A union dim that would otherwise SURVIVE into the broadcast result
// degrades that one axis to `number` (wide) — never a confident literal for
// either union member, and never the pre-fix leak (a raw `2 | ShapeError<...>`
// mixed union that `Guard`/`OkShape` collapsed to a confidently-wrong [2,3]).
type UA5 = Expect<Equal<Broadcast<readonly [2 | 7, 3], [2, 3]>, [number, 3]>>;

const uaBase = NDArray.zeros([2, 3]);
declare const uaUnionDimArg: NDArray<readonly [2 | 7, 3]>;
const uaAdded = uaBase.add(uaUnionDimArg); // must NOT error (no-claim, gradual)
type UA6 = Expect<Equal<(typeof uaAdded)["shape"], [number, 3]>>;
void uaAdded;

// --- Facette (b), corrected form: shape-union IN ONE TYPE PARAMETER --------
// (`x: NDArray<[2,3]|[7,3]>`) — uniform rank. Policy: gemischt (some members
// compatible, some not) -> accept, runtime backstop, result = union of the
// VALID members' results; ALL members incompatible -> reject at the
// argument with a COMBINED message (D-V1.4, tuple-wrapped `Guard`).

declare const baseAB: NDArray<[2, 3]>;
declare const mixedArg: NDArray<[2, 3] | [7, 3]>;
declare const allBadArg: NDArray<[9, 3] | [7, 3]>;

// gemischt: accepted, result is the union of valid members only (here: a
// single valid member, so it collapses to one confident shape).
const mixedAdded = baseAB.add(mixedArg);
type UB1 = Expect<Equal<(typeof mixedAdded)["shape"], [2, 3]>>;
void mixedAdded;

// uniform error union: ALL members incompatible -> rejected at the argument.
// The exact combined-message shape (D-V1.4's tuple-wrapped `infer` over a
// non-distributed union source infers the UNION of every matched branch's
// message — including the pre-existing "kreuzmultiplizierte Dim-Paare" nit,
// Kern-09 Nit 3: `ShowShape` of a whole shape-union operand also distributes
// independently of which specific member failed, so all 4 shape/message
// combinations appear, not just the 2 "real" ones).
type AllBadMsg =
  | "cannot broadcast shapes [2,3] and [7,3]: dims 2 and 7 are not broadcast-compatible (neither equal nor 1)"
  | "cannot broadcast shapes [2,3] and [7,3]: dims 2 and 9 are not broadcast-compatible (neither equal nor 1)"
  | "cannot broadcast shapes [2,3] and [9,3]: dims 2 and 7 are not broadcast-compatible (neither equal nor 1)"
  | "cannot broadcast shapes [2,3] and [9,3]: dims 2 and 9 are not broadcast-compatible (neither equal nor 1)";
type UB2 = Expect<Equal<Guard<Broadcast<[2, 3], [9, 3] | [7, 3]>, NDArray<[9, 3] | [7, 3]>>, { readonly __shapeError: AllBadMsg }>>;

// @ts-expect-error - both members of the argument's shape union are broadcast-incompatible with [2,3]
baseAB.add(allBadArg);
