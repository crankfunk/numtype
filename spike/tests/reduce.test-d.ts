import { type IsShapeError } from "../src/dim.ts";
import type { ReduceAxis, ReduceAxisKeepDims, Transpose } from "../src/reduce.ts";
import type { Equal, Expect } from "./test-utils.ts";

// --- Acceptance table: Reduce / Transpose --------------------------------

type T1 = Expect<Equal<ReduceAxis<[2, 3, 4], 1>, [2, 4]>>;
type T2 = Expect<Equal<ReduceAxis<[2, 3, 4], -1>, [2, 3]>>;
type T3 = Expect<Equal<ReduceAxisKeepDims<[2, 3, 4], 1>, [2, 1, 4]>>;
type T4 = Expect<Equal<ReduceAxis<[2, 3, 4]>, []>>;
type T5 = Expect<IsShapeError<ReduceAxis<[2, 3, 4], 3>>>;
type T6 = Expect<Equal<Transpose<[2, 3, 4]>, [4, 3, 2]>>;

// --- Extra sanity: other negative axes, other out-of-range directions ---

type T7 = Expect<Equal<ReduceAxis<[2, 3, 4], -3>, [3, 4]>>; // -3 === axis 0
type T8 = Expect<Equal<ReduceAxis<[2, 3, 4], 0>, [3, 4]>>;
type T9 = Expect<IsShapeError<ReduceAxis<[2, 3, 4], -4>>>; // too far negative
type T10 = Expect<Equal<ReduceAxisKeepDims<[2, 3, 4], -1>, [2, 3, 1]>>;
// no-axis + keepdims: reduce everything but preserve rank as all-1s (true
// NumPy `keepdims` semantics), not `[]` — deliberately distinct from the
// non-keepdims "no axis" case above.
type T11 = Expect<Equal<ReduceAxisKeepDims<[2, 3, 4]>, [1, 1, 1]>>;
type T12 = Expect<Equal<Transpose<[]>, []>>;
type T13 = Expect<Equal<Transpose<[5]>, [5]>>;

// --- Dynamic RANK / dynamic AXIS: degrade to `Dim[]` ----------------------
// Rank-unknown shapes can't be indexed; a non-literal `number` axis would
// otherwise hit `0 extends number` and silently remove axis 0.

type T14 = Expect<Equal<ReduceAxis<number[], 1>, readonly number[]>>;
type T15 = Expect<Equal<ReduceAxis<[2, 3, 4], number>, readonly number[]>>;
type T16 = Expect<Equal<ReduceAxis<number[]>, []>>; // full reduction is [] for EVERY rank, known or not
type T17 = Expect<Equal<ReduceAxisKeepDims<number[]>, 1[]>>; // rank unknown, but every dim is 1
type T18 = Expect<Equal<Transpose<number[]>, readonly number[]>>;

// =============================================================================
// Phase-D V1 (docs/phase-d-vorarbeiten-spec.md, Union-Guard-Fix): Facette (c)
// — a MIXED-rank shape union (`S["length"]` itself a proper union of rank
// literals, e.g. `[2,3] | [2,3,4]`) degrades uniformly at every rank-gate via
// `RankUnknowable`, exactly like a dynamic rank. This is the fix for the
// Kern-09 finding: pre-fix, `ReduceAxis<[2,3]|[2,3,4], 2>` silently resolved
// to the CONFIDENT `[2, 3]` (only the rank-3 member's removal succeeded; the
// rank-2 member's `ShapeError` got silently discarded by the distributive
// `Guard`/`OkShape` pipeline at the `NDArray.sum()` call site — see
// ndarray.test-d.ts for that call-site pin). `Transpose` is a disclosed,
// owner-decided precision trade-off rather than a bug fix: pre-fix it was
// already distributively CORRECT (`[3,2] | [4,3,2]`, a genuine per-member
// answer), but the uniform-degradation rule applies to every rank-gate
// without exception, for structural simplicity (docs/phase-d-vorarbeiten-spec.md,
// "Begründung der Uniform-Degradation").
// =============================================================================

type UC1 = Expect<Equal<ReduceAxis<[2, 3] | [2, 3, 4], 2>, readonly number[]>>;
type UC2 = Expect<Equal<Transpose<[2, 3] | [2, 3, 4]>, readonly number[]>>;

// Uniform-rank shape unions are NOT mixed-rank: `RankUnknowable` stays
// `false` for them, so natural per-member distribution still applies
// (Policy table row 2 — "natürliche Distribution bleibt"). Transpose of a
// uniform-rank union is genuinely, distributively correct.
type UC3 = Expect<Equal<Transpose<[2, 3] | [4, 5]>, [3, 2] | [5, 4]>>;
