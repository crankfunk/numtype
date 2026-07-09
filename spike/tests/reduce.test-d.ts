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
