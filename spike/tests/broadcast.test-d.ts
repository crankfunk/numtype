import type { Broadcast } from "../src/broadcast.ts";
import { type IsShapeError } from "../src/dim.ts";
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
