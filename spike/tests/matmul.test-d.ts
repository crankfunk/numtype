import { type IsShapeError, type ShapeError } from "../src/dim.ts";
import type { MatMul } from "../src/matmul.ts";
import type { Equal, Expect } from "./test-utils.ts";

// --- Acceptance table: MatMul -------------------------------------------

type T1 = Expect<Equal<MatMul<[2, 3], [3, 4]>, [2, 4]>>;
type T2 = Expect<Equal<MatMul<[3], [3]>, []>>;
type T3 = Expect<Equal<MatMul<[3], [3, 4]>, [4]>>;
type T4 = Expect<Equal<MatMul<[2, 3], [3]>, [2]>>;
type T5 = Expect<Equal<MatMul<[10, 2, 3], [10, 3, 4]>, [10, 2, 4]>>;
type T6 = Expect<Equal<MatMul<[10, 2, 3], [3, 4]>, [10, 2, 4]>>;
type T7 = Expect<Equal<MatMul<[2, 1, 2, 3], [7, 3, 4]>, [2, 7, 2, 4]>>;
type T8 = Expect<Equal<MatMul<[number, 3], [3, 4]>, [number, 4]>>;
type T9 = Expect<Equal<MatMul<[2, number], [3, 4]>, [2, 4]>>;

// Negative cases: must be a ShapeError.
type T10 = Expect<IsShapeError<MatMul<[2, 3], [4, 4]>>>;
type T11 = Expect<IsShapeError<MatMul<[], [3, 3]>>>;
type T12 = Expect<IsShapeError<MatMul<[2, 2, 3], [3, 3, 4]>>>;

// --- Extra sanity: the other scalar-operand direction, and both-scalar ---

type T13 = Expect<IsShapeError<MatMul<[3, 3], []>>>;
type T14 = Expect<IsShapeError<MatMul<[], []>>>;

// Exact message wording for the inner-dim-mismatch case — matches the
// spec's own example format verbatim: "matmul: inner dimensions 3 and 5 do
// not match" (docs/spike-01-type-layer-spec.md, hard design constraints).
type T15 = Expect<Equal<MatMul<[2, 3], [5, 4]>, ShapeError<"matmul: inner dimensions 3 and 5 do not match">>>;

// --- Dynamic RANK operands: degrade to `Dim[]`, never `never` -------------
// A `never` here would make `.matmul()` uncallable for rank-unknown arrays.

type T16 = Expect<Equal<MatMul<number[], [3, 4]>, readonly number[]>>;
type T17 = Expect<Equal<MatMul<[2, 3], number[]>, readonly number[]>>;
type T18 = Expect<Equal<MatMul<number[], number[]>, readonly number[]>>;

// The call-site DX check (what a *consumer* sees on `.matmul(...)`, not
// just the bare type alias) is exercised against the real NDArray class in
// ndarray-errors.test-d.ts — that's the actual, error-message-in-the-editor
// surface the spec asks to paste into the findings doc.

