import { type AnyNDArray, NDArray } from "../src/ndarray.ts";
import type { Equal, Expect } from "./test-utils.ts";

// --- const type params: callers never write `as const` ------------------

const zerosResult = NDArray.zeros([2, 3]);
type ZerosShape = (typeof zerosResult)["shape"];
type T1 = Expect<Equal<ZerosShape, [2, 3]>>; // clean tuple, no `readonly` noise, no `as const` needed

const onesResult = NDArray.ones([4]);
type T2 = Expect<Equal<(typeof onesResult)["shape"], [4]>>;

const fromArrayResult = NDArray.fromArray([2, 2], [1, 2, 3, 4]);
type T3 = Expect<Equal<(typeof fromArrayResult)["shape"], [2, 2]>>;

// --- add / matmul / sum / transpose: positive, type threads through -----

const a = NDArray.zeros([2, 3]);
const b = NDArray.zeros([3]);
const added = a.add(b);
type T4 = Expect<Equal<(typeof added)["shape"], [2, 3]>>;

const m1 = NDArray.zeros([2, 3]);
const m2 = NDArray.zeros([3, 4]);
const multiplied = m1.matmul(m2);
type T5 = Expect<Equal<(typeof multiplied)["shape"], [2, 4]>>;

const s = NDArray.zeros([2, 3, 4]);
const summed = s.sum(1);
type T6 = Expect<Equal<(typeof summed)["shape"], [2, 4]>>;

const summedAll = s.sum();
type T7 = Expect<Equal<(typeof summedAll)["shape"], []>>;

const transposed = s.transpose();
type T8 = Expect<Equal<(typeof transposed)["shape"], [4, 3, 2]>>;

// --- negative: bad shapes must error AT the offending argument ----------

const badAddArg = NDArray.zeros([4]);
// @ts-expect-error - [2,3] and [4] don't broadcast: error must land on `badAddArg`
a.add(badAddArg);

const badMatMulArg = NDArray.zeros([4, 4]);
// @ts-expect-error - inner dims 3 vs 4 mismatch: error must land on `badMatMulArg`
m1.matmul(badMatMulArg);

const scalarArg = NDArray.zeros([]);
// @ts-expect-error - rank-0 operand is a hard error for matmul
m1.matmul(scalarArg);

// @ts-expect-error - axis 3 is out of range for a rank-3 shape [2,3,4]
s.sum(3);

// --- gradual typing: a `number` (dynamic) dim never errors ---------------

declare const dynamicShape: readonly [2, number];
const dyn = NDArray.zeros(dynamicShape);
const dynAdded = dyn.add(NDArray.zeros([2, 3])); // must NOT error (gradual escape hatch)
type T9 = Expect<Equal<(typeof dynAdded)["shape"], [2, number]>>;

// --- gradual typing: dynamic RANK (`number[]`) and dynamic axis -----------
// All ops must stay callable on rank-unknown arrays and degrade the result
// to `number[]` — neither a confident wrong tuple nor an uncallable `never`.

declare const runtimeShape: number[];
const dynRank = NDArray.zeros(runtimeShape);
const dynRankAdded = dynRank.add(NDArray.zeros([2, 3]));
type T10 = Expect<Equal<(typeof dynRankAdded)["shape"], readonly number[]>>;

const dynRankMul = NDArray.zeros([2, 3]).matmul(dynRank);
type T11 = Expect<Equal<(typeof dynRankMul)["shape"], readonly number[]>>;

declare const dynamicAxis: number;
const dynAxisSum = NDArray.zeros([2, 3, 4]).sum(dynamicAxis);
type T12 = Expect<Equal<(typeof dynAxisSum)["shape"], readonly number[]>>;

const dynRankTransposed = dynRank.transpose();
type T13 = Expect<Equal<(typeof dynRankTransposed)["shape"], readonly number[]>>;

// --- erased top type -------------------------------------------------------
// `NDArray<Shape>` is NOT an implicit supertype: with sound dynamic-rank
// degradation, S occurs in method-parameter positions (argument-side error
// guards), so the class's measured variance is invariant. Pinned here so a
// future TS version or refactor that restores it makes this directive
// "unused" and alerts us. The supported erased handle is `AnyNDArray`.

// @ts-expect-error - invariant S: fixed shape not assignable to NDArray<Shape>
const erased: NDArray<readonly number[]> = NDArray.zeros([2, 3]);
void erased;

const anyErased: AnyNDArray = NDArray.zeros([2, 3]); // the supported pattern
const anyList: AnyNDArray[] = [NDArray.zeros([2, 3]), NDArray.zeros([7]), NDArray.zeros([])];
void anyErased;
void anyList;
