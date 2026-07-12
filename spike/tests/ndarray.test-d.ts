import type { Broadcast } from "../src/broadcast.ts";
import type { Shape } from "../src/dim.ts";
import { type AnyNDArray, type Guard, NDArray, type NDArrayView } from "../src/ndarray.ts";
import type { WNDArray } from "../src/wasm/resident.ts";
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

// --- keepdims (Kern 09): reduced axis kept as size-1, rank preserved -------
// Type layer (`ReduceAxis<S, Axis, KeepDims>`) already pinned in
// reduce.test-d.ts; these pin the `sum` method wiring (the `const KeepDims`
// param must reach the return type).
const summedKeep = s.sum(1, true);
type T7a = Expect<Equal<(typeof summedKeep)["shape"], [2, 1, 4]>>;

const summedAllKeep = s.sum(undefined, true);
type T7b = Expect<Equal<(typeof summedAllKeep)["shape"], [1, 1, 1]>>;

const summedNegKeep = s.sum(-1, true);
type T7c = Expect<Equal<(typeof summedNegKeep)["shape"], [2, 3, 1]>>;

const summedFalse = s.sum(1, false); // explicit `false` == default (axis removed)
type T7d = Expect<Equal<(typeof summedFalse)["shape"], [2, 4]>>;

// gradual: a dynamic (non-literal) boolean `keepdims` degrades to the union of
// keep/non-keep shapes — the same deliberate degradation as a dynamic axis
// (`const KeepDims` only pins LITERAL booleans; a variable stays `boolean`).
declare const dynKeep: boolean;
const summedDynKeep = s.sum(1, dynKeep);
type T7e = Expect<Equal<(typeof summedDynKeep)["shape"], [2, 4] | [2, 1, 4]>>;

// @ts-expect-error - axis 3 out of range even with keepdims: error stays at the axis argument
s.sum(3, true);

// resident twin: `WNDArray.sum` declares the `keepdims` signature independently
// of `NDArray.sum`, so pin its return type separately (shares `ReduceAxis` +
// `OkShape`, but a wiring typo in resident.ts wouldn't be caught above).
declare const rw: WNDArray<[2, 3, 4]>;
const rwKeep = rw.sum(1, true);
type T7f = Expect<Equal<(typeof rwKeep)["shape"], [2, 1, 4]>>;
const rwKeepAll = rw.sum(undefined, true);
type T7g = Expect<Equal<(typeof rwKeepAll)["shape"], [1, 1, 1]>>;

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

// --- NDArrayView<out S>: the safe, checker-enforced covariant read view ----
// (Spike 05, docs/spike-05-variance-design-spec.md). Unlike AnyNDArray
// (erasure — unsafe in both directions), the view's `out S` lets a concrete
// view WIDEN safely while still rejecting a narrowing assignment back.

declare const literalView: NDArrayView<[2, 3]>;
const widenedToShape: NDArrayView<Shape> = literalView; // widening: [2,3] -> Shape
const widenedToWideTuple: NDArrayView<readonly number[]> = literalView; // widening: [2,3] -> readonly number[]
void widenedToShape;
void widenedToWideTuple;

// A real NDArray<[2,3]> widens the same way — `NDArray<S> implements
// NDArrayView<S>` is the drift alarm on the class side (see ndarray.ts).
const nd23 = NDArray.zeros([2, 3]);
const nd45 = NDArray.zeros([4, 5]);
const nd23AsView: NDArrayView<Shape> = nd23;
void nd23AsView;

// Heterogeneous containers of the safe read-only top type — the same
// use case AnyNDArray[] serves above, but checker-enforced, not erased.
const heterogeneousViews: NDArrayView<Shape>[] = [nd23, nd45];
void heterogeneousViews;

// Downcast must still be rejected — `out` widens, it never narrows.
declare const wideView: NDArrayView<Shape>;
// @ts-expect-error - NDArrayView<Shape> is not assignable back to NDArrayView<[2, 3]>: downcast, not widening
const narrowedView: NDArrayView<[2, 3]> = wideView;
void narrowedView;

// Generic inference through the view: a generic function parameterized over
// NDArrayView<S> infers the exact literal tuple (probe evidence: the same
// held for a standalone `materialize<S>(v: View<S>)` free function).
declare function shapeOf<S extends Shape>(v: NDArrayView<S>): S;
const inferredShape = shapeOf(nd23);
type T14 = Expect<Equal<typeof inferredShape, [2, 3]>>;

// =============================================================================
// Phase-D V1 (docs/phase-d-vorarbeiten-spec.md, Union-Guard-Fix): call-site DX
// for Facette (c) (mixed-rank shape union) and the Kontroll-Pins for the
// (already-rejected, unaffected by V1) instance-union form.
// =============================================================================

// --- Facette (c): NDArray<[2,3]|[2,3,4]>.sum(2) -----------------------------
// Pre-fix (Kern-09 finding 1): silently accepted and returned the CONFIDENT
// `NDArray<[2,3]>` (only the rank-3 member's axis-2 removal actually
// succeeds; the rank-2 member's out-of-range `ShapeError` was silently
// discarded by the distributive `Guard`/`OkShape` pipeline — the axis
// PARAMETER's own `Guard<ReduceAxis<S,Axis>,Axis>` already accepted the call
// because ONE branch of the receiver-shape distribution was error-free).
// Post-fix: `RankUnknowable` degrades the WHOLE computation to `readonly
// Dim[]` before any per-member distribution happens — accepted (no-claim,
// runtime-backstopped), never a confident single-shape claim.
declare const mixedRankRecv: NDArray<[2, 3] | [2, 3, 4]>;
const mixedRankSummed = mixedRankRecv.sum(2);
type UC1 = Expect<Equal<(typeof mixedRankSummed)["shape"], readonly number[]>>;
void mixedRankSummed;

// --- Kontroll-Pins: union of whole NDArray<A>|NDArray<B> INSTANCES ---------
// (as opposed to a shape-union IN ONE type parameter, the form V1 actually
// fixes — see broadcast.test-d.ts's Facette-(b) pins). This form is
// rejected already TODAY by TS's own generic inference / class-invariance,
// unrelated to Guard/CompatDim/DimEq and unreachable/unfixable through this
// codebase's Guard design (Baustein-0 finding, spec's Adversariale
// Spec-Verifikation addendum). Pinned as a REGRESSION control, not a V1 fix.
declare const instanceUnionArg: NDArray<[2, 3]> | NDArray<[7, 3]>;
// @ts-expect-error - instance union NDArray<A>|NDArray<B> as ARGUMENT is rejected by TS's own generic inference (control pin, not a V1 fix target)
a.add(instanceUnionArg);
// @ts-expect-error - instance union NDArray<A>|NDArray<B> as RECEIVER is rejected too (no common call signature TS will synthesize)
instanceUnionArg.add(a);

// =============================================================================
// WNDArray-side: the SAME facette pins, at least for add/matmul/sum, as
// explicit WNDArray assertions (spec requirement — "gleiche importierte
// Maschinerie wird bewiesen, nicht angenommen": WNDArray consumes the exact
// same Guard/OkShape/Broadcast/MatMul/ReduceAxis/CompatDim/DimEq/RankUnknowable
// machinery from ndarray.ts/broadcast.ts/matmul.ts/reduce.ts/dim.ts, so a fix
// there covers WNDArray with zero resident.ts edits — proven here, not assumed).
// =============================================================================

// add: Facette (b) corrected, gemischt accepted.
declare const wAddBase: WNDArray<[2, 3]>;
declare const wAddMixedArg: WNDArray<[2, 3] | [7, 3]>;
const wAdded = wAddBase.add(wAddMixedArg);
type UW1 = Expect<Equal<(typeof wAdded)["shape"], [2, 3]>>;
void wAdded;

// add: Facette (b), WNDArray-side uniform-fehlerhaft -> combined message
// (mirrors broadcast.test-d.ts's UB2 construction/assertion idiom exactly,
// only the `Actual`/receiver type param swapped from NDArray to WNDArray;
// verify-round B-F4 closure. Empirically probed against the real
// Guard/Broadcast/WNDArray machinery via the marker-probe technique
// documented in docs/phase-d-vorarbeiten-v1-ergebnisse.md's "PRE-FIX-ROT-
// BEWEIS/Methodik" — the message came back byte-identical to UB2's
// `AllBadMsg`, confirming `Guard`'s error branch depends only on `Result`
// (Broadcast<S,B>), never on `Actual`).
type AllBadMsgW =
  | "cannot broadcast shapes [2,3] and [7,3]: dims 2 and 7 are not broadcast-compatible (neither equal nor 1)"
  | "cannot broadcast shapes [2,3] and [7,3]: dims 2 and 9 are not broadcast-compatible (neither equal nor 1)"
  | "cannot broadcast shapes [2,3] and [9,3]: dims 2 and 7 are not broadcast-compatible (neither equal nor 1)"
  | "cannot broadcast shapes [2,3] and [9,3]: dims 2 and 9 are not broadcast-compatible (neither equal nor 1)";
type UW4 = Expect<
  Equal<Guard<Broadcast<[2, 3], [9, 3] | [7, 3]>, WNDArray<[9, 3] | [7, 3]>>, { readonly __shapeError: AllBadMsgW }>
>;

// matmul: Facette (a), union dim at the CONTRACTION axis -> accepted,
// confident (the union dim is contracted away, never survives to output).
declare const wMatMul1: WNDArray<[2, 3 | 7]>;
declare const wMatMul2: WNDArray<[3, 4]>;
const wMatmulResult = wMatMul1.matmul(wMatMul2);
type UW2 = Expect<Equal<(typeof wMatmulResult)["shape"], [2, 4]>>;
void wMatmulResult;

// sum: Facette (c), mixed-rank receiver -> accepted, degraded to `readonly
// number[]` (never the pre-fix confident-but-wrong single shape).
declare const wMixedRankRecv: WNDArray<[2, 3] | [2, 3, 4]>;
const wMixedRankSummed = wMixedRankRecv.sum(2);
type UW3 = Expect<Equal<(typeof wMixedRankSummed)["shape"], readonly number[]>>;
void wMixedRankSummed;
