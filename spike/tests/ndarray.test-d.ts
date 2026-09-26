import type { Broadcast } from "../src/broadcast.ts";
import type { Shape } from "../src/dim.ts";
import { type AnyNDArray, type Guard, NDArray, type NDArrayView, type NestedArray, type NestedBoolValue, type NestedValue } from "../src/ndarray.ts";
import type { ReduceAxis } from "../src/reduce.ts";
import type { DType } from "../src/runtime.ts";
import type { ItemGuard, StackCheck, TopkCheck } from "../src/vector.ts";
import type { CoreExports } from "../src/wasm/loader.ts";
import { type AnyWNDArray, WNDArray } from "../src/wasm/resident.ts";
import type { Equal, Expect } from "./test-utils.ts";

// --- const type params: callers never write `as const` ------------------

// D-V2.3 (docs/phase-d-vorarbeiten-spec.md): `.shape` is now `Readonly<S>` —
// every literal-tuple `Equal<>` pin below is re-expressed intent-preservingly
// as `readonly [...]`, matching the new (accepted, documented) member-hover.
// The CLASS hover is unaffected (`S` itself, e.g. `[2, 3]`, stays a clean
// tuple) — only the WRAPPING at the `.shape` member changed.

const zerosResult = NDArray.zeros([2, 3]);
type ZerosShape = (typeof zerosResult)["shape"];
type T1 = Expect<Equal<ZerosShape, readonly [2, 3]>>; // clean tuple, no `readonly` noise, no `as const` needed

const onesResult = NDArray.ones([4]);
type T2 = Expect<Equal<(typeof onesResult)["shape"], readonly [4]>>;

const fromArrayResult = NDArray.fromArray([2, 2], [1, 2, 3, 4]);
type T3 = Expect<Equal<(typeof fromArrayResult)["shape"], readonly [2, 2]>>;

// --- add / matmul / sum / transpose: positive, type threads through -----

const a = NDArray.zeros([2, 3]);
const b = NDArray.zeros([3]);
const added = a.add(b);
type T4 = Expect<Equal<(typeof added)["shape"], readonly [2, 3]>>;

const m1 = NDArray.zeros([2, 3]);
const m2 = NDArray.zeros([3, 4]);
const multiplied = m1.matmul(m2);
type T5 = Expect<Equal<(typeof multiplied)["shape"], readonly [2, 4]>>;

const s = NDArray.zeros([2, 3, 4]);
const summed = s.sum(1);
type T6 = Expect<Equal<(typeof summed)["shape"], readonly [2, 4]>>;

const summedAll = s.sum();
type T7 = Expect<Equal<(typeof summedAll)["shape"], readonly []>>;

// --- keepdims (Kern 09): reduced axis kept as size-1, rank preserved -------
// Type layer (`ReduceAxis<S, Axis, KeepDims>`) already pinned in
// reduce.test-d.ts; these pin the `sum` method wiring (the `const KeepDims`
// param must reach the return type).
const summedKeep = s.sum(1, true);
type T7a = Expect<Equal<(typeof summedKeep)["shape"], readonly [2, 1, 4]>>;

const summedAllKeep = s.sum(undefined, true);
type T7b = Expect<Equal<(typeof summedAllKeep)["shape"], readonly [1, 1, 1]>>;

const summedNegKeep = s.sum(-1, true);
type T7c = Expect<Equal<(typeof summedNegKeep)["shape"], readonly [2, 3, 1]>>;

const summedFalse = s.sum(1, false); // explicit `false` == default (axis removed)
type T7d = Expect<Equal<(typeof summedFalse)["shape"], readonly [2, 4]>>;

// gradual: a dynamic (non-literal) boolean `keepdims` degrades to the union of
// keep/non-keep shapes — the same deliberate degradation as a dynamic axis
// (`const KeepDims` only pins LITERAL booleans; a variable stays `boolean`).
declare const dynKeep: boolean;
const summedDynKeep = s.sum(1, dynKeep);
type T7e = Expect<Equal<(typeof summedDynKeep)["shape"], readonly [2, 4] | readonly [2, 1, 4]>>;

// @ts-expect-error - axis 3 out of range even with keepdims: error stays at the axis argument
s.sum(3, true);

// resident twin: `WNDArray.sum` declares the `keepdims` signature independently
// of `NDArray.sum`, so pin its return type separately (shares `ReduceAxis` +
// `OkShape`, but a wiring typo in resident.ts wouldn't be caught above).
declare const rw: WNDArray<[2, 3, 4]>;
const rwKeep = rw.sum(1, true);
type T7f = Expect<Equal<(typeof rwKeep)["shape"], readonly [2, 1, 4]>>;
const rwKeepAll = rw.sum(undefined, true);
type T7g = Expect<Equal<(typeof rwKeepAll)["shape"], readonly [1, 1, 1]>>;

const transposed = s.transpose();
type T8 = Expect<Equal<(typeof transposed)["shape"], readonly [4, 3, 2]>>;

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
type T9 = Expect<Equal<(typeof dynAdded)["shape"], readonly [2, number]>>;

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
// HISTORY (three states, honestly tracked — docs/phase-d-vorarbeiten-spec.md
// / -v2-ergebnisse.md "Fund 2" + closure round, 2026-07-13): (1) pre-D-V2.3,
// this assignment was rejected — the class read as fully (but ACCIDENTALLY)
// invariant, a side effect of `sum(...).sum(...)`'s keepdims return-type
// machinery (`AllOnes<S>` producing a genuinely MUTABLE tuple like `[1, 1]`
// that a `readonly 1[]` result for the wide `S` case couldn't satisfy), not
// a deliberately engineered marker (`NDArray` carried no `in`/`out`
// annotation; its variance was always emergent/measured). (2) D-V2.3 wrapped
// `shape` in `Readonly<S>` for an unrelated reason (closing the
// `nd.shape[0] = 99` mutation hole) and, as a pure side effect, changed how
// this SAME comparison resolves: the widening silently opened up (isolated
// via an A/B probe on the real class) — VERIFIED SAFE on its own terms
// (M2: a wider claim is a less precise claim, never a wrong one) but
// ACCIDENTAL, so the V2 slice temporarily pinned this assignment as
// positive/valid. (3) The owner decided (2026-07-13, verify-round closure)
// that accidental variance is not something to ship on purpose — `NDArray`
// is re-invariantized DELIBERATELY via an explicit marker
// (`NDArray.__variance`, see ndarray.ts's doc comment there and on
// `AnyNDArray`), so this assignment is REJECTED again, now for good reason
// instead of by accident. `NDArrayView<out S>` (below) remains the ONE
// checker-enforced covariant surface in this codebase; `AnyNDArray` remains
// the supported both-ways-unsafe erased handle for calling ops on a
// heterogeneous handle — neither changes across any of these three states.
// @ts-expect-error - re-invariantized (2026-07-13): NDArray<[2, 3]> is not assignable to NDArray<readonly number[]> — deliberate, marker-enforced invariance, not the D-V2.3-era accidental widening (see history above)
const stillInvariant: NDArray<readonly number[]> = NDArray.zeros([2, 3]);
void stillInvariant;

// The actually load-bearing direction stays rejected: a dynamic-rank/wide
// handle must NEVER be claimed as a precise literal shape.
declare const wideNd: NDArray<readonly number[]>;
// @ts-expect-error - narrowing: NDArray<readonly number[]> is not assignable to NDArray<[2, 3]> (must stay rejected)
const narrowedNd: NDArray<[2, 3]> = wideNd;
void narrowedNd;

const anyErased: AnyNDArray = NDArray.zeros([2, 3]); // the supported pattern
const anyList: AnyNDArray[] = [NDArray.zeros([2, 3]), NDArray.zeros([7]), NDArray.zeros([])];
void anyErased;
void anyList;

// --- D-V2.3 deep-readonly `shape`: element mutation is a compile error -----
// (A-Auflage, verify-round closure, 2026-07-13). `Readonly<S>` on `.shape`
// blocks ELEMENT writes (`nd.shape[0] = ...`), not just property reassignment
// (`nd.shape = [...]`, already rejected before D-V2.3 too, since `shape` was
// always a `readonly` PROPERTY — only the element-write hole is new here).
// Honesty note (verify-round finding A): the V2 doc comments in ndarray.ts/
// resident.ts already CLAIMED this compile error at three sites
// (`nd.shape[0] = 99` / `view.shape[0] = 99` / the WNDArray equivalent), but
// the original V2 diff never actually pinned any of the three — these three
// assertions were added in the closure round, not present pre-closure.
//
// Honesty note (verify-round closure catchability check): the doc comments'
// illustrative `= 99` is intentionally NOT reused as the pinned value here.
// Empirically (scratch probe against a plain literal tuple, closure round):
// `declare const t: [2, 3]; t[0] = 99;` ALREADY errors (TS2322, "Type '99' is
// not assignable to type '2'") even WITHOUT any `readonly` involved — literal
// tuple-element narrowing alone rejects a MISMATCHED value on a MUTABLE
// tuple, which would make a `= 99` pin pass for the wrong reason and stay
// "red" even if `Readonly<S>` were reverted (confirmed: reverting `NDArray`'s
// `shape` field to plain `S` in a scratch mutant did NOT turn this pin's
// `@ts-expect-error` into "unused" — TS2322 fired instead of TS2540, an
// undetected false pass). `t[0] = 2` (the SAME literal the position already
// holds) has no such confound: it type-checks fine on a mutable tuple and
// ONLY a `readonly` modifier rejects it (TS2540, "Cannot assign ... because
// it is a read-only property" — verified in the same probe). Pinned with the
// matching value below so the assertion is unambiguously about readonly-ness,
// not an incidental side effect of literal narrowing.
declare const ndForMutation: NDArray<[2, 3]>;
// @ts-expect-error - nd.shape[0] = 2 must be a compile error (D-V2.3 deep-readonly; matching value isolates readonly from literal-type narrowing, see note above)
ndForMutation.shape[0] = 2;

declare const viewForMutation: NDArrayView<[2, 3]>;
// @ts-expect-error - view.shape[0] = 2 must be a compile error (D-V2.3 deep-readonly, NDArrayView; matching value, see note above)
viewForMutation.shape[0] = 2;

declare const wndForMutation: WNDArray<[2, 3]>;
// @ts-expect-error - wnd.shape[0] = 2 must be a compile error (D-V2.3 deep-readonly, WNDArray; matching value, see note above)
wndForMutation.shape[0] = 2;

// --- toNestedArray() return type: rank-computed NestedArray<S> (D-V2.2/B5's
// `unknown` pin REVERSED, docs/typed-nested-array-spec.md D6, Baustein-0
// finding 3: WNDArray's top type is `AnyWNDArray`, not `AnyNDArray`).
// `NDArrayView<S>` itself keeps `unknown` — D3, TS2636 under `out S` — so it
// is pinned separately; the concrete classes narrow the return type via
// `implements`, which the checker allows and does not itself verify, so the
// pins below are what catches drift between the three declarations.
type ToNestedArrayViewReturn = Expect<Equal<ReturnType<NDArrayView<[2, 3]>["toNestedArray"]>, unknown>>;

// NDArray: rank 0/1/2/3, `[number, 3]`, dynamic rank, `Shape`, mixed-rank
// union, same-rank union, `never`, rest tuple, top type.
type NDRank0 = Expect<Equal<ReturnType<NDArray<[]>["toNestedArray"]>, number>>;
type NDRank1 = Expect<Equal<ReturnType<NDArray<[3]>["toNestedArray"]>, number[]>>;
type NDRank2 = Expect<Equal<ReturnType<NDArray<[2, 3]>["toNestedArray"]>, number[][]>>;
type NDRank3 = Expect<Equal<ReturnType<NDArray<[2, 3, 4]>["toNestedArray"]>, number[][][]>>;
type NDMixedDim = Expect<Equal<ReturnType<NDArray<[number, 3]>["toNestedArray"]>, number[][]>>;
type NDDynamicRank = Expect<Equal<ReturnType<NDArray<number[]>["toNestedArray"]>, NestedValue>>;
type NDShape = Expect<Equal<ReturnType<NDArray<Shape>["toNestedArray"]>, NestedValue>>;
type NDMixedRankUnion = Expect<Equal<ReturnType<NDArray<[2] | [2, 3]>["toNestedArray"]>, NestedValue>>;
type NDSameRankUnion = Expect<Equal<ReturnType<NDArray<[2, 3] | [4, 5]>["toNestedArray"]>, number[][]>>;
type NDNever = Expect<Equal<ReturnType<NDArray<never>["toNestedArray"]>, NestedValue>>;
type NDRestTuple = Expect<Equal<ReturnType<NDArray<[2, ...number[]]>["toNestedArray"]>, NestedValue>>;
// dt1 (K5, docs/dtype-dt1-spec.md; A2 point 1): `AnyNDArray` widened from
// `NDArray<any>` to `NDArray<any, any>` (the fix for the confident-WRONG
// "any shape, only float64" top type, prototype finding F1) — its
// `toNestedArray()` now spans EVERY dtype's leaf type, `NestedValue |
// NestedBoolValue`, not just `NestedValue`. A mandatory, disclosed
// consequence of K5, not a behavior change to any float64 code path.
type NDAnyTop = Expect<Equal<ReturnType<AnyNDArray["toNestedArray"]>, NestedValue | NestedBoolValue>>;

// WNDArray: same expectation catalog, plus its own top type (`AnyWNDArray`).
type WNDRank0 = Expect<Equal<ReturnType<WNDArray<[]>["toNestedArray"]>, number>>;
type WNDRank1 = Expect<Equal<ReturnType<WNDArray<[3]>["toNestedArray"]>, number[]>>;
type WNDRank2 = Expect<Equal<ReturnType<WNDArray<[2, 3]>["toNestedArray"]>, number[][]>>;
type WNDRank3 = Expect<Equal<ReturnType<WNDArray<[2, 3, 4]>["toNestedArray"]>, number[][][]>>;
type WNDMixedDim = Expect<Equal<ReturnType<WNDArray<[number, 3]>["toNestedArray"]>, number[][]>>;
type WNDDynamicRank = Expect<Equal<ReturnType<WNDArray<number[]>["toNestedArray"]>, NestedValue>>;
type WNDShape = Expect<Equal<ReturnType<WNDArray<Shape>["toNestedArray"]>, NestedValue>>;
type WNDMixedRankUnion = Expect<Equal<ReturnType<WNDArray<[2] | [2, 3]>["toNestedArray"]>, NestedValue>>;
type WNDSameRankUnion = Expect<Equal<ReturnType<WNDArray<[2, 3] | [4, 5]>["toNestedArray"]>, number[][]>>;
type WNDNever = Expect<Equal<ReturnType<WNDArray<never>["toNestedArray"]>, NestedValue>>;
type WNDRestTuple = Expect<Equal<ReturnType<WNDArray<[2, ...number[]]>["toNestedArray"]>, NestedValue>>;
type WNDAnyTop = Expect<Equal<ReturnType<AnyWNDArray["toNestedArray"]>, NestedValue>>;

// Both classes agree for the same S (drift catcher between the two `implements NDArrayView<S>` sites).
type NDWNDAgree = Expect<Equal<ReturnType<NDArray<[2, 3]>["toNestedArray"]>, ReturnType<WNDArray<[2, 3]>["toNestedArray"]>>>;

// Indexed access under `noUncheckedIndexedAccess` (set in this corpus's root
// tsconfig): `[0][1]` is honestly `number | undefined`, not `number` — spec D7
// (the flag-free `number` case is pinned in both consumer smokes).
declare const nestedIndexRecv: NDArray<[2, 3]>;
const nestedIndexed = nestedIndexRecv.toNestedArray()[0]![1];
type NDNestedIndexUnderFlag = Expect<Equal<typeof nestedIndexed, number | undefined>>;

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
// D-V2.3 update: `shapeOf`'s OWN signature returns bare `S` (not `Readonly<S>`
// — it's a hypothetical free function, not a `.shape` read), yet the inferred
// `S` itself comes back `readonly [2, 3]`, not `[2, 3]`: TS infers `S` by
// unifying against `NDArrayView<S>`'s member types, and `nd23`'s `.shape` is
// now `Readonly<[2, 3]>` — the readonly-ness leaks into the inferred `S`
// itself, not just the member access. Re-expressed intent-preservingly.
declare function shapeOf<S extends Shape>(v: NDArrayView<S>): S;
const inferredShape = shapeOf(nd23);
type T14 = Expect<Equal<typeof inferredShape, readonly [2, 3]>>;

// =============================================================================
// Phase-D V2 (docs/phase-d-vorarbeiten-spec.md, D-V2.2): `WNDArray<S>` also
// `implements NDArrayView<S>` now — the resident backend gets the same
// checker-enforced covariant read view as `NDArray`, not just the naive one.
// =============================================================================

declare const wnd23: WNDArray<[2, 3]>;
const wnd23AsExactView: NDArrayView<[2, 3]> = wnd23; // exact-shape assignment
const wnd23AsWidenedView: NDArrayView<Shape> = wnd23; // widening, same as NDArray above
void wnd23AsExactView;
void wnd23AsWidenedView;

// Same downcast rejection as the NDArray-backed view — `out S` doesn't care
// which concrete class produced the view.
declare const wideWndView: NDArrayView<Shape>;
// @ts-expect-error - NDArrayView<Shape> is not assignable back to NDArrayView<[2, 3]>: downcast, not widening
const narrowedWndView: NDArrayView<[2, 3]> = wideWndView;
void narrowedWndView;

// Heterogeneous container mixing BOTH concrete backends behind the one safe
// top type — the point of a checker-enforced (not erased) view.
const mixedBackendViews: NDArrayView<Shape>[] = [nd23, wnd23];
void mixedBackendViews;

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
type UW1 = Expect<Equal<(typeof wAdded)["shape"], readonly [2, 3]>>;
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
type UW2 = Expect<Equal<(typeof wMatmulResult)["shape"], readonly [2, 4]>>;
void wMatmulResult;

// sum: Facette (c), mixed-rank receiver -> accepted, degraded to `readonly
// number[]` (never the pre-fix confident-but-wrong single shape).
declare const wMixedRankRecv: WNDArray<[2, 3] | [2, 3, 4]>;
const wMixedRankSummed = wMixedRankRecv.sum(2);
type UW3 = Expect<Equal<(typeof wMixedRankSummed)["shape"], readonly number[]>>;
void wMixedRankSummed;

// =============================================================================
// Union-Axis-Mini-Scheibe (docs/union-axis-mini-spec.md): Facette (1) call-site
// pins — a union AXIS argument (as opposed to Facette (c)'s union RECEIVER
// shape above) degrades to `readonly number[]`, both on the JS `NDArray` and
// the WASM-resident `WNDArray` twin (proves the same imported ReduceAxis
// machinery covers both surfaces, zero resident.ts edits — D-A.3).
// =============================================================================

// PRE-FIX: this was the confidently-wrong `NDArray<[3]>` (only the axis-0
// union member's result survived; the axis-2 out-of-range member's
// ShapeError was silently discarded by Guard/OkShape) even though the
// runtime value `2` makes `sumRuntime` throw. POST-FIX: degrades correctly.
declare const uAxisRecv: NDArray<[2, 3]>;
const uAxisSummed = uAxisRecv.sum(0 as 0 | 2);
type UA_CALL1 = Expect<Equal<(typeof uAxisSummed)["shape"], readonly number[]>>;
void uAxisSummed;

// WNDArray twin of the same call-site fix.
declare const wUAxisRecv: WNDArray<[2, 3]>;
const wUAxisSummed = wUAxisRecv.sum(0 as 0 | 2);
type UA_CALL2 = Expect<Equal<(typeof wUAxisSummed)["shape"], readonly number[]>>;
void wUAxisSummed;

// Workaround pin (Facette (2), release-relevant FOLLOWUPS item): an EXPLICIT
// type argument bypasses the optional-parameter inference stripping and
// reaches the same union-axis filter — degrades correctly post-fix, proving
// the workaround the FOLLOWUPS item recommends actually works today.
declare const uWorkaroundAxis: 0 | undefined;
const uWorkaroundSummed = uAxisRecv.sum<0 | undefined>(uWorkaroundAxis);
type UA_WORKAROUND = Expect<Equal<(typeof uWorkaroundSummed)["shape"], readonly number[]>>;
void uWorkaroundSummed;

// Facette (2) axis — CLOSED in Item 11 / S1 (sum-Overload-Umbau, KD-2). The
// realistic call form `a.sum(u)` with `u: 0 | undefined` and NO explicit type
// argument no longer resolves confidently to `NDArray<[3]>`: the multi-arg
// overloads make `axis` REQUIRED, so TS no longer strips `undefined` from the
// inferred `Axis` — the `0 | undefined` union now reaches `ReduceAxis`'s
// `IsUnion` filter and degrades to no-claim (`readonly number[]`). This pin was
// the `UA_GAP` sentinel that OBSERVED the open gap; per its own design it flips
// the moment the signature change lands — here it does, and is REVERSED to guard
// the CLOSURE. COVENANT v2 M2 note, axis facet — resolved.
declare const uGapAxis: 0 | undefined;
const uGapSummed = uAxisRecv.sum(uGapAxis);
type UA_AXIS_CLOSED = Expect<Equal<(typeof uGapSummed)["shape"], readonly number[]>>;
void uGapSummed;

// Facette (2) keepdims — CLOSED in Item 11 / S1 (KD-2). `a.sum(0, kd)` with
// `kd: true | undefined` no longer resolves confidently to `NDArray<[1, 3]>`
// (as if keepdims were surely `true`): the 2-arg overload makes `keepdims`
// REQUIRED, so `undefined` is no longer stripped from the inferred `KeepDims`,
// and reduce.ts's widened `KeepDims extends boolean | undefined` lets the union
// distribute HONESTLY through the `KeepDims extends true` conditionals to a real
// shape union (keepdims=true -> [1,3], keepdims=false -> [3]). COVENANT v2 M2
// note, keepdims facet — resolved.
declare const uGapKeep: true | undefined;
const uGapKeepSummed = uAxisRecv.sum(0, uGapKeep);
type UA_KEEP_CLOSED = Expect<Equal<(typeof uGapKeepSummed)["shape"], readonly [3] | readonly [1, 3]>>;
void uGapKeepSummed;

// Same facet at the full-reduction (undefined axis) branch.
const uGapKeepFull = uAxisRecv.sum(undefined, uGapKeep);
type UA_KEEP_CLOSED_FULL = Expect<Equal<(typeof uGapKeepFull)["shape"], readonly [] | readonly [1, 1]>>;
void uGapKeepFull;

// WNDArray twins of both M2 closures (axis + keepdims facet) — same overload
// umbau mirrored onto resident.ts, so the resident surface degrades identically.
declare const wUGapAxis: 0 | undefined;
const wUGapSummed = wUAxisRecv.sum(wUGapAxis);
type WUA_AXIS_CLOSED = Expect<Equal<(typeof wUGapSummed)["shape"], readonly number[]>>;
void wUGapSummed;

declare const wUGapKeep: true | undefined;
const wUGapKeepSummed = wUAxisRecv.sum(0, wUGapKeep);
type WUA_KEEP_CLOSED = Expect<Equal<(typeof wUGapKeepSummed)["shape"], readonly [3] | readonly [1, 3]>>;
void wUGapKeepSummed;

// Negative union member.
const uNegSummed = uAxisRecv.sum(-1 as -1 | 0);
type UA_NEG = Expect<Equal<(typeof uNegSummed)["shape"], readonly number[]>>;
void uNegSummed;

// ALL-invalid union (every member out of range for rank 2) — accepted, not
// statically rejected (documented incompleteness, mirrors union-DIM policy).
const uAllInvalidSummed = uAxisRecv.sum(2 as 2 | 5);
type UA_ALL_INVALID = Expect<Equal<(typeof uAllInvalidSummed)["shape"], readonly number[]>>;
void uAllInvalidSummed;

// Union axis x keepdims (true / false / dynamic boolean) — KeepDims never
// un-degrades a union AXIS.
const uKeepTrue = uAxisRecv.sum(0 as 0 | 2, true);
type UA_KEEP_TRUE = Expect<Equal<(typeof uKeepTrue)["shape"], readonly number[]>>;
void uKeepTrue;

const uKeepFalse = uAxisRecv.sum(0 as 0 | 2, false);
type UA_KEEP_FALSE = Expect<Equal<(typeof uKeepFalse)["shape"], readonly number[]>>;
void uKeepFalse;

declare const uDynKeep: boolean;
const uKeepDyn = uAxisRecv.sum(0 as 0 | 2, uDynKeep);
type UA_KEEP_DYN = Expect<Equal<(typeof uKeepDyn)["shape"], readonly number[]>>;
void uKeepDyn;

// already-safe control pin (Policy: KeepDims-`boolean` at a LITERAL axis is
// NOT touched by this slice — natural distribution already produces the
// correct per-member result union, no filter needed or added).
declare const uSafeKeepDims: boolean;
const uSafeKeepSummed = uAxisRecv.sum(1, uSafeKeepDims);
type UA_SAFE_KEEPDIMS = Expect<Equal<(typeof uSafeKeepSummed)["shape"], readonly [2] | readonly [2, 1]>>;
void uSafeKeepSummed;

// --- Kontroll-Pins (must NOT change, M3): literal single axis stays precise,
// dynamic axis stays unchanged, out-of-range single axis keeps its verbatim
// message at the offending argument. -----------------------------------------

const uControlLiteral = uAxisRecv.sum(0);
type UA_CONTROL_LITERAL = Expect<Equal<(typeof uControlLiteral)["shape"], readonly [3]>>;
void uControlLiteral;

declare const uControlDynamicAxis: number;
const uControlDynamic = uAxisRecv.sum(uControlDynamicAxis);
type UA_CONTROL_DYNAMIC = Expect<Equal<(typeof uControlDynamic)["shape"], readonly number[]>>;
void uControlDynamic;

// @ts-expect-error - axis 2 is out of range for rank-2 shape [2,3]: error stays at the argument, message unchanged from pre-fix
uAxisRecv.sum(2);

// =============================================================================
// Op-Scheibe W1 (docs/op-w1-argmax-topk-spec.md): `argmax`/`topk` type pins.
// `argmax(axis[, keepdims])` reuses `ReduceAxis`/`Guard`/`OkShape` UNCHANGED
// (same machinery `sum` above is already pinned against), so this section
// proves the WIRING (D2's overload shape + the niladic `number` deviation),
// not the underlying degradation rules a second time — mirrors the WNDArray
// section's own "same imported machinery, proven not assumed" precedent.
// `topk(k)` gets full first-class coverage (D3): its own `TopkCheck`/
// `TopkShape` guard is new machinery, not a reuse.
// =============================================================================

// --- argmax(): niladic -> plain `number` (D2), never `NDArray<[]>` --------

const wArg = NDArray.zeros([2, 3, 4]);
const argmaxFlat = wArg.argmax();
type ARGMAX_FLAT = Expect<Equal<typeof argmaxFlat, number>>;

// --- argmax(axis[, keepdims]): exact literal tuples ------------------------

const argmaxAxis1 = wArg.argmax(1);
type ARGMAX_AXIS1 = Expect<Equal<(typeof argmaxAxis1)["shape"], readonly [2, 4]>>;

const argmaxAxis1Keep = wArg.argmax(1, true);
type ARGMAX_AXIS1_KEEP = Expect<Equal<(typeof argmaxAxis1Keep)["shape"], readonly [2, 1, 4]>>;

const argmaxAxis1NoKeep = wArg.argmax(1, false); // explicit `false` == default
type ARGMAX_AXIS1_NOKEEP = Expect<Equal<(typeof argmaxAxis1NoKeep)["shape"], readonly [2, 4]>>;

// `argmax(undefined)` is the 1-ARG overload with an axis VALUE of `undefined`
// (full reduction), a DIFFERENT overload from the true 0-arg `argmax()`
// above — mirrors `sum(undefined)`'s own existing shape exactly (D2).
const argmaxUndefAxis = wArg.argmax(undefined);
type ARGMAX_UNDEF_AXIS = Expect<Equal<(typeof argmaxUndefAxis)["shape"], readonly []>>;

const argmaxUndefKeep = wArg.argmax(undefined, true);
type ARGMAX_UNDEF_KEEP = Expect<Equal<(typeof argmaxUndefKeep)["shape"], readonly [1, 1, 1]>>;

const argmaxNeg = wArg.argmax(-1);
type ARGMAX_NEG = Expect<Equal<(typeof argmaxNeg)["shape"], readonly [2, 3]>>;

// --- argmax(axis): degradations (same ReduceAxis machinery as sum) --------

declare const dynAxisArgmax: number;
const argmaxDynAxis = wArg.argmax(dynAxisArgmax);
type ARGMAX_DYN_AXIS = Expect<Equal<(typeof argmaxDynAxis)["shape"], readonly number[]>>;

declare const argmaxMixedRankRecv: NDArray<[2, 3] | [2, 3, 4]>;
const argmaxMixedSummed = argmaxMixedRankRecv.argmax(2);
type ARGMAX_MIXED_RANK = Expect<Equal<(typeof argmaxMixedSummed)["shape"], readonly number[]>>;

const argmaxUnionAxis = uAxisRecv.argmax(0 as 0 | 2);
type ARGMAX_UNION_AXIS = Expect<Equal<(typeof argmaxUnionAxis)["shape"], readonly number[]>>;

declare const argmaxDynKeep: true | undefined;
const argmaxKeepUnion = wArg.argmax(1, argmaxDynKeep);
type ARGMAX_KEEP_UNION = Expect<Equal<(typeof argmaxKeepUnion)["shape"], readonly [2, 4] | readonly [2, 1, 4]>>;

// @ts-expect-error - axis 5 is out of range for rank-3 shape [2,3,4]: error stays at the argument (ReduceAxis reused unmodified from sum, verbatim message)
wArg.argmax(5);

// Message-equality pin (T3d): the compile-time ShapeError for an
// out-of-range literal axis is the SAME `ReduceAxis` type `sum` already
// uses — this is a Guard/Equal proof that `argmax`'s axis machinery is the
// identical import, not a re-derivation, and that its wording is
// byte-for-byte the `reduce:` stem `argmaxRuntime` throws at runtime
// (runtime.ts, D4).
type ARGMAX_AXIS_OOB_MSG = Expect<
  Equal<Guard<ReduceAxis<[2, 3, 4], 5>, 5>, { readonly __shapeError: "reduce: axis 5 is out of range for shape [2,3,4] (rank 3)" }>
>;

// --- topk(k): exact literal tuples, incl. the k=0/k=D valid boundaries ----

const vTopk = NDArray.zeros([5]);

const topk3 = vTopk.topk(3);
type TOPK3_VALUES = Expect<Equal<(typeof topk3.values)["shape"], readonly [3]>>;
type TOPK3_INDICES = Expect<Equal<(typeof topk3.indices)["shape"], readonly [3]>>;

const topk0 = vTopk.topk(0); // k=0: VALID (D3 boundary)
type TOPK0_VALUES = Expect<Equal<(typeof topk0.values)["shape"], readonly [0]>>;
type TOPK0_INDICES = Expect<Equal<(typeof topk0.indices)["shape"], readonly [0]>>;

const topkD = vTopk.topk(5); // k=D: VALID (D3 boundary)
type TOPKD_VALUES = Expect<Equal<(typeof topkD.values)["shape"], readonly [5]>>;
type TOPKD_INDICES = Expect<Equal<(typeof topkD.indices)["shape"], readonly [5]>>;

// --- topk(k): compile errors AT the k argument (DotCheck precedent) -------

// @ts-expect-error - k=-1 is negative: error stays at the k argument
vTopk.topk(-1);

// @ts-expect-error - k=1.5 is non-integer (dot-form): error stays at the k argument
vTopk.topk(1.5);

// @ts-expect-error - k=6 exceeds the vector length 5: error stays at the k argument
vTopk.topk(6);

// @ts-expect-error - k = Number.MAX_SAFE_INTEGER vastly exceeds length 5: still a PROVABLE compile error (the digit machinery does not choke on large-but-representable literals)
vTopk.topk(9007199254740991);

const rank2Recv = NDArray.zeros([2, 3]);
// @ts-expect-error - rank-2 receiver: topk requires rank-1 (DotCheck precedent: the RECEIVER's problem surfaces at the k argument, same as dot/cosineSimilarity)
rank2Recv.topk(2);

const rank0Recv = NDArray.zeros([]);
// @ts-expect-error - rank-0 receiver: topk requires rank-1; error stays at the k argument
rank0Recv.topk(1);

// --- topk(k): degradations (never a confidently-wrong literal claim) ------

declare const dynK: number;
const topkDyn = vTopk.topk(dynK);
type TOPK_DYN_K_VALUES = Expect<Equal<(typeof topkDyn.values)["shape"], readonly [number]>>;
type TOPK_DYN_K_INDICES = Expect<Equal<(typeof topkDyn.indices)["shape"], readonly [number]>>;

declare const unionK: 2 | 3; // uniformly-valid union: still no-claim (union filter runs unconditionally)
const topkUnion = vTopk.topk(unionK);
type TOPK_UNION_K = Expect<Equal<(typeof topkUnion.values)["shape"], readonly [number]>>;

declare const unionKMixed: 2 | 10; // 10 alone would be a hard error; the union as a whole still degrades, never confidently accepts OR rejects
const topkUnionMixed = vTopk.topk(unionKMixed);
type TOPK_UNION_K_MIXED = Expect<Equal<(typeof topkUnionMixed.values)["shape"], readonly [number]>>;

// MAX_SAFE_INTEGER-adjacent edge: an exponent-form literal (`1e21`, beyond
// the digit machinery's plain-digit-string subset, CLAUDE.md's TS-limits
// section) degrades to no-claim rather than lying either way.
const topkExp = vTopk.topk(1e21);
type TOPK_EXP_K_VALUES = Expect<Equal<(typeof topkExp.values)["shape"], readonly [number]>>;
type TOPK_EXP_K_INDICES = Expect<Equal<(typeof topkExp.indices)["shape"], readonly [number]>>;

// --- topk(k): message-equality pins (T3d), via Guard/Equal directly -------

type TOPK_RANK_MSG = Expect<
  Equal<Guard<TopkCheck<[2, 3], 2>, 2>, { readonly __shapeError: "topk: expected a 1-D vector (got shape [2,3])" }>
>;
type TOPK_NEGATIVE_K_MSG = Expect<
  Equal<Guard<TopkCheck<[5], -1>, -1>, { readonly __shapeError: "topk: k must be a non-negative integer (got -1)" }>
>;
type TOPK_DOTFORM_K_MSG = Expect<
  Equal<Guard<TopkCheck<[5], 1.5>, 1.5>, { readonly __shapeError: "topk: k must be a non-negative integer (got 1.5)" }>
>;
type TOPK_BOUNDS_K_MSG = Expect<
  Equal<Guard<TopkCheck<[5], 10>, 10>, { readonly __shapeError: "topk: k=10 exceeds the vector length 5" }>
>;

// --- topk(k): RankUnknowable receiver -> uniform no-claim (policy pin) ----

declare const topkMixedRankRecv: NDArray<readonly [2, 3] | readonly [5]>;
// Deliberately COMPILES, with NO static claim: on a mixed-rank-union
// receiver even a provably-invalid k (-1 throws at runtime regardless of
// which rank is realized) degrades to no-claim, mirroring the uniform
// degrade of ALL seven rank gates (D-V1.3 house policy,
// docs/phase-d-vorarbeiten-v1-ergebnisse.md) instead of making a partial
// claim on an unknowable receiver. The runtime backstop (`topkRuntime`'s
// unconditional k validation) stays authoritative. Spec v4 corrected D3 to
// exactly this form (Verify-B finding F2) — this pin documents the policy;
// a future deliberate strengthening must consciously re-express it.
const topkMixedRankNegK = topkMixedRankRecv.topk(-1);
type TOPK_MIXEDRANK_NEG_K_VALUES = Expect<Equal<(typeof topkMixedRankNegK.values)["shape"], readonly [number]>>;
type TOPK_MIXEDRANK_NEG_K_INDICES = Expect<Equal<(typeof topkMixedRankNegK.indices)["shape"], readonly [number]>>;

// =============================================================================
// Op-Scheibe W2 (docs/op-w2-scalar-mean-spec.md): scalar-overload (`add`/
// `sub`/`mul`/`div`) + `mean` type pins.
// =============================================================================

// --- div(s): shape-preserving scalar overload (D2) — exact tuple, rank 0,
// wide/dynamic-rank receiver, readonly-S receiver -----------------------

const scalarBase = NDArray.zeros([2, 3]);
const scalarDivided = scalarBase.div(2);
type SCALAR_DIV_SHAPE = Expect<Equal<(typeof scalarDivided)["shape"], readonly [2, 3]>>;

const scalarRank0 = NDArray.zeros([]);
const scalarRank0Divided = scalarRank0.div(2);
type SCALAR_DIV_RANK0 = Expect<Equal<(typeof scalarRank0Divided)["shape"], readonly []>>;

// wide/dynamic-rank receiver: the scalar overload stays callable and
// degrades exactly like the binary overload already does — never a
// confident literal claim on an unknowable shape.
declare const scalarWide: NDArray<readonly number[]>;
const scalarWideDivided = scalarWide.div(2);
type SCALAR_DIV_WIDE = Expect<Equal<(typeof scalarWideDivided)["shape"], readonly number[]>>;

// Readonly-S receiver (a literal `readonly [...]` type argument threads
// through the scalar overload identically to every other op above).
declare const scalarReadonlyS: NDArray<readonly [4, 5]>;
const scalarReadonlySDivided = scalarReadonlyS.div(2);
type SCALAR_DIV_READONLY_S = Expect<Equal<(typeof scalarReadonlySDivided)["shape"], readonly [4, 5]>>;

// --- union-over-boundary (D2 v2): a UNION argument spanning both the scalar
// and the NDArray overload is rejected AS A WHOLE (TS2769), even though each
// member alone would be valid — the exact overload-resolution kink
// `NDArray.backend(kind)` already carries (see its doc comment, ndarray.ts).

declare const scalarOrArray: number | NDArray<[3]>;
// @ts-expect-error - a UNION argument spanning both the scalar overload and the NDArray overload is rejected as a whole (TS2769) even though each member alone is individually valid — documented D2 v2 kink, same precedent as NDArray.backend(kind) above
scalarBase.add(scalarOrArray);

// The documented narrowing workaround (`typeof x === "number" ? … : …`,
// same recipe `backend()`'s own doc comment recommends) actually compiles
// and resolves each branch to its own precise overload.
declare const narrowInput: number | NDArray<[2, 3]>;
if (typeof narrowInput === "number") {
  const narrowedNum = scalarBase.add(narrowInput);
  type SCALAR_NARROW_NUM = Expect<Equal<(typeof narrowedNum)["shape"], readonly [2, 3]>>;
  void narrowedNum;
} else {
  const narrowedArr = scalarBase.add(narrowInput);
  type SCALAR_NARROW_ARR = Expect<Equal<(typeof narrowedArr)["shape"], readonly [2, 3]>>;
  void narrowedArr;
}

// --- workaround path (D3): the OLD `[1]`-wrap call still compiles and still
// resolves through the ordinary binary overload, unaffected by the new
// scalar overload's addition (overload-set growth is additive, not
// replacing). ----------------------------------------------------------

const scalarWorkaround = scalarBase.div(NDArray.fromArray([1], [2]));
type SCALAR_DIV_WORKAROUND = Expect<Equal<(typeof scalarWorkaround)["shape"], readonly [2, 3]>>;

// `div(nd)` stays generic/unaffected: a plain NDArray argument still resolves
// through the broadcast overload, never mistaken for the scalar one.
const scalarDivByArray = scalarBase.div(NDArray.zeros([3]));
type SCALAR_DIV_BY_ARRAY = Expect<Equal<(typeof scalarDivByArray)["shape"], readonly [2, 3]>>;

// --- mean: overloads 0/1/2 mirror `sum`'s own shape (D4) — WIRING pins only
// (argmax precedent, docs/op-w1-argmax-topk-spec.md section above): proves
// `mean` reuses `ReduceAxis`/`Guard`/`OkShape` correctly, does NOT re-litigate
// the union-axis mini-scheibe's own 15-pin degradation family (sum-only).
// =============================================================================

// mean(): niladic -> NDArray<[]>-shaped, like `sum()` (D7) — NOT a bare
// `number` (D4: mean stays a chainable reduction, unlike argmax()).
const meanFlat = wArg.mean();
type MEAN_FLAT = Expect<Equal<(typeof meanFlat)["shape"], readonly []>>;

// mean(axis[, keepdims]): exact literal tuples (basic positive wiring proof).
const meanAxis1 = wArg.mean(1);
type MEAN_AXIS1 = Expect<Equal<(typeof meanAxis1)["shape"], readonly [2, 4]>>;

const meanAxis1Keep = wArg.mean(1, true);
type MEAN_AXIS1_KEEP = Expect<Equal<(typeof meanAxis1Keep)["shape"], readonly [2, 1, 4]>>;

const meanNeg = wArg.mean(-1);
type MEAN_NEG = Expect<Equal<(typeof meanNeg)["shape"], readonly [2, 3]>>;

// mean(axis): degradations — same ReduceAxis machinery as sum/argmax, only
// the WIRING is proven here (dyn axis, mixed rank, union axis, keepdims-union).

declare const dynAxisMean: number;
const meanDynAxis = wArg.mean(dynAxisMean);
type MEAN_DYN_AXIS = Expect<Equal<(typeof meanDynAxis)["shape"], readonly number[]>>;

declare const meanMixedRankRecv: NDArray<[2, 3] | [2, 3, 4]>;
const meanMixedSummed = meanMixedRankRecv.mean(2);
type MEAN_MIXED_RANK = Expect<Equal<(typeof meanMixedSummed)["shape"], readonly number[]>>;

const meanUnionAxis = uAxisRecv.mean(0 as 0 | 2);
type MEAN_UNION_AXIS = Expect<Equal<(typeof meanUnionAxis)["shape"], readonly number[]>>;

declare const meanDynKeep: true | undefined;
const meanKeepUnion = wArg.mean(1, meanDynKeep);
type MEAN_KEEP_UNION = Expect<Equal<(typeof meanKeepUnion)["shape"], readonly [2, 4] | readonly [2, 1, 4]>>;

// @ts-expect-error - axis 5 is out of range for rank-3 shape [2,3,4]: error stays at the argument (ReduceAxis reused unmodified from sum, verbatim message)
wArg.mean(5);

// Message-equality pin (mirrors ARGMAX_AXIS_OOB_MSG above): the compile-time
// ShapeError for an out-of-range literal axis is the SAME `ReduceAxis` type
// sum/argmax already use — proves `mean`'s axis machinery is the identical
// import, not a re-derivation, wording byte-for-byte the `reduce:` stem
// `sumRuntime`/`meanRuntime` throw at runtime (runtime.ts, D5).
type MEAN_AXIS_OOB_MSG = Expect<
  Equal<Guard<ReduceAxis<[2, 3, 4], 5>, 5>, { readonly __shapeError: "reduce: axis 5 is out of range for shape [2,3,4] (rank 3)" }>
>;

// WNDArray twin note (superseded — originally D7 v2, structural, "not a
// pin"): `WNDArray` gained add/sub/mul/div in WASM parity S1 and `mean` in
// WASM parity S2 — both now have their own WNDArray-side pin sections
// (search "WASM parity S1"/"WASM parity S2" below) — no longer a documented
// absence.

// =============================================================================
// Op-Scheibe W3 (docs/op-w3-sqrt-spec.md, D3): `sqrt()` — shape-PRESERVING at
// every rank (5 Equal pins: literal, rank 0, wide, readonly-S, dynamic rank),
// plus a non-vacuous niladic-arity pin (sqrt takes no argument).
// =============================================================================

const sqrtLiteral = NDArray.zeros([2, 3]).sqrt();
type SQRT_LITERAL = Expect<Equal<(typeof sqrtLiteral)["shape"], readonly [2, 3]>>;

const sqrtRank0 = NDArray.zeros([]).sqrt();
type SQRT_RANK0 = Expect<Equal<(typeof sqrtRank0)["shape"], readonly []>>;

declare const sqrtWide: NDArray<readonly number[]>;
const sqrtWideResult = sqrtWide.sqrt();
type SQRT_WIDE = Expect<Equal<(typeof sqrtWideResult)["shape"], readonly number[]>>;

declare const sqrtReadonlyS: NDArray<readonly [4, 5]>;
const sqrtReadonlySResult = sqrtReadonlyS.sqrt();
type SQRT_READONLY_S = Expect<Equal<(typeof sqrtReadonlySResult)["shape"], readonly [4, 5]>>;

declare const sqrtDynRank: NDArray<Shape>;
const sqrtDynRankResult = sqrtDynRank.sqrt();
type SQRT_DYN_RANK = Expect<Equal<(typeof sqrtDynRankResult)["shape"], Readonly<Shape>>>;

// @ts-expect-error - sqrt() is niladic: it takes no argument, unlike add/sub/mul/div's scalar overload
sqrtLiteral.sqrt(1);

// =============================================================================
// Op-Scheibe W4 (docs/op-w4-stack-spec.md, D5; Baustein-0-Addendum F1-F8):
// `NDArray.stack(rows)` — exact tuple shapes, heterogeneous-tuple/rank/empty
// rejections with message-equality pins (`StackCheck` operates directly on
// `readonly Shape[]` — no `NDArray` wrapper needed for these pins, the same
// precedent `TopkCheck`'s own message pins above already use), array-input
// degradations (F5/F6/F7/F8), `RankUnknowable` no-claim (both a tuple
// member and an array element), and a `const`-tuple-inference positive pin
// (no `as const` needed).
// =============================================================================

// --- exact tuple shapes: [2,3] (two [3] rows) and [3,4] (three [4] rows) ---

const stackA = NDArray.zeros([3]);
const stackB = NDArray.zeros([3]);
const stacked23 = NDArray.stack([stackA, stackB]);
type STACK_23 = Expect<Equal<(typeof stacked23)["shape"], readonly [2, 3]>>;

const stackC = NDArray.zeros([4]);
const stackD = NDArray.zeros([4]);
const stackE = NDArray.zeros([4]);
const stacked34 = NDArray.stack([stackC, stackD, stackE]);
type STACK_34 = Expect<Equal<(typeof stacked34)["shape"], readonly [3, 4]>>;

// --- heterogeneous tuple: mismatched literal D -> rejected AT the argument ---

const stackMismatchA = NDArray.zeros([3]);
const stackMismatchB = NDArray.zeros([4]);
// @ts-expect-error - [3] vs [4] row length mismatch: error stays at the rows argument
NDArray.stack([stackMismatchA, stackMismatchB]);

// Message-equality pin (D5): the compile-time ShapeError mirrors
// `stackRuntime`'s own throw verbatim (runtime.ts) — pinned directly by a
// unit test in scalar-mean.test.ts too, not just here.
type STACK_MISMATCH_MSG = Expect<
  Equal<
    Guard<StackCheck<readonly [readonly [3], readonly [4]]>, readonly [readonly [3], readonly [4]]>,
    { readonly __shapeError: "stack: row length mismatch (expected 3, got 4 at index 1)" }
  >
>;

// --- rank != 1 tuple member -> rejected AT the argument ---------------------

const stackRank2 = NDArray.zeros([2, 3]);
const stackRank1 = NDArray.zeros([3]);
// @ts-expect-error - rank-2 member at index 0: error stays at the rows argument
NDArray.stack([stackRank2, stackRank1]);

type STACK_RANK_MSG = Expect<
  Equal<
    Guard<StackCheck<readonly [readonly [2, 3], readonly [3]]>, readonly [readonly [2, 3], readonly [3]]>,
    { readonly __shapeError: "stack: expected 1-D rows (got shape [2,3] at index 0)" }
  >
>;

// --- empty tuple literal -> rejected AT the argument (F3) -------------------

// @ts-expect-error - empty tuple literal must be rejected (F3): a `[]` call is a guaranteed runtime throw ("expected at least one row")
NDArray.stack([]);

type STACK_EMPTY_MSG = Expect<
  Equal<Guard<StackCheck<readonly []>, readonly []>, { readonly __shapeError: "stack: expected at least one row" }>
>;

// --- array input: honest [number, D] (F5) ------------------------------------

declare const stackArr3: readonly NDArray<[3]>[];
const stackedArr3 = NDArray.stack(stackArr3);
type STACK_ARRAY_D = Expect<Equal<(typeof stackedArr3)["shape"], readonly [number, 3]>>;

// --- array with a per-row dynamic dim -> wide [number, number] (F6) --------

declare const stackArrDyn: readonly NDArray<[number]>[];
const stackedArrDyn = NDArray.stack(stackArrDyn);
type STACK_ARRAY_WIDE = Expect<Equal<(typeof stackedArrDyn)["shape"], readonly [number, number]>>;

// --- array with a uniform, provably-wrong literal rank -> rejected (F7) -----

declare const stackArrRank2: readonly NDArray<[2, 3]>[];
// @ts-expect-error - every possible call of this array type is a guaranteed throw (F7: even the empty array throws "expected at least one row")
NDArray.stack(stackArrRank2);

// --- array with a union element type -> wide [number, number] (F8) ---------

declare const stackArrUnion: readonly (NDArray<[3]> | NDArray<[4]>)[];
const stackedArrUnion = NDArray.stack(stackArrUnion);
type STACK_ARRAY_UNION = Expect<Equal<(typeof stackedArrUnion)["shape"], readonly [number, number]>>;

// --- RankUnknowable member -> no-claim, never confidently wrong ------------

declare const stackDynRankRow: NDArray<number[]>;
declare const stackKnownRow: NDArray<[3]>;
// Compiles (no @ts-expect-error): an unknowable-rank row can't be proven
// wrong, so N stays the literal tuple length (2) while D widens to `number`
// — never a confidently-wrong literal D.
const stackedUnknowableRank = NDArray.stack([stackDynRankRow, stackKnownRow]);
type STACK_UNKNOWABLE_RANK = Expect<Equal<(typeof stackedUnknowableRank)["shape"], readonly [2, number]>>;

declare const stackArrDynRank: readonly NDArray<number[]>[];
const stackedArrDynRank = NDArray.stack(stackArrDynRank); // compiles: array of unknowable-rank rows is no-claim, never rejected
type STACK_ARRAY_UNKNOWABLE_RANK = Expect<Equal<(typeof stackedArrDynRank)["shape"], readonly [number, number]>>;

// --- const-tuple inference: an inline literal stays a TUPLE, no `as const` ---
// (D4: `const Rows` means callers never write `as const` — same rationale
// `zeros`/`ones`/`fromArray`'s own `const S` already documents. The array
// literal must be written INLINE at the call site for `const` to preserve
// its tuple-ness — same as any other `const` type parameter, a value first
// bound to a plain variable widens to a regular array before it ever
// reaches `stack`.)

function stackMakeRow(): NDArray<[3]> {
  return NDArray.zeros([3]);
}
const stackedInferred = NDArray.stack([stackMakeRow(), stackMakeRow()]);
type STACK_CONST_INFERENCE = Expect<Equal<(typeof stackedInferred)["shape"], readonly [2, 3]>>;

// --- D=0 rows are valid --------------------------------------------------------

const stackZero1 = NDArray.zeros([0]);
const stackZero2 = NDArray.zeros([0]);
const stackedZero = NDArray.stack([stackZero1, stackZero2]);
type STACK_D0 = Expect<Equal<(typeof stackedZero)["shape"], readonly [2, 0]>>;

// =============================================================================
// W4-Nachtrag: Verify-Runde Baustein B (BLOCKER-class M2 finding, fixed in
// StackFold — see its doc comment in vector.ts for the full root-cause
// trace): a row whose OWN static type is a union of same-rank shapes (an
// ordinary ternary, no `stack`-specific machinery needed to produce one)
// used to distribute through `StackFold`'s naked `Head extends readonly
// [infer D]` check, forking the REST of the fold into parallel per-member
// continuations that could reach DIFFERENT verdicts — `Guard`'s tuple-
// wrapped uniform-error-only rejection then accepted the resulting MIXED
// union, and `StackShape`'s `Extract<..., Dim>` silently dropped the
// `ShapeError` member and kept a specific literal `D` that the runtime call
// could (and did) reject. Fixed by gating `IsUnion<Head>` BEFORE the naked
// destructure (the same `ReduceAxis` load-bearing-position precedent, D-
// V1.3/Union-Axis-Mini-Scheibe) — a union-shaped row now ALWAYS widens to
// no-claim, never distributes, never produces a `ShapeError` (not even on a
// double mismatch where EVERY union member would individually fail — a
// deliberate, simpler-than-necessary uniform degradation per D2's "no-claim
// statt Falsch-Claim" policy, not a missed opportunity: proving "every
// member fails" would need extra machinery for a marginal case, and
// `stackRuntime` stays the authoritative backstop regardless).
// =============================================================================

// --- B's exact repro: an ordinary ternary-typed row + a fixed row ----------

declare const stackUnionFlag: boolean;
const stackFixed3 = NDArray.zeros([3]);
const stackTernaryRow = stackUnionFlag ? NDArray.zeros([3]) : NDArray.zeros([4]); // NDArray<[3]> | NDArray<[4]>, an ORDINARY union, no cast

// Compiles (no @ts-expect-error): before the fix this resolved to a
// CONFIDENT `readonly [2, 3]` and threw at runtime for the `[4]` branch —
// the exact Verify-B repro. Now widens honestly to `readonly [2, number]`.
const stackReproResult = NDArray.stack([stackFixed3, stackTernaryRow]);
type STACK_UNION_ROW_REPRO = Expect<Equal<(typeof stackReproResult)["shape"], readonly [2, number]>>;

// Symmetry pin: same union row, swapped position — the fix must not be
// order-sensitive (StackFold folds left-to-right; the union could appear at
// ANY row index).
const stackReproResultRev = NDArray.stack([stackTernaryRow, stackFixed3]);
type STACK_UNION_ROW_REPRO_REV = Expect<Equal<(typeof stackReproResultRev)["shape"], readonly [2, number]>>;

// --- double-mismatch union: [5]-fixed row + [3]|[4]-union row --------------
// Every possible resolution of the union row (3 or 4) individually mismatches
// the fixed row's length (5) — a case where a MORE SOPHISTICATED analysis
// could in principle prove a guaranteed throw. `StackFold`'s fix does NOT
// attempt that: it degrades uniformly to no-claim regardless, same as any
// other union `Head`. Compiles (no @ts-expect-error); `stackRuntime` throws
// at runtime for whichever branch actually runs.

const stackFixed5 = NDArray.zeros([5]);
const stackDoubleMismatch = NDArray.stack([stackFixed5, stackTernaryRow]);
type STACK_UNION_ROW_DOUBLE_MISMATCH = Expect<Equal<(typeof stackDoubleMismatch)["shape"], readonly [2, number]>>;

// --- F-ADV-2 closure: direct StackDimMerge-wide coverage, both orders ------
// The only PRE-EXISTING coverage of `StackDimMerge`'s own wide-sentinel
// branch ran indirectly through a `RankUnknowable` row (STACK_UNKNOWABLE_RANK
// above, which short-circuits in `StackFold` BEFORE `StackDimMerge` is ever
// called). These two pins reach `StackDimMerge` itself with a row whose rank
// IS known (rank 1) but whose OWN dim is dynamic (`NDArray<[number]>`) —
// the `IsDynamicDim` branch inside `StackDimMerge`, not `RankUnknowable`.

declare const stackDynDimRow: NDArray<[number]>;
declare const stackKnownDimRow: NDArray<[3]>;
const stackDynDimResult = NDArray.stack([stackDynDimRow, stackKnownDimRow]);
type STACK_DIM_MERGE_WIDE = Expect<Equal<(typeof stackDynDimResult)["shape"], readonly [2, number]>>;

const stackDynDimResultRev = NDArray.stack([stackKnownDimRow, stackDynDimRow]);
type STACK_DIM_MERGE_WIDE_REV = Expect<Equal<(typeof stackDynDimResultRev)["shape"], readonly [2, number]>>;

// --- Verify-C coverage gap: array element union of DIFFERENT ranks ---------
// `readonly (NDArray<[3]> | NDArray<[2,3]>)[]` — checked EMPIRICALLY first
// (isolated tsc probe against a live symlink of spike/src, same technique
// used to catch the original F8 bug): `Shapes[number]` for this array is
// `readonly [3] | readonly [2, 3]`, a MIXED-rank union, so `RankUnknowable`
// (`IsUnion<Shapes[number]["length"]>` = `IsUnion<1 | 2>` = true) fires
// FIRST in both `StackCheckArray` and `StackShapeArray` — never reaches the
// rank-1 destructure or `ArrayRowD` at all. Confirmed empirically:
// `readonly [number, number]` (no-claim on BOTH `N` — already honest for
// any array — AND `D`, since the row's own rank isn't even confidently 1).

declare const stackArrMixedRankUnion: readonly (NDArray<[3]> | NDArray<[2, 3]>)[];
const stackMixedRankArrResult = NDArray.stack(stackArrMixedRankUnion);
type STACK_ARRAY_MIXED_RANK_UNION = Expect<Equal<(typeof stackMixedRankArrResult)["shape"], readonly [number, number]>>;

// =============================================================================
// Op-Scheibe W5 (docs/op-w5-item-spec.md): NDArray.item(...indices) type
// pins. `ItemGuard<S, Idx>` (vector.ts, Baustein-0 addendum F1-F8) is used
// DIRECTLY as item's rest-parameter type (F1 — a `Guard<>`-wrapped
// rest-parameter is a permanent TS2370 at the declaration), so its own
// message-equality pin below checks `ItemGuard` directly (the
// `SliceSpecsGuard` precedent in slice.test-d.ts), not the `Guard<>` wrapper
// other ops use. D6 cost note: a single `Equal<ItemGuard<...>, ...>`
// comparison against a hand-written expected tuple measured EXPENSIVE in
// isolation (~1,700 instantiations, an order of magnitude above a bare
// reference or an identity self-compare — `tsc`'s assignability check
// forces full structural normalization of the computed side, unlike a
// trivial `extends unknown` check) — so ONE combined case below proves BOTH
// custom stems (dot-form AND out-of-bounds) simultaneously via a single
// two-error-position tuple, rather than one comparison per edge.
// =============================================================================

// --- D5: valid full-indexing calls compile, return type is `number` --------

const itemM = NDArray.zeros([2, 3]);
const itemResult = itemM.item(0, 0);
type ITEM_RETURNS_NUMBER = Expect<Equal<typeof itemResult, number>>;

// A valid NEGATIVE literal index compiles too (NumPy-parity, Spike 03).
const itemNegResult = itemM.item(-1, -1);
type ITEM_NEGATIVE_LITERAL_OK = Expect<Equal<typeof itemNegResult, number>>;

// Rank 0: item() (zero arguments) compiles, returns `number`.
const itemR0 = NDArray.zeros([]);
const itemR0Result = itemR0.item();
type ITEM_RANK0_OK = Expect<Equal<typeof itemR0Result, number>>;

// --- D5: arity errors, BOTH directions — native TS2554 (F3), not a custom
// message (there is no argument position for a MISSING argument) ------------

// @ts-expect-error - too few indices for rank-2 shape [2,3]: native TS2554
itemM.item(0);

// @ts-expect-error - too many indices for rank-2 shape [2,3]: native TS2554
itemM.item(0, 0, 0);

// @ts-expect-error - rank-0 receiver called with an index: native TS2554
itemR0.item(0);

// --- D5: `ItemGuard` message-equality pin — BOTH custom stems, one shot ----
// index 0.5 (axis 0, dot-form, F5) AND index 3 (axis 1, out-of-bounds) in the
// SAME call: TS7's own "one diagnostic per call for multiple invalid
// positions" limit (F7, M3) means the underlying *guard type* still flags
// EVERY offending position — this single comparison proves both stems'
// exact wording at once (word-for-word M3, mirrored by itemRuntime's own
// throws, see runtime.ts).
type ItemDotFormMsg = `item: index 0.5 for axis 0 is not an integer (shape [2,3])`;
type ItemOobAxis1Msg = `item: index 3 is out of bounds for axis 1 with dim 3 (shape [2,3])`;
type IG1 = Expect<
  Equal<ItemGuard<[2, 3], [0.5, 3]>, [{ readonly __shapeError: ItemDotFormMsg }, { readonly __shapeError: ItemOobAxis1Msg }]>
>;

// The @ts-expect-error compile-reject counterparts (the method call itself,
// not just the bare `ItemGuard` type) — positive OOB, negative OOB
// (NumPy-parity sign handling), and dot-form, each at the offending
// argument:

// @ts-expect-error - index 2 out of bounds for axis 0 with dim 2 (positive)
itemM.item(2, 0);

// @ts-expect-error - index -3 out of bounds for axis 0 with dim 2 (negative, past the front)
itemM.item(-3, 0);

// @ts-expect-error - index 0.5 for axis 0 is not an integer (dot-form)
itemM.item(0.5, 0);

// --- D5: no-claim degrades — wide rank, union index, mixed-rank S, dynamic
// spread all compile (never wrongly rejected) --------------------------------

declare const itemWideRankNd: NDArray<number[]>;
const itemWideRankResult = itemWideRankNd.item(0, 1, 2); // wide rank: IsDynamicRank<S> -> RankUnknowable -> Idx passed through unchanged
type ITEM_WIDE_RANK_OK = Expect<Equal<typeof itemWideRankResult, number>>;

declare const itemUnionIdx: 1 | 5;
itemM.item(itemUnionIdx, 0); // union index: IsUnion pre-gate in ItemMark -> no-claim, compiles

// POLICY pin (W5 verify round, Verify-B finding 1): even a UNIFORMLY-INVALID
// union index (5 | 9 on dim 3 — every member is a guaranteed throw) compiles
// as no-claim. This mirrors reduce.ts's union-AXIS policy verbatim ("literal,
// negative, even ALL-invalid members degrade like the dynamic axis") and is
// what makes ItemMark's IsUnion pre-gate LOAD-BEARING: without the gate,
// LiteralIndexBounds' own tuple-wrapped subset check would classify 5|9 as a
// uniform "out" verdict and REJECT — sound, but a deliberate completeness
// trade against house-policy uniformity (Spec v3 addendum documents the
// choice). Removing the pre-gate turns exactly this line red.
declare const itemUnionIdxUniformInvalid: 5 | 9;
itemM.item(itemUnionIdxUniformInvalid, 0);

// Mixed-rank S (RankUnknowable's OWN union-of-ranks branch, dim.ts): a
// shape union `[2,3] | [2,3,4]` passed through `ItemGuard` directly (not
// via a class instance — cheaper than exercising full method-call
// resolution on a union RECEIVER, and this is the exact `S` union
// `ItemGuard`'s own `RankUnknowable<S>` gate is built to catch) degrades
// wholly to no-claim, `Idx` unchanged.
type ITEM_MIXED_RANK_S = Expect<Equal<ItemGuard<[2, 3] | [2, 3, 4], [0, 0]>, [0, 0]>>;

declare const itemDynIndices: number[];
itemM.item(...itemDynIndices); // F4: dynamic-length spread on a FIXED-rank receiver compiles (IsDynamicRank<Idx> gate, not just RankUnknowable<S>)

// =============================================================================
// WASM parity S1 (docs/wasm-parity-scalar-spec.md, D6): WNDArray scalar-
// overload (`add`/`sub`/`mul`/`div`) type pins — WNDArray-side twin of the
// NDArray W2 pins above (SCALAR_DIV_SHAPE etc.), proving the same
// scalar-overload machinery on the resident class (second call site of the
// already-proven `number | Guard<Broadcast<S,B>,...>` overload form — no
// new type machinery). Plus the T4b-mandated pin (Baustein-0 finding F1):
// an @ts-expect-error that actually CALLS the overloaded scalar method with
// a shape-incompatible WNDArray argument — something UW1-UW4 above never do
// (those probe `Guard<Broadcast<...>>` as a standalone type alias, never
// through a real method call).
// =============================================================================

declare const wScalarBase: WNDArray<[2, 3]>;
const wScalarDivided = wScalarBase.div(2);
type WSCALAR_DIV_SHAPE = Expect<Equal<(typeof wScalarDivided)["shape"], readonly [2, 3]>>;

declare const wScalarRank0: WNDArray<[]>;
const wScalarRank0Divided = wScalarRank0.div(2);
type WSCALAR_DIV_RANK0 = Expect<Equal<(typeof wScalarRank0Divided)["shape"], readonly []>>;

// wide/dynamic-rank receiver: the scalar overload stays callable and
// degrades exactly like the binary overload already does — never a
// confident literal claim on an unknowable shape.
declare const wScalarWide: WNDArray<readonly number[]>;
const wScalarWideDivided = wScalarWide.div(2);
type WSCALAR_DIV_WIDE = Expect<Equal<(typeof wScalarWideDivided)["shape"], readonly number[]>>;

// Readonly-S receiver (a literal `readonly [...]` type argument threads
// through the scalar overload identically to every other op above).
declare const wScalarReadonlyS: WNDArray<readonly [4, 5]>;
const wScalarReadonlySDivided = wScalarReadonlyS.div(2);
type WSCALAR_DIV_READONLY_S = Expect<Equal<(typeof wScalarReadonlySDivided)["shape"], readonly [4, 5]>>;

// union-over-boundary (D2 v2 kink, same as the NDArray side above): a UNION
// argument spanning both the scalar overload and the WNDArray overload is
// rejected AS A WHOLE (TS2769), even though each member alone would be valid.
declare const wScalarOrArray: number | WNDArray<[3]>;
// @ts-expect-error - a UNION argument spanning both the scalar overload and the WNDArray overload is rejected as a whole (TS2769) even though each member alone is individually valid — same documented D2 v2 kink as the NDArray side above
wScalarBase.add(wScalarOrArray);

// The old `[1]`-wrap workaround path still compiles and still resolves
// through the ordinary binary overload, unaffected by the new scalar
// overload's addition (overload-set growth is additive, not replacing).
declare const wScalarWrap: WNDArray<[1]>;
const wScalarWorkaround = wScalarBase.div(wScalarWrap);
type WSCALAR_DIV_WORKAROUND = Expect<Equal<(typeof wScalarWorkaround)["shape"], readonly [2, 3]>>;

// T4b-mandated pin (Baustein-0 finding F1): exercises the overloaded method
// with an actual shape-INCOMPATIBLE WNDArray argument — [2,3] and [9,9] are
// not broadcast-compatible at the trailing dim (3 vs 9, neither equal nor
// 1) — proving the overload SET rejects it, not just `Guard<Broadcast<...>>`
// probed in isolation (which is all UW1-UW4 above ever do).
declare const wScalarIncompatible: WNDArray<[9, 9]>;
// @ts-expect-error - shapes [2,3] and [9,9] are not broadcast-compatible (3 vs 9); calling the overloaded scalar method with an incompatible WNDArray argument must still be rejected through the real overload set (T4b)
wScalarBase.add(wScalarIncompatible);

// =============================================================================
// WASM parity S2 (docs/wasm-parity-mean-spec.md, D4/T6): `WNDArray.mean`
// type pins — WNDArray-side twin of the NDArray W2 mean pins above
// (MEAN_FLAT etc.), a THIRD call site of the ReduceAxis/Guard/OkShape
// machinery (after sum's own WNDArray pins T7f/T7g above and mean's NDArray
// pins above) — proves the WIRING, not the underlying degradation rules a
// second time (same house convention the WNDArray sum/scalar sections above
// already establish; does NOT re-litigate the union-axis-mini-scheibe's own
// 15-pin degradation family, which is sum-only).
// =============================================================================

// mean(): niladic -> WNDArray<[]>-shaped, like sum() and NDArray.mean() (D4)
// — NOT a bare `number` (mean stays a chainable reduction, unlike argmax()).
const wMeanFlat = rw.mean();
type WMEAN_FLAT = Expect<Equal<(typeof wMeanFlat)["shape"], readonly []>>;

// mean(axis[, keepdims]): exact literal tuples (basic positive wiring proof).
const wMeanAxis1 = rw.mean(1);
type WMEAN_AXIS1 = Expect<Equal<(typeof wMeanAxis1)["shape"], readonly [2, 4]>>;

const wMeanAxis1Keep = rw.mean(1, true);
type WMEAN_AXIS1_KEEP = Expect<Equal<(typeof wMeanAxis1Keep)["shape"], readonly [2, 1, 4]>>;

const wMeanNeg = rw.mean(-1);
type WMEAN_NEG = Expect<Equal<(typeof wMeanNeg)["shape"], readonly [2, 3]>>;

// mean(axis): degradations — same ReduceAxis machinery as sum/mean(NDArray),
// only the WIRING is proven here (dyn axis, mixed rank, union axis,
// keepdims-union) — reusing the SAME declared receivers `wMixedRankRecv`/
// `wUAxisRecv` the WNDArray sum section above already declares (proves the
// identical machinery, not a re-derivation).

declare const wDynAxisMean: number;
const wMeanDynAxis = rw.mean(wDynAxisMean);
type WMEAN_DYN_AXIS = Expect<Equal<(typeof wMeanDynAxis)["shape"], readonly number[]>>;

const wMeanMixedSummed = wMixedRankRecv.mean(2);
type WMEAN_MIXED_RANK = Expect<Equal<(typeof wMeanMixedSummed)["shape"], readonly number[]>>;

const wMeanUnionAxis = wUAxisRecv.mean(0 as 0 | 2);
type WMEAN_UNION_AXIS = Expect<Equal<(typeof wMeanUnionAxis)["shape"], readonly number[]>>;

declare const wMeanDynKeep: true | undefined;
const wMeanKeepUnion = rw.mean(1, wMeanDynKeep);
type WMEAN_KEEP_UNION = Expect<Equal<(typeof wMeanKeepUnion)["shape"], readonly [2, 4] | readonly [2, 1, 4]>>;

// OOB pin (T6): axis 5 is out of range for rank-3 shape [2,3,4] — error
// stays at the argument, through the SAME `Guard`/`__shapeError` the
// MEAN_AXIS_OOB_MSG pin above already checks at the type-alias level
// (ReduceAxis is class-agnostic — this call site proves the second class's
// overload set surfaces the identical `reduce: axis …` stem, word-for-word
// `sum`'s own throw, M3).
// @ts-expect-error - axis 5 is out of range for rank-3 shape [2,3,4]: error stays at the argument (ReduceAxis reused unmodified from sum, verbatim message — same Guard the MEAN_AXIS_OOB_MSG pin above already checks)
rw.mean(5);

// =============================================================================
// WASM parity S3 (docs/wasm-parity-item-stack-spec.md, D6/T6): WNDArray.item
// type pins — second call site of the UNCHANGED `ItemGuard<S, Idx>` (vector.ts;
// W5's own FOLLOWUPS entry predicted exactly this), mirrors the NDArray-side
// W5 pins above (ITEM_RETURNS_NUMBER etc.). Budget-Disziplin (W5-D6-Lektion):
// `ItemGuard`'s own message-equality content is ALREADY pinned bit-for-bit by
// the NDArray-side IG1 pin above (the type is UNCHANGED, D6: "null new type
// machinery") — re-running the same ~1,700-instantiation `Equal<>` structural
// comparison here would cost budget for zero new information, so this
// section proves the WIRING (return type, arity, rejection-at-the-argument,
// degradation) via cheap `@ts-expect-error`/`Equal<>`-on-a-`number`
// comparisons only, not a second copy of the message-content proof.
// =============================================================================

declare const witemM: WNDArray<[2, 3]>;
const witemResult = witemM.item(0, 0);
type WITEM_RETURNS_NUMBER = Expect<Equal<typeof witemResult, number>>;

// A valid NEGATIVE literal index compiles too (NumPy-parity, Spike 03).
const witemNegResult = witemM.item(-1, -1);
type WITEM_NEGATIVE_LITERAL_OK = Expect<Equal<typeof witemNegResult, number>>;

// Rank 0: item() (zero arguments) compiles, returns `number`.
declare const witemR0: WNDArray<[]>;
const witemR0Result = witemR0.item();
type WITEM_RANK0_OK = Expect<Equal<typeof witemR0Result, number>>;

// --- arity errors, BOTH directions — native TS2554 (F3), not a custom message ---

// @ts-expect-error - too few indices for rank-2 shape [2,3]: native TS2554
witemM.item(0);

// @ts-expect-error - too many indices for rank-2 shape [2,3]: native TS2554
witemM.item(0, 0, 0);

// @ts-expect-error - rank-0 receiver called with an index: native TS2554
witemR0.item(0);

// --- rejection AT the offending argument (both custom stems) ----------------

// @ts-expect-error - index 2 out of bounds for axis 0 with dim 2 (positive)
witemM.item(2, 0);

// @ts-expect-error - index -3 out of bounds for axis 0 with dim 2 (negative, past the front)
witemM.item(-3, 0);

// @ts-expect-error - index 0.5 for axis 0 is not an integer (dot-form)
witemM.item(0.5, 0);

// --- no-claim degrades: wide rank, union index, dynamic spread --------------

declare const witemWideRankNd: WNDArray<number[]>;
const witemWideRankResult = witemWideRankNd.item(0, 1, 2); // wide rank: IsDynamicRank<S> -> RankUnknowable -> Idx passed through unchanged
type WITEM_WIDE_RANK_OK = Expect<Equal<typeof witemWideRankResult, number>>;

declare const witemUnionIdx: 1 | 5;
witemM.item(witemUnionIdx, 0); // union index: IsUnion pre-gate in ItemMark -> no-claim, compiles

declare const witemDynIndices: number[];
witemM.item(...witemDynIndices); // F4: dynamic-length spread on a fixed-rank receiver compiles

// =============================================================================
// WASM parity S3 (docs/wasm-parity-item-stack-spec.md, D6/T6): WNDArray.stack
// type pins — second call site of the UNCHANGED `StackCheck`/`StackShape`
// (vector.ts), routed through the two NEW resident.ts-private mirrors
// `UnwrapWRow`/`WRowShapesOf`. The F2 (homomorphic, not non-homomorphic) and
// F8 (fresh nested generic, not inline) properties are pinned HERE via the
// REAL `WNDArray.stack(...)` call path (not by referencing the file-private
// `UnwrapWRow`/`WRowShapesOf` directly, which are inaccessible from this
// file by design, D6) — a regression in resident.ts's OWN mirror would not
// be caught by the NDArray-side STACK_* pins above, which exercise
// `ndarray.ts`'s `RowShapesOf` instead.
// =============================================================================

declare const wcore: CoreExports;

// --- exact tuple shape: [2,3] (two [3] rows) --------------------------------

declare const wstackA: WNDArray<[3]>;
declare const wstackB: WNDArray<[3]>;
const wstacked23 = WNDArray.stack(wcore, [wstackA, wstackB]);
type WSTACK_23 = Expect<Equal<(typeof wstacked23)["shape"], readonly [2, 3]>>;

// --- empty tuple literal -> rejected AT the argument (F3) -------------------

// @ts-expect-error - empty tuple literal must be rejected (F3): a `[]` call is a guaranteed runtime throw ("expected at least one row")
WNDArray.stack(wcore, []);

// --- array input: honest [number, D] (F5) -----------------------------------

declare const wstackArr3: readonly WNDArray<[3]>[];
const wstackedArr3 = WNDArray.stack(wcore, wstackArr3);
type WSTACK_ARRAY_D = Expect<Equal<(typeof wstackedArr3)["shape"], readonly [number, 3]>>;

// --- D=0 rows are valid ------------------------------------------------------

declare const wstackZero1: WNDArray<[0]>;
declare const wstackZero2: WNDArray<[0]>;
const wstackedZero = WNDArray.stack(wcore, [wstackZero1, wstackZero2]);
type WSTACK_D0 = Expect<Equal<(typeof wstackedZero)["shape"], readonly [2, 0]>>;

// --- F2 pin: heterogeneous tuple rejected AT the argument, through the REAL
// WNDArray.stack call path (proves WRowShapesOf resolves each row's OWN
// shape individually — a non-homomorphic collapse to `never` would surface
// as a DIFFERENT, generic rejection instead of this specific one). --------

declare const wstackMismatchA: WNDArray<[3]>;
declare const wstackMismatchB: WNDArray<[4]>;
// @ts-expect-error - [3] vs [4] row length mismatch: error stays at the rows argument (F2 pin, length axis)
WNDArray.stack(wcore, [wstackMismatchA, wstackMismatchB]);

declare const wstackRank2: WNDArray<[2, 3]>;
declare const wstackRank1: WNDArray<[3]>;
// @ts-expect-error - rank-2 member at index 0: error stays at the rows argument (F2 pin, rank axis)
WNDArray.stack(wcore, [wstackRank2, wstackRank1]);

// --- F2 pin, positive form: an unknowable-rank row mixed with a known-shape
// row widens HONESTLY to [2, number] — never collapses to `never`, never a
// wrongly-confident literal. Direct WNDArray-side mirror of the NDArray-side
// STACK_UNKNOWABLE_RANK pin above. -------------------------------------------

declare const wstackDynRankRow: WNDArray<number[]>;
declare const wstackKnownRow: WNDArray<[3]>;
const wstackedUnknowableRank = WNDArray.stack(wcore, [wstackDynRankRow, wstackKnownRow]);
type WSTACK_UNKNOWABLE_RANK = Expect<Equal<(typeof wstackedUnknowableRank)["shape"], readonly [2, number]>>;

// --- F8 pin: an ARRAY whose ELEMENT type is itself a union of DIFFERENT
// shapes widens to [number, number] — never collapses to `never`. Direct
// WNDArray-side mirror of the NDArray-side STACK_ARRAY_UNION pin above,
// proving `UnwrapWRow`'s fresh-nested-generic fix specifically on the
// resident.ts mirror (not merely re-exercising ndarray.ts's own). ----------

declare const wstackArrUnion: readonly (WNDArray<[3]> | WNDArray<[4]>)[];
const wstackedArrUnion = WNDArray.stack(wcore, wstackArrUnion);
type WSTACK_ARRAY_UNION = Expect<Equal<(typeof wstackedArrUnion)["shape"], readonly [number, number]>>;

// =============================================================================
// WASM parity S4 (docs/wasm-parity-argmax-spec.md, D2/D7): `WNDArray.argmax`
// type pins. `WNDArray` declares its overloads independently of `NDArray`, so
// a wiring typo in resident.ts would NOT be caught by the W1 `NDArray.argmax`
// section above — the same reason `WNDArray.sum`'s keepdims pins exist
// separately (T7f/T7g). The underlying `ReduceAxis`/`Guard`/`OkShape`
// degradation RULES are the identical imports, already proven there; what is
// pinned here is the WIRING plus D2's niladic `number` deviation.
//
// Budget note (W5-D6: `Equal<...>` MESSAGE pins cost ≈1,700 instantiations
// each): the out-of-range message wording is pinned exactly once, by
// `ARGMAX_AXIS_OOB_MSG` above, over the very type expression BOTH surfaces
// declare (`Guard<ReduceAxis<S, Axis>, Axis>`). Duplicating it here would buy
// no new information at ≈1,700 instantiations, so the WNDArray side pins the
// error's POSITION (`@ts-expect-error` at the axis argument) instead, and the
// message CONTENT is covered end-to-end by the cross-surface runtime stem
// tests plus the real-tsc diagnostic probe in special-values.test.ts.
// =============================================================================

// --- argmax(): niladic -> plain `number` (D2), never WNDArray<[]> ---------
// `rw` is the existing `WNDArray<[2, 3, 4]>` receiver declared for the
// `sum` keepdims pins above — reused deliberately rather than re-declared.
const wArgmaxFlat = rw.argmax();
type W_ARGMAX_FLAT = Expect<Equal<typeof wArgmaxFlat, number>>;

// --- argmax(axis[, keepdims]): exact literal tuples -----------------------

const wArgmaxAxis1 = rw.argmax(1);
type W_ARGMAX_AXIS1 = Expect<Equal<(typeof wArgmaxAxis1)["shape"], readonly [2, 4]>>;

const wArgmaxAxis1Keep = rw.argmax(1, true);
type W_ARGMAX_AXIS1_KEEP = Expect<Equal<(typeof wArgmaxAxis1Keep)["shape"], readonly [2, 1, 4]>>;

const wArgmaxAxis1NoKeep = rw.argmax(1, false); // explicit `false` == default
type W_ARGMAX_AXIS1_NOKEEP = Expect<Equal<(typeof wArgmaxAxis1NoKeep)["shape"], readonly [2, 4]>>;

const wArgmaxNeg = rw.argmax(-1);
type W_ARGMAX_NEG = Expect<Equal<(typeof wArgmaxNeg)["shape"], readonly [2, 3]>>;

// `argmax(undefined)` is the 1-ARG overload with an axis VALUE of `undefined`
// (full reduction) — a DIFFERENT overload from the true 0-arg `argmax()`
// above, exactly as on the NDArray surface (D2).
const wArgmaxUndefAxis = rw.argmax(undefined);
type W_ARGMAX_UNDEF_AXIS = Expect<Equal<(typeof wArgmaxUndefAxis)["shape"], readonly []>>;

const wArgmaxUndefKeep = rw.argmax(undefined, true);
type W_ARGMAX_UNDEF_KEEP = Expect<Equal<(typeof wArgmaxUndefKeep)["shape"], readonly [1, 1, 1]>>;

// --- degradations: every edge ends in an honest no-claim, never a wrong
// literal and never `never` (same ReduceAxis machinery as `WNDArray.sum`) --

declare const wArgmaxDynAxis: number;
const wArgmaxDyn = rw.argmax(wArgmaxDynAxis);
type W_ARGMAX_DYN_AXIS = Expect<Equal<(typeof wArgmaxDyn)["shape"], readonly number[]>>;

const wArgmaxUnionAxis = rw.argmax(0 as 0 | 2);
type W_ARGMAX_UNION_AXIS = Expect<Equal<(typeof wArgmaxUnionAxis)["shape"], readonly number[]>>;

declare const wArgmaxMixedRankRecv: WNDArray<[2, 3] | [2, 3, 4]>;
const wArgmaxMixedRank = wArgmaxMixedRankRecv.argmax(1);
type W_ARGMAX_MIXED_RANK = Expect<Equal<(typeof wArgmaxMixedRank)["shape"], readonly number[]>>;

declare const wArgmaxDynRankRecv: WNDArray<number[]>;
const wArgmaxDynRank = wArgmaxDynRankRecv.argmax(0);
type W_ARGMAX_DYN_RANK = Expect<Equal<(typeof wArgmaxDynRank)["shape"], readonly number[]>>;

declare const wArgmaxDynKeep: true | undefined;
const wArgmaxKeepUnion = rw.argmax(1, wArgmaxDynKeep);
type W_ARGMAX_KEEP_UNION = Expect<Equal<(typeof wArgmaxKeepUnion)["shape"], readonly [2, 4] | readonly [2, 1, 4]>>;

// --- negative: an out-of-range LITERAL axis is rejected AT the axis
// argument, in BOTH the 1-arg and the 2-arg form (Arbeitsregel 2: the
// guard-carrying overload is declared LAST, so it carries the diagnostic) --

// @ts-expect-error - axis 5 is out of range for rank-3 shape [2,3,4]: error stays at the axis argument
rw.argmax(5);

// @ts-expect-error - axis 3 out of range even with keepdims: error stays at the axis argument
rw.argmax(3, true);

// =============================================================================
// WASM parity S5 (docs/wasm-parity-topk-spec.md, D2/D7): `WNDArray.topk` type
// pins — the last op of the S0-S5 campaign. `WNDArray` declares its own
// signature independently of `NDArray`, so a wiring typo in resident.ts would
// NOT be caught by the W1 `NDArray.topk` section above; what is pinned here is
// the WIRING (same `TopkCheck`/`TopkShape` machinery, second call site) plus
// the two-handle return shape.
//
// Budget note (W5-D6: `Equal<...>` MESSAGE pins cost ≈1,700 instantiations
// each): the four `topk` message wordings are pinned exactly once, by
// `TOPK_RANK_MSG`/`TOPK_NEGATIVE_K_MSG`/`TOPK_DOTFORM_K_MSG`/
// `TOPK_BOUNDS_K_MSG` above, over the very type expression BOTH surfaces
// declare (`Guard<TopkCheck<S, K>, K>`). Duplicating them here would buy no
// new information, so the WNDArray side pins the error's POSITION
// (`@ts-expect-error` at the k argument) instead; the message CONTENT is
// covered end-to-end by the cross-surface runtime stem tests and the real-tsc
// diagnostic probe in resident.test.ts, which also pins the COLUMN.
// =============================================================================

declare const wTopkV: WNDArray<[5]>;

// --- topk(k): exact literal tuples on BOTH handles, incl. the k=0/k=D
// valid boundaries -------------------------------------------------------

const wTopk3 = wTopkV.topk(3);
type W_TOPK3_VALUES = Expect<Equal<(typeof wTopk3.values)["shape"], readonly [3]>>;
type W_TOPK3_INDICES = Expect<Equal<(typeof wTopk3.indices)["shape"], readonly [3]>>;

const wTopk0 = wTopkV.topk(0); // k=0: VALID boundary
type W_TOPK0_VALUES = Expect<Equal<(typeof wTopk0.values)["shape"], readonly [0]>>;
type W_TOPK0_INDICES = Expect<Equal<(typeof wTopk0.indices)["shape"], readonly [0]>>;

const wTopkD = wTopkV.topk(5); // k=D: VALID boundary
type W_TOPKD_VALUES = Expect<Equal<(typeof wTopkD.values)["shape"], readonly [5]>>;
type W_TOPKD_INDICES = Expect<Equal<(typeof wTopkD.indices)["shape"], readonly [5]>>;

// Both handles are real `WNDArray`s (not `NDArray`s, and not the erased
// `AnyWNDArray`) — the wiring claim this whole section exists for.
type W_TOPK_HANDLE_KIND = Expect<Equal<typeof wTopk3.values, WNDArray<[3]>>>;
type W_TOPK_HANDLE_KIND_IDX = Expect<Equal<typeof wTopk3.indices, WNDArray<[3]>>>;

// --- topk(k): compile errors AT the k argument (DotCheck precedent) -------

// @ts-expect-error - k=-1 is negative: error stays at the k argument
wTopkV.topk(-1);

// @ts-expect-error - k=1.5 is non-integer (dot-form): error stays at the k argument
wTopkV.topk(1.5);

// @ts-expect-error - k=6 exceeds the vector length 5: error stays at the k argument
wTopkV.topk(6);

// @ts-expect-error - k = Number.MAX_SAFE_INTEGER vastly exceeds length 5: still a PROVABLE compile error
wTopkV.topk(9007199254740991);

declare const wTopkRank2: WNDArray<[2, 3]>;
// @ts-expect-error - rank-2 receiver: topk requires rank-1 (the RECEIVER's problem surfaces at the k argument, DotCheck precedent)
wTopkRank2.topk(2);

declare const wTopkRank0: WNDArray<[]>;
// @ts-expect-error - rank-0 receiver: topk requires rank-1; error stays at the k argument
wTopkRank0.topk(1);

// --- topk(k): degradations (never a confidently-wrong literal claim) ------

declare const wDynK: number;
const wTopkDyn = wTopkV.topk(wDynK);
type W_TOPK_DYN_K_VALUES = Expect<Equal<(typeof wTopkDyn.values)["shape"], readonly [number]>>;
type W_TOPK_DYN_K_INDICES = Expect<Equal<(typeof wTopkDyn.indices)["shape"], readonly [number]>>;

declare const wUnionK: 2 | 3; // uniformly-valid union: still no-claim (the union filter runs unconditionally)
const wTopkUnion = wTopkV.topk(wUnionK);
type W_TOPK_UNION_K = Expect<Equal<(typeof wTopkUnion.values)["shape"], readonly [number]>>;

declare const wUnionKMixed: 2 | 10; // 10 alone would be a hard error; the union degrades, never confidently accepts OR rejects
const wTopkUnionMixed = wTopkV.topk(wUnionKMixed);
type W_TOPK_UNION_K_MIXED = Expect<Equal<(typeof wTopkUnionMixed.indices)["shape"], readonly [number]>>;

// Exponent-form literal: beyond the digit machinery's plain-digit subset, so
// no-claim rather than a lie in either direction.
const wTopkExp = wTopkV.topk(1e21);
type W_TOPK_EXP_K = Expect<Equal<(typeof wTopkExp.values)["shape"], readonly [number]>>;

// --- topk(k): RankUnknowable receiver -> uniform no-claim (policy pin) ----
// Deliberately COMPILES with NO static claim, mirroring the NDArray-side
// TOPK_MIXEDRANK pins above: on a mixed-rank-union receiver even a
// provably-invalid k degrades to no-claim (D-V1.3 house policy); the runtime
// backstop stays authoritative.
declare const wTopkMixedRankRecv: WNDArray<readonly [2, 3] | readonly [5]>;
const wTopkMixedRankNegK = wTopkMixedRankRecv.topk(-1);
type W_TOPK_MIXEDRANK_NEG_K_VALUES = Expect<Equal<(typeof wTopkMixedRankNegK.values)["shape"], readonly [number]>>;
type W_TOPK_MIXEDRANK_NEG_K_INDICES = Expect<Equal<(typeof wTopkMixedRankNegK.indices)["shape"], readonly [number]>>;

declare const wTopkDynRankRecv: WNDArray<number[]>;
const wTopkDynRank = wTopkDynRankRecv.topk(2);
type W_TOPK_DYN_RANK = Expect<Equal<(typeof wTopkDynRank.values)["shape"], readonly [2]>>;

// =============================================================================
// dt1 post-review fixes (docs/dtype-dt1-spec.md): F3 closes a comment that
// claimed a pin existed here without it actually existing; F4 adds a minimal
// set of type-level pins for dt1's new surface (construction, conversion,
// movement ops, and the two dtype-sensitive leaf/JSON members), none of
// which had ANY compile-time pin before this fix.
// =============================================================================

// --- F3: the bare-Uint8Array / ArrayLike<number> ambiguity rejections -------
// (scalar-mean.test.ts's "a bare Uint8Array without an explicit dtype throws
// at runtime" test references this exact pin by name — that reference was
// FALSE before this fix, since neither compile rejection was pinned anywhere
// in this file. `NDArray.fromArray`'s three no-`dtype` overloads (readonly
// number[] | Float64Array, Float32Array, Int32Array) deliberately omit BOTH
// `Uint8Array` (ambiguous: bool 0/1 vs. raw bytes, D3) and the wider
// `ArrayLike<number>` (not one of the three concrete source shapes) — a
// caller must pass an explicit `{ dtype }` for either.

// @ts-expect-error - a bare Uint8Array without `{ dtype }` matches no overload (D3 ambiguity: bool 0/1 vs. raw bytes)
NDArray.fromArray([3], new Uint8Array([1, 0, 1]));

declare const bareArrayLike: ArrayLike<number>;
// @ts-expect-error - a bare ArrayLike<number> (not `readonly number[]`/a concrete typed array) without `{ dtype }` matches no overload
NDArray.fromArray([3], bareArrayLike);

// The `{ dtype }` escape hatch actually works for the ambiguous Uint8Array
// case — same shape, no error (the `{ dtype }` overload's own source union is
// `readonly number[] | Float64Array | Float32Array | Int32Array | Uint8Array`,
// which does NOT include the wider `ArrayLike<number>` either — that source
// stays rejected even WITH an explicit dtype, a stricter rule than the no-
// dtype overloads', not pinned here since F3 only calls out the no-dtype gap).
const fromUint8WithDtype = NDArray.fromArray([3], new Uint8Array([1, 0, 1]), { dtype: "bool" });
type DT_FROM_UINT8_DTYPE = Expect<Equal<(typeof fromUint8WithDtype)["dtype"], "bool">>;

// --- F4: minimal type-level pins for dt1's new surface ----------------------

// zeros/ones with an explicit dtype: the second type parameter threads through.
const dtZerosInt32 = NDArray.zeros([2, 3], "int32");
type DT_ZEROS_INT32 = Expect<Equal<typeof dtZerosInt32, NDArray<[2, 3], "int32">>>;

// fromArray: with vs. without an explicit `{ dtype }` option.
const dtFromArrayExplicit = NDArray.fromArray([2], [1, 2], { dtype: "int32" });
type DT_FROMARRAY_EXPLICIT = Expect<Equal<typeof dtFromArrayExplicit, NDArray<[2], "int32">>>;
const dtFromArrayDefault = NDArray.fromArray([2], [1, 2]);
type DT_FROMARRAY_DEFAULT = Expect<Equal<typeof dtFromArrayDefault, NDArray<[2], "float64">>>;

// astype: the result carries the TARGET dtype, receiver's S unchanged.
const dtAstypeBase = NDArray.zeros([2], "int32");
const dtAstypeResult = dtAstypeBase.astype("float32");
type DT_ASTYPE_RESULT = Expect<Equal<typeof dtAstypeResult, NDArray<[2], "float32">>>;

// transpose(): dtype-neutral (D5 "D unverändert") — an int32 receiver's
// transpose stays int32, only the shape changes.
const dtTransposeBase = NDArray.zeros([2, 3], "int32");
const dtTransposeResult = dtTransposeBase.transpose();
type DT_TRANSPOSE_PRESERVES_INT32 = Expect<Equal<typeof dtTransposeResult, NDArray<[3, 2], "int32">>>;

// toNestedArray() on a bool array: boolean leaves, not number leaves (D5).
const dtBoolArr = NDArray.zeros([2, 3], "bool");
type DT_TONESTED_BOOL = Expect<Equal<ReturnType<(typeof dtBoolArr)["toNestedArray"]>, boolean[][]>>;

// toJSON(): `data` is `boolean[]` for a bool array, `number[]` for every
// other dtype (D3) — pinned for both ends of that split.
type DT_TOJSON_BOOL = Expect<Equal<ReturnType<(typeof dtBoolArr)["toJSON"]>, { shape: number[]; data: boolean[] }>>;
const dtFloat64Arr = NDArray.zeros([2, 3]);
type DT_TOJSON_FLOAT64 = Expect<Equal<ReturnType<(typeof dtFloat64Arr)["toJSON"]>, { shape: number[]; data: number[] }>>;

// item() on a bool array: returns `boolean`, not `number` (D5).
const dtBoolItem = dtBoolArr.item(0, 0);
type DT_ITEM_BOOL = Expect<Equal<typeof dtBoolItem, boolean>>;

// A union dtype INCLUDING float64 (`NDArray<[3], "int32" | "float64">`)
// calling `add`'s scalar overload: DOCUMENTED no-claim path. dt1 got this
// FOR FREE from `DTypeLock<D, Op>` being a naked conditional on `D` (it
// distributed over the union to `true | ShapeError<...>`, and `Guard`'s own
// tuple-wrapped check doesn't match a mixed union). dt2 (P1/P3,
// docs/dtype-dt2-spec.md) replaces that mechanism with an EXPLICIT
// `IsUnion<D>` gate in `ArithScalarOperand` (same discipline `Promote`
// itself uses, D4 "Dasselbe Gate gilt für jede dtype-Funktion") — confirmed
// (Baustein 0) that this exact pin's ASSERTION stays valid unchanged even
// though the underlying mechanism is now deliberate rather than a naked-
// distribution side effect. This deliberately COMPILES (no
// `@ts-expect-error`): the point of the pin is that the union case is
// honest no-claim, not a false accept limited to one branch. The runtime
// backstop for an actual int32 instance at this call is pinned in
// scalar-mean.test.ts's dt1/dt2 sections.
declare const dtUnionRecv: NDArray<[3], "int32" | "float64">;
const dtUnionAdded = dtUnionRecv.add(1); // must compile clean (no-claim), never a false accept OR a false reject
type DT_UNION_ADD_SHAPE = Expect<Equal<(typeof dtUnionAdded)["shape"], readonly [3]>>;
// dt2 addition: the RESULT dtype is `D` VERBATIM (the scalar overload's
// return type is `NDArray<S, D>`, unconditionally) — for a union `D` this is
// the PRECISE original union, never widened to the full `DType`, and never
// falsely narrowed to a single member either.
type DT_UNION_ADD_DTYPE = Expect<Equal<(typeof dtUnionAdded)["dtype"], "int32" | "float64">>;

// =============================================================================
// dt2 (docs/dtype-dt2-spec.md, P5): Promote<A,B>/PromoteDiv<A,B> type-level
// pins against the ONE runtime source (`PROMOTE_NUMERIC`, runtime.ts) — the
// SAME table scalar-mean.test.ts's dt2 section pins at the runtime layer via
// `promoteDType`/`promoteDTypeDiv`. Appended at the end of this file, no new
// file (dt2 spec: "Tests an bestehende Dateien anhängen").
// =============================================================================
import type { OkDType, Promote, PromoteDiv } from "../src/ndarray.ts";
import type { ShapeError } from "../src/dim.ts";

// --- Promote<A,B>: the full 3x3 numeric table, hand-written oracle ----------

type PROMOTE_F64_F64 = Expect<Equal<Promote<"float64", "float64">, "float64">>;
type PROMOTE_F64_F32 = Expect<Equal<Promote<"float64", "float32">, "float64">>;
type PROMOTE_F64_I32 = Expect<Equal<Promote<"float64", "int32">, "float64">>;
type PROMOTE_F32_F64 = Expect<Equal<Promote<"float32", "float64">, "float64">>;
type PROMOTE_F32_F32 = Expect<Equal<Promote<"float32", "float32">, "float32">>;
type PROMOTE_F32_I32 = Expect<Equal<Promote<"float32", "int32">, "float64">>;
type PROMOTE_I32_F64 = Expect<Equal<Promote<"int32", "float64">, "float64">>;
type PROMOTE_I32_F32 = Expect<Equal<Promote<"int32", "float32">, "float64">>;
type PROMOTE_I32_I32 = Expect<Equal<Promote<"int32", "int32">, "int32">>;

// --- PromoteDiv<A,B>: ALWAYS floating-point, float32⊕float32 is the ONLY
// cell that stays float32 (D5) — every other cell, INCLUDING int32⊕int32
// (which Promote keeps int32), widens to float64.

type PROMOTEDIV_F64_F64 = Expect<Equal<PromoteDiv<"float64", "float64">, "float64">>;
type PROMOTEDIV_F32_F32 = Expect<Equal<PromoteDiv<"float32", "float32">, "float32">>;
type PROMOTEDIV_I32_I32 = Expect<Equal<PromoteDiv<"int32", "int32">, "float64">>;
type PROMOTEDIV_F32_I32 = Expect<Equal<PromoteDiv<"float32", "int32">, "float64">>;
type PROMOTEDIV_I32_F32 = Expect<Equal<PromoteDiv<"int32", "float32">, "float64">>;

// --- bool rejects (either side) via the SAME ShapeError/Guard mechanism a
// shape mismatch uses — never a plain `never`/silent narrowing.

type PROMOTE_BOOL_LEFT = Expect<Equal<Promote<"bool", "int32">, ShapeError<"dtype 'bool' has no arithmetic — use astype() to convert first">>>;
type PROMOTE_BOOL_RIGHT = Expect<Equal<Promote<"int32", "bool">, ShapeError<"dtype 'bool' has no arithmetic — use astype() to convert first">>>;
type PROMOTEDIV_BOOL_LEFT = Expect<Equal<PromoteDiv<"bool", "float64">, ShapeError<"dtype 'bool' has no arithmetic — use astype() to convert first">>>;
type PROMOTEDIV_BOOL_RIGHT = Expect<Equal<PromoteDiv<"float64", "bool">, ShapeError<"dtype 'bool' has no arithmetic — use astype() to convert first">>>;

// --- Union-Gate (D4): a union on EITHER side degrades the WHOLE result to
// the wide `DType` — no claim, never a false accept OR a false narrow-reject,
// even when one union member is bool (the exact Baustein-0 Blocker F4 class:
// `Promote<"bool" | "int32", "int32">` must NOT silently drop the bool
// rejection by distributing to `ShapeError<...> | "int32"`).

type PROMOTE_UNION_LEFT = Expect<Equal<Promote<"bool" | "int32", "int32">, DType>>;
type PROMOTE_UNION_RIGHT = Expect<Equal<Promote<"int32", "bool" | "int32">, DType>>;
type PROMOTE_UNION_BOTH = Expect<Equal<Promote<DType, DType>, DType>>;
type PROMOTEDIV_UNION_LEFT = Expect<Equal<PromoteDiv<"bool" | "int32", "int32">, DType>>;
type PROMOTEDIV_UNION_WIDE = Expect<Equal<PromoteDiv<DType, "float32">, DType>>;
// G2 fix (post-verify coverage gap): both existing PromoteDiv union pins
// above put the union on the LEFT (A) operand only (`"bool" | "int32"` or
// the wide `DType` as A, always a concrete "float32"/"int32" as B) — unlike
// `Promote`, which has both `PROMOTE_UNION_LEFT` AND `PROMOTE_UNION_RIGHT`.
// A mutant deleting PromoteDiv's `IsUnion<B> extends true ? DType :` branch
// (the ARGUMENT side) left `pnpm check` green with only the two pins above
// — a real, provable M2 hole (a union ARGUMENT dtype could silently drop a
// bool member the same way the Baustein-0 Blocker F4 class did for
// `Promote`). Mirrors `PROMOTE_UNION_RIGHT` exactly, union on B this time.
type PROMOTEDIV_UNION_RIGHT = Expect<Equal<PromoteDiv<"int32", "bool" | "int32">, DType>>;

// OkDType<P>: strips the ShapeError branch down to a real DType, the
// union-degraded ACCEPT case included (never collapses to `never`).
type OKDTYPE_ACCEPT = Expect<Equal<OkDType<Promote<"int32", "int32">>, "int32">>;
type OKDTYPE_UNION_ACCEPT = Expect<Equal<OkDType<Promote<"bool" | "int32", "int32">>, DType>>;
type OKDTYPE_REJECT = Expect<Equal<OkDType<Promote<"bool", "int32">>, never>>;

// --- End-to-end through the public methods: add/sub/mul/div's RESULT dtype
// matches Promote/PromoteDiv exactly for a representative set of dtype pairs
// (the runtime layer's own promotion-table test in scalar-mean.test.ts
// covers the FULL 3x3 grid; this is the type-level companion "beides gegen
// die eine Quelle" the spec asks for, P5).

const dtF64 = NDArray.zeros([3]);
const dtF32 = NDArray.zeros([3], "float32");
const dtI32 = NDArray.zeros([3], "int32");
const dtBool = NDArray.zeros([3], "bool");

const addF64F32 = dtF64.add(dtF32);
type ADD_F64_F32_DTYPE = Expect<Equal<typeof addF64F32.dtype, "float64">>;
const addF32I32 = dtF32.add(dtI32);
type ADD_F32_I32_DTYPE = Expect<Equal<typeof addF32I32.dtype, "float64">>;
const mulI32I32 = dtI32.mul(dtI32);
type MUL_I32_I32_DTYPE = Expect<Equal<typeof mulI32I32.dtype, "int32">>;
const divI32I32 = dtI32.div(dtI32);
type DIV_I32_I32_DTYPE = Expect<Equal<typeof divI32I32.dtype, "float64">>;
const divF32F32 = dtF32.div(dtF32);
type DIV_F32_F32_DTYPE = Expect<Equal<typeof divF32F32.dtype, "float32">>;

// A union dtype ARGUMENT (not just receiver): `NDArray<[3], "bool" |
// "int32">` on the array overload must also degrade to no-claim (Union-Gate
// applies to BOTH operands, D4) — never a false accept of the bool member.
declare const dtUnionArg: NDArray<[3], "bool" | "int32">;
const addUnionArg = dtF64.add(dtUnionArg);
type ADD_UNION_ARG_DTYPE = Expect<Equal<typeof addUnionArg.dtype, DType>>;

// G2 fix: the method-level companion to PROMOTEDIV_UNION_RIGHT above — a
// union dtype ARGUMENT on `div`'s array overload (PromoteDiv's own table,
// not Promote's) must also degrade to no-claim, never a false accept.
const divUnionArg = dtF64.div(dtUnionArg);
type DIV_UNION_ARG_DTYPE = Expect<Equal<typeof divUnionArg.dtype, DType>>;

// A CONCRETE bool argument/receiver is still rejected (P4, permanent).
// @ts-expect-error - bool has no arithmetic (array form, argument)
dtF64.add(dtBool);
// @ts-expect-error - bool has no arithmetic (array form, receiver)
dtBool.add(dtF64);
// @ts-expect-error - bool has no arithmetic (scalar form) — M3 v9 named
// exception: tsc shows the masked generic TS2769, not this own message, but
// the call is STILL a genuine compile error (M2), which is all `@ts-expect-
// error` needs to see.
dtBool.add(1);

// --- D6 scalar rule pins: int32 keeps D, a dot-form non-integer literal is
// a compile error, an integer literal/exponent-form/wide number is fine.

const addI32Int = dtI32.add(2);
type ADD_I32_INT_DTYPE = Expect<Equal<typeof addI32Int.dtype, "int32">>;
const addI32Exp = dtI32.add(1e3); // exponent form: no static claim, must compile
type ADD_I32_EXP_DTYPE = Expect<Equal<typeof addI32Exp.dtype, "int32">>;
declare const wideScalar: number;
const addI32Wide = dtI32.add(wideScalar); // wide number: no static claim
type ADD_I32_WIDE_DTYPE = Expect<Equal<typeof addI32Wide.dtype, "int32">>;
// @ts-expect-error - 2.5 is a PROVEN non-integer dot-form literal on int32 (D6) — M3 v9 named exception (masked generic message), still a genuine compile error.
dtI32.add(2.5);

// div's scalar overload has NO integer restriction at all (D5) — a
// fractional literal on an int32 receiver is FINE, unlike add/sub/mul.
const divI32Frac = dtI32.div(2.5);
type DIV_I32_FRAC_DTYPE = Expect<Equal<typeof divI32Frac.dtype, "float64">>;
const divF32Scalar = dtF32.div(2);
type DIV_F32_SCALAR_DTYPE = Expect<Equal<typeof divF32Scalar.dtype, "float32">>;
// @ts-expect-error - div's scalar overload still rejects bool unconditionally (P4).
dtBool.div(1);

// =============================================================================
// dt2 Commit D (G1 fix, post-3b-verify regression): `AnyNDArray` (the erased
// `NDArray<any, any>` handle) must NOT be rejected by add/sub/mul/div's
// array overload. Regression: `Promote<A,B>`/`PromoteDiv<A,B>` collapsed to
// the literal `any` type whenever EITHER operand's dtype was `any` (an
// indexed-access-with-`any`-key artifact — `IsUnion<any>` itself resolves
// to `false`, so it never caught this case), which made the surrounding
// `Guard<any, Actual>` reject UNCONDITIONALLY (`[any] extends
// [ShapeError<M>]` is deterministically `true`) — every call below failed
// to compile with TS2769 ("Property '__shapeError' is missing") before the
// fix, even though the pre-dt2 base (`e7087b5`) compiled the equivalent
// AnyNDArray patterns clean (A/B-checked against that commit, same
// session). Fixed by `IsAnyDType<T>` (ndarray.ts, gated BEFORE the union
// gate and the table on both `Promote` and `PromoteDiv`): `any` now
// degrades to the wide `DType`/`readonly number[]` — no claim, same
// no-false-reject policy the union gate already gives a union dtype —
// never back to a false `any` result (the `Equal<>` pins below use the
// generic-function-comparison form of `Equal`, which DOES distinguish
// `any` from `DType`, unlike a naive mutual-assignability check).
// =============================================================================

declare const g1AnyRecv: AnyNDArray;
declare const g1KnownArg: NDArray<[3], "float32">;
declare const g1AnyArg: AnyNDArray;
declare const g1KnownRecv: NDArray<[3], "float32">;

// add: AnyNDArray receiver, AnyNDArray argument, both, and the exact
// function-boundary pattern the regression report used.
const g1Add1 = g1AnyRecv.add(g1KnownArg);
type G1_ADD_ANY_RECV_SHAPE = Expect<Equal<(typeof g1Add1)["shape"], readonly number[]>>;
type G1_ADD_ANY_RECV_DTYPE = Expect<Equal<(typeof g1Add1)["dtype"], DType>>;

const g1Add2 = g1KnownRecv.add(g1AnyArg);
type G1_ADD_ANY_ARG_SHAPE = Expect<Equal<(typeof g1Add2)["shape"], readonly number[]>>;
type G1_ADD_ANY_ARG_DTYPE = Expect<Equal<(typeof g1Add2)["dtype"], DType>>;

const g1Add3 = g1AnyRecv.add(g1AnyArg);
type G1_ADD_ANY_BOTH_SHAPE = Expect<Equal<(typeof g1Add3)["shape"], readonly number[]>>;
type G1_ADD_ANY_BOTH_DTYPE = Expect<Equal<(typeof g1Add3)["dtype"], DType>>;

function g1Combine(a: AnyNDArray, b: NDArray<[3], "float32">) {
  return a.add(b);
}
void g1Combine;

// sub/mul: identical array-overload machinery/regression class as add — one
// AnyNDArray-operand pin apiece is enough to catch a reintroduction.
const g1Sub = g1AnyRecv.sub(g1KnownArg);
type G1_SUB_ANY_DTYPE = Expect<Equal<(typeof g1Sub)["dtype"], DType>>;
const g1Mul = g1AnyRecv.mul(g1KnownArg);
type G1_MUL_ANY_DTYPE = Expect<Equal<(typeof g1Mul)["dtype"], DType>>;

// div: `PromoteDiv`'s CURRENT hand-written leaf never actually collapsed to
// `any` (verified empirically — this call already compiled clean before
// the `IsAnyDType` gate existed), but the gate was added defensively
// alongside `Promote`'s fix because Commit E (G3) rewrites this leaf onto
// the same indexed-access shape `Promote` uses, which WOULD reintroduce the
// hole without it. Pinned so a future regression here is caught too.
const g1Div = g1AnyRecv.div(g1KnownArg);
type G1_DIV_ANY_DTYPE = Expect<Equal<(typeof g1Div)["dtype"], DType>>;
