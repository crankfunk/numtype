import type { Broadcast } from "../src/broadcast.ts";
import type { Shape } from "../src/dim.ts";
import { type AnyNDArray, type Guard, NDArray, type NDArrayView } from "../src/ndarray.ts";
import type { WNDArray } from "../src/wasm/resident.ts";
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

// --- toNestedArray() return type is pinned against narrowing (B5, closure --
// round, 2026-07-13). `toNestedArray(): unknown` is intentional (D-V2.2's
// doc comment on `NDArrayView`: no `data`-shaped member, keeps the interface
// satisfiable by any backend) — a future edit that narrows the return type
// on any ONE of the three declarations (`NDArrayView`, `NDArray`, `WNDArray`)
// without updating the others would silently break `implements
// NDArrayView<S>` (narrower return types are fine for `implements`, so the
// compiler would not itself flag the drift) or just quietly change the
// documented contract. These three `Equal` pins catch either.
type ToNestedArrayViewReturn = Expect<Equal<ReturnType<NDArrayView<[2, 3]>["toNestedArray"]>, unknown>>;
type ToNestedArrayNDReturn = Expect<Equal<ReturnType<NDArray<[2, 3]>["toNestedArray"]>, unknown>>;
type ToNestedArrayWNDReturn = Expect<Equal<ReturnType<WNDArray<[2, 3]>["toNestedArray"]>, unknown>>;

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
