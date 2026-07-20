import type { Broadcast } from "../src/broadcast.ts";
import type { Shape } from "../src/dim.ts";
import { type AnyNDArray, type Guard, NDArray, type NDArrayView } from "../src/ndarray.ts";
import type { ReduceAxis } from "../src/reduce.ts";
import type { TopkCheck } from "../src/vector.ts";
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

// WNDArray twin note (D7 v2, structural — not a pin): `WNDArray` has NEITHER
// add/sub/mul/div NOR mean today, so there is no WUA-style mirror section to
// write here — a documented absence, not an oversight (see spec D7 v2).

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
