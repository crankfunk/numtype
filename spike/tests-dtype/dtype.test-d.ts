// dtype type-level pins (dt1-dt4) -- relocated verbatim from spike/tests/ndarray.test-d.ts
// (infra-02, docs/infra-02-dtype-pin-korpus.md). A SEPARATELY MEASURED corpus: the
// instantiation cost of these pins no longer inflates the Root `check:diag` pin
// (precedent: spike/tests-stress, docs/infra-01-stress-split.md). They are still CHECKED:
// `pnpm check` (the Verbund) runs `tsc --noEmit -p spike/tests-dtype`. The K5
// `AnyNDArray["toNestedArray"]` pin from ndarray.test-d.ts's toNestedArray section lives
// here too (its first section below). New dtype pins belong HERE, not in ndarray.test-d.ts.
import type { Shape } from "../src/dim.ts";
import { type AnyNDArray, NDArray, type NestedBoolValue, type NestedValue } from "../src/ndarray.ts";
import type { DType } from "../src/runtime.ts";
import type { Equal, Expect } from "../tests/test-utils.ts";

// dt1 (K5, docs/dtype-dt1-spec.md; A2 point 1): `AnyNDArray` widened from
// `NDArray<any>` to `NDArray<any, any>` (the fix for the confident-WRONG
// "any shape, only float64" top type, prototype finding F1) — its
// `toNestedArray()` now spans EVERY dtype's leaf type, `NestedValue |
// NestedBoolValue`, not just `NestedValue`. A mandatory, disclosed
// consequence of K5, not a behavior change to any float64 code path.
type NDAnyTop = Expect<Equal<ReturnType<AnyNDArray["toNestedArray"]>, NestedValue | NestedBoolValue>>;

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
// @ts-expect-error - div's scalar overload still rejects bool unconditionally (P4). M3 v9 named exception applies here too (same masked-generic-TS2769 mechanism as the add/sub/mul scalar overloads above, via `DivScalarOperand`'s own `Guard`/`ShapeError`) -- the runtime backstop (`scalarDivTyped`) still throws BOOL_ARITHMETIC_MESSAGE.
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

// =============================================================================
// dt3 (docs/dtype-dt3-spec.md, R1-R4, R6): `ReduceDType<D>` and the dtype-aware
// signatures of `sum`/`mean` (E3), `matmul`/`dot`/`cosineSimilarity` (O1, via
// `PromoteDiv`) and `norm`. Appended at the end of this file (no new file) — the
// RUNTIME half of every claim below is pinned in scalar-mean.test.ts's dt3
// section, against the same two tables (`REDUCE_DTYPE`/`PROMOTE_DIV`).
// =============================================================================
import type { ReduceDType } from "../src/ndarray.ts";

// --- ReduceDType<D>: the full table, then the degradation cases --------------
type DT3_RD_F64 = Expect<Equal<ReduceDType<"float64">, "float64">>;
type DT3_RD_F32 = Expect<Equal<ReduceDType<"float32">, "float32">>;
type DT3_RD_I32 = Expect<Equal<ReduceDType<"int32">, "float64">>;
type DT3_RD_BOOL = Expect<Equal<ReduceDType<"bool">, "float64">>;
// Union gate: a union receiver degrades to the wide `DType` (no claim) — never a per-member distribution that
// would claim e.g. `"float32" | "float64"` for `"float32" | "int32"` (precision, not soundness: see the
// mutant note in the dt3 results) and never a false narrow.
type DT3_RD_UNION = Expect<Equal<ReduceDType<"float32" | "int32">, DType>>;
type DT3_RD_UNION_BOOL = Expect<Equal<ReduceDType<"bool" | "float32">, DType>>;
type DT3_RD_WIDE = Expect<Equal<ReduceDType<DType>, DType>>;
// G1-class gate: an `any` dtype must degrade to `DType`, never collapse the indexed-access leaf to `any`
// (which `Guard`-less return types would silently accept as "anything").
type DT3_RD_ANY = Expect<Equal<ReduceDType<any>, DType>>;

declare const d3f64: NDArray<[2, 3]>;
declare const d3f32: NDArray<[2, 3], "float32">;
declare const d3i32: NDArray<[2, 3], "int32">;
declare const d3bl: NDArray<[2, 3], "bool">;
declare const d3any: AnyNDArray;
declare const d3uni: NDArray<[2, 3], DType>;
declare const d3uni2: NDArray<[2, 3], "float32" | "int32">;
declare const d3uni3: NDArray<[2, 3], "bool" | "float32">;
declare const d3m32: NDArray<[3, 4], "float32">;
declare const d3m64: NDArray<[3, 4]>;
declare const d3mi: NDArray<[3, 4], "int32">;
declare const d3mb: NDArray<[3, 4], "bool">;
declare const d3v32: NDArray<[3], "float32">;
declare const d3v64: NDArray<[3]>;
declare const d3vi: NDArray<[3], "int32">;
declare const d3vb: NDArray<[3], "bool">;

// --- sum: every dtype, 0-arg / axis / keepdims -------------------------------
const d3s1 = d3f64.sum();
const d3s2 = d3f32.sum();
const d3s3 = d3i32.sum();
const d3s4 = d3bl.sum();
const d3s5 = d3f32.sum(0);
const d3s6 = d3i32.sum(1);
const d3s7 = d3bl.sum(0);
const d3s8 = d3f64.sum(0, true);
const d3s9 = d3f32.sum(1, true);
type DT3_SUM_F64 = Expect<Equal<typeof d3s1, NDArray<[], "float64">>>;
type DT3_SUM_F32 = Expect<Equal<typeof d3s2, NDArray<[], "float32">>>;
type DT3_SUM_I32 = Expect<Equal<typeof d3s3, NDArray<[], "float64">>>;
type DT3_SUM_BOOL = Expect<Equal<typeof d3s4, NDArray<[], "float64">>>;
type DT3_SUM_F32_AXIS = Expect<Equal<typeof d3s5, NDArray<[3], "float32">>>;
type DT3_SUM_I32_AXIS = Expect<Equal<typeof d3s6, NDArray<[2], "float64">>>;
type DT3_SUM_BOOL_AXIS = Expect<Equal<typeof d3s7, NDArray<[3], "float64">>>;
type DT3_SUM_F64_KEEP = Expect<Equal<typeof d3s8, NDArray<[1, 3], "float64">>>;
type DT3_SUM_F32_KEEP = Expect<Equal<typeof d3s9, NDArray<[2, 1], "float32">>>;

// --- mean: same dtype rule ---------------------------------------------------
const d3me1 = d3f64.mean();
const d3me2 = d3f32.mean();
const d3me3 = d3i32.mean();
const d3me4 = d3bl.mean();
const d3me5 = d3f32.mean(0);
const d3me6 = d3i32.mean(1);
const d3me8 = d3f32.mean(1, true);
type DT3_MEAN_F64 = Expect<Equal<typeof d3me1, NDArray<[], "float64">>>;
type DT3_MEAN_F32 = Expect<Equal<typeof d3me2, NDArray<[], "float32">>>;
type DT3_MEAN_I32 = Expect<Equal<typeof d3me3, NDArray<[], "float64">>>;
type DT3_MEAN_BOOL = Expect<Equal<typeof d3me4, NDArray<[], "float64">>>;
type DT3_MEAN_F32_AXIS = Expect<Equal<typeof d3me5, NDArray<[3], "float32">>>;
type DT3_MEAN_I32_AXIS = Expect<Equal<typeof d3me6, NDArray<[2], "float64">>>;
type DT3_MEAN_F32_KEEP = Expect<Equal<typeof d3me8, NDArray<[2, 1], "float32">>>;

// --- sum/mean: `any` and union receivers degrade to the wide DType (no claim, never a false accept) ---
const d3a1 = d3any.sum();
const d3a2 = d3any.mean(0);
const d3u1 = d3uni.sum(0);
const d3u2 = d3uni2.sum(0);
const d3u3 = d3uni3.mean();
const d3u4 = d3uni.mean(1, true);
type DT3_ANY_SUM = Expect<Equal<(typeof d3a1)["dtype"], DType>>;
type DT3_ANY_MEAN = Expect<Equal<(typeof d3a2)["dtype"], DType>>;
type DT3_WIDE_SUM = Expect<Equal<typeof d3u1, NDArray<[3], DType>>>;
type DT3_UNION_SUM = Expect<Equal<typeof d3u2, NDArray<[3], DType>>>;
type DT3_UNION_BOOL_MEAN = Expect<Equal<typeof d3u3, NDArray<[], DType>>>;
type DT3_WIDE_MEAN_KEEP = Expect<Equal<typeof d3u4, NDArray<[2, 1], DType>>>;

// --- matmul: PromoteDiv — float32 only for float32 x float32 (O1) ------------
const d3x1 = d3f32.matmul(d3m32);
const d3x2 = d3f32.matmul(d3mi);
const d3x3 = d3i32.matmul(d3mi);
const d3x4 = d3f64.matmul(d3m32);
const d3x5 = d3f64.matmul(d3m64);
type DT3_MM_F32_F32 = Expect<Equal<typeof d3x1, NDArray<[2, 4], "float32">>>;
type DT3_MM_F32_I32 = Expect<Equal<typeof d3x2, NDArray<[2, 4], "float64">>>;
type DT3_MM_I32_I32 = Expect<Equal<typeof d3x3, NDArray<[2, 4], "float64">>>; // widens, unlike add/sub/mul
type DT3_MM_F64_F32 = Expect<Equal<typeof d3x4, NDArray<[2, 4], "float64">>>;
type DT3_MM_F64_F64 = Expect<Equal<typeof d3x5, NDArray<[2, 4], "float64">>>;
// batch broadcast and 1-D promotion keep the dtype rule
declare const d3b32: NDArray<[5, 2, 3], "float32">;
declare const d3bi: NDArray<[3, 4], "int32">;
const d3x7 = d3b32.matmul(d3bi);
const d3x9 = d3f32.matmul(d3v32);
type DT3_MM_BATCH_MIXED = Expect<Equal<typeof d3x7, NDArray<[5, 2, 4], "float64">>>;
type DT3_MM_VEC_RIGHT = Expect<Equal<typeof d3x9, NDArray<[2], "float32">>>;
// `any` / union on either side degrade to DType (G1 class: PromoteDiv's gates apply here too)
const d3xa = d3f32.matmul(d3any);
const d3xb = d3any.matmul(d3m32);
const d3xc = d3uni2.matmul(d3m32);
type DT3_MM_ANY_ARG = Expect<Equal<(typeof d3xa)["dtype"], DType>>;
type DT3_MM_ANY_RECV = Expect<Equal<(typeof d3xb)["dtype"], DType>>;
type DT3_MM_UNION_RECV = Expect<Equal<(typeof d3xc)["dtype"], DType>>;

// --- dot / cosineSimilarity / norm: still a plain `number` --------------------
const d3d1 = d3v32.dot(d3v32);
const d3d2 = d3v32.dot(d3vi);
const d3d5 = d3v32.dot(d3any);
const d3c1 = d3v32.cosineSimilarity(d3v32);
const d3c2 = d3vi.cosineSimilarity(d3v64);
const d3c3 = d3v32.cosineSimilarity(d3any);
type DT3_DOT_F32 = Expect<Equal<typeof d3d1, number>>;
type DT3_DOT_MIXED = Expect<Equal<typeof d3d2, number>>;
type DT3_DOT_ANY = Expect<Equal<typeof d3d5, number>>;
type DT3_COS_F32 = Expect<Equal<typeof d3c1, number>>;
type DT3_COS_MIXED = Expect<Equal<typeof d3c2, number>>;
type DT3_COS_ANY = Expect<Equal<typeof d3c3, number>>;
const d3n2 = d3v32.norm();
const d3n3 = d3vi.norm();
type DT3_NORM_F32 = Expect<Equal<typeof d3n2, number>>;
type DT3_NORM_I32 = Expect<Equal<typeof d3n3, number>>;
// A3(a) (COVENANT M2 v10): `norm()` on bool COMPILES — it is niladic, there is no argument to hang a rejection on;
// the permanent rejection is runtime-only (BOOL_ARITHMETIC_MESSAGE, pinned in scalar-mean.test.ts). No
// `@ts-expect-error` here on purpose: if a later change ever makes this a compile error, the call below fails.
const d3n1 = d3vb.norm();
type DT3_NORM_BOOL_COMPILES = Expect<Equal<typeof d3n1, number>>;

// --- bool is rejected at the ARGUMENT for the two-operand ops, receiver or argument (O1 / `PromoteDiv`) ---
// @ts-expect-error - bool argument (matmul)
d3f32.matmul(d3mb);
// @ts-expect-error - bool receiver (matmul)
d3mb.matmul(d3m32 as unknown as NDArray<[4, 3], "float32">);
// @ts-expect-error - bool receiver (dot)
d3vb.dot(d3v32);
// @ts-expect-error - bool argument (dot)
d3v32.dot(d3vb);
// @ts-expect-error - bool receiver (cosineSimilarity)
d3vb.cosineSimilarity(d3v32);
// @ts-expect-error - bool argument (cosineSimilarity)
d3v32.cosineSimilarity(d3vb);
// A shape error with a bool operand is still a compile error (WHICH message wins — the shape one — is pinned
// per diagnostic content in scalar-mean.test.ts, "dt3 diagnostic CONTENTS").
// @ts-expect-error - inner dimensions 3 and 5 do not match (and the argument is bool)
d3f32.matmul(d3mb as unknown as NDArray<[5, 4], "bool">);

// --- dt3 fix round (G2): pins restored after the gate-driven trim, plus targeted pins for the 18 type mutants
// that survived the first verify (each pin below names the mutant class it kills) ---------------------------
// Restored (the eight pins C2 had dropped to fit the +5,000 gate; the owner accepted the overshoot instead).
const d3x6 = d3b32.matmul(d3m32);
const d3x8 = d3v32.matmul(d3m32);
const d3xd = d3uni3.matmul(d3m32);
const d3s10 = d3i32.sum(-1, true);
const d3s11 = d3bl.sum(0, false);
const d3me7 = d3bl.mean(0);
const d3d3 = d3vi.dot(d3vi);
const d3d4 = d3v64.dot(d3v32);
type DT3_MM_BATCH_F32 = Expect<Equal<typeof d3x6, NDArray<[5, 2, 4], "float32">>>; // rank-3 receiver keeps float32
type DT3_MM_VEC_LEFT = Expect<Equal<typeof d3x8, NDArray<[4], "float32">>>; // left 1-D receiver keeps float32
type DT3_MM_UNION_BOOL_RECV = Expect<Equal<(typeof d3xd)["dtype"], DType>>;
type DT3_SUM_I32_NEGKEEP = Expect<Equal<typeof d3s10, NDArray<[2, 1], "float64">>>; // int32 -> float64 with keepdims
type DT3_SUM_BOOL_NOKEEP = Expect<Equal<typeof d3s11, NDArray<[3], "float64">>>;
type DT3_MEAN_BOOL_AXIS = Expect<Equal<typeof d3me7, NDArray<[3], "float64">>>; // bool -> float64 (axis overload of mean)
type DT3_DOT_I32 = Expect<Equal<typeof d3d3, number>>; // int32 receiver accepted
type DT3_DOT_F64_F32 = Expect<Equal<typeof d3d4, number>>;

// dot / cosineSimilarity: every dtype pair that must be ACCEPTED (guard must not over-reject) ...
const d3g1 = d3v64.dot(d3vi); // float64 receiver, int32 argument
const d3g2 = d3vi.dot(d3v64); // int32 receiver, float64 argument
const d3g3 = d3v64.cosineSimilarity(d3vi); // float64 receiver, int32 argument
const d3g4 = d3vi.cosineSimilarity(d3vi);
type DT3_DOT_F64_I32 = Expect<Equal<typeof d3g1, number>>;
type DT3_DOT_I32_F64 = Expect<Equal<typeof d3g2, number>>;
type DT3_COS_F64_I32 = Expect<Equal<typeof d3g3, number>>;
type DT3_COS_I32_I32 = Expect<Equal<typeof d3g4, number>>;
// ... and every bool position that must be REJECTED (guard must not under-reject), for non-float64 partners too.
// @ts-expect-error - int32 receiver, bool argument (matmul)
d3i32.matmul(d3mb);
// @ts-expect-error - bool receiver, int32 argument (dot)
d3vb.dot(d3vi);
// @ts-expect-error - float64 receiver, bool argument (dot)
d3v64.dot(d3vb);
// @ts-expect-error - int32 receiver, bool argument (dot)
d3vi.dot(d3vb);
// @ts-expect-error - bool receiver, bool argument (dot)
d3vb.dot(d3vb);
// @ts-expect-error - int32 receiver, bool argument (cosineSimilarity)
d3vi.cosineSimilarity(d3vb);
// @ts-expect-error - float64 receiver, bool argument (cosineSimilarity)
d3v64.cosineSimilarity(d3vb);
// @ts-expect-error - bool receiver, int32 argument (cosineSimilarity)
d3vb.cosineSimilarity(d3vi);
// @ts-expect-error - bool receiver, bool argument (cosineSimilarity)
d3vb.cosineSimilarity(d3vb);

// sum/mean with a NON-literal axis / keepdims: the result dtype must stay the table's answer, never a claim of
// float64 for a float32 receiver, never widened to DType for a concrete receiver.
declare const d3WideAxis: number;
declare const d3WideKeep: boolean;
declare const d3OptAxis: 0 | undefined;
const d3w1 = d3f32.sum(d3WideAxis);
const d3w2 = d3f64.sum(d3WideAxis);
const d3w3 = d3f32.mean(d3WideAxis);
const d3w4 = d3f64.mean(d3WideAxis);
const d3w5 = d3f32.sum(0, d3WideKeep);
const d3w6 = d3f32.mean(0, d3WideKeep);
const d3w7 = d3i32.sum(0, d3WideKeep);
const d3w8 = d3f32.sum(d3OptAxis);
const d3w9 = d3f32.mean(d3OptAxis);
const d3w10 = d3uni2.sum(0, true);
const d3w11 = d3uni2.mean(0, d3WideKeep);
type DT3_SUM_WIDE_AXIS_F32 = Expect<Equal<(typeof d3w1)["dtype"], "float32">>;
type DT3_SUM_WIDE_AXIS_F64 = Expect<Equal<(typeof d3w2)["dtype"], "float64">>;
type DT3_MEAN_WIDE_AXIS_F32 = Expect<Equal<(typeof d3w3)["dtype"], "float32">>;
type DT3_MEAN_WIDE_AXIS_F64 = Expect<Equal<(typeof d3w4)["dtype"], "float64">>;
type DT3_SUM_WIDE_KEEP_F32 = Expect<Equal<(typeof d3w5)["dtype"], "float32">>;
type DT3_MEAN_WIDE_KEEP_F32 = Expect<Equal<(typeof d3w6)["dtype"], "float32">>;
type DT3_SUM_WIDE_KEEP_I32 = Expect<Equal<(typeof d3w7)["dtype"], "float64">>;
type DT3_SUM_OPT_AXIS_F32 = Expect<Equal<(typeof d3w8)["dtype"], "float32">>;
type DT3_MEAN_OPT_AXIS_F32 = Expect<Equal<(typeof d3w9)["dtype"], "float32">>;
type DT3_SUM_UNION_KEEP = Expect<Equal<(typeof d3w10)["dtype"], DType>>; // union receiver on the (axis, keepdims) overload
type DT3_MEAN_UNION_WIDE_KEEP = Expect<Equal<(typeof d3w11)["dtype"], DType>>;

// =============================================================================
// dt4 (docs/dtype-dt4-spec.md): comparisons `gt/ge/lt/le/eq/ne`, `NDArray.where`,
// `any`/`all`. Pin strategy (A4, measured in Baustein 0): the CHECK ALIASES are
// pinned DIRECTLY — `CompareCheck`, `WhereCheck`, `CompareScalarOperand` — because
// the `IsAnyDType`/`IsUnion` gates inside them are behaviour-redundant at call
// level (a call-level pin cannot kill a mutant that only removes such a gate).
// The wiring (each op's own overloads, the guards' attribution to the right
// ARGUMENT, the result types) is pinned by a few calls per op; the order shape ->
// mask -> a/b and the diagnostic CONTENTS are additionally pinned at runtime by
// real tsc fixtures (scalar-mean.test.ts, "dt4 diagnostic CONTENTS").
// =============================================================================
import type { IsShapeError } from "../src/dim.ts";
import type { CompareCheck, CompareScalarOperand, WhereCheck } from "../src/ndarray.ts";
import type { COMPARE_BOOL_MESSAGE, WHERE_BOOL_MIX_MESSAGE, WHERE_MASK_MESSAGE } from "../src/runtime.ts";

type Dt4CmpErr = ShapeError<typeof COMPARE_BOOL_MESSAGE>;
type Dt4MaskErr = ShapeError<typeof WHERE_MASK_MESSAGE>;
type Dt4MixErr = ShapeError<typeof WHERE_BOOL_MIX_MESSAGE>;

// --- CompareCheck: all 16 dtype pairs (bool, as receiver OR argument, is the ONLY rejection) ---------------------
type DT4_CT_00 = Expect<Equal<CompareCheck<[3], [3], "float64", "float64">, true>>;
type DT4_CT_01 = Expect<Equal<CompareCheck<[3], [3], "float64", "float32">, true>>;
type DT4_CT_02 = Expect<Equal<CompareCheck<[3], [3], "float64", "int32">, true>>;
type DT4_CT_03 = Expect<Equal<CompareCheck<[3], [3], "float64", "bool">, Dt4CmpErr>>;
type DT4_CT_04 = Expect<Equal<CompareCheck<[3], [3], "float32", "float64">, true>>;
type DT4_CT_05 = Expect<Equal<CompareCheck<[3], [3], "float32", "float32">, true>>;
type DT4_CT_06 = Expect<Equal<CompareCheck<[3], [3], "float32", "int32">, true>>;
type DT4_CT_07 = Expect<Equal<CompareCheck<[3], [3], "float32", "bool">, Dt4CmpErr>>;
type DT4_CT_08 = Expect<Equal<CompareCheck<[3], [3], "int32", "float64">, true>>;
type DT4_CT_09 = Expect<Equal<CompareCheck<[3], [3], "int32", "float32">, true>>;
type DT4_CT_10 = Expect<Equal<CompareCheck<[3], [3], "int32", "int32">, true>>;
type DT4_CT_11 = Expect<Equal<CompareCheck<[3], [3], "int32", "bool">, Dt4CmpErr>>;
type DT4_CT_12 = Expect<Equal<CompareCheck<[3], [3], "bool", "float64">, Dt4CmpErr>>;
type DT4_CT_13 = Expect<Equal<CompareCheck<[3], [3], "bool", "float32">, Dt4CmpErr>>;
type DT4_CT_14 = Expect<Equal<CompareCheck<[3], [3], "bool", "int32">, Dt4CmpErr>>;
type DT4_CT_15 = Expect<Equal<CompareCheck<[3], [3], "bool", "bool">, Dt4CmpErr>>;
// shape FIRST: a shape error with a bool operand is the shape error, never the bool message
type DT4_CT_SHAPE_FIRST = Expect<Equal<IsShapeError<CompareCheck<[3], [4], "bool", "bool">>, true>>;
type DT4_CT_SHAPE_NOT_BOOL_MSG = Expect<Equal<CompareCheck<[3], [4], "bool", "bool"> extends Dt4CmpErr ? true : false, false>>;
// `any` and unions give NO claim (accept), even when a union MEMBER is bool — both sides, and the wide DType
type DT4_CT_ANY_L = Expect<Equal<CompareCheck<[3], [3], any, "bool">, true>>;
type DT4_CT_ANY_R = Expect<Equal<CompareCheck<[3], [3], "bool", any>, true>>;
type DT4_CT_UNION_L = Expect<Equal<CompareCheck<[3], [3], "bool" | "int32", "float64">, true>>;
type DT4_CT_UNION_R = Expect<Equal<CompareCheck<[3], [3], "float64", "bool" | "int32">, true>>;
type DT4_CT_UNION_BOTH = Expect<Equal<CompareCheck<[3], [3], "bool" | "int32", "bool" | "int32">, true>>;
type DT4_CT_WIDE = Expect<Equal<CompareCheck<[3], [3], DType, DType>, true>>;
// broadcasting is the add/sub/mul one: a broadcastable pair passes, rank may differ
type DT4_CT_BROADCAST_OK = Expect<Equal<CompareCheck<[2, 3], [3], "float32", "int32">, true>>;
type DT4_CT_BROADCAST_ERR = Expect<Equal<IsShapeError<CompareCheck<[2, 3], [4], "float32", "int32">>, true>>;

// --- CompareScalarOperand: only a bool RECEIVER rejects; any scalar value otherwise (int32 compares 2.5 exactly) ---
type DT4_CS_BOOL = Expect<Equal<CompareScalarOperand<"bool", 1>, { readonly __shapeError: typeof COMPARE_BOOL_MESSAGE }>>;
type DT4_CS_F64 = Expect<Equal<CompareScalarOperand<"float64", 2.5>, 2.5>>;
type DT4_CS_F32 = Expect<Equal<CompareScalarOperand<"float32", 2.5>, 2.5>>;
type DT4_CS_I32 = Expect<Equal<CompareScalarOperand<"int32", 2.5>, 2.5>>; // no D6-style integer rule: nothing is stored in int32
type DT4_CS_I32_WIDE = Expect<Equal<CompareScalarOperand<"int32", number>, number>>;
type DT4_CS_UNION = Expect<Equal<CompareScalarOperand<"bool" | "int32", 1>, 1>>; // union receiver: no claim
type DT4_CS_UNION_DTYPE = Expect<Equal<CompareScalarOperand<DType, number>, number>>;

// --- WhereCheck: the whole verdict (shape -> mask -> a/b pair), result dtype on success --------------------------
// a non-bool mask rejects, whatever the value operands are (the mask outranks a mix error)
type DT4_WT_MASK_F64 = Expect<Equal<WhereCheck<[3], "float64", [3], "float64", [3], "float64">, Dt4MaskErr>>;
type DT4_WT_MASK_F64_MIX = Expect<Equal<WhereCheck<[3], "float64", [3], "bool", [3], "int32">, Dt4MaskErr>>;
type DT4_WT_MASK_F32 = Expect<Equal<WhereCheck<[3], "float32", [3], "float64", [3], "float64">, Dt4MaskErr>>;
type DT4_WT_MASK_F32_MIX = Expect<Equal<WhereCheck<[3], "float32", [3], "bool", [3], "int32">, Dt4MaskErr>>;
type DT4_WT_MASK_I32 = Expect<Equal<WhereCheck<[3], "int32", [3], "float64", [3], "float64">, Dt4MaskErr>>;
type DT4_WT_MASK_I32_MIX = Expect<Equal<WhereCheck<[3], "int32", [3], "bool", [3], "int32">, Dt4MaskErr>>;
// numeric x numeric: Promote's table, both operand orders (float32 only for float32 x float32, int32 stays int32)
type DT4_WT_NUM_F64_F64 = Expect<Equal<WhereCheck<[3], "bool", [3], "float64", [3], "float64">, "float64">>;
type DT4_WT_NUM_F64_F32 = Expect<Equal<WhereCheck<[3], "bool", [3], "float64", [3], "float32">, "float64">>;
type DT4_WT_NUM_F64_I32 = Expect<Equal<WhereCheck<[3], "bool", [3], "float64", [3], "int32">, "float64">>;
type DT4_WT_NUM_F32_F64 = Expect<Equal<WhereCheck<[3], "bool", [3], "float32", [3], "float64">, "float64">>;
type DT4_WT_NUM_F32_F32 = Expect<Equal<WhereCheck<[3], "bool", [3], "float32", [3], "float32">, "float32">>;
type DT4_WT_NUM_F32_I32 = Expect<Equal<WhereCheck<[3], "bool", [3], "float32", [3], "int32">, "float64">>;
type DT4_WT_NUM_I32_F64 = Expect<Equal<WhereCheck<[3], "bool", [3], "int32", [3], "float64">, "float64">>;
type DT4_WT_NUM_I32_F32 = Expect<Equal<WhereCheck<[3], "bool", [3], "int32", [3], "float32">, "float64">>;
type DT4_WT_NUM_I32_I32 = Expect<Equal<WhereCheck<[3], "bool", [3], "int32", [3], "int32">, "int32">>;
// A5: bool x bool stays bool (pure selection); a bool/numeric mix has its OWN message, both orders
type DT4_WT_BOOL_BOOL = Expect<Equal<WhereCheck<[3], "bool", [3], "bool", [3], "bool">, "bool">>;
type DT4_WT_MIX_BOOL_F64 = Expect<Equal<WhereCheck<[3], "bool", [3], "bool", [3], "float64">, Dt4MixErr>>;
type DT4_WT_MIX_BOOL_F32 = Expect<Equal<WhereCheck<[3], "bool", [3], "bool", [3], "float32">, Dt4MixErr>>;
type DT4_WT_MIX_BOOL_I32 = Expect<Equal<WhereCheck<[3], "bool", [3], "bool", [3], "int32">, Dt4MixErr>>;
type DT4_WT_MIX_F64_BOOL = Expect<Equal<WhereCheck<[3], "bool", [3], "float64", [3], "bool">, Dt4MixErr>>;
type DT4_WT_MIX_F32_BOOL = Expect<Equal<WhereCheck<[3], "bool", [3], "float32", [3], "bool">, Dt4MixErr>>;
type DT4_WT_MIX_I32_BOOL = Expect<Equal<WhereCheck<[3], "bool", [3], "int32", [3], "bool">, Dt4MixErr>>;
// order: shape outranks the mask error and the mix error
type DT4_WT_SHAPE_FIRST = Expect<Equal<IsShapeError<WhereCheck<[3], "float64", [4], "float64", [3], "float64">>, true>>;
type DT4_WT_SHAPE_NOT_MASK_MSG = Expect<Equal<WhereCheck<[3], "float64", [4], "float64", [3], "float64"> extends Dt4MaskErr ? true : false, false>>;
type DT4_WT_SHAPE_NOT_MIX_MSG = Expect<Equal<WhereCheck<[3], "bool", [3], "bool", [4], "float64"> extends Dt4MixErr ? true : false, false>>;
type DT4_WT_SHAPE_BEATS_MIX = Expect<Equal<IsShapeError<WhereCheck<[3], "bool", [3], "bool", [4], "float64">>, true>>;
// three-way broadcast: pairwise folding (mask, a) then (., b); a conflict that only the triple produces is found
type DT4_WT_3WAY_OK = Expect<Equal<WhereCheck<[2, 1], "bool", [1, 3], "float64", [2, 3], "float64">, "float64">>;
type DT4_WT_3WAY_ERR = Expect<Equal<IsShapeError<WhereCheck<[2, 1], "bool", [1, 3], "float64", [4, 3], "float64">>, true>>;
type DT4_WT_PERM_1 = Expect<Equal<IsShapeError<WhereCheck<[2], "bool", [1], "float64", [3], "float64">>, true>>;
type DT4_WT_PERM_2 = Expect<Equal<IsShapeError<WhereCheck<[1], "bool", [2], "float64", [3], "float64">>, true>>;
type DT4_WT_PERM_3 = Expect<Equal<IsShapeError<WhereCheck<[3], "bool", [1], "float64", [2], "float64">>, true>>;
type DT4_WT_PERM_4 = Expect<Equal<IsShapeError<WhereCheck<[1], "bool", [3], "float64", [2], "float64">>, true>>;
type DT4_WT_PERM_5 = Expect<Equal<IsShapeError<WhereCheck<[2], "bool", [3], "float64", [1], "float64">>, true>>;
type DT4_WT_PERM_6 = Expect<Equal<IsShapeError<WhereCheck<[3], "bool", [2], "float64", [1], "float64">>, true>>;
type DT4_WT_PERM_OK_1 = Expect<Equal<WhereCheck<[2], "bool", [1], "float64", [2], "float64">, "float64">>;
type DT4_WT_PERM_OK_2 = Expect<Equal<WhereCheck<[1], "bool", [2], "float64", [2], "float64">, "float64">>;
type DT4_WT_PERM_OK_3 = Expect<Equal<WhereCheck<[2], "bool", [2], "float64", [1], "float64">, "float64">>;
// `number` dims: never a false rejection. A triple that is guaranteed to fail at runtime (2 vs 3) is still ACCEPTED when
// `number` hides it from the pairwise fold — documented incompleteness, not a claim
type DT4_WT_DYNAMIC_ACCEPTED = Expect<Equal<WhereCheck<[number], "bool", [2], "float64", [3], "float64">, "float64">>;
// `any` and unions give no claim: a mask gate passes through to the a/b verdict, an a/b gate degrades the RESULT to DType
type DT4_WT_ANY_MASK = Expect<Equal<WhereCheck<[3], any, [3], "float64", [3], "float64">, "float64">>;
type DT4_WT_ANY_MASK_MIX = Expect<Equal<WhereCheck<[3], any, [3], "bool", [3], "int32">, Dt4MixErr>>;
type DT4_WT_UNION_MASK = Expect<Equal<WhereCheck<[3], "float32" | "bool", [3], "float64", [3], "float64">, "float64">>;
type DT4_WT_ANY_A = Expect<Equal<WhereCheck<[3], "bool", [3], any, [3], "float64">, DType>>;
type DT4_WT_ANY_B = Expect<Equal<WhereCheck<[3], "bool", [3], "float64", [3], any>, DType>>;
type DT4_WT_UNION_A = Expect<Equal<WhereCheck<[3], "bool", [3], "bool" | "int32", [3], "float64">, DType>>;
type DT4_WT_UNION_B = Expect<Equal<WhereCheck<[3], "bool", [3], "float64", [3], "bool" | "int32">, DType>>;
type DT4_WT_WIDE_AB = Expect<Equal<WhereCheck<[3], "bool", [3], DType, [3], DType>, DType>>;

// --- the six comparisons, wired through the class: receiver/argument dtypes, result bool, the guard on EVERY op ---
declare const dt4f64: NDArray<[2, 3]>;
declare const dt4f32: NDArray<[2, 3], "float32">;
declare const dt4i32: NDArray<[2, 3], "int32">;
declare const dt4bl: NDArray<[2, 3], "bool">;
declare const dt4v64: NDArray<[3]>;
declare const dt4v32: NDArray<[3], "float32">;
declare const dt4vi: NDArray<[3], "int32">;
declare const dt4vbl: NDArray<[3], "bool">;
declare const dt4any: AnyNDArray;
declare const dt4uni: NDArray<[2, 3], DType>;
declare const dt4uniBool: NDArray<[2, 3], "bool" | "float32">;
declare const dt4uniV: NDArray<[3], DType>;
const dt4g1 = dt4f32.gt(dt4vi);
const dt4g2 = dt4i32.gt(2.5);
const dt4e1 = dt4f32.ge(dt4vi);
const dt4e2 = dt4i32.ge(2.5);
const dt4l1 = dt4f32.lt(dt4vi);
const dt4l2 = dt4i32.lt(2.5);
const dt4m1 = dt4f32.le(dt4vi);
const dt4m2 = dt4i32.le(2.5);
const dt4q1 = dt4f32.eq(dt4vi);
const dt4q2 = dt4i32.eq(2.5);
const dt4n1 = dt4f32.ne(dt4vi);
const dt4n2 = dt4i32.ne(2.5);
type DT4_GT_ARRAY = Expect<Equal<typeof dt4g1, NDArray<[2, 3], "bool">>>;
type DT4_GT_SCALAR = Expect<Equal<typeof dt4g2, NDArray<[2, 3], "bool">>>;
type DT4_GE_ARRAY = Expect<Equal<typeof dt4e1, NDArray<[2, 3], "bool">>>;
type DT4_GE_SCALAR = Expect<Equal<typeof dt4e2, NDArray<[2, 3], "bool">>>;
type DT4_LT_ARRAY = Expect<Equal<typeof dt4l1, NDArray<[2, 3], "bool">>>;
type DT4_LT_SCALAR = Expect<Equal<typeof dt4l2, NDArray<[2, 3], "bool">>>;
type DT4_LE_ARRAY = Expect<Equal<typeof dt4m1, NDArray<[2, 3], "bool">>>;
type DT4_LE_SCALAR = Expect<Equal<typeof dt4m2, NDArray<[2, 3], "bool">>>;
type DT4_EQ_ARRAY = Expect<Equal<typeof dt4q1, NDArray<[2, 3], "bool">>>;
type DT4_EQ_SCALAR = Expect<Equal<typeof dt4q2, NDArray<[2, 3], "bool">>>;
type DT4_NE_ARRAY = Expect<Equal<typeof dt4n1, NDArray<[2, 3], "bool">>>;
type DT4_NE_SCALAR = Expect<Equal<typeof dt4n2, NDArray<[2, 3], "bool">>>;
// @ts-expect-error - gt: bool argument
dt4f64.gt(dt4vbl);
// @ts-expect-error - gt: bool receiver, array form
dt4bl.gt(dt4v64);
// @ts-expect-error - gt: bool receiver, scalar form (masked generic TS2769 — M3 v9 exception, v11)
dt4bl.gt(1);
// @ts-expect-error - ge: bool argument
dt4f64.ge(dt4vbl);
// @ts-expect-error - ge: bool receiver, scalar form
dt4bl.ge(1);
// @ts-expect-error - lt: bool argument
dt4f64.lt(dt4vbl);
// @ts-expect-error - lt: bool receiver, scalar form
dt4bl.lt(1);
// @ts-expect-error - le: bool argument
dt4f64.le(dt4vbl);
// @ts-expect-error - le: bool receiver, scalar form
dt4bl.le(1);
// @ts-expect-error - eq: bool argument
dt4f64.eq(dt4vbl);
// @ts-expect-error - eq: bool receiver, scalar form
dt4bl.eq(1);
// @ts-expect-error - ne: bool argument
dt4f64.ne(dt4vbl);
// @ts-expect-error - ne: bool receiver, scalar form
dt4bl.ne(1);
// @ts-expect-error - a broadcast mismatch is a compile error AT the argument
dt4f64.gt(NDArray.zeros([4]));
// the result is ALWAYS bool, also for union / `any` operands (a correct claim, no degradation to DType)
const dt4u1 = dt4uni.gt(1);
const dt4u2 = dt4uni.gt(dt4f64);
const dt4u3 = dt4f32.le(dt4uni);
const dt4u4 = dt4uniBool.eq(dt4i32);
const dt4u5 = dt4any.gt(1);
const dt4u6 = dt4any.gt(dt4f64);
type DT4_UNION_SCALAR_BOOL = Expect<Equal<typeof dt4u1, NDArray<[2, 3], "bool">>>;
type DT4_UNION_RECV_BOOL = Expect<Equal<typeof dt4u2, NDArray<[2, 3], "bool">>>;
type DT4_UNION_ARG_BOOL = Expect<Equal<typeof dt4u3, NDArray<[2, 3], "bool">>>;
type DT4_UNION_WITH_BOOL_MEMBER = Expect<Equal<typeof dt4u4, NDArray<[2, 3], "bool">>>; // compiles: a union with a bool member gives no claim
type DT4_ANY_SCALAR_BOOL = Expect<Equal<(typeof dt4u5)["dtype"], "bool">>;
type DT4_ANY_ARRAY_BOOL = Expect<Equal<(typeof dt4u6)["dtype"], "bool">>;

// --- the grid (fix round F1): for EVERY op the result shape of the array form (the LARGER operand is the argument: a
// result of S instead of Broadcast<S, B> shows) and the bool result for union-dtype operands in each position —
// scalar form with a union receiver, array form with a union receiver, array form with a union argument. ---
const dt4x_gt1 = dt4v64.gt(dt4f64);
const dt4x_gt2 = dt4uni.gt(1);
const dt4x_gt3 = dt4uni.gt(dt4f64);
const dt4x_gt4 = dt4f32.gt(dt4uniV);
type DT4_GRID_GT_SHAPE = Expect<Equal<typeof dt4x_gt1, NDArray<[2, 3], "bool">>>; // [3] gt [2, 3]: the argument is the larger operand
type DT4_GRID_GT_UNION_SCALAR = Expect<Equal<typeof dt4x_gt2, NDArray<[2, 3], "bool">>>;
type DT4_GRID_GT_UNION_RECV = Expect<Equal<typeof dt4x_gt3, NDArray<[2, 3], "bool">>>;
type DT4_GRID_GT_UNION_ARG = Expect<Equal<typeof dt4x_gt4, NDArray<[2, 3], "bool">>>;
const dt4x_ge1 = dt4v64.ge(dt4f64);
const dt4x_ge2 = dt4uni.ge(1);
const dt4x_ge3 = dt4uni.ge(dt4f64);
const dt4x_ge4 = dt4f32.ge(dt4uniV);
type DT4_GRID_GE_SHAPE = Expect<Equal<typeof dt4x_ge1, NDArray<[2, 3], "bool">>>; // [3] ge [2, 3]: the argument is the larger operand
type DT4_GRID_GE_UNION_SCALAR = Expect<Equal<typeof dt4x_ge2, NDArray<[2, 3], "bool">>>;
type DT4_GRID_GE_UNION_RECV = Expect<Equal<typeof dt4x_ge3, NDArray<[2, 3], "bool">>>;
type DT4_GRID_GE_UNION_ARG = Expect<Equal<typeof dt4x_ge4, NDArray<[2, 3], "bool">>>;
const dt4x_lt1 = dt4v64.lt(dt4f64);
const dt4x_lt2 = dt4uni.lt(1);
const dt4x_lt3 = dt4uni.lt(dt4f64);
const dt4x_lt4 = dt4f32.lt(dt4uniV);
type DT4_GRID_LT_SHAPE = Expect<Equal<typeof dt4x_lt1, NDArray<[2, 3], "bool">>>; // [3] lt [2, 3]: the argument is the larger operand
type DT4_GRID_LT_UNION_SCALAR = Expect<Equal<typeof dt4x_lt2, NDArray<[2, 3], "bool">>>;
type DT4_GRID_LT_UNION_RECV = Expect<Equal<typeof dt4x_lt3, NDArray<[2, 3], "bool">>>;
type DT4_GRID_LT_UNION_ARG = Expect<Equal<typeof dt4x_lt4, NDArray<[2, 3], "bool">>>;
const dt4x_le1 = dt4v64.le(dt4f64);
const dt4x_le2 = dt4uni.le(1);
const dt4x_le3 = dt4uni.le(dt4f64);
const dt4x_le4 = dt4f32.le(dt4uniV);
type DT4_GRID_LE_SHAPE = Expect<Equal<typeof dt4x_le1, NDArray<[2, 3], "bool">>>; // [3] le [2, 3]: the argument is the larger operand
type DT4_GRID_LE_UNION_SCALAR = Expect<Equal<typeof dt4x_le2, NDArray<[2, 3], "bool">>>;
type DT4_GRID_LE_UNION_RECV = Expect<Equal<typeof dt4x_le3, NDArray<[2, 3], "bool">>>;
type DT4_GRID_LE_UNION_ARG = Expect<Equal<typeof dt4x_le4, NDArray<[2, 3], "bool">>>;
const dt4x_eq1 = dt4v64.eq(dt4f64);
const dt4x_eq2 = dt4uni.eq(1);
const dt4x_eq3 = dt4uni.eq(dt4f64);
const dt4x_eq4 = dt4f32.eq(dt4uniV);
type DT4_GRID_EQ_SHAPE = Expect<Equal<typeof dt4x_eq1, NDArray<[2, 3], "bool">>>; // [3] eq [2, 3]: the argument is the larger operand
type DT4_GRID_EQ_UNION_SCALAR = Expect<Equal<typeof dt4x_eq2, NDArray<[2, 3], "bool">>>;
type DT4_GRID_EQ_UNION_RECV = Expect<Equal<typeof dt4x_eq3, NDArray<[2, 3], "bool">>>;
type DT4_GRID_EQ_UNION_ARG = Expect<Equal<typeof dt4x_eq4, NDArray<[2, 3], "bool">>>;
const dt4x_ne1 = dt4v64.ne(dt4f64);
const dt4x_ne2 = dt4uni.ne(1);
const dt4x_ne3 = dt4uni.ne(dt4f64);
const dt4x_ne4 = dt4f32.ne(dt4uniV);
type DT4_GRID_NE_SHAPE = Expect<Equal<typeof dt4x_ne1, NDArray<[2, 3], "bool">>>; // [3] ne [2, 3]: the argument is the larger operand
type DT4_GRID_NE_UNION_SCALAR = Expect<Equal<typeof dt4x_ne2, NDArray<[2, 3], "bool">>>;
type DT4_GRID_NE_UNION_RECV = Expect<Equal<typeof dt4x_ne3, NDArray<[2, 3], "bool">>>;
type DT4_GRID_NE_UNION_ARG = Expect<Equal<typeof dt4x_ne4, NDArray<[2, 3], "bool">>>;

// --- any / all: three overloads each, always bool, the ReduceAxis guard ------------------------------------------
const dt4a1 = dt4f64.any();
const dt4a2 = dt4bl.any(0);
const dt4a3 = dt4f32.any(-1, true);
const dt4b1 = dt4i32.all();
const dt4b2 = dt4f32.all(1);
const dt4b3 = dt4bl.all(0, true);
type DT4_ANY_0 = Expect<Equal<typeof dt4a1, NDArray<[], "bool">>>;
type DT4_ANY_AXIS = Expect<Equal<typeof dt4a2, NDArray<[3], "bool">>>;
type DT4_ANY_KEEP = Expect<Equal<typeof dt4a3, NDArray<[2, 1], "bool">>>;
type DT4_ALL_0 = Expect<Equal<typeof dt4b1, NDArray<[], "bool">>>;
type DT4_ALL_AXIS = Expect<Equal<typeof dt4b2, NDArray<[2], "bool">>>;
type DT4_ALL_KEEP = Expect<Equal<typeof dt4b3, NDArray<[1, 3], "bool">>>;
// @ts-expect-error - any: axis out of range
dt4f64.any(2);
// @ts-expect-error - all: axis out of range
dt4f64.all(2);
const dt4a4 = dt4uni.any(0);
const dt4b4 = dt4any.all();
type DT4_ANY_UNION = Expect<Equal<typeof dt4a4, NDArray<[3], "bool">>>;
type DT4_ALL_ANY = Expect<Equal<(typeof dt4b4)["dtype"], "bool">>;

// the third overload (axis, keepdims) — fix round F1: the result is bool for a NON-bool receiver too (a result of `D`
// shows), and an out-of-range axis is rejected AT the axis argument on this overload as well (the guard is not optional)
const dt4a5 = dt4f32.all(0, true);
const dt4a6 = dt4i32.any(1, true);
type DT4_ALL_KEEP_NONBOOL = Expect<Equal<typeof dt4a5, NDArray<[1, 3], "bool">>>;
type DT4_ANY_KEEP_NONBOOL = Expect<Equal<typeof dt4a6, NDArray<[2, 1], "bool">>>;
dt4f64.any(
  // @ts-expect-error - any(axis, keepdims): axis out of range stands at the axis argument
  5,
  true,
);
dt4f64.all(
  // @ts-expect-error - all(axis, keepdims): axis out of range stands at the axis argument
  5,
  true,
);

// --- NDArray.where: result type, and every rejection stands at the ARGUMENT that is wrong (shape -> mask -> a/b) --
const dt4w1 = NDArray.where(dt4vbl, dt4v32, dt4v32);
const dt4w2 = NDArray.where(dt4vbl, dt4vi, dt4vi);
const dt4w3 = NDArray.where(dt4vbl, dt4vbl, dt4vbl);
const dt4w4 = NDArray.where(dt4vbl, dt4v32, dt4vi);
const dt4w5 = NDArray.where(NDArray.zeros([2, 1], "bool"), NDArray.zeros([1, 3], "float32"), NDArray.zeros([4, 1, 1]));
type DT4_WHERE_F32 = Expect<Equal<typeof dt4w1, NDArray<[3], "float32">>>; // float32 only for float32 x float32
type DT4_WHERE_I32 = Expect<Equal<typeof dt4w2, NDArray<[3], "int32">>>; // int32 stays int32
type DT4_WHERE_BOOL = Expect<Equal<typeof dt4w3, NDArray<[3], "bool">>>; // A5: bool x bool
type DT4_WHERE_MIXED = Expect<Equal<typeof dt4w4, NDArray<[3], "float64">>>; // float32 x int32 widens
type DT4_WHERE_THREE_SHAPES = Expect<Equal<typeof dt4w5, NDArray<[4, 2, 3], "float64">>>; // the b shape takes part in the result shape
// `any` / unions: shape claimed, dtype degraded to DType (no claim, never a false reject)
const dt4w6 = NDArray.where(dt4vbl, dt4any, dt4v64);
const dt4w7 = NDArray.where(dt4vbl, dt4uniV, dt4v64);
type DT4_WHERE_ANY = Expect<Equal<(typeof dt4w6)["dtype"], DType>>;
type DT4_WHERE_UNION = Expect<Equal<(typeof dt4w7)["dtype"], DType>>;
NDArray.where(
  // @ts-expect-error - the mask is not bool: the error stands at the MASK argument
  dt4v64,
  dt4v64,
  dt4v64,
);
NDArray.where(
  dt4vbl,
  // @ts-expect-error - a is the bool one (b is numeric): the mix error stands at a
  dt4vbl,
  dt4v64,
);
NDArray.where(
  dt4vbl,
  dt4v64,
  // @ts-expect-error - b is the bool one (a is numeric): the mix error stands at b
  dt4vbl,
);
NDArray.where(
  dt4vbl,
  dt4v64,
  // @ts-expect-error - shape error: stands at b
  NDArray.zeros([4]),
);
NDArray.where(
  // @ts-expect-error - the mask outranks the mix error: stands at the mask
  dt4v64,
  dt4vbl,
  dt4v64,
);
NDArray.where(
  NDArray.zeros([2], "bool"),
  NDArray.zeros([1], "float64"),
  // @ts-expect-error - a conflict only the TRIPLE produces (2 (+) 1 = 2 vs 3): stands at b
  NDArray.zeros([3]),
);
NDArray.where(
  dt4v64,
  dt4v64,
  // @ts-expect-error - shape outranks the non-bool mask: stands at b, not at the mask
  NDArray.zeros([4]),
);

// --- known limit (spec D7; measured): a generic helper whose `S`/`D` are still OPEN type parameters is rejected for the array
// form of the comparisons and for `where`, like `add`/`div` — the check cannot resolve. The scalar form compiles when only `S`
// is open (D concrete); with an open `D` it is rejected too (as `div`'s scalar form is). `any`/`all` have no such limit. ---
function dt4OpenScalar<S extends Shape>(x: NDArray<S, "float32">) {
  return x.gt(1);
}
type DT4_OPEN_SCALAR = Expect<Equal<ReturnType<typeof dt4OpenScalar<[3]>>, NDArray<[3], "bool">>>;
function dt4OpenArray<S extends Shape>(x: NDArray<S, "float32">, y: NDArray<S, "float32">) {
  // @ts-expect-error - known limit: array form with an open S
  return x.gt(y);
}
function dt4OpenWhere<S extends Shape>(m: NDArray<S, "bool">, x: NDArray<S, "float32">, y: NDArray<S, "float32">) {
  // @ts-expect-error - known limit: where with an open S (stands at the first argument whose guard cannot resolve)
  return NDArray.where(m, x, y);
}
function dt4OpenAny<S extends Shape, D extends DType>(x: NDArray<S, D>) {
  return x.any();
}
void [dt4OpenArray, dt4OpenWhere, dt4OpenAny];


// =============================================================================
// dt5 pins (docs/dtype-dt5-spec.md): `sqrt` / `argmax` / `topk` / `stack` on every dtype. Alias-level pins on
// `SqrtDType` / `StackDTypeCheck` (a gate that is only observable on the alias cannot be killed from a call site, dt4
// lesson), then call pins per op and per overload including the result SHAPE, union and `any` operands, the openly
// disclosed `never` follow-ups of `sqrt()` on bool (A6), and the measured known limits of open type parameters (D-6).
// Diagnostic CONTENTS and their order (shape before dtype) are NOT pinned by `@ts-expect-error` here -- it cannot tell
// two errors apart; they are pinned by the tsc-fixture test in scalar-mean.test.ts ("dt5 diagnostic CONTENTS").
// =============================================================================
import type { SqrtDType, StackDTypeCheck } from "../src/ndarray.ts";

declare const dt5v64: NDArray<[3]>;
declare const dt5v32: NDArray<[3], "float32">;
declare const dt5vi: NDArray<[3], "int32">;
declare const dt5vbl: NDArray<[3], "bool">;
declare const dt5m64: NDArray<[2, 3]>;
declare const dt5m32: NDArray<[2, 3], "float32">;
declare const dt5mi: NDArray<[2, 3], "int32">;
declare const dt5mbl: NDArray<[2, 3], "bool">;
declare const dt5c32: NDArray<[2, 3, 4], "float32">;
declare const dt5uniV: NDArray<[3], "float32" | "int32">;
declare const dt5uniM: NDArray<[2, 3], "float32" | "int32">;
declare const dt5anyV: NDArray<[3], any>;
declare const dt5anyM: NDArray<[2, 3], any>;
declare const dt5top: AnyNDArray;

// --- SqrtDType, alias level: IsAnyDType first, then IsUnion, then bool -> never, then the table ---
type DT5_SQRTD_F64 = Expect<Equal<SqrtDType<"float64">, "float64">>;
type DT5_SQRTD_F32 = Expect<Equal<SqrtDType<"float32">, "float32">>;
type DT5_SQRTD_I32 = Expect<Equal<SqrtDType<"int32">, "float64">>;
type DT5_SQRTD_BOOL = Expect<Equal<SqrtDType<"bool">, never>>;
type DT5_SQRTD_ANY = Expect<Equal<SqrtDType<any>, DType>>; // the Baustein-0 blocker D-1: without the any-gate this was `never`
type DT5_SQRTD_UNION = Expect<Equal<SqrtDType<"float32" | "int32">, DType>>;
type DT5_SQRTD_UNION_BOOL = Expect<Equal<SqrtDType<"float32" | "bool">, DType>>; // a union containing bool is NOT `never`
type DT5_SQRTD_ALL = Expect<Equal<SqrtDType<DType>, DType>>;

// --- sqrt(): niladic, per receiver dtype, shape kept ---
type DT5_SQRT_F64 = Expect<Equal<ReturnType<NDArray<[3]>["sqrt"]>, NDArray<[3], "float64">>>;
type DT5_SQRT_F32 = Expect<Equal<ReturnType<NDArray<[3], "float32">["sqrt"]>, NDArray<[3], "float32">>>;
type DT5_SQRT_I32 = Expect<Equal<ReturnType<NDArray<[3], "int32">["sqrt"]>, NDArray<[3], "float64">>>;
type DT5_SQRT_BOOL = Expect<Equal<ReturnType<NDArray<[3], "bool">["sqrt"]>, NDArray<[3], never>>>; // runtime-only rejection (M2 v10)
type DT5_SQRT_RANK0 = Expect<Equal<ReturnType<NDArray<[], "float32">["sqrt"]>, NDArray<[], "float32">>>;
type DT5_SQRT_RANK2 = Expect<Equal<ReturnType<typeof dt5mi.sqrt>, NDArray<[2, 3], "float64">>>;
type DT5_SQRT_ZERO = Expect<Equal<ReturnType<NDArray<[0, 3], "float32">["sqrt"]>, NDArray<[0, 3], "float32">>>;
// any / union receivers (T-2): no claim, never `never`
type DT5_SQRT_ANY_RECV = Expect<Equal<ReturnType<typeof dt5anyV.sqrt>, NDArray<[3], DType>>>;
type DT5_SQRT_TOP_RECV = Expect<Equal<ReturnType<AnyNDArray["sqrt"]>["dtype"], DType>>;
type DT5_SQRT_UNION_RECV = Expect<Equal<ReturnType<typeof dt5uniV.sqrt>, NDArray<[3], DType>>>;
// chains through the `never` result degrade quietly and add no compile error of their own
const dt5chain = dt5mbl.sqrt().sum(0);
type DT5_SQRT_BOOL_CHAIN = Expect<Equal<typeof dt5chain, NDArray<[3], DType>>>;
const dt5chainT = dt5mbl.sqrt().transpose();
type DT5_SQRT_BOOL_TRANSPOSE = Expect<Equal<typeof dt5chainT, NDArray<[3, 2], never>>>;

// --- the three named `never` follow-ups of sqrt() on bool (A6 / COVENANT M3 v13): a native message through the private
// `__varianceD` marker, in code that throws at the sqrt() anyway. Pinned so a change in them is noticed. ---
const dt5never = dt5mbl.sqrt();
// @ts-expect-error - A6 (1): assigning the `never`-dtype result to AnyNDArray
const dt5neverTop: AnyNDArray = dt5never;
declare const dt5neverRow: ReturnType<NDArray<[3], "bool">["sqrt"]>;
NDArray.stack(
  // @ts-expect-error - A6 (2): `stack` with such a row
  [dt5neverRow],
);
// @ts-expect-error - A6 (3): the scalar form of add on it (TS2769, generic text)
dt5never.add(1);
void dt5neverTop;

// --- argmax(), overload 1: niladic -> plain number for every dtype, unions and `any` included ---
const dt5a0 = [dt5v64.argmax(), dt5v32.argmax(), dt5vi.argmax(), dt5vbl.argmax(), dt5uniV.argmax(), dt5anyV.argmax(), dt5top.argmax()] as const;
type DT5_ARGMAX_NILADIC = Expect<Equal<typeof dt5a0, readonly [number, number, number, number, number, number, number]>>;

// --- argmax(axis), overload 2: int32 indices of the reduced shape, EVERY receiver dtype (E6, float64 included) ---
const dt5a1 = dt5m32.argmax(1);
const dt5a2 = dt5m32.argmax(-1);
const dt5a3 = dt5m64.argmax(0);
const dt5a4 = dt5mbl.argmax(1);
const dt5a5 = dt5c32.argmax(1);
const dt5a6 = dt5m32.argmax(undefined);
const dt5a7 = dt5uniM.argmax(1);
const dt5a8 = dt5anyM.argmax(1);
const dt5a9 = dt5mi.argmax(0);
type DT5_ARGMAX_AXIS_F32 = Expect<Equal<typeof dt5a1, NDArray<[2], "int32">>>;
type DT5_ARGMAX_AXIS_NEG = Expect<Equal<typeof dt5a2, NDArray<[2], "int32">>>;
type DT5_ARGMAX_F64_INT32 = Expect<Equal<typeof dt5a3, NDArray<[3], "int32">>>; // E6: float64 receivers return int32 now (0.3.0: float64)
type DT5_ARGMAX_BOOL = Expect<Equal<typeof dt5a4, NDArray<[2], "int32">>>;
type DT5_ARGMAX_RANK3 = Expect<Equal<typeof dt5a5, NDArray<[2, 4], "int32">>>;
type DT5_ARGMAX_UNDEFINED = Expect<Equal<typeof dt5a6, NDArray<[], "int32">>>;
type DT5_ARGMAX_UNION_RECV = Expect<Equal<typeof dt5a7, NDArray<[2], "int32">>>;
type DT5_ARGMAX_ANY_RECV = Expect<Equal<typeof dt5a8, NDArray<[2], "int32">>>;
type DT5_ARGMAX_I32 = Expect<Equal<typeof dt5a9, NDArray<[3], "int32">>>;

// --- argmax(axis, keepdims), overload 3 ---
const dt5k1 = dt5m32.argmax(1, true);
const dt5k2 = dt5mbl.argmax(0, true);
const dt5k3 = dt5mi.argmax(1, false);
const dt5k4 = dt5m64.argmax(undefined, true);
const dt5k5 = dt5uniM.argmax(0, true);
type DT5_ARGMAX_KEEP_F32 = Expect<Equal<typeof dt5k1, NDArray<[2, 1], "int32">>>;
type DT5_ARGMAX_KEEP_BOOL = Expect<Equal<typeof dt5k2, NDArray<[1, 3], "int32">>>;
type DT5_ARGMAX_KEEP_FALSE = Expect<Equal<typeof dt5k3, NDArray<[2], "int32">>>;
type DT5_ARGMAX_KEEP_UNDEF = Expect<Equal<typeof dt5k4, NDArray<[1, 1], "int32">>>;
type DT5_ARGMAX_KEEP_UNION = Expect<Equal<typeof dt5k5, NDArray<[1, 3], "int32">>>;
// the shape guard stays (the dtype lock is gone): out-of-range axis stands at the axis argument, both overloads, any dtype
dt5m32.argmax(
  // @ts-expect-error - overload 2: axis out of range
  5,
);
dt5mbl.argmax(
  // @ts-expect-error - overload 3: axis out of range, bool receiver
  5,
  true,
);
dt5m64.argmax(
  // @ts-expect-error - overload 2: axis out of range, float64 receiver
  -3,
);

// --- topk(k): values keep the receiver dtype, indices are int32 for every receiver (E6) ---
const dt5t1 = dt5v32.topk(2);
const dt5t2 = dt5vbl.topk(3);
const dt5t3 = dt5v64.topk(1);
const dt5t4 = dt5uniV.topk(2);
const dt5t5 = dt5vi.topk(0);
const dt5t6 = dt5anyV.topk(2);
type DT5_TOPK_F32 = Expect<Equal<typeof dt5t1, { values: NDArray<[2], "float32">; indices: NDArray<[2], "int32"> }>>;
type DT5_TOPK_BOOL = Expect<Equal<typeof dt5t2, { values: NDArray<[3], "bool">; indices: NDArray<[3], "int32"> }>>;
type DT5_TOPK_F64_INT32 = Expect<Equal<typeof dt5t3.indices, NDArray<[1], "int32">>>; // E6 for float64 too
type DT5_TOPK_F64_VALUES = Expect<Equal<typeof dt5t3.values, NDArray<[1], "float64">>>;
type DT5_TOPK_UNION = Expect<Equal<typeof dt5t4, { values: NDArray<[2], "float32" | "int32">; indices: NDArray<[2], "int32"> }>>;
type DT5_TOPK_K0 = Expect<Equal<typeof dt5t5, { values: NDArray<[0], "int32">; indices: NDArray<[0], "int32"> }>>;
type DT5_TOPK_ANY = Expect<Equal<typeof dt5t6, { values: NDArray<[2], any>; indices: NDArray<[2], "int32"> }>>;
dt5v32.topk(
  // @ts-expect-error - k beyond the length stands at the k argument (no dtype lock any more)
  5,
);
dt5m32.topk(
  // @ts-expect-error - a rank-2 receiver is reported at the k argument
  1,
);
dt5vbl.topk(
  // @ts-expect-error - negative k, bool receiver
  -1,
);

// --- StackDTypeCheck, alias level ---
type DT5_STACKC_SAME = Expect<Equal<StackDTypeCheck<["float32", "float32"]>, true>>;
type DT5_STACKC_SINGLE = Expect<Equal<StackDTypeCheck<["bool"]>, true>>;
type DT5_STACKC_EMPTY = Expect<Equal<StackDTypeCheck<[]>, true>>; // the empty literal has its own SHAPE message
type DT5_STACKC_MIXED = Expect<
  Equal<StackDTypeCheck<["float32", "int32"]>, ShapeError<"stack: rows must share one dtype (got 'float32' and 'int32') — use astype() first">>
>;
type DT5_STACKC_LATE = Expect<
  Equal<StackDTypeCheck<["bool", "bool", "float64"]>, ShapeError<"stack: rows must share one dtype (got 'bool' and 'float64') — use astype() first">>
>;
// union / any rows are SKIPPED and the fold continues (D-3): the verdict does not depend on where they stand
type DT5_STACKC_UNION_ROW = Expect<Equal<StackDTypeCheck<["float32", "float32" | "int32"]>, true>>;
type DT5_STACKC_UNION_ONLY_DISJOINT = Expect<Equal<StackDTypeCheck<["float64", "float32" | "int32"]>, true>>; // no member equals float64, still no claim
type DT5_STACKC_UNION_FIRST = Expect<Equal<StackDTypeCheck<["float32" | "int32", "float32", "int32"]>, ShapeError<"stack: rows must share one dtype (got 'float32' and 'int32') — use astype() first">>>;
type DT5_STACKC_ANY_FIRST = Expect<Equal<StackDTypeCheck<[any, "float32", "int32"]>, ShapeError<"stack: rows must share one dtype (got 'float32' and 'int32') — use astype() first">>>;
type DT5_STACKC_ANY_MID = Expect<Equal<StackDTypeCheck<["float32", any, "int32"]>, ShapeError<"stack: rows must share one dtype (got 'float32' and 'int32') — use astype() first">>>;
type DT5_STACKC_ANY_OK = Expect<Equal<StackDTypeCheck<["float32", any, "float32"]>, true>>;
type DT5_STACKC_ALL_UNION = Expect<Equal<StackDTypeCheck<["float32" | "int32", "float32" | "int32"]>, true>>;
// arrays of rows (no tuple structure): no claim; a variadic tail is checked on its fixed leading rows
type DT5_STACKC_ARRAY = Expect<Equal<StackDTypeCheck<readonly ("float32" | "int32")[]>, true>>;
type DT5_STACKC_ARRAY_ONE = Expect<Equal<StackDTypeCheck<readonly "int32"[]>, true>>;
type DT5_STACKC_VARIADIC_OK = Expect<Equal<StackDTypeCheck<readonly ["float32", ...("int32")[]]>, true>>;
type DT5_STACKC_VARIADIC_ERR = Expect<
  Equal<StackDTypeCheck<readonly ["float32", "int32", ...("int32")[]]>, ShapeError<"stack: rows must share one dtype (got 'float32' and 'int32') — use astype() first">>
>;

// --- NDArray.stack(rows): result shape [N, D] and the shared dtype ---
const dt5s1 = NDArray.stack([dt5v32, dt5v32]);
const dt5s2 = NDArray.stack([dt5vi, dt5vi, dt5vi]);
const dt5s3 = NDArray.stack([dt5vbl]);
const dt5s4 = NDArray.stack([dt5v64, dt5v64]);
const dt5s5 = NDArray.stack([dt5v32, dt5uniV]);
const dt5s6 = NDArray.stack([dt5v32, dt5anyV]);
const dt5s7 = NDArray.stack([dt5uniV, dt5uniV]);
const dt5s8 = NDArray.stack([dt5uniV, NDArray.zeros([3], "float64")]);
type DT5_STACK_F32 = Expect<Equal<typeof dt5s1, NDArray<readonly [2, 3], "float32">>>;
type DT5_STACK_I32 = Expect<Equal<typeof dt5s2, NDArray<readonly [3, 3], "int32">>>;
type DT5_STACK_BOOL = Expect<Equal<typeof dt5s3, NDArray<readonly [1, 3], "bool">>>;
type DT5_STACK_F64 = Expect<Equal<typeof dt5s4, NDArray<readonly [2, 3], "float64">>>;
type DT5_STACK_UNION_ROW = Expect<Equal<typeof dt5s5, NDArray<readonly [2, 3], "float32" | "int32">>>;
type DT5_STACK_ANY_ROW = Expect<Equal<(typeof dt5s6)["dtype"], any>>;
type DT5_STACK_UNION_BOTH = Expect<Equal<typeof dt5s7, NDArray<readonly [2, 3], "float32" | "int32">>>;
type DT5_STACK_UNION_DISJOINT = Expect<Equal<typeof dt5s8, NDArray<readonly [2, 3], "float32" | "int32" | "float64">>>; // accepted, runtime decides
declare const dt5arr: NDArray<[3], "int32">[];
declare const dt5arrMixed: (NDArray<[3], "float32"> | NDArray<[3], "int32">)[];
declare const dt5arrAny: AnyNDArray[];
const dt5s9 = NDArray.stack(dt5arr);
const dt5s10 = NDArray.stack(dt5arrMixed);
const dt5s11 = NDArray.stack(dt5arrAny);
type DT5_STACK_ARRAY = Expect<Equal<typeof dt5s9, NDArray<readonly [number, 3], "int32">>>;
type DT5_STACK_ARRAY_MIXED = Expect<Equal<typeof dt5s10, NDArray<readonly [number, 3], "float32" | "int32">>>;
type DT5_STACK_ARRAY_ANY = Expect<Equal<(typeof dt5s11)["dtype"], any>>;
// a dtype mismatch is an error at the rows argument (the exact text and the order shape-before-dtype: fixture test)
NDArray.stack(
  // @ts-expect-error - float32 and int32
  [dt5v32, dt5vi],
);
NDArray.stack(
  // @ts-expect-error - the mismatch is the LAST row
  [dt5v64, dt5v64, dt5vbl],
);
NDArray.stack(
  // @ts-expect-error - a union row in front is skipped, the definite rows still disagree
  [dt5uniV, dt5v32, dt5vi],
);
NDArray.stack(
  // @ts-expect-error - an any row in front is skipped, the definite rows still disagree
  [dt5anyV, dt5v32, dt5vi],
);
NDArray.stack(
  // @ts-expect-error - a variadic tail: the fixed leading rows disagree
  [dt5v32, dt5vi, ...dt5arr],
);
NDArray.stack(
  // @ts-expect-error - shape error (rank) first
  [dt5v32, dt5mbl],
);
NDArray.stack(
  // @ts-expect-error - an empty literal keeps its own message
  [],
);

// --- known limits (D-6, measured): a generic helper with an OPEN shape or dtype parameter cannot prove the call, so it is
// rejected -- like `add`/`where` (dt2/dt4). `argmax(axis)` with a CONCRETE shape and an open dtype compiles (the result does not
// depend on D), as do `topk` and `sqrt` with an open dtype. ---
function dt5OpenStack<S extends Shape, D extends DType>(x: NDArray<S, D>) {
  // @ts-expect-error - known limit: stack with an open S and an open D
  return NDArray.stack([x, x]);
}
function dt5OpenStackD<D extends DType>(x: NDArray<[3], D>) {
  // @ts-expect-error - known limit: stack with a concrete S but an open D (the dtype fold cannot resolve)
  return NDArray.stack([x, x]);
}
function dt5OpenArgmaxS<S extends Shape>(x: NDArray<S, "float32">) {
  // @ts-expect-error - known limit: argmax(axis) with an open S (like sum(axis))
  return x.argmax(0);
}
function dt5OpenArgmaxD<D extends DType>(x: NDArray<[2, 3], D>) {
  return x.argmax(0);
}
function dt5OpenTopk<D extends DType>(x: NDArray<[3], D>) {
  return x.topk(2);
}
function dt5OpenSqrt<D extends DType>(x: NDArray<[3], D>) {
  return x.sqrt();
}
type DT5_OPEN_ARGMAX_D = Expect<Equal<ReturnType<typeof dt5OpenArgmaxD<"bool">>, NDArray<[3], "int32">>>;
type DT5_OPEN_TOPK_D = Expect<Equal<ReturnType<typeof dt5OpenTopk<"int32">>, { values: NDArray<[2], "int32">; indices: NDArray<[2], "int32"> }>>;
type DT5_OPEN_SQRT_D = Expect<Equal<ReturnType<typeof dt5OpenSqrt<"int32">>, NDArray<[3], "float64">>>;
type DT5_OPEN_SQRT_D_BOOL = Expect<Equal<ReturnType<typeof dt5OpenSqrt<"bool">>, NDArray<[3], never>>>;
void [dt5OpenStack, dt5OpenStackD, dt5OpenArgmaxS, dt5OpenArgmaxD, dt5OpenTopk, dt5OpenSqrt];
