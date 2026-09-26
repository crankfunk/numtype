/**
 * Public function surface: `NDArray<S>` + constructors + the four ops
 * (add, matmul, sum, transpose). See docs/spike-01-ergebnisse.md for the
 * error-surfacing pattern write-up and the alternative considered.
 *
 * Error-surfacing pattern (chosen): each op's "other-shape" parameter is
 * typed as a conditional — when the computed result type is a
 * `ShapeError<Message>`, the parameter type becomes an object requiring a
 * `__shapeError: Message` property. The actual argument (a plain
 * `NDArray<B>`) obviously lacks that property, so `tsc` reports a "missing
 * property" error *at that argument*, and the property's type is the
 * literal message string — so the message (naming the offending shapes)
 * appears verbatim in the error. When shapes ARE compatible, the
 * conditional resolves to plain `NDArray<B>` and the call is unconstrained.
 * See docs/spike-01-ergebnisse.md for the alternative (return-type-only
 * error surfacing) and why it was rejected: it type-checks fine at the
 * mistake's call site and only errors later, wherever the result is first
 * consumed — which can be far from the actual mistake, especially in
 * chained calls.
 */

import type { Broadcast } from "./broadcast.ts";
import { type Dim, type Mutable, type RankUnknowable, type Shape, type ShapeError } from "./dim.ts";
import type { MatMul } from "./matmul.ts";
import type { ReduceAxis, Transpose } from "./reduce.ts";
import type { ReshapeCheck } from "./reshape.ts";
import {
  argmaxRuntime,
  assertFloat64Locked,
  assertReshapeArgs,
  assertVectorPair,
  astypeConvert,
  computeStrides,
  convertToDType,
  type DataOf,
  type DataOfRuntime,
  dotRuntime,
  type DType,
  elementwiseBinary,
  formatNDArrayDisplay,
  itemRuntime,
  keepDimsShape,
  matmulRuntime,
  meanRuntime,
  normalizeSliceSpecs,
  normSqRuntime,
  onesData,
  product,
  scalarElementwiseRuntime,
  sliceDtyped,
  type SliceSpec,
  sqrtRuntime,
  stackRuntime,
  sumRuntime,
  topkRuntime,
  transposeDtyped,
  zerosData,
} from "./runtime.ts";
import type { LiteralShapeProduct } from "./literal-arithmetic.ts";
import type { SliceShape, SliceSpecInput, SliceSpecsGuard } from "./slice.ts";
import type { DotCheck, ItemGuard, StackCheck, StackShape, TopkCheck, TopkShape } from "./vector.ts";
import { checkThreadedEnv, WasmBackend, type BackendKind, type ThreadedBackendOptions } from "./wasm/backend-api.ts";
import { initCore } from "./wasm/loader.ts";
import type { ThreadedBackend } from "./wasm/threaded.ts";

/** Narrow a possibly-erroring computed shape down to a real `Shape`,
 * excluding the `ShapeError` branch. Only ever evaluated at call sites
 * where the compatible branch already applies (the incompatible branch is
 * rejected earlier, at the argument itself, by `Guard`).
 *
 * Exported (type-only, Kern 02): `spike/src/wasm/resident.ts`'s `WNDArray`
 * reuses this exact type machinery so its ops surface shape errors at the
 * argument identically to `NDArray` — see docs/kern-02-residency-spec.md. */
export type OkShape<S> = S extends ShapeError<string> ? never : S extends Shape ? S : never;

/** The argument-side guard: forces a "missing property" error, naming the
 * shape mismatch, at the *argument* when `Result` is a `ShapeError`.
 *
 * Tuple-wrapped (D-V1.4, docs/phase-d-vorarbeiten-spec.md): `[Result]
 * extends [ShapeError<infer Message>]` checks `Result` as a WHOLE, never
 * distributing over a union `Result` the way a naked `Result extends
 * ShapeError<...>` check would. This matters when `Result` is itself a
 * union produced by a shape-union operand (uniform rank, e.g.
 * `Broadcast<S, [2,3] | [7,3]>`): a MIXED union (some members ok, some
 * `ShapeError`) now resolves the tuple-wrapped check to `false` as a whole
 * (not every member extends `ShapeError`) and falls through to `Actual` —
 * accepted, gradual, runtime-backstopped, same policy as before but via a
 * cleaner mechanism. A UNIFORM error union (every member a `ShapeError`)
 * resolves the check to `true`, and `infer Message` over a non-distributed
 * union source infers the UNION of every matched branch's message, so the
 * rejection is ONE combined `{ __shapeError: M1 | M2 | ... }` object naming
 * every member's failure, instead of the old distributive form's union of
 * SEPARATE single-message objects (which also rejected, just with a
 * structurally messier diagnostic). A non-union `Result` (the overwhelming
 * common case) is unaffected either way — nothing to distribute over.
 *
 * Exported (type-only, Kern 02): see `OkShape` above. */
export type Guard<Result, Actual> = [Result] extends [ShapeError<infer Message>] ? { readonly __shapeError: Message } : Actual;

/**
 * Op-Scheibe W4 (docs/op-w4-stack-spec.md, D2/F1/F2): the `NDArray` ->
 * `Shape` unwrap `StackCheck`/`StackShape` (vector.ts) need, kept OUT of
 * vector.ts itself (F1 — vector.ts never imports `NDArray`, to avoid an
 * import cycle: dim.ts's own file-header precedent already documents why
 * that import direction is one-way). A HOMOMORPHIC mapped type (`{ [I in
 * keyof Rows]: ... }`, the same `[K in keyof S]` idiom `reduce.ts`'s
 * `AllOnes`/`slice.ts`'s `ErrorTuple` already use) — deliberately NOT a
 * `Rows[number] extends NDArray<infer S> ? S : never` extraction (F2, a
 * BLOCKER-class finding): that non-homomorphic form collapses to `never` on
 * a HETEROGENEOUS tuple, because `NDArray`'s own `__variance` marker makes
 * it INVARIANT (see the class doc comment below) — an invariant generic
 * indexed by `number` over a tuple of DIFFERENT `NDArray<S>` instantiations
 * has no single common `S` for `infer` to land on, so the conditional's
 * `NDArray<infer S>` branch never matches and the whole expression widens
 * to `never`. The homomorphic form sidesteps this entirely: it maps EACH
 * position independently (`Rows[I] extends NDArray<infer S> ? S : never`
 * per-index, never over the collapsed `Rows[number]` union), so a
 * heterogeneous tuple like `[NDArray<[3]>, NDArray<[4]>]` maps to
 * `[readonly [3], readonly [4]]` — the exact per-position information
 * `StackFold` (vector.ts) needs to catch a length mismatch, not `never`.
 *
 * Also homomorphic over an ARRAY `Rows` (F5): `{ [I in keyof Rows]: ... }`
 * applied to `readonly NDArray<[3]>[]` preserves the array shape, yielding
 * `readonly (readonly [3])[]` — never collapsing to a tuple — exactly what
 * `StackCheckArray`/`StackShapeArray` (vector.ts) expect on that path.
 *
 * Factored into this ONE named type (Baustein-0 measurement, verified
 * sketch): inlining the mapped type separately at both the
 * `StackCheck`/`StackShape` call sites in the `stack` method signature
 * below roughly DOUBLED the measured instantiation cost (≈1,428 vs ≈801) —
 * TS does not automatically dedupe two textually-identical-but-separately-
 * written mapped-type expressions the way it dedupes two references to the
 * SAME named type alias. Not exported — only `stack`'s own signature below
 * needs it.
 *
 * SECOND manifestation of F2's own root cause, caught empirically verifying
 * this exact type (probe: an ARRAY whose ELEMENT type is itself a union,
 * e.g. `readonly (NDArray<[3]>|NDArray<[4]>)[]`, the F8 test case):
 * inlining `Rows[I] extends NDArray<infer S> ? S : never` directly in the
 * mapped type's body does NOT distribute over that per-element union the
 * way it distributes per-POSITION over a heterogeneous TUPLE — for an
 * ARRAY, the homomorphic mapped type evaluates the element-type expression
 * ONCE against the array's single (here: union) element type, and `Rows[I]`
 * at that evaluation is an indexed-access expression, not a naked type
 * parameter reference — so `(NDArray<[3]>|NDArray<[4]>) extends
 * NDArray<infer S>` runs NON-distributively, and (`NDArray`'s own
 * invariance, same as F2) no single `S` satisfies both members at once ->
 * collapses to `never`, silently. `UnwrapRow` below reintroduces a FRESH
 * generic with its own naked parameter purely to force distribution again
 * at that call site (the identical "extra generic" idiom vector.ts's own
 * `ArrayRowD` already uses for the same reason) — this restores the
 * per-tuple-position behavior the doc comment above describes UNCHANGED
 * (each position's own type is still passed through `UnwrapRow` one at a
 * time) while fixing the array-union-element case to distribute to
 * `readonly [3]|readonly [4]`, letting `StackShapeArray`'s own `IsUnion`
 * filter (vector.ts, F8) degrade it to wide `number` deliberately, rather
 * than silently miscomputing `never`.
 */
type UnwrapRow<R> = R extends NDArray<infer S> ? S : never;
type RowShapesOf<Rows extends readonly NDArray<any>[]> = { [I in keyof Rows]: UnwrapRow<Rows[I]> };

/**
 * dt1 (K4, O2(a), docs/dtype-dt1-spec.md): the compile-time half of the lock
 * for every op not yet implemented for `D != "float64"` in this slice
 * (add/sub/mul/div/matmul/dot/cosineSimilarity/sum/mean/argmax/topk/sqrt/
 * norm — dt2-dt5 unlock these progressively; `stack` has its own disclosed
 * M3 exception, see its doc comment above). `D extends DType` is an
 * ordinary in-scope generic reference — the SAME mechanism `Broadcast<S,B>`
 * already is, never a `this`-parameter (Owner decision O2(a): a
 * `this`-parameter's TS2684 diagnostic is opaque and not in the M3
 * exception list). Resolves to `true` for `D = "float64"`; otherwise a
 * `ShapeError` naming the op and the offending dtype, WORD-IDENTICAL to
 * `lockedOpMessage` (runtime.ts, M3 message parity) — attached to WHATEVER
 * argument the op already has (message-table order: independent of, and
 * combined with, any shape check the same argument might also carry, the
 * same pattern `add`'s own inline conditionals below already use for shape
 * errors).
 *
 * Known gap, disclosed (M2 v8 "übergangsweise nur zur Laufzeit gesperrte
 * Ops", never a false accept): a genuinely NILADIC op (`sqrt()`, `norm()`,
 * the 0-argument overloads of `mean()`/`argmax()`) has no argument position
 * to attach this to at all — those forms keep their pre-dtype signature and
 * rely solely on `assertFloat64Locked` (runtime.ts) as their ONLY backstop.
 */
type DTypeLock<D extends DType, Op extends string> = D extends "float64"
  ? true
  : ShapeError<`${Op}: dtype '${D}' is not implemented for non-float64 arrays yet (use astype("float64") first)`>;

/** `DTypeLock`'s two-operand form, for a locked op whose argument is itself
 * an `NDArray<B, Dd>` (`matmul`/`dot`/`cosineSimilarity`) — the receiver's
 * own `D` is checked FIRST (message-table order), then the argument's `Dd`,
 * so a call is rejected if EITHER side isn't float64. */
type DTypeLockPair<D extends DType, Dd extends DType, Op extends string> = D extends "float64" ? DTypeLock<Dd, Op> : DTypeLock<D, Op>;

/** dt1 (K3(b), docs/dtype-dt1-spec.md): a same-typed-array-class copy of
 * `data` — the dtype-generic successor of the plain `new
 * Float64Array(this.data)` inline copy `reshape`/`flatten` used before dt1.
 * Deliberately kept INLINE in this file (not appended to `runtime.ts`,
 * spec K3(b)): it needs no `Guard`/type-level machinery, only an
 * `instanceof` dispatch over the four typed-array kinds. */
function copySameKindArray(data: DataOfRuntime): DataOfRuntime {
  if (data instanceof Float32Array) return new Float32Array(data);
  if (data instanceof Int32Array) return new Int32Array(data);
  if (data instanceof Uint8Array) return new Uint8Array(data);
  return new Float64Array(data);
}

/**
 * A minimal, checker-ENFORCED covariant read view (Spike 05,
 * docs/spike-05-variance-design-spec.md). `out S` makes the compiler itself
 * prove, at every compile, that widening a concrete `NDArrayView<[2, 3]>` to
 * `NDArrayView<Shape>` (or to `NDArrayView<readonly number[]>`) is sound —
 * the annotation is a declaration-site regression pin for every
 * instantiation at once. WITH ONE KNOWN LOOPHOLE the maintainer must carry
 * (verified empirically, Spike-05 fresh-context pass): TypeScript checks
 * METHOD-SHORTHAND parameters bivariantly (the same long-standing exemption
 * `strictFunctionTypes` grants methods), so a future member written
 * shorthand-style with `S` in argument position — `resizeTo(s: S): void` —
 * would COMPILE despite genuinely breaking covariance. Only a
 * property-typed function member (`resizeTo: (s: S) => void`) triggers
 * TS2636. House rule, therefore: this view must NEVER gain a member that
 * consumes `S`, and any future function-typed member is to be declared
 * property-style, where the annotation actually enforces.
 *
 * Kept to EXACTLY these three members — the omissions are load-bearing,
 * not oversights (probe evidence in the spec):
 *  - No op methods (add/matmul/sum/transpose/slice/…): sound dynamic-rank
 *    degradation needs `S` in ARGUMENT position (the `Guard<Result, Actual>`
 *    pattern above, on `NDArray` itself) to surface shape errors at the
 *    call site — consuming `S` genuinely breaks covariance (whether or not
 *    the checker catches the particular syntax, per the loophole above), so
 *    guard-bearing members and a covariant view are mutually exclusive
 *    (the two-of-three rule, docs/spike-01-ergebnisse.md, applied to a
 *    read-only view). Op consumers stay on generic `NDArray<S>` — a plain
 *    generic function never needs variance, it re-derives `S` per call.
 *  - No shape-COMPUTING members either (e.g. a hypothetical
 *    `transpose(): NDArrayView<Transpose<S>>`), even though such a member
 *    takes no `S`-typed argument: measured this session (scratchpad
 *    `variance-probes{,-2}.ts`), that member declaration itself fails with
 *    TS2636. The `out` check is ABSTRACT — it must hold for any two related
 *    S/super-S, not merely the shapes this codebase happens to build — and
 *    `Transpose` is not *provably* monotone under that abstract check, only
 *    *factually* monotone for concrete instantiations. Proof of the
 *    distinction: the identical member WITHOUT the `out` annotation type-
 *    checks fine (the structural widening check runs on concrete
 *    instantiations only, where `Transpose` happens to behave) — but that
 *    is covariance-BY-ACCIDENT, exactly the failure mode Spike 01 rejected
 *    for `NDArray` itself: unenforced, silently breakable by a future
 *    non-monotone shape op, and only catchable point-wise by tests. The
 *    `out` annotation trades that away for a compile-time proof, which is
 *    only available to a view with no computing members.
 *  - No `data` field: a `Float64Array` is naive-backend-specific; keeping
 *    the view to shape/strides/nested-array keeps it satisfiable by
 *    resident/strided backends (`WNDArray`-style) too, for a later phase
 *    (FOLLOWUPS.md) — out of scope here.
 *
 * One honest caveat on "read view" remains (a second, about `shape` element
 * mutability, was CLOSED by D-V2.3 below — `readonly shape: Readonly<S>` now
 * blocks `view.shape[0] = 99` too, not just property reassignment):
 *  - This is ordinary STRUCTURAL typing: any object with these three
 *    members satisfies the view, real `NDArray` or not — unlike
 *    `AnyNDArray` below, which stays de-facto nominal because `NDArray`'s
 *    private constructor blocks structural impostors. For a read-only
 *    surface that looseness is by design, but the capability difference
 *    between the two top types is real and worth knowing.
 *
 * Residency caveat (D-V2.2, docs/phase-d-vorarbeiten-spec.md): `WNDArray`
 * (`spike/src/wasm/resident.ts`) also `implements NDArrayView<S>`. Unlike
 * `NDArray`, a `WNDArray` handle has a disposal lifecycle — the interface
 * itself promises no LIVENESS: any member call on a disposed handle may
 * throw (`WNDArray.strides`/`shape` remain readable post-dispose since
 * they're plain fields, but `toNestedArray()` throws immediately, naming the
 * operation, before touching WASM memory). A caller holding only the
 * `NDArrayView<S>` supertype cannot tell whether the concrete handle behind
 * it is a live `NDArray` (never throws) or a disposed `WNDArray` (throws on
 * `toNestedArray()`) — know your concrete backend if liveness matters.
 *
 * `strides` (D-V2.1): a `readonly` PROPERTY, not a method — this harmonizes
 * `NDArray.strides()` (previously a method) with `WNDArray.strides` (always
 * a field, semantically load-bearing for views) onto one shape. The two
 * concrete backends differ in one honest way the interface does NOT paper
 * over: `NDArray`'s `strides` is a GETTER that computes a fresh array from
 * `shape` on every access (`a.strides !== a.strides` — row-major is always
 * derivable, so nothing is cached); `WNDArray`'s `strides` is an identity-
 * stable field set once at construction (it can be genuinely non-row-major
 * for a transpose/slice view, so it cannot be recomputed from `shape`
 * alone). The view CONTRACT promises neither identity stability nor
 * freshness — only the current values — so both implementations are
 * conforming; do not rely on identity across repeated `.strides` reads
 * through the `NDArrayView` interface.
 *
 * `shape: Readonly<S>` (D-V2.3, gated on the pre-flight probe in
 * docs/phase-d-vorarbeiten-spec.md — `Readonly<S>` under `out S` does NOT
 * throw TS2636 on TS 7.0.2, verified against a known-bad control): closes
 * the previously-documented latent hole above — `view.shape[0] = 99` is now
 * a compile error, not a silent no-op. `Readonly<S>` is homomorphic over the
 * tuple/array `S` (adds the `readonly` tuple-element modifier, doesn't
 * change element TYPES), so this is a pure precision gain, not a behavior
 * change: every previously-valid read still type-checks identically, only
 * element-write attempts newly error. Deep-readonly is all-or-nothing across
 * `NDArrayView`/`NDArray`/`WNDArray` (a partial rollout would collide with
 * `implements`: `Readonly<S>` is not assignable to a plain `shape: S`
 * interface member) — see `NDArray`/`WNDArray` for the mirrored change.
 * Member-level hovers now read `readonly [2, 3]` instead of `[2, 3]`; the
 * CLASS-level hover (`NDArray<[2, 3]>`) is unaffected — `S` itself is still
 * the clean tuple, only the wrapping `Readonly<...>` at the `.shape` member
 * changes (the clean-hover house rule binds the class hover, not every
 * member hover).
 */
/** Recursive fallback for `NestedArray<S>` whenever the rank of `S` cannot
 * be pinned down statically (dynamic rank, or a rank-mixed union) — an
 * honest recursive type instead of a wrong claim (docs/typed-nested-array-spec.md
 * D1/D5, Covenant M2). */
export type NestedValue = number | NestedValue[];

/** dt1 (K3, D5 "bool -> boolean-Blätter"): the `bool`-leaf twin of
 * `NestedValue` above, for the SAME dynamic-rank/rank-mixed-union fallback
 * but on a bool-dtyped array — using plain `NestedValue` (number leaves)
 * there would be a confident-WRONG claim for a bool array, not merely
 * incomplete (M2). Exported like `NestedValue` (docs/dtype-dt1-spec.md K3:
 * "NestedBoolValue rekursiv exportiert wie NestedValue"). */
export type NestedBoolValue = boolean | NestedBoolValue[];

/** dt1 (D5): the leaf type for one dtype's `NestedArray` — `bool` renders as
 * `boolean`, every other dtype (float64/float32/int32, all read out of their
 * typed array as a plain JS `number`) renders as `number`. */
type NestedLeafOf<D extends DType> = D extends "bool" ? boolean : number;

/** `toNestedArray()`'s return type, computed from the RANK of `S` alone —
 * never from its dim VALUES (a per-dim-value tuple would be the forbidden
 * tuple-length arithmetic over large dimensions, CLAUDE.md's TS limits).
 * `[2, 3]` → `number[][]`, `[]` → `number`, `[number, 3]` → `number[][]`.
 * The `[S] extends [never]` branch must run before anything else, or
 * `NestedOfRank` recurses forever trying to compute `never["length"]`.
 * `RankUnknowable` (dim.ts) degrades dynamic rank AND rank-mixed unions to
 * `NestedValue`/`NestedBoolValue` (by `D`) before any destructuring happens
 * (Arbeitsregel 3); a union of shapes with the SAME rank still resolves
 * through `S["length"]` to one concrete type (docs/typed-nested-array-spec.md
 * D1 v2 — the v1 recursive-decomposition design produced one instantiation
 * per union branch instead, `number[][] | number[][]`, which TS does not
 * dedupe). dt1: `D`'s default keeps every pre-dtype 1-type-argument call
 * (`NestedArray<S>`) meaning exactly what it meant before (float64, `number`
 * leaves). */
export type NestedArray<S extends Shape, D extends DType = "float64"> = [S] extends [never]
  ? D extends "bool"
    ? NestedBoolValue
    : NestedValue
  : RankUnknowable<S> extends true
    ? D extends "bool"
      ? NestedBoolValue
      : NestedValue
    : NestedOfRank<S["length"], [], NestedLeafOf<D>>;

/** Private, tail-recursive rank-to-nesting builder (accumulator pattern,
 * CLAUDE.md TS limits: tail-recursive types tolerate ~1000 instantiation
 * depth vs. ~100 non-tail-recursive) — `NestedOfRank<0> = number`,
 * `NestedOfRank<2> = number[][]`. Rank ceiling measured at 999 (TS2589 from
 * 999 on, one below the pre-existing shape machinery's ceiling of 1024 —
 * docs/typed-nested-array-spec.md D1), practically irrelevant. dt1: the leaf
 * type `T` defaults to `number` (float64/float32/int32) but is passed
 * `NestedLeafOf<D>` (`boolean` for bool) by `NestedArray` above. */
type NestedOfRank<N extends number, Acc extends readonly unknown[] = [], T = number> = Acc["length"] extends N
  ? T
  : NestedOfRank<N, [...Acc, unknown], T[]>;

export interface NDArrayView<out S extends Shape> {
  readonly shape: Readonly<S>;
  readonly strides: readonly number[];
  /** Stays `unknown`, unlike `NDArray`/`WNDArray`'s own `toNestedArray()`
   * (docs/typed-nested-array-spec.md D3): a rank-computed `NestedArray<S>`
   * return type under this interface's checker-enforced `out S` throws
   * TS2636 ("NestedArray<sub-S> is not assignable to NestedArray<super-S>"),
   * the same variance class as Spike 05's `Transpose` — `NestedArray<S>` is
   * a genuinely computing type function, not provably monotone in `S` the
   * way covariance needs (measured against real tsc 7.0.2, Baustein 0). The
   * concrete classes narrow the return type via `implements`, which is
   * unaffected by the interface member's own declared type. */
  toNestedArray(): unknown;
}

/**
 * Erased top type for heterogeneous containers and non-generic helpers that
 * also need to CALL ops (add/matmul/sum/transpose/slice/…) on a
 * shape-erased handle — a deliberately UNSAFE escape hatch, unsafe in BOTH
 * directions: `any` as the type argument bypasses the variance comparison
 * entirely (TS's own idiom for a variance-erased handle), so nothing stops
 * an `AnyNDArray` from flowing back into a precisely-shaped
 * `NDArray<[2, 3]>` binding either.
 *
 * HISTORY — D-V2.3 accidentally OPENED this class's variance, the owner
 * CLOSED it again on purpose (docs/phase-d-vorarbeiten-spec.md /
 * -v2-ergebnisse.md "Fund 2" + closure round, 2026-07-13): pre-D-V2.3, this
 * class read as fully invariant, but BY ACCIDENT — it carries no `in`/`out`
 * annotation, and invariance was an emergent side effect of
 * `sum(...).sum(...)`'s keepdims return-type machinery (`AllOnes<S>`
 * produces a genuinely MUTABLE tuple like `[1, 1]`, which blocked exactly
 * the comparison a wide `S`'s `readonly` result needed to satisfy for
 * widening to succeed). D-V2.3 wrapped `shape` in `Readonly<S>` for an
 * unrelated reason (closing the `nd.shape[0] = 99` mutation hole) and, as a
 * pure side effect, changed how that SAME comparison resolves: the widening
 * direction (`NDArray<[2, 3]>` assignable to `NDArray<readonly number[]>`)
 * silently opened up (isolated via an A/B probe on the real class — field
 * type `S` vs `Readonly<S>`, all else held constant; verified safe on its
 * own terms too — a wider static claim is only a LESS precise claim, never a
 * wrong one, the same shape as COVENANT.md's M2 principle, though M2 itself
 * is written about `Guard`/`OkShape`'s compile-time rejection semantics, not
 * about class assignability — this is an analogy worth drawing, not a
 * certificate M2 itself issues — see docs/phase-d-vorarbeiten-v2-ergebnisse.md
 * for the full investigation). But ACCIDENTAL, unowned invariance/variance is
 * exactly the failure mode this codebase otherwise refuses to ship (compare
 * `NDArrayView`'s checker-ENFORCED `out S` above, which is a PROVEN
 * annotation, not a measured accident) — so the owner decided (2026-07-13,
 * verify-round closure) to re-invariantize `NDArray` DELIBERATELY instead of
 * leaving its variance as a by-product of unrelated type machinery that
 * could just as easily drift open (or shut) again the next time some
 * unrelated member's type changes. The mechanism is the `__variance` member
 * on the class itself (see its own doc comment) — `NDArrayView<out S>`
 * remains the ONE checker-enforced covariant surface in this codebase;
 * `NDArray`/`WNDArray` are both deliberately invariant again. `AnyNDArray`
 * below is unaffected by any of this either way — `any` bypasses the
 * variance comparison regardless of what the class's own variance measures
 * or enforces, which is exactly why it stays the escape hatch for the
 * genuinely both-ways-unsafe use case.
 *
 * Reach for `AnyNDArray` ONLY when you need to CALL ops on a
 * heterogeneously-shaped array (`NDArray<Shape>`'s ops still work — dynamic
 * rank always degraded gracefully — but a `Guard`-typed argument position
 * consuming `S` is exactly what invariance, measured or enforced, cannot
 * make sound in the OTHER direction; `AnyNDArray`'s `any` stays the
 * deliberate, documented both-ways-unsafe tool for that). For read-only
 * access (shape/strides/toNestedArray) prefer `NDArrayView<Shape>` (above)
 * instead — it is the safe, checker-enforced top type: covariant by a proven
 * annotation, not by erasure, so it cannot silently unerase in the write
 * direction the way `any` can.
 *
 * dt1 (K5, docs/dtype-dt1-spec.md): fixed to `NDArray<any, any>` — D1
 * requires this top type to erase EVERY type parameter, dtype included; the
 * prototype (docs/dtype-design-ergebnisse.md, "Defekte") left this at
 * `NDArray<any>`, which left `D` at its default `"float64"` (the same
 * 1-arg-omitted-default mechanism `RowShapesOf`'s `UnwrapRow` below relies
 * on for `stack`), silently narrowing "any dtype" down to "any shape, only
 * float64" — a confident-WRONG top type this fix removes (Baustein 0,
 * finding F1). */
export type AnyNDArray = NDArray<any, any>;

/** dt1 (K1, D1): `D extends DType = "float64"` — the default keeps every
 * existing 1-type-argument use (`NDArray<[2, 3]>`) valid and meaning
 * float64, unchanged. */
export class NDArray<S extends Shape, D extends DType = "float64"> implements NDArrayView<S> {
  /** Deliberate invariance marker (re-invariantization owner-decision,
   * 2026-07-13 — full history in the `AnyNDArray` doc comment above). `S`
   * occurs in both a contravariant (parameter) and a covariant (return)
   * position of ONE property-typed function, which forces TS's structural
   * comparison between two instantiations of this class to require BOTH
   * directions to hold — i.e. invariance. Property-style is load-bearing,
   * not stylistic: TypeScript checks METHOD-SHORTHAND parameters
   * bivariantly (the same long-standing exemption documented on
   * `NDArrayView` above), so a marker written as a method
   * (`__variance(s: S): S`) would compile but do NOTHING — proven
   * empirically (verify-round Baustein B's Probe 3: the property-style form
   * closed the widening, the method-shorthand form did not). `declare`
   * means the member has no runtime representation (nothing initializes or
   * assigns it — a pure compile-time device, never a real field);
   * `private` means it never appears in the public surface or in a hover.
   * Neither affects `implements NDArrayView<S>` (an interface only requires
   * ITS OWN declared members; a class may carry extra private ones) nor the
   * class-level hover (`NDArray<[2, 3]>` — this member has no shape of its
   * own to display). */
  private declare readonly __variance: (s: S) => S;

  /** dt1 D-variance probe (docs/dtype-design-spec.md D11 "Varianz-Probe",
   * prototype commit 244ce04): measured (own probe, real tsc 7.0.2) that
   * `NDArray<S, D>` is ALREADY invariant in `D` once `data: DataOf<D>`
   * exists — but EMERGENTLY (a structural accident of the typed arrays'
   * per-kind `Symbol.toStringTag`, unrelated to dtype semantics), the exact
   * failure mode `AnyNDArray`'s own doc comment warns about for `S`
   * (accidental, unowned variance, liable to drift the next time an
   * unrelated member's type changes — e.g. a future dt-slice's own scalar
   * rule). This marker makes the invariance DELIBERATE and
   * declaration-site enforced instead: `D` in both a contravariant
   * (parameter) and covariant (return) position of one property-typed
   * function — same mechanism, same property-vs-method-shorthand bivariance
   * caveat as `__variance` above. */
  private declare readonly __varianceD: (d: D) => D;

  /** D-V2.3: deep-readonly (see `NDArrayView` doc comment above) — element
   * writes like `nd.shape[0] = 99` are now a compile error, not a silent
   * no-op. The stored value is unchanged; only the static type tightened. */
  readonly shape: Readonly<S>;
  /** dt1 (K1, D2): the RUNTIME dtype tag, needed alongside the compile-time
   * `D` because every op decides its OWN behavior (which typed array to
   * allocate, whether to wrap/truncate/fround) from a real value, not from
   * an erased type parameter. `readonly` — an `NDArray` never changes dtype
   * in place, `astype` always returns a fresh instance. */
  readonly dtype: D;
  readonly data: DataOf<D>;

  private constructor(shape: S, dtype: D, data: DataOf<D>) {
    this.shape = shape;
    this.dtype = dtype;
    this.data = data;
  }

  /** An all-zeros array of the given shape and dtype (dt1 K2, D3; default
   * `"float64"` keeps every existing 1-argument call unchanged). `const S`
   * means callers never need `as const` — `NDArray.zeros([2, 3])` infers `S`
   * from the literal `[2, 3]`. `Mutable<S>` strips the `readonly` a `const`
   * type param would otherwise attach, so the hover matches every other
   * op's plain-tuple display: `NDArray<[2, 3], "float64">`, not
   * `NDArray<readonly [2, 3], "float64">`. */
  static zeros<const S extends Shape, D extends DType = "float64">(shape: S, dtype?: D): NDArray<Mutable<S>, D> {
    const dt = (dtype ?? "float64") as D;
    return new NDArray<Mutable<S>, D>([...shape] as Mutable<S>, dt, zerosData(dt, product(shape)) as DataOf<D>);
  }

  /** An all-ones array of the given shape and dtype (dt1 K2, D3). */
  static ones<const S extends Shape, D extends DType = "float64">(shape: S, dtype?: D): NDArray<Mutable<S>, D> {
    const dt = (dtype ?? "float64") as D;
    return new NDArray<Mutable<S>, D>([...shape] as Mutable<S>, dt, onesData(dt, product(shape)) as DataOf<D>);
  }

  /** Build an array from flat row-major values (dt1 K2, D3). Four overloads:
   *  - no `dtype` option, a plain list or `Float64Array` source: float64,
   *    the pre-dtype default, unchanged.
   *  - no `dtype` option, a `Float32Array`/`Int32Array` source: the dtype is
   *    INFERRED from the source array's own kind.
   *  - an EXPLICIT `{ dtype }` option, any numeric source including
   *    `Uint8Array`: validated + converted (`convertToDType` — int32 must be
   *    an in-range integer, bool must be 0/1, else throw).
   * `Uint8Array` is deliberately ABSENT from the first two overloads' input
   * union — passing one without an explicit `dtype` is a COMPILE ERROR (D3:
   * a bare `Uint8Array` is ambiguous, since it could mean bool 0/1 or raw
   * byte data). The typed-array paths copy via the copy constructor
   * (memcpy-fast); forcing typed-array callers through `number[]` would
   * cost ~100x at the boundary (docs/kern-02-ergebnisse.md, chain-bench
   * finding). The input is always copied, never aliased. Throws at runtime
   * if `values.length` doesn't match the shape's element count. */
  static fromArray<const S extends Shape>(shape: S, values: readonly number[] | Float64Array): NDArray<Mutable<S>, "float64">;
  static fromArray<const S extends Shape>(shape: S, values: Float32Array): NDArray<Mutable<S>, "float32">;
  static fromArray<const S extends Shape>(shape: S, values: Int32Array): NDArray<Mutable<S>, "int32">;
  static fromArray<const S extends Shape, D extends DType>(
    shape: S,
    values: readonly number[] | Float64Array | Float32Array | Int32Array | Uint8Array,
    opts: { dtype: D },
  ): NDArray<Mutable<S>, D>;
  static fromArray<const S extends Shape, D extends DType>(
    shape: S,
    values: readonly number[] | Float64Array | Float32Array | Int32Array | Uint8Array,
    opts?: { dtype?: D },
  ): NDArray<Mutable<S>, D> {
    const size = product(shape);
    if (values.length !== size) {
      throw new Error(`fromArray: expected ${size} values for shape [${shape.join(",")}], got ${values.length}`);
    }
    let dtype: DType;
    let data: DataOfRuntime;
    if (opts?.dtype !== undefined) {
      dtype = opts.dtype;
      data = convertToDType(dtype, values as ArrayLike<number>);
    } else if (values instanceof Float32Array) {
      dtype = "float32";
      data = new Float32Array(values);
    } else if (values instanceof Int32Array) {
      dtype = "int32";
      data = new Int32Array(values);
    } else if (values instanceof Uint8Array) {
      // Unreachable through the typed overloads above (a bare `Uint8Array`
      // requires an explicit `{ dtype }` there) — runtime backstop for a
      // caller that bypasses the type layer, same "never silently wrong"
      // discipline the rest of this file follows (M2).
      throw new Error(`fromArray: a Uint8Array source requires an explicit { dtype } option (ambiguous — could mean bool 0/1 or raw byte data)`);
    } else if (values instanceof Float64Array) {
      dtype = "float64";
      data = new Float64Array(values);
    } else {
      dtype = "float64";
      data = Float64Array.from(values as readonly number[]);
    }
    return new NDArray<Mutable<S>, D>([...shape] as Mutable<S>, dtype as D, data as DataOf<D>);
  }

  /** dt1 (K2, D3): dtype-cast to `DataOf<T>`, same conversion rules
   * `astypeConvert` documents (float32 via `Math.fround`; float->int32
   * truncates toward zero and throws on NaN/±Infinity/out-of-range; ->bool
   * is `x !== 0`, NaN -> `true`; bool->numeric is plain 0/1). Always a fresh
   * copy (house invariant: `NDArray` never aliases), even when `T` equals
   * `D`. */
  astype<T extends DType>(dtype: T): NDArray<S, T> {
    const data = astypeConvert(dtype, this.data as unknown as DataOfRuntime);
    return new NDArray<S, T>(this.shape as unknown as S, dtype, data as DataOf<T>);
  }

  /** dt1 (K1, D2): the dtype-typed read-out `NDArray` never had a dedicated
   * method for (unlike `WNDArray.toArray()`, resident.ts) — `.data` is
   * already public, this is a same-shape alias for parity with the wider
   * design and for symmetry with `item`/`toNestedArray` below. */
  toArray(): DataOf<D> {
    return this.data;
  }

  /** Stack N independently-built rank-1 rows into a rank-2 `[N, D]` matrix
   * (Op-Scheibe W4, docs/op-w4-stack-spec.md; wishlist evidence F5,
   * docs/dogfooding-rag-ergebnisse.md — `embedMatrix`'s hand-rolled
   * `Float64Array#set`-at-row-offset flatten helper in
   * examples/rag-demo/embedding.ts is the exact algorithm `stackRuntime`
   * below reuses). NumPy's `np.stack([...])`/`np.array([row for row in
   * ...])` reflex — "fromRows" would be an equally fitting name (mentioned
   * here as a doc-only alias, not a second export): D1 scopes this method
   * to NO general axis/higher-rank stack, no `concat`/`vstack`/`hstack` —
   * see the spec's Nicht-Ziele. Inserted here, right after `fromArray`
   * (Baustein-0 recommendation): `stack` is conceptually a constructor too
   * — it never reads `this`, only builds a fresh `NDArray` from its rows.
   *
   * Two call shapes (D2, `StackCheck`/`StackShape`, vector.ts):
   *  - a literal TUPLE of rows (`NDArray.stack([a, b])` — `const Rows`
   *    keeps it a tuple rather than widening to a plain array, same
   *    rationale `zeros`/`ones`/`fromArray`'s own `const S` already
   *    documents) -> `N` = `Rows["length"]` (a literal), `D` = every row's
   *    dim, checked pairwise equal; a proven length mismatch, a proven
   *    non-rank-1 row, or an empty tuple literal (F3) is a compile error AT
   *    THE `rows` ARGUMENT (`Guard`);
   *  - a `readonly NDArray<[3]>[]` ARRAY of unknown length -> `N` degrades
   *    honestly to `number`; `D` stays the shared literal unless a row's
   *    own dim is dynamic, or the array's element type is itself a union of
   *    shapes (F8) — both degrade `D` to `number` too.
   *
   * `RowShapesOf<Rows>` (above) is the `NDArray` -> `Shape` unwrap
   * `StackCheck`/`StackShape` need — vector.ts never imports `NDArray` (F1,
   * cycle-risk precedent documented on `RowShapesOf` itself).
   *
   * Runtime (D3, `stackRuntime`): validates (>= 1 row; every row rank-1;
   * every row the same length) with the exact three stems `StackCheck`
   * mirrors, then a row-major `Float64Array#set` copy per row into a fresh
   * buffer — never aliasing any row's own `data` (D=0 rows are valid:
   * `[[], []]` stacks to `[2, 0]`).
   *
   * Surface asymmetry (D1, disclosed, same shape as `argmax`/`topk`/the W2
   * scalar overloads/`mean`/`sqrt`): `stack` exists ONLY on this naive
   * `NDArray` — no WASM kernel, no `WNDArray` parity yet (FOLLOWUPS.md
   * tracks the follow-up).
   *
   * dt1 (K4, docs/dtype-dt1-spec.md; M3 v8 "befristete Ausnahme stack"):
   * `Rows` is effectively float64-only already — `NDArray<any>` (in
   * `RowShapesOf`'s `UnwrapRow` below) fills the omitted `D` with its
   * default `"float64"`, so a non-float64 row is rejected by a NATIVE
   * structural-mismatch diagnostic (`data: Int32Array` not assignable to
   * `data: Float64Array`), not this codebase's own `DTypeLock` message — a
   * DISCLOSED, TIME-BOXED M3 exception until dt5 gives `stack` real
   * cross-dtype support (every row sharing one dtype; Promotion across rows
   * stays a non-goal, D5). `assertFloat64Locked` per row is still the
   * honest RUNTIME backstop, word-identical to every other locked op (M3
   * message parity holds at the runtime boundary; only the editor
   * diagnostic is the disclosed exception). */
  static stack<const Rows extends readonly NDArray<any>[]>(
    rows: Guard<StackCheck<RowShapesOf<Rows>>, Rows>,
  ): NDArray<OkShape<StackShape<RowShapesOf<Rows>>>> {
    const rs = rows as unknown as readonly NDArray<any>[];
    for (const r of rs) assertFloat64Locked("stack", r.dtype);
    const { shape, data } = stackRuntime(rs.map((r) => ({ shape: r.shape as readonly number[], data: r.data })));
    return new NDArray<OkShape<StackShape<RowShapesOf<Rows>>>>(shape as unknown as OkShape<StackShape<RowShapesOf<Rows>>>, "float64", data);
  }

  /** Explicit, opt-in performance backends (Item 10 — Backend-Wahl-API,
   * docs/item-10-backend-api-spec.md, D1). `NDArray` itself stays the
   * synchronous, browser-safe JS default and carries the USP (compile-time
   * shape checks) everywhere; this is the discoverable async entry point
   * into the WASM-resident world (`WNDArray`, `spike/src/wasm/resident.ts`)
   * and its further Node-only worker-thread opt-in. Overloaded so each
   * `kind` resolves to its own precise backend type — no widened union at
   * the call site (D1, overload resolution verified empirically).
   *
   * `"wasm"`: instantiates a FRESH WASM core (`initCore()`) and wraps it in
   * a `WasmBackend` — `backend.fromArray/zeros/ones` hand that `core`
   * straight through to the existing `WNDArray.*(core, ...)` statics,
   * unchanged (D1).
   *
   * `"threaded"`: env-detected Node-only opt-in (D2). `./wasm/threaded.ts`
   * has top-level `node:os`/`node:fs/promises`/`node:worker_threads`
   * imports — a STATIC import of it here would contaminate the
   * browser-safe `NDArray` default and `backend("wasm")` path, so it is
   * loaded with a DYNAMIC `import()` that runs strictly AFTER the env
   * check below passes: missing Node or the threads artifact throws with
   * the pinned message stem (detail decision 4), never a silent fallback
   * and never a bare crash from the module's own top-level imports.
   *
   * `kind` must be a LITERAL (`"wasm"` or `"threaded"`) at the call site —
   * ordinary TS overload resolution, nothing special to this method: a
   * caller holding a dynamically-typed `BackendKind` value (the `"wasm" |
   * "threaded"` union) gets rejected ("No overload matches this call"),
   * because a union isn't assignable to either overload's literal
   * parameter type (verified empirically). Narrow with an `if`/`switch` on
   * `kind` first, so each branch calls `backend` with a literal. */
  static async backend(kind: "wasm"): Promise<WasmBackend>;
  static async backend(kind: "threaded", opts?: ThreadedBackendOptions): Promise<ThreadedBackend>;
  static async backend(kind: BackendKind, opts?: ThreadedBackendOptions): Promise<WasmBackend | ThreadedBackend> {
    if (kind === "wasm") {
      const core = await initCore();
      return new WasmBackend(core);
    }
    const reason = await checkThreadedEnv();
    if (reason !== null) {
      throw new Error(`NDArray.backend("threaded"): threaded backend requires Node with the threads artifact (${reason})`);
    }
    const mod = await import("./wasm/threaded.ts");
    const pool = await mod.initThreadedCore(opts?.workers, opts?.matmulTimeoutMs);
    return new mod.ThreadedBackend(pool, opts?.minPoolWork);
  }

  /** Broadcasting elementwise add.
   *
   * Scalar overload (Op-Scheibe W2, docs/op-w2-scalar-mean-spec.md, D1/D2):
   * `x.add(s)` for a plain `number` `s` is shape-PRESERVING (NumPy-scalar
   * semantics, not a `[1]`-broadcast) — `NDArray<S>` in, `NDArray<S>` out,
   * unchanged even at rank 0 (`[]` stays `[]`; the old `x.add(fromArray([1],
   * [s]))` workaround would have turned `[]` into `[1]`, which is
   * NumPy-false). No guard on the scalar: every finite/non-finite `number`
   * is valid, IEEE propagation only (NaN/±Infinity), same as the existing
   * broadcast path. A UNION argument spanning BOTH overloads (`x: number |
   * NDArray<B>`) is rejected by TS as a whole (TS2769) even when every
   * member would individually be valid — an inherent property of real
   * overloads, the exact precedent already living on `NDArray.backend(kind)`
   * above (see its doc comment): narrow with `typeof x === "number"` first
   * if a caller genuinely needs to accept both forms through one variable.
   *
   * Declaration ORDER of the two overloads is LOAD-BEARING (Verify-B finding
   * F1, W2): on a failed overload set TS surfaces the error of the LAST
   * candidate, so the generic `Guard`-carrying overload must be declared
   * LAST — otherwise a plain broadcast mismatch reports the scalar decoy
   * ("not assignable to type 'number'") instead of the shape-naming
   * `__shapeError` (M3). Resolution is unaffected: a `number` argument
   * matches the scalar overload FIRST. Pinned by the diagnostic-quality
   * test in scalar-mean.test.ts (asserts the broadcast stem in real tsc
   * output — an `@ts-expect-error` alone cannot see message content).
   *
   * dt1 (K4, O2(a)): locked for `D != "float64"` in this slice (dt2 gives
   * `add` real Promotion + the D6 scalar rule) — both overloads' arguments
   * carry `DTypeLock`/`DTypeLockPair`, plus a runtime `assertFloat64Locked`
   * backstop (word-identical message, M3). `<DD extends DType = D>` on the
   * scalar overload (measured TS 7.0.2 quirk, prototype stage 7): an
   * overload signature with NO local type parameter of its own, whose
   * argument type directly references the ENCLOSING class's `D` inside a
   * `Guard`-wrapped conditional, fails TS2394 ("not compatible with its
   * implementation signature") — even though the implementation accepts a
   * strict superset. Adding a local generic that merely DEFAULTS to `D`
   * (never otherwise used) resolves it; the array overload already has a
   * local generic (`B`) for another reason and needs no such workaround
   * (mirrored on `Dd` below, one more for the argument's own dtype). */
  add<DD extends DType = D>(s: Guard<DTypeLock<DD, "add">, number>): NDArray<S>;
  add<B extends Shape, Dd extends DType>(
    other: Guard<Broadcast<S, B> extends ShapeError<string> ? Broadcast<S, B> : DTypeLockPair<D, Dd, "add">, NDArray<B, Dd>>,
  ): NDArray<OkShape<Broadcast<S, B>>>;
  add<B extends Shape, Dd extends DType>(
    other: number | Guard<Broadcast<S, B> extends ShapeError<string> ? Broadcast<S, B> : DTypeLockPair<D, Dd, "add">, NDArray<B, Dd>>,
  ): NDArray<OkShape<Broadcast<S, B>>> | NDArray<S> {
    assertFloat64Locked("add", this.dtype);
    if (typeof other === "number") {
      const data = scalarElementwiseRuntime("add", this.data as unknown as Float64Array, other);
      return new NDArray<S>(this.shape as unknown as S, "float64", data);
    }
    const o = other as unknown as NDArray<B, Dd>;
    assertFloat64Locked("add", o.dtype);
    const { shape, data } = elementwiseBinary(this.shape, this.data as unknown as Float64Array, o.shape, o.data as unknown as Float64Array, (x, y) => x + y);
    return new NDArray<OkShape<Broadcast<S, B>>>(shape as OkShape<Broadcast<S, B>>, "float64", data);
  }

  /** Broadcasting elementwise subtract (Kern 07). Structural mirror of
   * `add` — same `Broadcast`/`Guard`/`OkShape` pattern, pinned closure
   * `(x, y) => x - y`.
   *
   * Scalar overload (Op-Scheibe W2, docs/op-w2-scalar-mean-spec.md, D1/D2):
   * structural mirror of `add`'s own scalar overload — shape-preserving
   * NumPy-scalar semantics (rank 0 stays `[]`), no guard on the scalar
   * (IEEE propagation only), pinned closure `x - s`. Same documented
   * union-over-boundary rejection (TS2769), the same LOAD-BEARING overload
   * order (scalar first, generic Guard-carrier last — Verify-B F1), and
   * `NDArray.backend(kind)` precedent as `add` above — see its doc comment.
   *
   * dt1 (K4, O2(a)): locked for `D != "float64"` in this slice — same
   * mechanism, and the same `<DD = D>` TS2394 workaround, as `add` above. */
  sub<DD extends DType = D>(s: Guard<DTypeLock<DD, "sub">, number>): NDArray<S>;
  sub<B extends Shape, Dd extends DType>(
    other: Guard<Broadcast<S, B> extends ShapeError<string> ? Broadcast<S, B> : DTypeLockPair<D, Dd, "sub">, NDArray<B, Dd>>,
  ): NDArray<OkShape<Broadcast<S, B>>>;
  sub<B extends Shape, Dd extends DType>(
    other: number | Guard<Broadcast<S, B> extends ShapeError<string> ? Broadcast<S, B> : DTypeLockPair<D, Dd, "sub">, NDArray<B, Dd>>,
  ): NDArray<OkShape<Broadcast<S, B>>> | NDArray<S> {
    assertFloat64Locked("sub", this.dtype);
    if (typeof other === "number") {
      const data = scalarElementwiseRuntime("sub", this.data as unknown as Float64Array, other);
      return new NDArray<S>(this.shape as unknown as S, "float64", data);
    }
    const o = other as unknown as NDArray<B, Dd>;
    assertFloat64Locked("sub", o.dtype);
    const { shape, data } = elementwiseBinary(this.shape, this.data as unknown as Float64Array, o.shape, o.data as unknown as Float64Array, (x, y) => x - y);
    return new NDArray<OkShape<Broadcast<S, B>>>(shape as OkShape<Broadcast<S, B>>, "float64", data);
  }

  /** Broadcasting elementwise multiply (Kern 07). Structural mirror of
   * `add` — pinned closure `(x, y) => x * y`.
   *
   * Scalar overload (Op-Scheibe W2, docs/op-w2-scalar-mean-spec.md, D1/D2):
   * structural mirror of `add`'s own scalar overload — shape-preserving
   * NumPy-scalar semantics (rank 0 stays `[]`), no guard on the scalar
   * (IEEE propagation only), pinned closure `x * s`. Same documented
   * union-over-boundary rejection (TS2769), the same LOAD-BEARING overload
   * order (scalar first, generic Guard-carrier last — Verify-B F1), and
   * `NDArray.backend(kind)` precedent as `add` above — see its doc comment.
   *
   * dt1 (K4, O2(a)): locked for `D != "float64"` in this slice — same
   * mechanism, and the same `<DD = D>` TS2394 workaround, as `add` above. */
  mul<DD extends DType = D>(s: Guard<DTypeLock<DD, "mul">, number>): NDArray<S>;
  mul<B extends Shape, Dd extends DType>(
    other: Guard<Broadcast<S, B> extends ShapeError<string> ? Broadcast<S, B> : DTypeLockPair<D, Dd, "mul">, NDArray<B, Dd>>,
  ): NDArray<OkShape<Broadcast<S, B>>>;
  mul<B extends Shape, Dd extends DType>(
    other: number | Guard<Broadcast<S, B> extends ShapeError<string> ? Broadcast<S, B> : DTypeLockPair<D, Dd, "mul">, NDArray<B, Dd>>,
  ): NDArray<OkShape<Broadcast<S, B>>> | NDArray<S> {
    assertFloat64Locked("mul", this.dtype);
    if (typeof other === "number") {
      const data = scalarElementwiseRuntime("mul", this.data as unknown as Float64Array, other);
      return new NDArray<S>(this.shape as unknown as S, "float64", data);
    }
    const o = other as unknown as NDArray<B, Dd>;
    assertFloat64Locked("mul", o.dtype);
    const { shape, data } = elementwiseBinary(this.shape, this.data as unknown as Float64Array, o.shape, o.data as unknown as Float64Array, (x, y) => x * y);
    return new NDArray<OkShape<Broadcast<S, B>>>(shape as OkShape<Broadcast<S, B>>, "float64", data);
  }

  /** Broadcasting elementwise divide (Kern 07). Structural mirror of `add`
   * — pinned closure `(x, y) => x / y`. Pure IEEE 754: no zero checks, no
   * throws (`x/0 -> +/-Infinity`, `0/0 -> NaN`, signed zeros/infinities
   * propagate per the standard — a documented divergence from NumPy, which
   * additionally warns; see spec).
   *
   * Scalar overload (Op-Scheibe W2, docs/op-w2-scalar-mean-spec.md, D1/D2):
   * `x.div(s)` reads as "divide by `s`" — shape-preserving NumPy-scalar
   * semantics (rank 0 stays `[]`, no `[1]`-broadcast temp), no guard on the
   * scalar (same pure-IEEE contract as above: `x/0 -> +/-Infinity`, `0/0 ->
   * NaN`, no special-casing). Same documented union-over-boundary rejection
   * (TS2769), the same LOAD-BEARING overload order (scalar first, generic
   * Guard-carrier last — Verify-B F1), and `NDArray.backend(kind)`
   * precedent as `add` above — see its doc comment. The old
   * `x.div(fromArray([1], [s]))` `[1]`-wrap workaround still compiles and
   * still works (byte-identical to this overload for rank >= 1, D3), just
   * no longer necessary.
   *
   * dt1 (K4, O2(a)): locked for `D != "float64"` in this slice — same
   * mechanism, and the same `<DD = D>` TS2394 workaround, as `add` above. */
  div<DD extends DType = D>(s: Guard<DTypeLock<DD, "div">, number>): NDArray<S>;
  div<B extends Shape, Dd extends DType>(
    other: Guard<Broadcast<S, B> extends ShapeError<string> ? Broadcast<S, B> : DTypeLockPair<D, Dd, "div">, NDArray<B, Dd>>,
  ): NDArray<OkShape<Broadcast<S, B>>>;
  div<B extends Shape, Dd extends DType>(
    other: number | Guard<Broadcast<S, B> extends ShapeError<string> ? Broadcast<S, B> : DTypeLockPair<D, Dd, "div">, NDArray<B, Dd>>,
  ): NDArray<OkShape<Broadcast<S, B>>> | NDArray<S> {
    assertFloat64Locked("div", this.dtype);
    if (typeof other === "number") {
      const data = scalarElementwiseRuntime("div", this.data as unknown as Float64Array, other);
      return new NDArray<S>(this.shape as unknown as S, "float64", data);
    }
    const o = other as unknown as NDArray<B, Dd>;
    assertFloat64Locked("div", o.dtype);
    const { shape, data } = elementwiseBinary(this.shape, this.data as unknown as Float64Array, o.shape, o.data as unknown as Float64Array, (x, y) => x / y);
    return new NDArray<OkShape<Broadcast<S, B>>>(shape as OkShape<Broadcast<S, B>>, "float64", data);
  }

  /** Full NumPy `matmul`: 2-D product, 1-D promotion, batch broadcasting.
   *
   * dt1 (K4, O2(a)): locked for `D != "float64"` in this slice (dt3 gives
   * `matmul` real dtype support, including O1's int32-widening decision). */
  matmul<B extends Shape, Dd extends DType>(
    other: Guard<MatMul<S, B> extends ShapeError<string> ? MatMul<S, B> : DTypeLockPair<D, Dd, "matmul">, NDArray<B, Dd>>,
  ): NDArray<OkShape<MatMul<S, B>>> {
    assertFloat64Locked("matmul", this.dtype);
    const o = other as unknown as NDArray<B, Dd>;
    assertFloat64Locked("matmul", o.dtype);
    const { shape, data } = matmulRuntime(this.shape, this.data as unknown as Float64Array, o.shape, o.data as unknown as Float64Array);
    return new NDArray<OkShape<MatMul<S, B>>>(shape as OkShape<MatMul<S, B>>, "float64", data);
  }

  /** Sum-reduce along `axis` (negative counts from the end); omit `axis` to
   * sum every element down to a rank-0 array. Pass `keepdims = true` (NumPy
   * `keepdims`) to keep the reduced axis as size-1 instead of removing it
   * (rank preserved) — `undefined` axis + keepdims reduces every axis to an
   * all-ones shape. keepdims is pure shape metadata: the summed DATA is
   * byte-identical to the non-keepdims result (Kern 09).
   *
   * dt1 (K4, O2(a)): locked for `D != "float64"` in this slice (dt3 gives
   * `sum` real dtype support, E3: int32/bool widen to float64). The 0-arg
   * overload is NILADIC (same disclosed gap as `norm()` below — no argument
   * position to hang a compile-time `DTypeLock` on); the axis-bearing
   * overloads combine `DTypeLock` into their existing `ReduceAxis` `Guard`. */
  sum(): NDArray<OkShape<ReduceAxis<S, undefined, false>>>;
  sum<const Axis extends number | undefined>(
    axis: Guard<ReduceAxis<S, Axis> extends ShapeError<string> ? ReduceAxis<S, Axis> : DTypeLock<D, "sum">, Axis>,
  ): NDArray<OkShape<ReduceAxis<S, Axis, false>>>;
  sum<const Axis extends number | undefined, const KeepDims extends boolean | undefined>(
    axis: Guard<ReduceAxis<S, Axis> extends ShapeError<string> ? ReduceAxis<S, Axis> : DTypeLock<D, "sum">, Axis>,
    keepdims: KeepDims,
  ): NDArray<OkShape<ReduceAxis<S, Axis, KeepDims>>>;
  sum<const Axis extends number | undefined = undefined, const KeepDims extends boolean = false>(
    axis?: Guard<ReduceAxis<S, Axis> extends ShapeError<string> ? ReduceAxis<S, Axis> : DTypeLock<D, "sum">, Axis>,
    keepdims?: KeepDims,
  ): NDArray<any> {
    assertFloat64Locked("sum", this.dtype);
    const axisNum = axis as unknown as Axis | undefined;
    const { shape, data } = sumRuntime(this.shape, this.data as unknown as Float64Array, axisNum);
    const outShape = keepdims ? keepDimsShape(this.shape, axisNum) : shape;
    return new NDArray<OkShape<ReduceAxis<S, Axis, KeepDims>>>(
      outShape as OkShape<ReduceAxis<S, Axis, KeepDims>>,
      "float64",
      data,
    );
  }

  /** 1-D inner product (Kern 07): `a.dot(b)` for two rank-1 arrays of equal
   * length. Returns a plain `number` — deliberately leaves the `NDArray`
   * world (see spec's "dot(other)" design note: `sum()` is a reduction
   * that stays chainable, `dot`/`norm`/`cosineSimilarity` are scalar
   * consumer ops that terminate a chain). Rank != 1 (either operand) or a
   * length mismatch is a compile error at the argument (`DotCheck`) and a
   * runtime throw (`assertVectorPair`) for the gradual/dynamic cases the
   * type layer couldn't check statically.
   *
   * dt1 (K4, O2(a)): locked for `D != "float64"` in this slice (dt3). */
  dot<B extends Shape, Dd extends DType>(
    other: Guard<DotCheck<S, B, "dot"> extends ShapeError<string> ? DotCheck<S, B, "dot"> : DTypeLockPair<D, Dd, "dot">, NDArray<B, Dd>>,
  ): number {
    assertFloat64Locked("dot", this.dtype);
    const o = other as unknown as NDArray<B, Dd>;
    assertFloat64Locked("dot", o.dtype);
    assertVectorPair("dot", this.shape, o.shape);
    return dotRuntime(this.shape, this.data as unknown as Float64Array, o.shape, o.data as unknown as Float64Array);
  }

  /** L2/Frobenius norm over ALL elements (Kern 07), any rank (mirrors
   * `np.linalg.norm`'s default: flatten, then L2) — no guard, since a
   * niladic method has no argument to hang one on and every rank is valid
   * by this op's own semantics. `Math.sqrt` is IEEE-correctly-rounded, so
   * this is bit-identical to `WNDArray.norm` iff the underlying sum of
   * squares is (which the differential suite asserts).
   *
   * dt1 (K4, O2(a)): locked for `D != "float64"` — NILADIC, the disclosed
   * gap `DTypeLock`'s own doc comment names: no argument position to hang a
   * compile-time `Guard` on, `assertFloat64Locked` is the ONLY backstop
   * (M2: an honest no-claim, never a false accept). */
  norm(): number {
    assertFloat64Locked("norm", this.dtype);
    return Math.sqrt(normSqRuntime(this.data as unknown as Float64Array));
  }

  /** Cosine similarity (Kern 07): same rank-1/equal-length operand contract
   * as `dot` (own error-message prefix). The pinned expression (spec,
   * identical on both surfaces): `num = dot(a,b)`, `den =
   * sqrt(normSq(a)) * sqrt(normSq(b))`, `return num / den`. Pure IEEE, no
   * epsilon guards: a zero vector on either side makes `den` (or both
   * `num` and `den`) `0`, yielding `NaN`; an adversarial magnitude split
   * can underflow `den` to `0` with `num != 0`, yielding `+/-Infinity`.
   *
   * dt1 (K4, O2(a)): locked for `D != "float64"` in this slice — same
   * mechanism as `dot` above. */
  cosineSimilarity<B extends Shape, Dd extends DType>(
    other: Guard<
      DotCheck<S, B, "cosineSimilarity"> extends ShapeError<string> ? DotCheck<S, B, "cosineSimilarity"> : DTypeLockPair<D, Dd, "cosineSimilarity">,
      NDArray<B, Dd>
    >,
  ): number {
    assertFloat64Locked("cosineSimilarity", this.dtype);
    const o = other as unknown as NDArray<B, Dd>;
    assertFloat64Locked("cosineSimilarity", o.dtype);
    assertVectorPair("cosineSimilarity", this.shape, o.shape);
    const num = dotRuntime(this.shape, this.data as unknown as Float64Array, o.shape, o.data as unknown as Float64Array);
    const den = Math.sqrt(normSqRuntime(this.data as unknown as Float64Array)) * Math.sqrt(normSqRuntime(o.data as unknown as Float64Array));
    return num / den;
  }

  /** Reverse every axis (NumPy's `.T` generalized to N-D).
   *
   * dt1 (K3, D5 "D unverändert"): dtype-neutral — works for every dtype,
   * unlike the prototype (which locked this op to float64-only). Every
   * dtype's data moves through `transposeDtyped` (runtime.ts), which
   * allocates its output in the SAME typed-array class as the input; no
   * dtype lock, since this is pure data movement, no arithmetic. */
  transpose(): NDArray<Transpose<S>, D> {
    const { shape, data } = transposeDtyped(this.shape, this.data as unknown as DataOfRuntime);
    return new NDArray<Transpose<S>, D>(shape as Transpose<S>, this.dtype, data as DataOf<D>);
  }

  /** Basic (NumPy-style) slicing: one spec per leading axis, trailing axes
   * taken in full — see docs/kern-05-slicing-spec.md for the full semantics
   * table. `const Specs` means callers never write `as const` (same
   * rationale as `zeros`/`ones`/`fromArray`'s `const S`), which also lets
   * literal `start`/`stop`/`step` values reach the type layer. Always a
   * fresh COPY (naive reference; `WNDArray.slice` is the O(1) view twin —
   * both share `normalizeSliceSpecs`, see its doc comment for why that's a
   * deliberate, documented differential blind spot). Too many specs is a
   * compile error at the offending argument (`SliceSpecsGuard`, see
   * slice.ts) and a runtime throw (`normalizeSliceSpecs`) for gradual/
   * dynamic-rank callers the type layer couldn't check statically.
   *
   * dt1 (K3, D5 "D unverändert"): dtype-neutral, same rationale as
   * `transpose` above — `sliceDtyped` (runtime.ts) allocates its output in
   * the SAME typed-array class as the input. */
  slice<const Specs extends readonly SliceSpecInput[]>(
    ...specs: SliceSpecsGuard<S, Specs>
  ): NDArray<OkShape<SliceShape<S, Specs>>, D> {
    const rawSpecs = specs as unknown as readonly SliceSpec[];
    const norm = normalizeSliceSpecs(this.shape, rawSpecs);
    const { shape, data } = sliceDtyped(this.shape, this.data as unknown as DataOfRuntime, norm);
    return new NDArray<OkShape<SliceShape<S, Specs>>, D>(shape as OkShape<SliceShape<S, Specs>>, this.dtype, data as DataOf<D>);
  }

  /** Same elements, new shape (Kern 08, docs/kern-08-reshape-flatten-spec.md):
   * NumPy semantics minus `-1` inference (FOLLOWUPS). Always a fresh COPY
   * (house invariant: `NDArray` never aliases) — logical row-major order is
   * preserved, so this is a straight `Float64Array` copy under new shape
   * metadata, never a per-element reorder. Compile error at the argument
   * when both shapes' literal element products are known and differ
   * (`ReshapeCheck`); a provably-invalid literal dim of the new shape is
   * ALSO a compile error (the Kern-08 stretch, `LiteralReshapeDimInvalid`)
   * — both mirror `assertReshapeArgs`'s own runtime throw verbatim.
   * Gradual/dynamic callers fall through to that same runtime backstop.
   *
   * dt1 (K3, D5 "D unverändert"): dtype-neutral — the inline copy this
   * method already did (`new Float64Array(this.data)`) becomes
   * `copySameKindArray`, a same-typed-array-class copy (dt1 spec K3(b):
   * "reshape/flatten kopieren inline in ndarray.ts" — still inline here,
   * not a `runtime.ts` addition, since it needs no `Guard`/type machinery,
   * only an `instanceof` dispatch). */
  reshape<const NS extends Shape>(shape: Guard<ReshapeCheck<S, NS>, NS>): NDArray<Mutable<NS>, D> {
    const ns = shape as unknown as NS;
    assertReshapeArgs(this.shape, ns);
    const copy = copySameKindArray(this.data as unknown as DataOfRuntime);
    return new NDArray<Mutable<NS>, D>([...ns] as Mutable<NS>, this.dtype, copy as DataOf<D>);
  }

  /** Rank-1 copy of every element (Kern 08): `a.flatten()` behaves exactly
   * like `a.reshape([product(a.shape)])` (always valid — no guard, same
   * niladic-method reasoning as `norm()` in Kern 07). Return type is the
   * Spike-04 payoff: a statically computed literal rank-1 shape (hover
   * `NDArray<[1048576]>` for `[1024, 1024]`) whenever every dim of `S` is a
   * supported literal, degrading to the honest `NDArray<[number]>`
   * whenever the product itself degrades.
   *
   * dt1 (K3, D5 "D unverändert"): dtype-neutral, same `copySameKindArray`
   * inline copy as `reshape` above. */
  flatten(): NDArray<[LiteralShapeProduct<S>], D> {
    const size = product(this.shape);
    const copy = copySameKindArray(this.data as unknown as DataOfRuntime);
    return new NDArray<[LiteralShapeProduct<S>], D>([size] as unknown as [LiteralShapeProduct<S>], this.dtype, copy as DataOf<D>);
  }

  /** Row-major strides for the current shape (introspection helper,
   * D-V2.1: readonly property, not a method — harmonizes with `WNDArray`'s
   * field and the `NDArrayView` interface). A GETTER, not a cached value:
   * recomputed fresh from `shape` on every access (`a.strides !== a.strides`
   * — see the `NDArrayView` doc comment above for why that's fine under the
   * view contract and why `WNDArray`'s field differs). */
  get strides(): readonly number[] {
    return computeStrides(this.shape);
  }

  /** Read back as a plain nested JS array, rank- AND dtype-typed via
   * `NestedArray<S, D>` (docs/typed-nested-array-spec.md D2; dtype-korrekt
   * per dt1 K3, D5) — bool leaves read as `boolean` (`v !== 0`), every
   * other dtype as a plain `number` (already what a typed-array numeric
   * read produces). */
  toNestedArray(): NestedArray<S, D> {
    const strides = computeStrides(this.shape);
    const isBool = this.dtype === "bool";
    const build = (axis: number, offset: number): unknown => {
      if (axis === this.shape.length) {
        const v = this.data[offset] ?? 0;
        return isBool ? v !== 0 : v;
      }
      const dim: Dim = this.shape[axis] ?? 0;
      const stride = strides[axis] ?? 0;
      const out: unknown[] = [];
      for (let i = 0; i < dim; i++) out.push(build(axis + 1, offset + i * stride));
      return out;
    };
    return build(0, 0) as NestedArray<S, D>;
  }

  /** Index of the maximum element (Op-Scheibe W1,
   * docs/op-w1-argmax-topk-spec.md): `argmax()` (no axis) returns the index
   * into the ROW-MAJOR FLATTENING of every element (NumPy's `np.argmax(a)`
   * without an axis) — a deliberate departure from every other op above,
   * which stays inside the `NDArray` world: `argmax()`, like `dot`/`norm`/
   * `cosineSimilarity`, is a scalar-consumer op that TERMINATES a chain
   * (D2). `argmax(axis)`/`argmax(axis, keepdims)` instead stay inside the
   * `NDArray` world, mirroring `sum`'s own Arity-0/1/2 overload shape
   * exactly (same `ReduceAxis`/`Guard`/`OkShape` machinery, UNCHANGED —
   * reduce.ts is not touched by this slice). Total order (D4, pinned): NaN
   * counts as MAXIMAL (NumPy `argmax` behavior); ties (including `0`/`-0`,
   * compared via plain `>`, never `Object.is`) are broken by the FIRST
   * index. A niladic call has no argument to hang a compile-time guard on
   * (same reasoning as `norm()` above) — an empty receiver is a pure
   * runtime throw, never a compile-time claim.
   *
   * Result data is always an f64-encoded INTEGRAL index (this codebase is
   * f64-only throughout — `NDArray.data` is always a `Float64Array`, so an
   * index like `3` is stored as the exact double `3.0`, safely
   * round-trippable since every index here stays far below
   * `Number.MAX_SAFE_INTEGER`).
   *
   * Surface asymmetry (D1, disclosed): `argmax`/`topk` exist ONLY on this
   * naive `NDArray` — no WASM kernel, no `WNDArray`/threaded parity yet
   * (FOLLOWUPS.md tracks the follow-up).
   *
   * dt1 (K4, O2(a)): locked for `D != "float64"` in this slice (dt5). The
   * 0-arg overload is NILADIC (same disclosed gap as `norm()`); the
   * axis-bearing overloads combine `DTypeLock` into their existing
   * `ReduceAxis` `Guard`. */
  argmax(): number;
  argmax<const Axis extends number | undefined>(
    axis: Guard<ReduceAxis<S, Axis> extends ShapeError<string> ? ReduceAxis<S, Axis> : DTypeLock<D, "argmax">, Axis>,
  ): NDArray<OkShape<ReduceAxis<S, Axis, false>>>;
  argmax<const Axis extends number | undefined, const KeepDims extends boolean | undefined>(
    axis: Guard<ReduceAxis<S, Axis> extends ShapeError<string> ? ReduceAxis<S, Axis> : DTypeLock<D, "argmax">, Axis>,
    keepdims: KeepDims,
  ): NDArray<OkShape<ReduceAxis<S, Axis, KeepDims>>>;
  argmax<const Axis extends number | undefined = undefined, const KeepDims extends boolean = false>(
    axis?: Guard<ReduceAxis<S, Axis> extends ShapeError<string> ? ReduceAxis<S, Axis> : DTypeLock<D, "argmax">, Axis>,
    keepdims?: KeepDims,
  ): NDArray<any> | number {
    assertFloat64Locked("argmax", this.dtype);
    // `arguments.length`, not `axis === undefined`: the TRULY niladic
    // overload (`argmax()`, zero arguments -> `number`) is a DIFFERENT
    // overload from the 1-/2-arg forms with an axis value that happens to
    // BE `undefined` (`argmax(undefined)` / `argmax(undefined, true)` ->
    // full-reduction `NDArray<...>`, mirroring `sum(undefined[, keepdims])`
    // above exactly) — TS's own overload resolution already distinguishes
    // these by ARGUMENT COUNT at the call site (D2), so the implementation
    // must too, or a 2-arg `argmax(undefined, true)` call would silently
    // fall through to the bare-`number` branch and drop `keepdims`.
    if (arguments.length === 0) {
      const flat = argmaxRuntime(this.shape, this.data as unknown as Float64Array, undefined);
      return flat.data[0] ?? 0;
    }
    const axisNum = axis as unknown as Axis | undefined;
    const { shape, data } = argmaxRuntime(this.shape, this.data as unknown as Float64Array, axisNum);
    const outShape = keepdims ? keepDimsShape(this.shape, axisNum) : shape;
    return new NDArray<OkShape<ReduceAxis<S, Axis, KeepDims>>>(
      outShape as OkShape<ReduceAxis<S, Axis, KeepDims>>,
      "float64",
      data,
    );
  }

  /** Top-`k` values + indices along a rank-1 receiver (Op-Scheibe W1,
   * docs/op-w1-argmax-topk-spec.md, D3): `torch.topk`'s shape — BOTH
   * `values` and `indices`, since retrieval-style ranking needs both and
   * there is no `gather`/`take` op (yet) to recover one from the other.
   * Rank-1-only, `DotCheck`-family precedent for WHERE the error surfaces:
   * a receiver-rank problem is reported AT THE `k` ARGUMENT (`TopkCheck`,
   * vector.ts), same reasoning `dot`/`cosineSimilarity` already establish
   * above. `k = 0` and `k = length` are both valid (an empty result / the
   * whole vector, sorted). Total order (D4, pinned, same NaN-is-maximal
   * rule as `argmax`): NaN entries first (by ascending index among
   * themselves), then descending by value, ties broken by ascending index —
   * `values[i] === data[indices[i]]` exactly (a `Float64Array`-to-
   * `Float64Array` copy, so a NaN's exact bit payload survives).
   *
   * Same f64-index and surface-asymmetry notes as `argmax` above apply to
   * `indices`.
   *
   * dt1 (K4, O2(a)): locked for `D != "float64"` in this slice (dt5) —
   * combined into the SAME `Guard` as `TopkCheck` (rank check first,
   * message-table order). */
  topk<const K extends number>(
    k: Guard<TopkCheck<S, K> extends ShapeError<string> ? TopkCheck<S, K> : DTypeLock<D, "topk">, K>,
  ): { values: NDArray<OkShape<TopkShape<S, K>>>; indices: NDArray<OkShape<TopkShape<S, K>>> } {
    assertFloat64Locked("topk", this.dtype);
    const kNum = k as unknown as K;
    const { values, indices } = topkRuntime(this.shape, this.data as unknown as Float64Array, kNum);
    return {
      values: new NDArray<OkShape<TopkShape<S, K>>>([kNum] as unknown as OkShape<TopkShape<S, K>>, "float64", values),
      indices: new NDArray<OkShape<TopkShape<S, K>>>([kNum] as unknown as OkShape<TopkShape<S, K>>, "float64", indices),
    };
  }

  /** Mean-reduce along `axis` (negative axes count from the end); omit
   * `axis` to average every element down to a rank-0 array (Op-Scheibe W2,
   * docs/op-w2-scalar-mean-spec.md, D1/D4): overloads 0/1/2 are EXACTLY
   * `sum`'s own shape (same `ReduceAxis`/`Guard`/`OkShape` machinery,
   * `reduce.ts` unchanged) — `mean` is a reduction like `sum`, not a
   * scalar-consumer op like `dot`/`norm`/`argmax()`, so it stays chainable
   * and returns `NDArray<...>`, never a bare `number`. Pass `keepdims =
   * true` for the same size-1-instead-of-removed semantics `sum` documents.
   *
   * Runtime composition (D5, pinned order): `meanRuntime` = `sumRuntime`,
   * then EXACTLY ONE division per output element by `n` (`shape[axis]` for
   * the axis form, the total input element count for the full-reduction
   * form) — deliberately NOT `sum * (1/n)`, which rounds differently in f64;
   * see `runtime.ts`'s `meanRuntime` doc comment for the full determinism
   * note. Because the axis validation is entirely `sumRuntime`'s own, a
   * bad literal/dynamic axis throws the identical `reduce: axis …` stem
   * `sum` throws (M3) — no separate `mean`-specific message exists.
   *
   * size-0 disclosure (D5): the mean of an empty receiver, or of a size-0
   * axis, is `0/0 -> NaN` (NumPy-conformant), never a throw — unlike
   * `argmax()`, which throws on the same input. A caller relying on `mean`
   * to reject an empty input the way `argmax` does will be surprised;
   * this is a deliberate, disclosed divergence between the two reductions,
   * not an oversight.
   *
   * dt1 (K4, O2(a)): locked for `D != "float64"` in this slice (dt3, E3:
   * int32/bool widen to float64). The 0-arg overload is NILADIC (same
   * disclosed gap as `norm()`); the axis-bearing overloads combine
   * `DTypeLock` into their existing `ReduceAxis` `Guard`. */
  mean(): NDArray<OkShape<ReduceAxis<S, undefined, false>>>;
  mean<const Axis extends number | undefined>(
    axis: Guard<ReduceAxis<S, Axis> extends ShapeError<string> ? ReduceAxis<S, Axis> : DTypeLock<D, "mean">, Axis>,
  ): NDArray<OkShape<ReduceAxis<S, Axis, false>>>;
  mean<const Axis extends number | undefined, const KeepDims extends boolean | undefined>(
    axis: Guard<ReduceAxis<S, Axis> extends ShapeError<string> ? ReduceAxis<S, Axis> : DTypeLock<D, "mean">, Axis>,
    keepdims: KeepDims,
  ): NDArray<OkShape<ReduceAxis<S, Axis, KeepDims>>>;
  mean<const Axis extends number | undefined = undefined, const KeepDims extends boolean = false>(
    axis?: Guard<ReduceAxis<S, Axis> extends ShapeError<string> ? ReduceAxis<S, Axis> : DTypeLock<D, "mean">, Axis>,
    keepdims?: KeepDims,
  ): NDArray<any> {
    assertFloat64Locked("mean", this.dtype);
    const axisNum = axis as unknown as Axis | undefined;
    const { shape, data } = meanRuntime(this.shape, this.data as unknown as Float64Array, axisNum);
    const outShape = keepdims ? keepDimsShape(this.shape, axisNum) : shape;
    return new NDArray<OkShape<ReduceAxis<S, Axis, KeepDims>>>(
      outShape as OkShape<ReduceAxis<S, Axis, KeepDims>>,
      "float64",
      data,
    );
  }

  /** Elementwise square root (Op-Scheibe W3, docs/op-w3-sqrt-spec.md, D1/D2):
   * shape-PRESERVING at every rank, including rank 0 (`[]` stays `[]`) — the
   * missing last step of the `mul -> sum(axis) -> sqrt -> reshape -> div` L2-
   * normalization chain (`corpusSumSquares.sqrt()` instead of the old
   * hand-loop over `.data`, see `norm()`'s own doc comment and
   * examples/rag-demo/main.ts's FRICTION F1). No guard: `sqrt` is niladic —
   * same reasoning `norm()`/`flatten()` above already give (no argument to
   * hang a compile-time claim on, every shape is valid by this op's own
   * semantics).
   *
   * IEEE-754 exactness (the basis for excluding `sqrt` from the
   * transcendental non-goal, docs/op-w3-sqrt-spec.md): ECMA-262
   * `sec-math.sqrt` defines `Math.sqrt` via the exact correctly-rounded real
   * square root — the SAME correctly-rounded contract `+`/`-`/`*`/`/` carry,
   * unlike every transcendental `Math.*` method (`exp`/`log`/`sin`/...),
   * which the spec explicitly marks "implementation-approximated". `sqrt` is
   * therefore bit-deterministic, not merely "close enough" — the same
   * guarantee `norm()` already relies on above.
   *
   * NaN/sign disclosure (D2, pinned by tests): negative finite inputs yield
   * `NaN` (IEEE `sqrt` is undefined for negatives — no throw, gradual/
   * runtime propagation only, same house style as `div`'s zero handling);
   * `sqrt(-0) === -0` is a genuine IEEE edge case (`Object.is`-distinguished
   * from `+0`), not a bug.
   *
   * Surface asymmetry (D1, disclosed, same shape as `argmax`/`topk`/the W2
   * scalar overloads and `mean`): `sqrt` exists ONLY on this naive `NDArray`
   * — no WASM kernel, no `WNDArray` parity yet (FOLLOWUPS.md tracks the
   * follow-up).
   *
   * dt1 (K4, O2(a)): locked for `D != "float64"` in this slice (dt5's
   * int32/float32 rule for `sqrt` is a LATER dt-slice) — NILADIC, same
   * disclosed gap as `norm()` above (no argument position for a
   * compile-time `Guard`). */
  sqrt(): NDArray<S> {
    assertFloat64Locked("sqrt", this.dtype);
    const data = sqrtRuntime(this.data as unknown as Float64Array);
    return new NDArray<S>(this.shape as unknown as S, "float64", data);
  }

  /** Op-Scheibe W5 (docs/op-w5-item-spec.md): the direct scalar read, NumPy's
   * own `x.item(i, j, ...)` — the friction-log's last remaining Wunschlisten-
   * Platz (docs/dogfooding-rag-ergebnisse.md W5/F3: a scalar read out of a
   * score matrix, e.g. `similarities.item(qi, docIdx)`, previously needed
   * either `slice(qi).slice(docIdx)` — two fresh-copy allocations for one
   * number — or hand-rolled flat-index arithmetic over `.data` directly,
   * bypassing this class's own strided-read logic entirely). FULL indexing
   * only (D1): exactly one index per axis, rank 0 included (`item()`, zero
   * arguments, reads the sole element). No partial indexing (that's
   * `slice()`), no setter, no `at` alias (D1: a single name — `at` invites
   * confusion with `Array.prototype.at`'s single-axis semantics).
   *
   * `ItemGuard<S, Idx>` (vector.ts, Baustein-0 addendum F1-F8) is used
   * DIRECTLY as the rest-parameter's declared type (F1: wrapping it in the
   * `Guard<>` helper above is a permanent TS2370 at this very declaration —
   * confirmed empirically, the reason `slice()`'s own `SliceSpecsGuard`
   * already exists as a parallel, non-`Guard`-based mechanism). Two
   * DIFFERENT compile-time mechanisms cover the two provable-mistake
   * classes (F3, disclosed asymmetry, not an oversight):
   *  - arity (wrong number of indices for a statically-known rank) is a
   *    NATIVE `tsc` diagnostic, TS2554 ("Expected N arguments, but got M")
   *    — `ItemGuard`'s own fold pins the declared rest-parameter type to
   *    exactly `S["length"]` elements, so TS's own arity check does the
   *    work; there is no argument position to hang a custom message on for
   *    a MISSING argument;
   *  - a literal index PROVABLY invalid for its own axis (out of bounds
   *    per `LiteralIndexBounds`'s NumPy-negative-aware Spike-03 semantics,
   *    or a dot-form non-integer like `1.5` per `IsDotFormStep`) is a
   *    custom `{ __shapeError }` message AT that exact argument, stems
   *    word-for-word identical to `itemRuntime`'s own runtime throws (M3).
   * Wide rank/dim/index, union index, and dynamic-length spread calls
   * (`item(...someNumberArray)`, F4) all degrade to no-claim, gradual,
   * runtime-checked (`itemRuntime` stays authoritative for everything the
   * static guard can't prove).
   *
   * Surface asymmetry (same disclosed shape as `argmax`/`topk`/the W2
   * scalar overloads/`mean`/`sqrt`/`stack`): `item` exists ONLY on this
   * naive `NDArray` — no WASM kernel, no `WNDArray` parity yet (M1 v5:
   * kernel-less by design, a plain strided read; FOLLOWUPS.md tracks the
   * parity follow-up). Reuses Spike 03's own negative-index-normalization +
   * bounds-check semantics (`docs/spike-03-index-bounds-ergebnisse.md`),
   * `computeStrides` for the flat offset — no new arithmetic invented.
   *
   * dt1 (K3, D5 "item liefert boolean bei bool, sonst number"): dtype-
   * neutral, like `transpose`/`slice` above — no lock, pure strided read. */
  item<const Idx extends readonly number[]>(...indices: ItemGuard<S, Idx>): NestedLeafOf<D> {
    const v = itemRuntime(this.shape, this.data as unknown as Float64Array, indices as unknown as readonly number[]);
    return (this.dtype === "bool" ? v !== 0 : v) as NestedLeafOf<D>;
  }

  /** D3 (docs/release-0.3.0-spec.md): lossless round-trip via
   * `NDArray.fromArray(json.shape, json.data)`. `this.data` is already
   * logical row-major (the class invariant — see the constructor above),
   * so no strided read is needed here, unlike `WNDArray`'s version.
   * Disclosed, undodged limitation: `JSON.stringify` itself serializes
   * `NaN`/`±Infinity` as `null` (standard behavior, not worked around).
   *
   * dt1 (K3, D5 "toJSON `data` als number[] bzw. boolean[]"): a bool array's
   * 0/1 bytes read as `boolean` (`v !== 0`) — otherwise the type would say
   * `boolean[]` while the runtime handed back `0`/`1` numbers (M2). */
  toJSON(): { shape: number[]; data: D extends "bool" ? boolean[] : number[] } {
    const raw = Array.from(this.data as unknown as ArrayLike<number>);
    const data = (this.dtype === "bool" ? raw.map((v) => v !== 0) : raw) as D extends "bool" ? boolean[] : number[];
    return { shape: [...this.shape], data };
  }

  /** D3: Node's console/`util.inspect` custom hook, e.g.
   * `NDArray<[2, 3]> [[1, 2, 3], [4, 5, 6]]`. `Symbol.for(...)` needs no
   * Node import (browser-safe) even though the property name originates
   * from Node's inspect protocol. */
  [Symbol.for("nodejs.util.inspect.custom")](): string {
    return formatNDArrayDisplay("NDArray", this.shape, this.toNestedArray());
  }

  /** D3: same rendering as the inspect hook above, for plain string
   * contexts (`` `${arr}` ``, `String(arr)`, template literals). */
  toString(): string {
    return formatNDArrayDisplay("NDArray", this.shape, this.toNestedArray());
  }
}
