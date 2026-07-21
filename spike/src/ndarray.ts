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
import { type Dim, type Mutable, type Shape, type ShapeError } from "./dim.ts";
import type { MatMul } from "./matmul.ts";
import type { ReduceAxis, Transpose } from "./reduce.ts";
import type { ReshapeCheck } from "./reshape.ts";
import {
  argmaxRuntime,
  assertReshapeArgs,
  assertVectorPair,
  computeStrides,
  dotRuntime,
  elementwiseBinary,
  itemRuntime,
  keepDimsShape,
  matmulRuntime,
  meanRuntime,
  normalizeSliceSpecs,
  normSqRuntime,
  product,
  scalarElementwiseRuntime,
  sliceRuntime,
  type SliceSpec,
  sqrtRuntime,
  stackRuntime,
  sumRuntime,
  topkRuntime,
  transposeRuntime,
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
export interface NDArrayView<out S extends Shape> {
  readonly shape: Readonly<S>;
  readonly strides: readonly number[];
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
 */
export type AnyNDArray = NDArray<any>;

export class NDArray<S extends Shape> implements NDArrayView<S> {
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

  /** D-V2.3: deep-readonly (see `NDArrayView` doc comment above) — element
   * writes like `nd.shape[0] = 99` are now a compile error, not a silent
   * no-op. The stored value is unchanged; only the static type tightened. */
  readonly shape: Readonly<S>;
  readonly data: Float64Array;

  private constructor(shape: S, data: Float64Array) {
    this.shape = shape;
    this.data = data;
  }

  /** An all-zeros array of the given shape. `const S` means callers never
   * need `as const` — `NDArray.zeros([2, 3])` infers `S` from the literal
   * `[2, 3]`. `Mutable<S>` strips the `readonly` a `const` type param would
   * otherwise attach, so the hover matches every other op's plain-tuple
   * display: `NDArray<[2, 3]>`, not `NDArray<readonly [2, 3]>`. */
  static zeros<const S extends Shape>(shape: S): NDArray<Mutable<S>> {
    return new NDArray<Mutable<S>>([...shape] as Mutable<S>, new Float64Array(product(shape)));
  }

  /** An all-ones array of the given shape. */
  static ones<const S extends Shape>(shape: S): NDArray<Mutable<S>> {
    return new NDArray<Mutable<S>>([...shape] as Mutable<S>, new Float64Array(product(shape)).fill(1));
  }

  /** Build an array from flat row-major values. Accepts a plain list or a
   * `Float64Array` — the typed-array path copies via the copy constructor
   * (memcpy-fast); forcing typed-array callers through `number[]` would
   * cost ~100x at the boundary (docs/kern-02-ergebnisse.md, chain-bench
   * finding). The input is always copied, never aliased. Throws at runtime
   * if `values.length` doesn't match the shape's element count. */
  static fromArray<const S extends Shape>(shape: S, values: readonly number[] | Float64Array): NDArray<Mutable<S>> {
    const size = product(shape);
    if (values.length !== size) {
      throw new Error(`fromArray: expected ${size} values for shape [${shape.join(",")}], got ${values.length}`);
    }
    const data = values instanceof Float64Array ? new Float64Array(values) : Float64Array.from(values);
    return new NDArray<Mutable<S>>([...shape] as Mutable<S>, data);
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
   * tracks the follow-up). */
  static stack<const Rows extends readonly NDArray<any>[]>(
    rows: Guard<StackCheck<RowShapesOf<Rows>>, Rows>,
  ): NDArray<OkShape<StackShape<RowShapesOf<Rows>>>> {
    const rs = rows as unknown as readonly NDArray<any>[];
    const { shape, data } = stackRuntime(rs.map((r) => ({ shape: r.shape as readonly number[], data: r.data })));
    return new NDArray<OkShape<StackShape<RowShapesOf<Rows>>>>(shape as unknown as OkShape<StackShape<RowShapesOf<Rows>>>, data);
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
   * output — an `@ts-expect-error` alone cannot see message content). */
  add(s: number): NDArray<S>;
  add<B extends Shape>(other: Guard<Broadcast<S, B>, NDArray<B>>): NDArray<OkShape<Broadcast<S, B>>>;
  add<B extends Shape>(other: number | Guard<Broadcast<S, B>, NDArray<B>>): NDArray<OkShape<Broadcast<S, B>>> | NDArray<S> {
    if (typeof other === "number") {
      const data = scalarElementwiseRuntime("add", this.data, other);
      return new NDArray<S>(this.shape as unknown as S, data);
    }
    const o = other as unknown as NDArray<B>;
    const { shape, data } = elementwiseBinary(this.shape, this.data, o.shape, o.data, (x, y) => x + y);
    return new NDArray<OkShape<Broadcast<S, B>>>(shape as OkShape<Broadcast<S, B>>, data);
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
   * `NDArray.backend(kind)` precedent as `add` above — see its doc comment. */
  sub(s: number): NDArray<S>;
  sub<B extends Shape>(other: Guard<Broadcast<S, B>, NDArray<B>>): NDArray<OkShape<Broadcast<S, B>>>;
  sub<B extends Shape>(other: number | Guard<Broadcast<S, B>, NDArray<B>>): NDArray<OkShape<Broadcast<S, B>>> | NDArray<S> {
    if (typeof other === "number") {
      const data = scalarElementwiseRuntime("sub", this.data, other);
      return new NDArray<S>(this.shape as unknown as S, data);
    }
    const o = other as unknown as NDArray<B>;
    const { shape, data } = elementwiseBinary(this.shape, this.data, o.shape, o.data, (x, y) => x - y);
    return new NDArray<OkShape<Broadcast<S, B>>>(shape as OkShape<Broadcast<S, B>>, data);
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
   * `NDArray.backend(kind)` precedent as `add` above — see its doc comment. */
  mul(s: number): NDArray<S>;
  mul<B extends Shape>(other: Guard<Broadcast<S, B>, NDArray<B>>): NDArray<OkShape<Broadcast<S, B>>>;
  mul<B extends Shape>(other: number | Guard<Broadcast<S, B>, NDArray<B>>): NDArray<OkShape<Broadcast<S, B>>> | NDArray<S> {
    if (typeof other === "number") {
      const data = scalarElementwiseRuntime("mul", this.data, other);
      return new NDArray<S>(this.shape as unknown as S, data);
    }
    const o = other as unknown as NDArray<B>;
    const { shape, data } = elementwiseBinary(this.shape, this.data, o.shape, o.data, (x, y) => x * y);
    return new NDArray<OkShape<Broadcast<S, B>>>(shape as OkShape<Broadcast<S, B>>, data);
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
   * no longer necessary. */
  div(s: number): NDArray<S>;
  div<B extends Shape>(other: Guard<Broadcast<S, B>, NDArray<B>>): NDArray<OkShape<Broadcast<S, B>>>;
  div<B extends Shape>(other: number | Guard<Broadcast<S, B>, NDArray<B>>): NDArray<OkShape<Broadcast<S, B>>> | NDArray<S> {
    if (typeof other === "number") {
      const data = scalarElementwiseRuntime("div", this.data, other);
      return new NDArray<S>(this.shape as unknown as S, data);
    }
    const o = other as unknown as NDArray<B>;
    const { shape, data } = elementwiseBinary(this.shape, this.data, o.shape, o.data, (x, y) => x / y);
    return new NDArray<OkShape<Broadcast<S, B>>>(shape as OkShape<Broadcast<S, B>>, data);
  }

  /** Full NumPy `matmul`: 2-D product, 1-D promotion, batch broadcasting. */
  matmul<B extends Shape>(other: Guard<MatMul<S, B>, NDArray<B>>): NDArray<OkShape<MatMul<S, B>>> {
    const o = other as unknown as NDArray<B>;
    const { shape, data } = matmulRuntime(this.shape, this.data, o.shape, o.data);
    return new NDArray<OkShape<MatMul<S, B>>>(shape as OkShape<MatMul<S, B>>, data);
  }

  /** Sum-reduce along `axis` (negative counts from the end); omit `axis` to
   * sum every element down to a rank-0 array. Pass `keepdims = true` (NumPy
   * `keepdims`) to keep the reduced axis as size-1 instead of removing it
   * (rank preserved) — `undefined` axis + keepdims reduces every axis to an
   * all-ones shape. keepdims is pure shape metadata: the summed DATA is
   * byte-identical to the non-keepdims result (Kern 09). */
  sum(): NDArray<OkShape<ReduceAxis<S, undefined, false>>>;
  sum<const Axis extends number | undefined>(
    axis: Guard<ReduceAxis<S, Axis>, Axis>,
  ): NDArray<OkShape<ReduceAxis<S, Axis, false>>>;
  sum<const Axis extends number | undefined, const KeepDims extends boolean | undefined>(
    axis: Guard<ReduceAxis<S, Axis>, Axis>,
    keepdims: KeepDims,
  ): NDArray<OkShape<ReduceAxis<S, Axis, KeepDims>>>;
  sum<const Axis extends number | undefined = undefined, const KeepDims extends boolean = false>(
    axis?: Guard<ReduceAxis<S, Axis>, Axis>,
    keepdims?: KeepDims,
  ): NDArray<any> {
    const axisNum = axis as unknown as Axis | undefined;
    const { shape, data } = sumRuntime(this.shape, this.data, axisNum);
    const outShape = keepdims ? keepDimsShape(this.shape, axisNum) : shape;
    return new NDArray<OkShape<ReduceAxis<S, Axis, KeepDims>>>(
      outShape as OkShape<ReduceAxis<S, Axis, KeepDims>>,
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
   * type layer couldn't check statically. */
  dot<B extends Shape>(other: Guard<DotCheck<S, B, "dot">, NDArray<B>>): number {
    const o = other as unknown as NDArray<B>;
    assertVectorPair("dot", this.shape, o.shape);
    return dotRuntime(this.shape, this.data, o.shape, o.data);
  }

  /** L2/Frobenius norm over ALL elements (Kern 07), any rank (mirrors
   * `np.linalg.norm`'s default: flatten, then L2) — no guard, since a
   * niladic method has no argument to hang one on and every rank is valid
   * by this op's own semantics. `Math.sqrt` is IEEE-correctly-rounded, so
   * this is bit-identical to `WNDArray.norm` iff the underlying sum of
   * squares is (which the differential suite asserts). */
  norm(): number {
    return Math.sqrt(normSqRuntime(this.data));
  }

  /** Cosine similarity (Kern 07): same rank-1/equal-length operand contract
   * as `dot` (own error-message prefix). The pinned expression (spec,
   * identical on both surfaces): `num = dot(a,b)`, `den =
   * sqrt(normSq(a)) * sqrt(normSq(b))`, `return num / den`. Pure IEEE, no
   * epsilon guards: a zero vector on either side makes `den` (or both
   * `num` and `den`) `0`, yielding `NaN`; an adversarial magnitude split
   * can underflow `den` to `0` with `num != 0`, yielding `+/-Infinity`. */
  cosineSimilarity<B extends Shape>(other: Guard<DotCheck<S, B, "cosineSimilarity">, NDArray<B>>): number {
    const o = other as unknown as NDArray<B>;
    assertVectorPair("cosineSimilarity", this.shape, o.shape);
    const num = dotRuntime(this.shape, this.data, o.shape, o.data);
    const den = Math.sqrt(normSqRuntime(this.data)) * Math.sqrt(normSqRuntime(o.data));
    return num / den;
  }

  /** Reverse every axis (NumPy's `.T` generalized to N-D). */
  transpose(): NDArray<Transpose<S>> {
    const { shape, data } = transposeRuntime(this.shape, this.data);
    return new NDArray<Transpose<S>>(shape as Transpose<S>, data);
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
   * dynamic-rank callers the type layer couldn't check statically. */
  slice<const Specs extends readonly SliceSpecInput[]>(
    ...specs: SliceSpecsGuard<S, Specs>
  ): NDArray<OkShape<SliceShape<S, Specs>>> {
    const rawSpecs = specs as unknown as readonly SliceSpec[];
    const norm = normalizeSliceSpecs(this.shape, rawSpecs);
    const { shape, data } = sliceRuntime(this.shape, this.data, norm);
    return new NDArray<OkShape<SliceShape<S, Specs>>>(shape as OkShape<SliceShape<S, Specs>>, data);
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
   * Gradual/dynamic callers fall through to that same runtime backstop. */
  reshape<const NS extends Shape>(shape: Guard<ReshapeCheck<S, NS>, NS>): NDArray<Mutable<NS>> {
    const ns = shape as unknown as NS;
    assertReshapeArgs(this.shape, ns);
    return new NDArray<Mutable<NS>>([...ns] as Mutable<NS>, new Float64Array(this.data));
  }

  /** Rank-1 copy of every element (Kern 08): `a.flatten()` behaves exactly
   * like `a.reshape([product(a.shape)])` (always valid — no guard, same
   * niladic-method reasoning as `norm()` in Kern 07). Return type is the
   * Spike-04 payoff: a statically computed literal rank-1 shape (hover
   * `NDArray<[1048576]>` for `[1024, 1024]`) whenever every dim of `S` is a
   * supported literal, degrading to the honest `NDArray<[number]>`
   * whenever the product itself degrades. */
  flatten(): NDArray<[LiteralShapeProduct<S>]> {
    const size = product(this.shape);
    return new NDArray<[LiteralShapeProduct<S>]>([size] as unknown as [LiteralShapeProduct<S>], new Float64Array(this.data));
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

  /** Read back as a plain nested JS array (any rank), for printing/tests. */
  toNestedArray(): unknown {
    const strides = computeStrides(this.shape);
    const build = (axis: number, offset: number): unknown => {
      if (axis === this.shape.length) return this.data[offset] ?? 0;
      const dim: Dim = this.shape[axis] ?? 0;
      const stride = strides[axis] ?? 0;
      const out: unknown[] = [];
      for (let i = 0; i < dim; i++) out.push(build(axis + 1, offset + i * stride));
      return out;
    };
    return build(0, 0);
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
   * (FOLLOWUPS.md tracks the follow-up). */
  argmax(): number;
  argmax<const Axis extends number | undefined>(
    axis: Guard<ReduceAxis<S, Axis>, Axis>,
  ): NDArray<OkShape<ReduceAxis<S, Axis, false>>>;
  argmax<const Axis extends number | undefined, const KeepDims extends boolean | undefined>(
    axis: Guard<ReduceAxis<S, Axis>, Axis>,
    keepdims: KeepDims,
  ): NDArray<OkShape<ReduceAxis<S, Axis, KeepDims>>>;
  argmax<const Axis extends number | undefined = undefined, const KeepDims extends boolean = false>(
    axis?: Guard<ReduceAxis<S, Axis>, Axis>,
    keepdims?: KeepDims,
  ): NDArray<any> | number {
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
      const flat = argmaxRuntime(this.shape, this.data, undefined);
      return flat.data[0] ?? 0;
    }
    const axisNum = axis as unknown as Axis | undefined;
    const { shape, data } = argmaxRuntime(this.shape, this.data, axisNum);
    const outShape = keepdims ? keepDimsShape(this.shape, axisNum) : shape;
    return new NDArray<OkShape<ReduceAxis<S, Axis, KeepDims>>>(
      outShape as OkShape<ReduceAxis<S, Axis, KeepDims>>,
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
   * `indices`. */
  topk<const K extends number>(
    k: Guard<TopkCheck<S, K>, K>,
  ): { values: NDArray<OkShape<TopkShape<S, K>>>; indices: NDArray<OkShape<TopkShape<S, K>>> } {
    const kNum = k as unknown as K;
    const { values, indices } = topkRuntime(this.shape, this.data, kNum);
    return {
      values: new NDArray<OkShape<TopkShape<S, K>>>([kNum] as unknown as OkShape<TopkShape<S, K>>, values),
      indices: new NDArray<OkShape<TopkShape<S, K>>>([kNum] as unknown as OkShape<TopkShape<S, K>>, indices),
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
   * not an oversight. */
  mean(): NDArray<OkShape<ReduceAxis<S, undefined, false>>>;
  mean<const Axis extends number | undefined>(
    axis: Guard<ReduceAxis<S, Axis>, Axis>,
  ): NDArray<OkShape<ReduceAxis<S, Axis, false>>>;
  mean<const Axis extends number | undefined, const KeepDims extends boolean | undefined>(
    axis: Guard<ReduceAxis<S, Axis>, Axis>,
    keepdims: KeepDims,
  ): NDArray<OkShape<ReduceAxis<S, Axis, KeepDims>>>;
  mean<const Axis extends number | undefined = undefined, const KeepDims extends boolean = false>(
    axis?: Guard<ReduceAxis<S, Axis>, Axis>,
    keepdims?: KeepDims,
  ): NDArray<any> {
    const axisNum = axis as unknown as Axis | undefined;
    const { shape, data } = meanRuntime(this.shape, this.data, axisNum);
    const outShape = keepdims ? keepDimsShape(this.shape, axisNum) : shape;
    return new NDArray<OkShape<ReduceAxis<S, Axis, KeepDims>>>(
      outShape as OkShape<ReduceAxis<S, Axis, KeepDims>>,
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
   * follow-up). */
  sqrt(): NDArray<S> {
    const data = sqrtRuntime(this.data);
    return new NDArray<S>(this.shape as unknown as S, data);
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
   * `computeStrides` for the flat offset — no new arithmetic invented. */
  item<const Idx extends readonly number[]>(...indices: ItemGuard<S, Idx>): number {
    return itemRuntime(this.shape, this.data, indices as unknown as readonly number[]);
  }
}
