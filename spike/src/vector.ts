/**
 * Kern 07: `DotCheck<S, B, Op>` — the compile-time guard shared by `dot` and
 * `cosineSimilarity` (both rank-1 x rank-1, equal length; see
 * docs/kern-07-elementwise-vector-spec.md's error-message table). `Op` is
 * the message stem (`"dot"` or `"cosineSimilarity"`) so both call sites
 * reuse one guard instead of two near-duplicates.
 *
 * Structure mirrors two existing house patterns rather than inventing a
 * third:
 *  - the `RankUnknowable` guard on EITHER operand (dynamic rank OR a
 *    mixed-rank shape union, D-V1.3, docs/phase-d-vorarbeiten-spec.md),
 *    checked first and unconditionally degrading to "no claim" (pass) —
 *    identical to `Broadcast`/`MatMul`'s own `RankUnknowable` gate
 *    (broadcast.ts, matmul.ts);
 *  - the rank-1 destructuring check `S extends readonly [infer D extends Dim]`
 *    — MatMul's own "rank(A) === 1" idiom (matmul.ts) — used here as a NAKED
 *    check on the bare type parameter so a union OF SHAPES (e.g.
 *    `[2,3] | [4]`) distributes and is processed member-by-member, the same
 *    natural distribution `MatMulStatic`/`ProductAcc` already rely on
 *    elsewhere in this codebase (never special-cased).
 *
 * Deliberately does NOT reuse `DimEq` (dim.ts) for the length comparison:
 * `DimEq`'s `A extends B` check distributes over a union DIM VALUE (e.g. a
 * single axis typed `2 | 3`) with no filter — the pre-existing `MatMul`
 * latent hazard this design pass discovered and deliberately leaves
 * unfixed (FOLLOWUPS). `VectorLenCheck` below filters union dims FIRST via
 * the newly-exported `IsUnion` (literal-arithmetic.ts, Spike-04/06 house rule) —
 * a union dim on either side is *never* misread as a verdict, it degrades
 * to "no claim" instead, same as a dynamic (`number`) dim.
 */
import { type Dim, type IsDynamicDim, type RankUnknowable, type Shape, type ShapeError, type ShowShape } from "./dim.ts";
import { type IsUnion } from "./literal-arithmetic.ts";

/** Non-error sentinel `DotCheck`/`VectorLenCheck` resolve to on every "pass"
 * branch (dynamic rank, dynamic dim, union dim, or equal literal lengths).
 * `Guard<Result, Actual>` (ndarray.ts) only branches on `Result extends
 * ShapeError<string>`, so any non-`ShapeError` type works here — `true` is
 * chosen for readability at each branch. */
type Pass = true;

/**
 * Per-axis length check for the (already rank-1-confirmed) single dim of
 * each operand. Checked in this order: union dim (either side) -> no claim;
 * dynamic dim (either side) -> no claim; equal literals -> pass; unequal
 * literals -> the length-mismatch message (spec table, row 3).
 */
type VectorLenCheck<SD extends Dim, BD extends Dim, Op extends string> = IsUnion<SD> extends true
  ? Pass
  : IsUnion<BD> extends true
    ? Pass
    : IsDynamicDim<SD> extends true
      ? Pass
      : IsDynamicDim<BD> extends true
        ? Pass
        : SD extends BD
          ? BD extends SD
            ? Pass
            : ShapeError<`${Op}: vector lengths ${SD} and ${BD} do not match`>
          : ShapeError<`${Op}: vector lengths ${SD} and ${BD} do not match`>;

/**
 * Static (both operands already known non-dynamic-rank) half of `DotCheck`.
 * Message-table order: first-operand rank, then second-operand rank, then
 * length. `S`/`B` are checked NAKED (not via `S["length"]`) so a union of
 * whole SHAPES distributes across this conditional's branches, matching
 * `MatMulStatic`'s own precedent — only a union DIM WITHIN one shape needs
 * the explicit `IsUnion` filter (`VectorLenCheck` above), never a union of
 * whole shapes.
 */
type DotCheckStatic<S extends Shape, B extends Shape, Op extends string> = S extends readonly [infer SD extends Dim]
  ? B extends readonly [infer BD extends Dim]
    ? VectorLenCheck<SD, BD, Op>
    : ShapeError<`${Op}: expected a 1-D vector as the second operand (got shape ${ShowShape<B>})`>
  : ShapeError<`${Op}: expected a 1-D vector as the first operand (got shape ${ShowShape<S>})`>;

/**
 * The `dot`/`cosineSimilarity` operand guard: both operands must be rank-1
 * with equal length. Dynamic RANK on either side, OR a MIXED-rank shape
 * union (D-V1.3, `RankUnknowable`, docs/phase-d-vorarbeiten-spec.md),
 * degrades unconditionally to "no claim" (pass; the runtime backstop is
 * `assertVectorPair` in runtime.ts). Used as `Guard<DotCheck<S, B, "dot">, NDArray<B>>` (resp.
 * `WNDArray<B>`, `"cosineSimilarity"`) — a receiver-side (`S`) violation
 * still surfaces AT THE ARGUMENT, same as every other op here, because
 * that is where `Guard` itself puts the compile error.
 */
export type DotCheck<S extends Shape, B extends Shape, Op extends string> = RankUnknowable<S> extends true
  ? Pass
  : RankUnknowable<B> extends true
    ? Pass
    : DotCheckStatic<S, B, Op>;

// ---------------------------------------------------------------------------
// Op-Scheibe W1 (docs/op-w1-argmax-topk-spec.md, D3): `TopkCheck<S, K>` +
// `TopkShape<S, K>` — the compile-time guard + resulting shape for
// `NDArray.topk`. Appended strictly after all pre-existing content in this
// file (freeze discipline, D5 — this is why the extra import below is its
// OWN statement rather than widening the `literal-arithmetic.ts` import
// line above: T5 requires this file's diff to show ONLY additions after the
// last pre-existing line). Rank-1-only (`DotCheck`'s own precedent above: a
// receiver-side rank problem surfaces AT THE `k` ARGUMENT, since `topk` has
// no other place to hang a `Guard` on). Reuses the EXISTING digit-string
// machinery from literal-arithmetic.ts (`Compare`, `NonNegDigits`,
// `LiteralReshapeDimInvalid`, `IsUnion`, already imported above) — no new
// arithmetic primitives, per D3's binding decision. `LiteralIndexBounds` is
// DELIBERATELY not used here: it has the wrong semantics for a k-bounds
// check in BOTH directions (`k == D` classifies as `"out"` under
// `LiteralIndexBounds`'s `i < d` index semantics; a negative `k` classifies
// as `"in"` under its NumPy negative-index normalization) — empirically
// proven during Baustein 0, see the spec's adversarial-verification
// addendum.
// ---------------------------------------------------------------------------
import type { Compare, LiteralReshapeDimInvalid, NonNegDigits } from "./literal-arithmetic.ts";

/** `topk`'s "wrong receiver rank" message — `DotCheck` precedent: the error
 * surfaces at the `k` argument (via `Guard`) even though the actual problem
 * is with the RECEIVER `S`, because a niladic-style rank check has nowhere
 * else to attach. Mirrors the runtime throw (`topkRuntime`, runtime.ts)
 * verbatim — pinned by string-equality unit tests, not just the
 * differential suite. */
type TopkRankMessage<S extends Shape> = `topk: expected a 1-D vector (got shape ${ShowShape<S>})`;

/** `topk`'s "k is not a valid non-negative integer" message — covers BOTH a
 * negative literal `k` and a dot-form (non-integer) literal `k` in ONE
 * check (`LiteralReshapeDimInvalid`, reused unmodified against the
 * singleton shape `[K]` — the Kern-08 reshape-dim classifier already proves
 * exactly this "provably invalid dim" verdict, and a `topk` count is
 * structurally the same kind of value as a reshape dim: a non-negative
 * integer, or wide/no-claim). Mirrors the runtime throw verbatim. */
type TopkInvalidKMessage<K extends number> = `topk: k must be a non-negative integer (got ${K})`;

/** `topk`'s "k exceeds the vector length" message. Mirrors the runtime
 * throw verbatim. `LiteralIndexBounds` is NOT used here (see file-section
 * header above) — this reuses `Compare` (+ `NonNegDigits` to obtain a
 * comparable digit string for `K`) directly instead, the Owner-decided
 * minimal-export path (D3 v2). */
type TopkBoundsMessage<K extends number, D extends Dim> = `topk: k=${K} exceeds the vector length ${D}`;

/**
 * `k`'s own validity, once the receiver is confirmed rank-1 with dim `D`:
 * dot-form/negative literal `k` -> error (structurally safe, doesn't depend
 * on `D` at all — checked FIRST, unconditionally); otherwise a bounds check
 * against `D`, itself gated by `IsUnion<D>`/`IsDynamicDim<D>` (same
 * defensive filter `VectorLenCheck` above already uses for a single dim
 * that might carry a union type) and by `NonNegDigits<K>` (wide/
 * exponent-form `k` -> no bounds claim, the same digit-machinery boundary
 * `literal-arithmetic.ts` already draws — D3's "MAX_SAFE_INTEGER-Kappe ...
 * jenseits -> no-claim"). `k = 0` and `k = D` are both VALID (`Compare`
 * yields `"lt"`/`"eq"`; only `"gt"` errors).
 */
type TopkCheckStatic<D extends Dim, K extends number> = LiteralReshapeDimInvalid<[K]> extends "ok"
  ? IsUnion<D> extends true
    ? Pass
    : IsDynamicDim<D> extends true
      ? Pass
      : NonNegDigits<K> extends infer KDigits extends string
        ? KDigits extends "unsupported"
          ? Pass
          : Compare<KDigits, `${D}`> extends "gt"
            ? ShapeError<TopkBoundsMessage<K, D>>
            : Pass
        : never
  : ShapeError<TopkInvalidKMessage<K>>;

/**
 * The `topk` operand guard: `S` must be rank-1 (mirrors `DotCheck`'s own
 * receiver-rank check above); `k` must classify as a valid non-negative
 * integer no larger than the vector's length. `IsUnion<K>` is filtered
 * FIRST, UNCONDITIONALLY — before even the rank-1 check — mirroring
 * reduce.ts's own union-AXIS filter position exactly (D3: "the filter runs
 * before any ShapeError is even produced"; a union `k` is a no-claim
 * degrade regardless of whether the receiver's rank happens to also be
 * wrong). `RankUnknowable<S>` degrades the SAME way `DotCheck` does above
 * (no claim; the runtime backstop, `topkRuntime`, stays authoritative).
 */
export type TopkCheck<S extends Shape, K extends number> = IsUnion<K> extends true
  ? Pass
  : RankUnknowable<S> extends true
    ? Pass
    : S extends readonly [infer D extends Dim]
      ? TopkCheckStatic<D, K>
      : ShapeError<TopkRankMessage<S>>;

/**
 * `topk`'s resulting shape (both `values` and `indices` share it). Always
 * rank-1 — unlike `ReduceAxis`, `topk`'s output rank never depends on the
 * receiver `S` at all (a successful call is ALWAYS `[k]`, regardless of
 * whether `S`'s rank was statically confirmed or only runtime-checked), so
 * `S` itself is UNUSED in the body below — kept as a parameter purely for
 * signature symmetry with every other shape-computing type in this codebase
 * (`SliceShape`, `ReduceAxis`, ...) and because it is what the `topk` call
 * site naturally has in scope. This mirrors `SliceShape`/`SliceSpecsGuard`'s
 * own precedent (slice.ts): the shape-computing type does not need to
 * re-validate what the paired `*Check` type already gates via `Guard` — an
 * invalid call never type-checks in the first place, so this type only
 * needs to be honest for calls the guard already accepted. Degrades to
 * `readonly [number]` (rank-1, dim unknown) for a union or wide/
 * unsupported-digit-form `K` — never `readonly Dim[]` (unbounded rank),
 * since the RANK is always known even when the exact length isn't.
 */
export type TopkShape<S extends Shape, K extends number> = IsUnion<K> extends true
  ? readonly [number]
  : NonNegDigits<K> extends "unsupported"
    ? readonly [number]
    : [K];
