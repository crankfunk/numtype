/**
 * Kern 07: `DotCheck<S, B, Op>` — the compile-time guard shared by `dot` and
 * `cosineSimilarity` (both rank-1 x rank-1, equal length; see
 * docs/kern-07-elementwise-vector-spec.md's error-message table). `Op` is
 * the message stem (`"dot"` or `"cosineSimilarity"`) so both call sites
 * reuse one guard instead of two near-duplicates.
 *
 * Structure mirrors two existing house patterns rather than inventing a
 * third:
 *  - the `IsDynamicRank` guard on EITHER operand, checked first and
 *    unconditionally degrading to "no claim" (pass) — identical to
 *    `Broadcast`/`MatMul`'s own dynamic-rank handling (broadcast.ts,
 *    matmul.ts);
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
 * the newly-exported `IsUnion` (slice-literal.ts, Spike-04/06 house rule) —
 * a union dim on either side is *never* misread as a verdict, it degrades
 * to "no claim" instead, same as a dynamic (`number`) dim.
 */
import { type Dim, type IsDynamicDim, type IsDynamicRank, type Shape, type ShapeError, type ShowShape } from "./dim.ts";
import { type IsUnion } from "./slice-literal.ts";

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
 * with equal length. Dynamic RANK on either side degrades unconditionally
 * to "no claim" (pass; the runtime backstop is `assertVectorPair` in
 * runtime.ts). Used as `Guard<DotCheck<S, B, "dot">, NDArray<B>>` (resp.
 * `WNDArray<B>`, `"cosineSimilarity"`) — a receiver-side (`S`) violation
 * still surfaces AT THE ARGUMENT, same as every other op here, because
 * that is where `Guard` itself puts the compile error.
 */
export type DotCheck<S extends Shape, B extends Shape, Op extends string> = IsDynamicRank<S> extends true
  ? Pass
  : IsDynamicRank<B> extends true
    ? Pass
    : DotCheckStatic<S, B, Op>;
