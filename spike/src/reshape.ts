/**
 * Kern 08 (docs/kern-08-reshape-flatten-spec.md): `ReshapeCheck<S, NS>` — the
 * compile-time guard for `NDArray.reshape`/`WNDArray.reshape`, consuming the
 * Spike-04 `LiteralShapeProduct<S>` enabler. New standalone file (keeps
 * `slice-literal.ts` untouched for the CORE check; see the stretch section
 * below for the one append it does need).
 *
 * Decision order (binding spec, never-wrong-only-incomplete):
 *  1. `P_old = LiteralShapeProduct<S>`, `P_new = LiteralShapeProduct<NS>`.
 *  2. Either product WIDE (`number extends P`) -> no claim (pass). This
 *     already subsumes dynamic rank, dynamic dims, negative/non-integer/
 *     exponent-form dims, and over-cap products — `LiteralShapeProduct`
 *     degrades all of those to `number` itself, so this file needs no
 *     separate handling for any of them.
 *  3. Either product a UNION (`IsUnion<P>`) -> no claim (a product verdict
 *     is an unbounded value — the Spike-04 rule: no subset-check is
 *     possible for an unbounded alphabet, so a union boundary-filters
 *     instead, exactly like `ProductAcc` itself does for a union DIM; here
 *     the union can arise one level up, from a union of whole OPERAND
 *     shapes distributing through `LiteralShapeProduct`).
 *  4. Both plain literals: equal -> pass; unequal -> `ShapeError` with the
 *     verbatim stem (mirrors `assertReshapeArgs`'s message 2 in runtime.ts).
 *
 * Stretch (Spike-03/06 idiom, droppable independently — see the spec's
 * "Stretch" section): `LiteralReshapeDimInvalid<NS>` (appended to
 * `slice-literal.ts`, its historical discipline — that file already owns
 * the private classification primitives, `IsPlainDigits`/dot-form
 * detection, this needs) lifts a PROVABLY invalid literal dim of the new
 * shape to a compile error, checked BEFORE the product check above (mirrors
 * `assertReshapeArgs`'s own check order: dim validity first, then product).
 */
import { type IsDynamicDim, type Shape, type ShapeError, type ShowShape } from "./dim.ts";
import { type IsUnion, type LiteralReshapeDimInvalid, type LiteralShapeProduct } from "./slice-literal.ts";

/** Non-error sentinel, same idiom as `vector.ts`'s own local `Pass` (each
 * guard file in this codebase defines its own — no shared export exists,
 * and `Guard<Result, Actual>` only cares that `Result` is not a
 * `ShapeError`, so any non-`ShapeError` type works). */
type Pass = true;

/**
 * The core check (spec steps 1-4 above): product equality between `S` and
 * `NS`, once both sides are known to be single plain literals (wide/union
 * already filtered by `ReshapeCheckCore` below).
 */
type ReshapeProductCheck<POld extends number, PNew extends number, NS extends Shape> = POld extends PNew
  ? PNew extends POld
    ? Pass
    : ShapeError<`reshape: cannot reshape array of size ${POld} into shape ${ShowShape<NS>}`>
  : ShapeError<`reshape: cannot reshape array of size ${POld} into shape ${ShowShape<NS>}`>;

/** Steps 1-3: compute both products, boundary-filter wide then union (same
 * order `LiteralShapeProduct`'s own `ProductAcc` establishes for a single
 * dim, applied here at the whole-product level), then hand off to the
 * literal-vs-literal check above. `extends infer ... extends number` binds
 * each product exactly once (the `ProductAcc`/`ResolveBoundaryDigits` idiom
 * already used throughout `slice-literal.ts`), avoiding recomputation. */
type ReshapeCheckCore<S extends Shape, NS extends Shape> = LiteralShapeProduct<S> extends infer POld extends number
  ? LiteralShapeProduct<NS> extends infer PNew extends number
    ? IsDynamicDim<POld> extends true
      ? Pass
      : IsDynamicDim<PNew> extends true
        ? Pass
        : IsUnion<POld> extends true
          ? Pass
          : IsUnion<PNew> extends true
            ? Pass
            : ReshapeProductCheck<POld, PNew, NS>
    : never // defensive: LiteralShapeProduct<NS> always yields a number
  : never; // defensive: LiteralShapeProduct<S> always yields a number

/**
 * The full guard, WITH the stretch: `LiteralReshapeDimInvalid<NS>` walks
 * `NS`'s dims left to right (mirrors the runtime's own per-axis loop) and
 * reports the FIRST provably-invalid one's OWN literal value, or the
 * sentinel `"ok"` if none found — deliberately not `never` (the standard
 * "never always matches every extends check" gotcha this codebase already
 * guards against elsewhere, e.g. `slice-literal.ts`'s `NonNegDigits`). The
 * message itself (mirroring `assertReshapeArgs`'s message-1 stem verbatim)
 * is built HERE, not in `slice-literal.ts` (which only returns bare
 * verdicts, by that file's own historical precedent and its append-only
 * freeze discipline — see `LiteralReshapeDimInvalid`'s own doc comment).
 * Falls through to the core product check only once no axis is provably
 * invalid.
 *
 * This is the file's single exported guard. Dropping the stretch (if a
 * budget gate strains) is a one-line revert: change this alias's body from
 * `ReshapeCheckWithStretch<S, NS>` to `ReshapeCheckCore<S, NS>` and delete
 * the now-unused `ReshapeCheckWithStretch` type and its
 * `LiteralReshapeDimInvalid` import — no other file needs to change (both
 * `ndarray.ts` and `resident.ts` only ever reference this one exported name).
 */
type ReshapeCheckWithStretch<S extends Shape, NS extends Shape> = LiteralReshapeDimInvalid<NS> extends infer DimVerdict
  ? DimVerdict extends "ok"
    ? ReshapeCheckCore<S, NS>
    : DimVerdict extends number
      ? ShapeError<`reshape: invalid dimension ${DimVerdict} in shape ${ShowShape<NS>} (dims must be non-negative integers)`>
      : never // defensive: LiteralReshapeDimInvalid always yields "ok" or a number
  : never;

/**
 * The `reshape()` method's argument guard. Used as
 * `Guard<ReshapeCheck<S, NS>, NS>` (resp. `WNDArray`) — a receiver-side
 * (`S`) violation still surfaces AT THE ARGUMENT, same as every other op in
 * this codebase, because that is where `Guard` itself places the compile
 * error.
 *
 * Known, documented, deliberately-not-fixed boundary (FOLLOWUPS, shared
 * with `DotCheck`/`MatMul`): a union of whole OPERAND types can bypass
 * argument-side guards in general; per-guard, step 3 above means a literal
 * union-of-shapes NEW-shape argument gets no claim — pinned by type tests
 * as no-claim, not silently assumed covered.
 */
export type ReshapeCheck<S extends Shape, NS extends Shape> = ReshapeCheckWithStretch<S, NS>;
