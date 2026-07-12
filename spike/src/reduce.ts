/**
 * Axis reduction + transpose, at the type level.
 *
 * `ReduceAxis<S, Axis, KeepDims>` removes (or, with KeepDims, replaces with
 * `1`) the given axis from S. Negative axes count from the end (`-1` = last
 * axis). No axis (`undefined`) means "reduce everything" -> `[]`.
 *
 * Design note: axis is a *rank-level* index (small int) — arithmetic on it
 * is explicitly allowed by the spec ("rank-level counting via tuple length
 * is fine"), unlike arithmetic on dim *values*. We still avoid a generic
 * two-argument `Subtract<Len, Pos>` (which risks silently producing `never`
 * for out-of-range values via distributive-conditional pitfalls) by instead
 * reusing `Reverse` for negative axes: `-k` becomes 0-based index `k-1` into
 * the *reversed* shape, remove/replace there, then reverse back. Only a
 * trivial `Decrement<N> = N-1` is needed, which is safe and simple.
 */

import { type Dim, type RankUnknowable, type Reverse, type Shape, type ShapeError, type ShowShape } from "./dim.ts";

// ---- rank-scale tuple counting helpers (never applied to dim values) ----

type BuildTuple<L extends number, Acc extends readonly unknown[] = []> = Acc["length"] extends L
  ? Acc
  : BuildTuple<L, readonly [...Acc, unknown]>;

/** N - 1 for a small non-negative integer N >= 1. */
type Decrement<N extends number> = BuildTuple<N> extends readonly [unknown, ...infer Rest] ? Rest["length"] : never;

// ---- index-based removal / replacement from the front ----

type RemoveAtFront<S extends readonly Dim[], Idx extends number, Passed extends readonly Dim[] = []> = Passed["length"] extends Idx
  ? S extends readonly [Dim, ...infer Rest extends readonly Dim[]]
    ? [...Passed, ...Rest]
    : ShapeError<"index out of range">
  : S extends readonly [infer Head extends Dim, ...infer Rest extends readonly Dim[]]
    ? RemoveAtFront<Rest, Idx, [...Passed, Head]>
    : ShapeError<"index out of range">;

type ReplaceAtFront<S extends readonly Dim[], Idx extends number, Passed extends readonly Dim[] = []> = Passed["length"] extends Idx
  ? S extends readonly [Dim, ...infer Rest extends readonly Dim[]]
    ? [...Passed, 1, ...Rest]
    : ShapeError<"index out of range">
  : S extends readonly [infer Head extends Dim, ...infer Rest extends readonly Dim[]]
    ? ReplaceAtFront<Rest, Idx, [...Passed, Head]>
    : ShapeError<"index out of range">;

type ApplyAt<S extends readonly Dim[], Idx extends number, KeepDims extends boolean> = KeepDims extends true
  ? ReplaceAtFront<S, Idx>
  : RemoveAtFront<S, Idx>;

/** Map every dim to `1`, preserving rank (homomorphic mapped tuple type) —
 * used for the `KeepDims` + "no axis" (reduce everything) combination. */
type AllOnes<S extends readonly Dim[]> = { [K in keyof S]: 1 };

type ResolveAndApply<S extends Shape, Axis extends number, KeepDims extends boolean> = `${Axis}` extends `-${infer Pos extends number}`
  ? Decrement<Pos> extends infer RIdx
    ? [RIdx] extends [never]
      ? ShapeError<"index out of range">
      : RIdx extends number
        ? Reverse<S> extends infer RS
          ? RS extends readonly Dim[]
            ? ApplyAt<RS, RIdx, KeepDims> extends infer RR
              ? RR extends ShapeError<string>
                ? RR
                : RR extends readonly Dim[]
                  ? Reverse<RR>
                  : never
              : never
            : never
          : never
        : never
    : never
  : ApplyAt<S, Axis, KeepDims>;

/**
 * Remove (or, with `KeepDims = true`, replace with `1`) the given axis.
 * `Axis = undefined` reduces every axis -> `[]`. Out-of-range axes (in
 * either direction) resolve to a `ShapeError`.
 */
export type ReduceAxis<S extends Shape, Axis extends number | undefined = undefined, KeepDims extends boolean = false> = [
  Axis,
] extends [undefined]
  ? KeepDims extends true
    ? AllOnes<S> // note: statically correct even for dynamic rank (`number[]` -> `1[]`)
    : [] // full reduction is `[]` for EVERY rank, known or not
  : Axis extends number
    ? RankUnknowable<S> extends true
      ? readonly Dim[] // dynamic rank OR mixed-rank union (D-V1.3): axis validity/position unknowable -> gradual, runtime-checked
      : number extends Axis
        ? readonly Dim[] // dynamic AXIS on a known shape: which dim goes is unknowable -> gradual (without this guard, `0 extends number` silently removes axis 0)
        : ResolveAndApply<S, Axis, KeepDims> extends infer R
          ? R extends ShapeError<string>
            ? ShapeError<`reduce: axis ${Axis} is out of range for shape ${ShowShape<S>} (rank ${S["length"]})`>
            : R
          : never
    : never;

/** Convenience alias for the keepdims variant (see ReduceAxis). */
export type ReduceAxisKeepDims<S extends Shape, Axis extends number | undefined = undefined> = ReduceAxis<S, Axis, true>;

/** Reverse all axes (NumPy's `.T` generalized to N-D). Dynamic rank cannot
 * be reversed axis-by-axis (Reverse would silently return `[]`) — degrade
 * to `Dim[]`. A MIXED-rank shape union (D-V1.3, `RankUnknowable`) degrades
 * the SAME way, by owner decision — even though `Reverse` would otherwise
 * distribute to a per-member-correct union (e.g. `[3,2] | [4,3,2]` for
 * `[2,3] | [2,3,4]`): one uniform rule at every rank-gate, disclosed
 * precision loss for this exotic case, docs/phase-d-vorarbeiten-spec.md. */
export type Transpose<S extends Shape> = RankUnknowable<S> extends true ? readonly Dim[] : Reverse<S>;
