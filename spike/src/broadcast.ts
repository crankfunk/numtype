/**
 * NumPy broadcasting, at the type level.
 *
 * Rule: align shapes from the right; two dims are compatible iff equal or
 * one of them is 1; the result dim is the non-1 one (or either, if equal).
 * Missing leading dims on the shorter shape behave as if they were 1 (i.e.
 * are simply absent from the compatibility check and pass through from the
 * longer shape unchanged).
 *
 * Implementation note: this consumes both tuples from the *tail* using the
 * native `[...Init, Last]` variadic pattern (no separate Reverse+re-reverse
 * needed — walking from the tail and prepending to the accumulator already
 * yields the result in the original left-to-right order). Each branch's
 * final expression is a direct, unwrapped recursive call — the accumulator
 * form required for TS's tail-call elimination (PR #45711).
 */

import { type CompatDim, type Dim, type RankUnknowable, type Shape, type ShapeError, type ShowShape } from "./dim.ts";

type BroadcastAcc<A extends readonly Dim[], B extends readonly Dim[], Acc extends readonly Dim[]> = A extends readonly [
  ...infer AInit extends readonly Dim[],
  infer ALast extends Dim,
]
  ? B extends readonly [...infer BInit extends readonly Dim[], infer BLast extends Dim]
    ? CompatDim<ALast, BLast> extends infer D
      ? D extends ShapeError<string>
        ? D
        : D extends Dim
          ? BroadcastAcc<AInit, BInit, [D, ...Acc]>
          : never
      : never
    : [...A, ...Acc]
  : B extends readonly [...infer BInit extends readonly Dim[], infer BLast extends Dim]
    ? [...B, ...Acc]
    : Acc;

/**
 * Broadcast two shapes per NumPy rules. Resolves to the broadcast shape, or
 * a `ShapeError<...>` naming the incompatible shapes. A dynamic-RANK operand
 * (`number[]`, `[2, ...number[]]`) OR a MIXED-rank shape union (D-V1.3,
 * `RankUnknowable`) makes the result rank unknowable — degrade to `Dim[]`
 * (gradual), never guess and never error.
 */
export type Broadcast<A extends Shape, B extends Shape> = RankUnknowable<A> extends true
  ? readonly Dim[]
  : RankUnknowable<B> extends true
    ? readonly Dim[]
    : BroadcastStatic<A, B>;

type BroadcastStatic<A extends Shape, B extends Shape> = BroadcastAcc<A, B, []> extends infer R
  ? R extends ShapeError<infer M>
    ? ShapeError<`cannot broadcast shapes ${ShowShape<A>} and ${ShowShape<B>}: ${M}`>
    : R
  : never;
