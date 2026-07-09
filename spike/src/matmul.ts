/**
 * Full NumPy `matmul` semantics, at the type level:
 *  - 2-D x 2-D: matrix product, inner dims must match.
 *  - 1-D promotion: a 1-D first operand is treated as [1, k] and the
 *    prepended 1 is squeezed from the result; a 1-D second operand is
 *    treated as [k, 1] and the appended 1 is squeezed from the result.
 *  - 1-D x 1-D: inner (dot) product, result is rank 0 (`[]`).
 *  - Batch dims (everything but the trailing 1 or 2 core dims) broadcast
 *    per NumPy rules.
 *  - Scalars (rank 0) are a hard error on either side.
 *
 * "Broadcasting an empty batch shape `[]` against anything always yields
 * that anything, unconditionally" (see broadcast.ts) — so the 1-D
 * promotion cases below can splice the *other* side's batch dims straight
 * into the result without a redundant `Broadcast<[], X>` call.
 */

import { type Dim, type DimEq, type IsDynamicRank, type Shape, type ShapeError, type ShowShape } from "./dim.ts";
import type { Broadcast } from "./broadcast.ts";

type InnerMismatch<K1 extends Dim, K2 extends Dim> = ShapeError<`matmul: inner dimensions ${K1} and ${K2} do not match`>;

/** A dynamic-RANK operand makes core-dim extraction and batch broadcasting
 * unknowable — degrade to `Dim[]` (gradual, runtime-checked). Without this
 * guard, `number[]` falls through every tuple destructure to `never`,
 * making the op uncallable. */
export type MatMul<A extends Shape, B extends Shape> = IsDynamicRank<A> extends true
  ? readonly Dim[]
  : IsDynamicRank<B> extends true
    ? readonly Dim[]
    : MatMulStatic<A, B>;

type MatMulStatic<A extends Shape, B extends Shape> = A extends readonly []
  ? ShapeError<`matmul: scalar operand (rank 0) is not allowed as the first argument (got shape ${ShowShape<A>})`>
  : B extends readonly []
    ? ShapeError<`matmul: scalar operand (rank 0) is not allowed as the second argument (got shape ${ShowShape<B>})`>
    : A extends readonly [infer K1 extends Dim] // rank(A) === 1
      ? B extends readonly [infer K2 extends Dim] // rank(B) === 1: dot product -> scalar
        ? DimEq<K1, K2> extends true
          ? []
          : InnerMismatch<K1, K2>
        : B extends readonly [...infer BatchB extends Shape, infer K2 extends Dim, infer N extends Dim] // rank(B) >= 2
          ? DimEq<K1, K2> extends true
            ? [...BatchB, N] // A promoted to [1,k]; squeeze the prepended 1
            : InnerMismatch<K1, K2>
          : never
      : A extends readonly [...infer BatchA extends Shape, infer M extends Dim, infer K1 extends Dim] // rank(A) >= 2
        ? B extends readonly [infer K2 extends Dim] // rank(B) === 1
          ? DimEq<K1, K2> extends true
            ? [...BatchA, M] // B promoted to [k,1]; squeeze the appended 1
            : InnerMismatch<K1, K2>
          : B extends readonly [...infer BatchB extends Shape, infer K2 extends Dim, infer N extends Dim] // rank(B) >= 2
            ? DimEq<K1, K2> extends true
              ? Broadcast<BatchA, BatchB> extends infer BR
                ? BR extends ShapeError<infer BatchMsg>
                  ? ShapeError<`matmul: batch dimensions incompatible (${BatchMsg})`>
                  : BR extends Shape
                    ? [...BR, M, N]
                    : never
                : never
              : InnerMismatch<K1, K2>
            : never
        : never;
