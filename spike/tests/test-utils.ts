/**
 * Hand-written type-testing helpers (zero deps — no `tsd`/vitest).
 * `tsc --noEmit` alone is the test runner: positive cases assert via
 * `Expect<Equal<X, Y>>`, negative cases via `@ts-expect-error` (an unused
 * directive is itself a compile error, so negatives are checked both ways).
 *
 * For "is this some ShapeError, regardless of exact wording" checks, use
 * `IsShapeError<T>` from `spike/src/dim.ts` directly: `Expect<IsShapeError<X>>`.
 */

/** Precise type equality (not mere mutual assignability) — the standard
 * "distributive-safe" check (handles literal-vs-wide, `any`, etc. correctly,
 * unlike a naive `X extends Y ? Y extends X ? true : false`). */
export type Equal<X, Y> = (<T>() => T extends X ? 1 : 2) extends <T>() => T extends Y ? 1 : 2 ? true : false;

export type Expect<T extends true> = T;
