/**
 * Spike 04 (docs/spike-04-shape-products-spec.md): type-level tests for
 * `LiteralShapeProduct<S>` — the literal product of a fixed-rank shape's
 * dims (reshape/flatten's Phase-B enabler). Pure type function, no runtime
 * `NDArray` involved (there is no runtime `reshape`/`flatten` yet — out of
 * scope per the spec); `tsc --noEmit` is the test runner, same house idioms
 * as every other `*.test-d.ts` file (`Expect<Equal<…>>` for positives,
 * `@ts-expect-error` for negatives — not needed here since every input is
 * a valid call, just some degrade to `number`).
 */
import type { LiteralShapeProduct } from "../src/slice-literal.ts";
import type { Equal, Expect } from "./test-utils.ts";

// --- identities -------------------------------------------------------------

// Empty shape: the empty product, matching NumPy's scalar size.
type P1 = Expect<Equal<LiteralShapeProduct<readonly []>, 1>>;
type P2 = Expect<Equal<LiteralShapeProduct<readonly [1, 1, 1]>, 1>>;
type P3 = Expect<Equal<LiteralShapeProduct<readonly [3]>, 3>>;

// --- zeros: digit strings are exact at any length; only the FINAL value ----
// is judged against the cap, so a `0` dim zeroes out even a shape that also
// carries huge dims elsewhere.

type P4 = Expect<Equal<LiteralShapeProduct<readonly [0, 5]>, 0>>;
// P5 (2^52 * 2^52 * 0, transiently way past the safe-integer cap) moved to
// spike/tests-stress/product-stress.test-d.ts (Infra 01 — digit-arithmetic
// stress split, docs/infra-01-stress-split.md).

// --- carry-heavy multi-digit multiplication ---------------------------------

type P6 = Expect<Equal<LiteralShapeProduct<readonly [999, 999]>, 998001>>;
type P7 = Expect<Equal<LiteralShapeProduct<readonly [1024, 768]>, 786432>>;
type P8 = Expect<Equal<LiteralShapeProduct<readonly [65536, 65536]>, 4294967296>>;
// P9 (7-digit operands) moved to spike/tests-stress/product-stress.test-d.ts
// (Infra 01 — digit-arithmetic stress split, docs/infra-01-stress-split.md).

// --- cap boundary, BOTH sides (Number.MAX_SAFE_INTEGER = 9007199254740991) --

// P10 (exactly AT the cap) and P11 (one past the cap) moved to
// spike/tests-stress/product-stress.test-d.ts (Infra 01 — digit-arithmetic
// stress split, docs/infra-01-stress-split.md).

// --- degrade rows: every one of the semantics table's "no claim" cases -----

// Dynamic RANK: checked FIRST, before any tuple recursion.
type P12 = Expect<Equal<LiteralShapeProduct<number[]>, number>>;
type P13 = Expect<Equal<LiteralShapeProduct<readonly [2, ...number[]]>, number>>;

// Dynamic DIM (a `number` dim, known rank): the whole product degrades, not
// just that axis — there's no partial literal to report.
type P14 = Expect<Equal<LiteralShapeProduct<readonly [2, number]>, number>>;

// Union dim: excluded at the boundary (a union has no finite verdict
// alphabet for an unbounded VALUE result — see the binding spec's union
// rule), never reaches the arithmetic.
type P15 = Expect<Equal<LiteralShapeProduct<readonly [2 | 3]>, number>>;

// `never` dim: needs its own guard in `ProductAcc` — not because of
// `IsUnion` (which answers `false` for `never` via its explicit first
// branch), but because `never` satisfies EVERY `extends` check and would
// otherwise slip through `NonNegDigits` and propagate an all-`never` result
// instead of this honest `number` degrade (ablation-confirmed).
type P16 = Expect<Equal<LiteralShapeProduct<readonly [never]>, number>>;

// Negative literal dim: outside the supported plain-digit subset.
type P17 = Expect<Equal<LiteralShapeProduct<readonly [-1, 5]>, number>>;

// Non-integer literal.
type P18 = Expect<Equal<LiteralShapeProduct<readonly [1.5]>, number>>;

// Exponent-form literal: an integer, but not provable from its template
// form (`${1e21}` stringifies to `"1e+21"`, not a plain digit string).
type P19 = Expect<Equal<LiteralShapeProduct<readonly [1e21]>, number>>;

// --- union of FIXED shapes: distributes, each member independently --------

type P20 = Expect<Equal<LiteralShapeProduct<readonly [2, 3] | readonly [4]>, 6 | 4>>;
