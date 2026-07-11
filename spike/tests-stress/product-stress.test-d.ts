/**
 * Infra 01 (docs/infra-01-stress-split.md): the digit-arithmetic STRESS
 * cases relocated out of `spike/tests/product.test-d.ts` — mandated
 * cap-/16-digit probe cases that were ~78% of the last check:diag pin
 * delta but carry no product cost of their own. Checked via this
 * directory's standalone `tsconfig.json` (own `pnpm check:diag:stress`
 * pin), still part of the compound `pnpm check`. Pure relocation: the
 * expectations below are character-identical to the originals; see
 * `spike/tests/product.test-d.ts` for the cases that stayed (P1-P4,
 * P6-P8, P12-P20) and for the semantic context of `LiteralShapeProduct`.
 */
import type { LiteralShapeProduct } from "../src/slice-literal.ts";
import type { Equal, Expect } from "../tests/test-utils.ts";

// --- zeros: digit strings are exact at any length; only the FINAL value ----
// is judged against the cap, so a `0` dim zeroes out even a shape that also
// carries huge dims elsewhere.

// 2^52 squared (~2e31, transiently way past the safe-integer cap) times a
// trailing 0: the true product is exactly 0, not a degrade — the cap check
// happens ONLY at exhaustion, never on an intermediate partial product.
type P5 = Expect<Equal<LiteralShapeProduct<readonly [4503599627370496, 4503599627370496, 0]>, 0>>;

// --- carry-heavy multi-digit multiplication ---------------------------------

// 7-digit operands: exercises the schoolbook accumulation over more digit
// positions than any of the above.
type P9 = Expect<Equal<LiteralShapeProduct<readonly [1000003, 1000003]>, 1000006000009>>;

// --- cap boundary, BOTH sides (Number.MAX_SAFE_INTEGER = 9007199254740991) --

// Exactly AT the cap: still an exact, reportable literal (a plain
// `Number.isSafeInteger`-style boundary — `==` the cap is safe).
type P10 = Expect<Equal<LiteralShapeProduct<readonly [6361, 69431, 20394401]>, 9007199254740991>>;
// One past the cap (2 * 2^52 = 2^53 = 9007199254740992): degrades to
// `number` — round-tripping this exact digit string through
// `` `${infer N extends number}` `` would double-round through float64 and
// silently claim a WRONG literal.
type P11 = Expect<Equal<LiteralShapeProduct<readonly [2, 4503599627370496]>, number>>;
