/**
 * Infra 01 (docs/infra-01-stress-split.md): the digit-arithmetic STRESS
 * cases relocated out of `spike/tests/reshape.test-d.ts` — the over-cap
 * `LiteralShapeProduct` guard case (RG17) and the method-level
 * MAX_SAFE_INTEGER cap cases (T7/T8), which reuse product.test-d.ts's
 * P10/P11 boundary at the `NDArray.flatten()` method level. Checked via
 * this directory's standalone `tsconfig.json` (own `pnpm check:diag:stress`
 * pin), still part of the compound `pnpm check`. Pure relocation: the
 * expectations below are character-identical to the originals; see
 * `spike/tests/reshape.test-d.ts` for everything that stayed (T6's
 * [1024,1024] headline case, RG18's exponent-form case, and the rest).
 */
import { type Guard, NDArray } from "../src/ndarray.ts";
import type { ReshapeCheck } from "../src/reshape.ts";
import type { Equal, Expect } from "../tests/test-utils.ts";

// --- over-cap product on either side -> no claim (LiteralShapeProduct's
// own MAX_SAFE_INTEGER boundary, degrades to `number` beyond the cap). -----

type RG17 = Expect<Equal<Guard<ReshapeCheck<readonly [2, 4503599627370496], [3, 2]>, [3, 2]>, [3, 2]>>;

// --- ok: flatten at the MAX_SAFE_INTEGER cap boundary, reused at the METHOD
// level from product.test-d.ts's P10/P11 (bare LiteralShapeProduct checks)
// — `declare const` avoids ever constructing an unrepresentable array while
// still exercising the real method's return-type wiring end to end. --------

// D-V2.3 (docs/phase-d-vorarbeiten-spec.md): `.shape` is now `Readonly<S>` —
// pins re-expressed intent-preservingly as `readonly [...]`.
declare const atCapArr: NDArray<readonly [6361, 69431, 20394401]>;
const atCapFlat = atCapArr.flatten();
type T7 = Expect<Equal<(typeof atCapFlat)["shape"], readonly [9007199254740991]>>; // exactly AT the cap: still exact

declare const overCapArr: NDArray<readonly [2, 4503599627370496]>;
const overCapFlat = overCapArr.flatten();
type T8 = Expect<Equal<(typeof overCapFlat)["shape"], readonly [number]>>; // one past the cap: honest `number` degrade

void atCapFlat;
void overCapFlat;
