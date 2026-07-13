/**
 * Stress & metrics cases from the acceptance table:
 *  - Rank-16 broadcast type-checks.
 *  - A chain of >=100 composed ops (alternating matmul/add/transpose on the
 *    *value* level, letting inference thread the shape through) type-checks
 *    without depth errors.
 *
 * `pnpm check:diag` numbers for this file (and the whole suite) are
 * recorded in docs/spike-01-ergebnisse.md, not here.
 */
import type { Broadcast } from "../src/broadcast.ts";
import { NDArray } from "../src/ndarray.ts";
import type { Equal, Expect } from "./test-utils.ts";

// --- Rank-16 broadcast ----------------------------------------------------

type Rank16A = [1, 2, 1, 4, 1, 6, 1, 8, 1, 10, 1, 12, 1, 14, 1, 16];
type Rank16B = [1, 1, 3, 1, 5, 1, 7, 1, 9, 1, 11, 1, 13, 1, 15, 1];
type Rank16Result = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
type RankCheck = Expect<Equal<Broadcast<Rank16A, Rank16B>, Rank16Result>>;

// --- >=100 composed ops, value-level chain, letting inference thread ------
// Generated (not hand-typed) via a throwaway script; alternates
// matmul/add/transpose on square [4,4] matrices so every intermediate shape
// stays [4,4] and the chain is easy to hand-verify. Exactly 100 op calls.

const chained100 = NDArray.zeros([4, 4])
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]))
  .add(NDArray.zeros([4, 4]))
  .transpose()
  .matmul(NDArray.zeros([4, 4]));

// After 100 alternating ops on square [4,4] matrices, the shape must still
// resolve cleanly to [4, 4] — not "conditional-type soup".
// D-V2.3 (docs/phase-d-vorarbeiten-spec.md): `.shape` is now `Readonly<S>` —
// pin re-expressed intent-preservingly as `readonly [...]`.
type Chained100Shape = (typeof chained100)["shape"];
type ChainCheck = Expect<Equal<Chained100Shape, readonly [4, 4]>>;
