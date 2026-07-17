/**
 * D-S3.2 (2) — Type-level package smoke. A stand-in CONSUMER `.ts`, checked
 * against the BUILT `dist/index.d.ts` under consumer-typical defaults
 * (`skipLibCheck: true`, `moduleResolution: bundler` — see ./tsconfig.json).
 * Proves:
 *  - Blocker 1 fixed: importing from the package raises NO "Cannot find
 *    module './x.ts'" — the `.d.ts` chain resolves (typecheck EXIT 0).
 *  - M3 preserved through the build: a valid call resolves clean, and an
 *    out-of-range axis is a COMPILE ERROR at the argument (the `@ts-expect-error`
 *    below fails the build if `sum(9)` ever stops erroring — non-vacuous).
 *  - D-S2.3 (a): the standard consumer does NOT need `@types/node`
 *    (skipLibCheck:true is the Vite/Next/CRA default; verified EXIT 0).
 *
 * Runs under `pnpm test:package` (`tsc -p spike/tests-package/consumer/
 * tsconfig.json`), after `build:dist`. NOT part of the root `pnpm check`
 * corpus — it needs a built `dist/`, which doesn't exist at check time; the
 * root tsconfig excludes `spike/tests-package`.
 */
import { NDArray } from "../../../dist/index.js";

const a = NDArray.fromArray([2, 3], [1, 2, 3, 4, 5, 6]);

// Valid calls resolve clean (no error):
a.sum();
a.sum(0);
a.sum(1, true);
a.add(NDArray.fromArray([2, 3], [6, 5, 4, 3, 2, 1]));

// M3: an out-of-range axis is a compile error AT the argument. If this ever
// stops erroring, `@ts-expect-error` becomes "unused" and tsc fails — so this
// pins the guard's survival through emit + post-rewrite, not just its presence.
// @ts-expect-error axis 9 is out of range for shape [2,3] (rank 2)
a.sum(9);
