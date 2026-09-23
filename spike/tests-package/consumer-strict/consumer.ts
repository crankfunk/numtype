/**
 * D2 (docs/release-0.3.0-spec.md) — second, STRICTER type-level package
 * smoke: `skipLibCheck: false` (see ./tsconfig.json) and no `@types/node`
 * anywhere in this repo's own devDependencies (project constraint — see
 * package.json). Exercises exactly the reachable-from-`index.d.ts` type
 * surface D2 fixed: `WNDArray` (D1, type-only export) and `ThreadedBackend`
 * (type-only, pre-existing export). Before the D2 fix, `ThreadedBackend`'s
 * declaration chain pulled `threaded.d.ts`'s
 * `import { Worker } from "node:worker_threads"` into the reachable set,
 * and `skipLibCheck: false` forces tsc to actually resolve that import —
 * TS2591 ("Cannot find module 'node:worker_threads'. Do you need to install
 * type definitions for node?") with no `@types/node` installed. Proven
 * non-vacuous during implementation: this file, unmodified, reproducibly
 * failed with that exact code on the pre-fix `threaded.ts` and passes
 * after — see docs/release-0.3.0-ergebnisse.md.
 *
 * Runs under `pnpm test:package`, after `build:dist`. Not part of the root
 * `pnpm check` corpus (needs a built `dist/`), same as ../consumer/.
 */
import type { NDArray, ThreadedBackend, WNDArray } from "../../../dist/index.js";

// D1: WNDArray is nameable as a type — its constructor stays private, so
// this is a type-position reference only (e.g. a helper's parameter type),
// never a new way to construct one.
declare function describeResident(arr: WNDArray<readonly [2, 3]>): string;

// D2: ThreadedBackend is nameable as a type without @types/node anywhere in
// scope or on disk.
declare function useThreaded(backend: ThreadedBackend): Promise<void>;

// A plain NDArray consumer stays green too (same baseline as ../consumer/).
declare const a: NDArray<readonly [2, 3]>;

// docs/typed-nested-array-spec.md D7: same `toNestedArray()` pin as
// ../consumer/ — this smoke doesn't set `noUncheckedIndexedAccess` either
// (see ./tsconfig.json), so the result is plain `number`, no cast.
const nested: number = a.toNestedArray()[0][1];

void a;
void describeResident;
void useThreaded;
void nested;
